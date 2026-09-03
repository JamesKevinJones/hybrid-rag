from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_rag.config import AuditSettings, BudgetSettings, RoutingSettings
from hybrid_rag.governance.audit import JsonlAuditSink, SourceAttribution
from hybrid_rag.governance.middleware import GovernanceEnvelope, GovernanceMiddleware
from hybrid_rag.llm.ollama_client import LocalInferenceError
from hybrid_rag.routing.budget import Priority, RollingWindowBudget, Verdict
from hybrid_rag.routing.routing_gate import (
    RetrievalSignal,
    RoutingGate,
    Stage,
    Tier,
    explicit_tag,
    score_heuristics,
)

STRONG = RetrievalSignal(
    confidence=0.91,
    n_results=4,
    top_score=0.95,
    sources=(SourceAttribution(doc_id="d", chunk_id="d::0", score=0.95, rank=0),),
)
WEAK = RetrievalSignal(confidence=0.41, n_results=4, top_score=0.55)

# One middleware instance for the whole module: the seal key is per-instance, so
# sharing it is what lets the gate verify envelopes these tests mint.
GOV = GovernanceMiddleware()


async def envelope_for(query: str) -> GovernanceEnvelope:
    """Mint a real, sealed envelope the way the pipeline would."""
    envelope, _ = await GOV.ingress(query)
    return envelope


def _budget(tmp_path: Path, **overrides: object) -> RollingWindowBudget:
    base: dict[str, object] = {
        "ledger_path": tmp_path / "budget.jsonl",
        "max_messages_per_window": 10,
        "max_tokens_per_window": 100_000,
        "weekly_message_cap": 100,
    }
    base.update(overrides)
    return RollingWindowBudget(BudgetSettings(**base))  # type: ignore[arg-type]


def _gate(tmp_path: Path, audit, generator=None, **routing: object) -> RoutingGate:
    return RoutingGate(
        settings=RoutingSettings(**routing),  # type: ignore[arg-type]
        budget=_budget(tmp_path),
        audit=audit,
        generator=generator,
        verifier=GOV,
        slm_model="stub-resident",
    )


# --------------------------------------------------------------- heuristics


@pytest.mark.parametrize(
    "query",
    [
        "Generate a dataset of chain-of-thought pairs for the retrieval reader",
        "Compare the trade-offs between QLoRA and full fine-tuning and explain why",
        "Which answer is better, and grade both against the rubric?",
    ],
)
def test_heuristics_flag_frontier_work(query: str) -> None:
    score, signals = score_heuristics(query)
    assert score >= 0.35
    assert any(s.fired for s in signals)


@pytest.mark.parametrize(
    "query",
    ["What is NF4 quantization?", "List the four NIST AI RMF functions", "Fix this typo"],
)
def test_heuristics_keep_lookups_local(query: str) -> None:
    score, _ = score_heuristics(query)
    assert score <= 0.15


def test_heuristic_score_is_clamped() -> None:
    score, _ = score_heuristics(
        "design an architecture to synthesize a training dataset "
        "and compare why each multi-hop approach differs " * 4
    )
    assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------ staging


async def test_force_local_short_circuits_everything(tmp_path, audit, generator) -> None:
    gate = _gate(tmp_path, audit, generator, force_local=True)
    decision = await gate.decide("Synthesize a full training dataset", retrieval=WEAK)

    assert decision.tier is Tier.LOCAL
    assert decision.decided_by is Stage.FORCED
    assert decision.reservation_id is None
    assert generator.prompts == [], "the SLM must not be consulted when Level 2 is disabled"


async def test_low_retrieval_confidence_escalates(tmp_path, audit, generator) -> None:
    gate = _gate(tmp_path, audit, generator)
    query = "Summarize the governance section"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )

    assert decision.tier is Tier.OPUS
    assert decision.decided_by is Stage.RETRIEVAL_CONFIDENCE
    assert decision.reservation_id is not None
    assert decision.estimated_tokens > 0


async def test_strong_retrieval_plus_lookup_phrasing_stays_local(
    tmp_path, audit, generator
) -> None:
    gate = _gate(tmp_path, audit, generator)
    decision = await gate.decide("What is the Measure function?", retrieval=STRONG)

    assert decision.tier is Tier.LOCAL
    assert decision.decided_by is Stage.HEURISTIC
    assert generator.prompts == []


async def test_ambiguous_query_defers_to_slm_vote(tmp_path, audit, generator) -> None:
    generator.default = "O"
    gate = _gate(tmp_path, audit, generator)
    query = "Explain how the adapter rank interacts with the paged optimizer"
    decision = await gate.decide(
        query, retrieval=STRONG, envelope=await envelope_for(query)
    )

    assert decision.decided_by is Stage.SLM_VOTE
    assert decision.slm_vote == "OPUS"
    assert decision.tier is Tier.OPUS
    assert len(generator.prompts) == 1


async def test_slm_failure_fails_safe_to_local(tmp_path, audit, generator) -> None:
    generator.raises = LocalInferenceError("ollama is not running")
    gate = _gate(tmp_path, audit, generator)
    decision = await gate.decide(
        "Explain how the adapter rank interacts with the paged optimizer", retrieval=STRONG
    )

    assert decision.tier is Tier.LOCAL, "a dead local model must never cost cloud tokens"
    assert decision.slm_vote is None


async def test_unparseable_vote_fails_safe_to_local(tmp_path, audit, generator) -> None:
    # Ignored the one-letter instruction and started a sentence instead.
    generator.default = "I think you should probably ask a bigger model about this"
    gate = _gate(tmp_path, audit, generator)
    decision = await gate.decide(
        "Explain how the adapter rank interacts with the paged optimizer", retrieval=STRONG
    )
    assert decision.tier is Tier.LOCAL


# ------------------------------------------------------------------- budget


async def test_budget_veto_downgrades_and_flags_degraded(tmp_path, audit, generator) -> None:
    gate = RoutingGate(
        settings=RoutingSettings(),
        budget=_budget(tmp_path, max_messages_per_window=0),
        audit=audit,
        generator=generator,
        verifier=GOV,
        slm_model="stub-resident",
    )
    query = "Synthesize a training dataset"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )

    assert decision.tier is Tier.LOCAL
    assert decision.degraded is True
    assert decision.decided_by is Stage.BUDGET_VETO
    assert decision.budget is not None and not decision.budget.allowed
    assert "degraded_local" in audit.records[-1].compliance_flags


async def test_batch_priority_cannot_touch_the_interactive_reserve(tmp_path, audit) -> None:
    budget = _budget(tmp_path, max_messages_per_window=10)
    for _ in range(8):
        d = budget.reserve(10, trace_id="t", priority=Priority.INTERACTIVE)
        assert d.allowed
        budget.commit(d.reservation_id or "", tokens_in=5, tokens_out=5)

    batch = budget.reserve(10, trace_id="t", priority=Priority.BATCH)
    assert not batch.allowed
    assert batch.verdict is Verdict.DENY_RESERVE_POOL

    interactive = budget.reserve(10, trace_id="t", priority=Priority.INTERACTIVE)
    assert interactive.allowed


async def test_reservation_is_released_when_the_cloud_call_raises(
    tmp_path, audit, generator
) -> None:
    gate = _gate(tmp_path, audit, generator)
    query = "Summarize the governance section"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )
    assert decision.reservation_id is not None

    with pytest.raises(RuntimeError):
        async with gate.escalation(decision):
            raise RuntimeError("Anthropic API timed out")

    assert decision.reservation_id is None, "capacity must go back on failure"


async def test_settlement_replaces_the_estimate_with_measured_spend(
    tmp_path, audit, generator
) -> None:
    budget = _budget(tmp_path)
    gate = RoutingGate(
        settings=RoutingSettings(),
        budget=budget,
        audit=audit,
        generator=generator,
        verifier=GOV,
        slm_model="stub-resident",
    )
    query = "Summarize the governance section"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )

    async with gate.escalation(decision) as spend:
        spend.record(1200, 800)

    snap = budget.snapshot()
    assert snap.tokens_used == 2000
    assert snap.messages_used == 1


# -------------------------------------------------------------------- audit


async def test_every_decision_is_logged_with_a_hash_chain(tmp_path, generator) -> None:
    sink = JsonlAuditSink(tmp_path / "audit.jsonl", schema_version=AuditSettings().schema_version)
    gate = RoutingGate(
        settings=RoutingSettings(),
        budget=_budget(tmp_path),
        audit=sink,
        generator=generator,
        verifier=GOV,
        slm_model="stub-resident",
    )
    await gate.decide("What is NF4?", retrieval=STRONG)
    await gate.decide("Synthesize a dataset of reasoning traces", retrieval=WEAK)

    lines = sink.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    intact, bad_line = sink.verify_chain()
    assert intact and bad_line == 0


async def test_tampering_with_the_ledger_is_detectable(tmp_path, generator) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path)
    gate = RoutingGate(
        settings=RoutingSettings(),
        budget=_budget(tmp_path),
        audit=sink,
        generator=generator,
        verifier=GOV,
        slm_model="stub-resident",
    )
    await gate.decide("What is NF4?", retrieval=STRONG)
    await gate.decide("What is NF4 again?", retrieval=STRONG)

    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["tier"] = "opus"  # rewrite history: claim the local decision went to Opus
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    intact, bad_line = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad_line == 1


async def test_prompt_text_is_never_written_to_the_ledger(tmp_path, generator) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path)
    gate = RoutingGate(
        settings=RoutingSettings(),
        budget=_budget(tmp_path),
        audit=sink,
        generator=generator,
        verifier=GOV,
        slm_model="stub-resident",
    )
    secret = "my api key is sk-ant-notarealkey-000"
    await gate.decide(f"What is {secret}?", retrieval=STRONG)

    assert secret not in path.read_text(encoding="utf-8")


# ------------------------------------------------------- invariant 1 (Module 2)


async def test_escalation_without_an_envelope_is_vetoed(tmp_path, audit, generator) -> None:
    gate = _gate(tmp_path, audit, generator)
    decision = await gate.decide("Summarize the governance section", retrieval=WEAK)

    assert decision.tier is Tier.LOCAL
    assert decision.decided_by is Stage.GOVERNANCE_VETO
    assert decision.degraded is True
    assert decision.reservation_id is None, "a vetoed request must not consume capacity"
    assert "uninspected" in decision.reason


async def test_an_envelope_minted_for_a_different_prompt_is_rejected(
    tmp_path, audit, generator
) -> None:
    gate = _gate(tmp_path, audit, generator)
    decision = await gate.decide(
        "Summarize the governance section",
        retrieval=WEAK,
        envelope=await envelope_for("a completely unrelated question"),
    )

    assert decision.decided_by is Stage.GOVERNANCE_VETO
    assert "did not verify" in decision.reason


async def test_a_blocked_envelope_cannot_escalate(tmp_path, audit, generator) -> None:
    gate = _gate(tmp_path, audit, generator)
    query = "Summarize the governance section using sk-ant-api03-AAAABBBBCCCCDDDDEEEE"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )

    assert decision.tier is Tier.LOCAL
    assert decision.decided_by is Stage.GOVERNANCE_VETO
    assert "rotate it" in decision.reason


async def test_the_router_never_sees_the_redacted_secret(tmp_path, audit, generator) -> None:
    generator.default = "O"
    gate = _gate(tmp_path, audit, generator)
    query = "Explain how kevin@example.com should configure the paged optimizer"
    await gate.decide(query, retrieval=STRONG, envelope=await envelope_for(query))

    assert generator.prompts, "this query should have reached the SLM vote"
    assert "kevin@example.com" not in generator.prompts[0]
    assert "<EMAIL_ADDRESS_1>" in generator.prompts[0]


async def test_governance_verdicts_land_in_the_routing_audit_record(
    tmp_path, audit, generator
) -> None:
    gate = _gate(tmp_path, audit, generator)
    query = "Summarize the governance section for kevin@example.com"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )

    record = audit.records[-1]
    assert record.pii_status == "scrubbed"
    assert record.guardrail_verdict == "allow"
    assert record.extra["envelope_id"] == decision.envelope_id
    assert "pii:scrubbed" in record.compliance_flags


async def test_disabling_the_requirement_is_possible_but_explicit(
    tmp_path, audit, generator
) -> None:
    gate = _gate(tmp_path, audit, generator, require_envelope=False)
    decision = await gate.decide("Summarize the governance section", retrieval=WEAK)

    assert decision.tier is Tier.OPUS
    assert decision.decided_by is Stage.RETRIEVAL_CONFIDENCE


# ------------------------------------------------ tier 1a: explicit tags


async def test_explicit_opus_tag_overrides_strong_retrieval(
    tmp_path, audit, generator
) -> None:
    gate = _gate(tmp_path, audit, generator)
    query = "@opus rewrite the whole retrieval layer"
    decision = await gate.decide(
        query, retrieval=STRONG, envelope=await envelope_for(query)
    )

    assert decision.tier is Tier.OPUS
    assert decision.decided_by is Stage.EXPLICIT_TAG
    assert generator.calls == [], "an explicit tag must not cost a model call"


async def test_explicit_local_tag_overrides_low_confidence(
    tmp_path, audit, generator
) -> None:
    """The one case where a tag beats the retrieval-confidence escalation."""
    gate = _gate(tmp_path, audit, generator)
    query = "@local just grep for the config key"
    decision = await gate.decide(
        query, retrieval=WEAK, envelope=await envelope_for(query)
    )

    assert decision.tier is Tier.LOCAL
    assert decision.decided_by is Stage.EXPLICIT_TAG
    assert decision.reservation_id is None


@pytest.mark.parametrize(
    "query",
    [
        "an email@opus.example.com is not a tag",
        "the opus dataset is large",
        "localhost is not a tag either",
    ],
)
def test_tag_matching_does_not_fire_on_ordinary_words(query: str) -> None:
    assert explicit_tag(query) is None


# ------------------------------------------- tier 3: the resident model


async def test_tier3_reuses_the_resident_model_with_a_one_token_cap(
    tmp_path, audit, generator
) -> None:
    """The whole point of the consolidation: no second model, no eviction."""
    generator.default = "O"
    gate = _gate(tmp_path, audit, generator)
    query = "Explain how the adapter rank interacts with the paged optimizer"
    await gate.decide(query, retrieval=STRONG, envelope=await envelope_for(query))

    assert len(generator.calls) == 1
    assert generator.calls[0]["model"] == "stub-resident"
    assert generator.calls[0]["max_tokens"] == 1, "one token, not a JSON object"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("O", "OPUS"), ("L", "LOCAL"), ("o", "OPUS"), (" L\n", "LOCAL"), ("", None), ("?", None)],
)
async def test_single_letter_votes_are_parsed(
    tmp_path, audit, generator, reply: str, expected: str | None
) -> None:
    generator.default = reply
    gate = _gate(tmp_path, audit, generator)
    query = "Explain how the adapter rank interacts with the paged optimizer"
    decision = await gate.decide(
        query, retrieval=STRONG, envelope=await envelope_for(query)
    )
    assert decision.slm_vote == expected
