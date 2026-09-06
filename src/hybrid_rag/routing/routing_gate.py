"""Module 1a - the Token Routing Gatekeeper.

The single choke point between Level 0 (free, local) and Level 2 (Claude Opus
5, metered). Everything that could burn cloud quota passes through
`RoutingGate.decide()` first, and every decision - including the ones that
stayed local - lands in the governance ledger.

A three-tier waterfall, and the tier numbers are latency budgets:

  Tier 1   0 ms      explicit `@opus` / `@local` tags, then a regex heuristic
                     table. Deterministic, inspectable, no model call.
  Tier 2   0 ms      calibrated retrieval confidence, already computed by the
                     RAG pass that ran before this. Below threshold escalates.
  Tier 3   ~300 ms   only when tiers 1 and 2 are both inconclusive: ask the
                     3B generator that is ALREADY RESIDENT in VRAM for a
                     one-token verdict.

Tier 3 used to be a separate 1.5B classifier. It was consolidated onto the
resident generator because a one-token vote is less inference than a JSON vote,
and because it deletes a model, a config field and a class of bug where the two
models disagree.

The reason originally given - that the two could not co-reside in 5.2 GB and
Ollama evicted one per query - was never measured and is wrong. Three models
totalling 4.1 GB run at 100% GPU on this card. Decision 23 corrects it.

Two vetoes sit after the waterfall and can overrule it:

  Governance  no verified GovernanceEnvelope, no crossing the boundary.
  Budget      the rolling window holds a hard veto over the router.

Both record `degraded_local` rather than silently downgrading, because "we
answered locally because the local answer was good" and "we answered locally
because we were out of quota" are different facts an auditor must distinguish.

Fail-safe direction is always LOCAL. A dead Ollama, an unparseable vote or a
corrupt ledger costs latency, never tokens.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

from hybrid_rag.config import RoutingSettings
from hybrid_rag.governance.audit import (
    AuditRecord,
    AuditSink,
    GuardrailVerdict,
    PIIStatus,
    SourceAttribution,
    TokenSource,
    hash_prompt,
)
from hybrid_rag.governance.middleware import EnvelopeVerifier, GovernanceEnvelope
from hybrid_rag.llm.ollama_client import LocalInferenceError, TextGenerator
from hybrid_rag.routing.budget import (
    BudgetDecision,
    Priority,
    RollingWindowBudget,
    estimate_tokens,
)


class Tier(StrEnum):
    LOCAL = "local"
    OPUS = "opus"


class Stage(StrEnum):
    FORCED = "forced_local"
    EXPLICIT_TAG = "tier1_explicit_tag"
    HEURISTIC = "tier1_heuristic"
    RETRIEVAL_CONFIDENCE = "tier2_retrieval_confidence"
    SLM_VOTE = "tier3_resident_slm"
    GOVERNANCE_VETO = "governance_veto"
    BUDGET_VETO = "budget_veto"


@dataclass(slots=True, frozen=True)
class RetrievalSignal:
    """What the local RAG engine reported. Structural, not an import of the engine."""

    confidence: float
    n_results: int
    top_score: float = 0.0
    sources: Sequence[SourceAttribution] = field(default_factory=tuple)

    @classmethod
    def empty(cls) -> RetrievalSignal:
        return cls(confidence=0.0, n_results=0, top_score=0.0, sources=())


class Signal(BaseModel):
    name: str
    fired: bool
    weight: float
    detail: str = ""


class RoutingDecision(BaseModel):
    """The complete, auditable reason a request went where it went."""

    trace_id: str
    tier: Tier
    decided_by: Stage
    reason: str
    confidence: float
    heuristic_score: float
    retrieval_confidence: float | None = None
    slm_vote: str | None = None
    degraded: bool = False
    envelope_id: str | None = None
    reservation_id: str | None = None
    estimated_tokens: int = 0
    budget: BudgetDecision | None = None
    signals: list[Signal] = Field(default_factory=list)
    latency_ms: float = 0.0
    audit_event_id: str | None = None

    @property
    def escalated(self) -> bool:
        return self.tier is Tier.OPUS


# --------------------------------------------------------------------------
# Stage 1 - lexical heuristics
#
# Weights are additive and clamped to [0, 1]. They are intentionally boring
# and inspectable: a regex table beats a learned classifier here because the
# whole point of this stage is to be explainable in an audit and to cost
# nothing. Tune the numbers, do not replace the mechanism.
# --------------------------------------------------------------------------

_ESCALATE_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"\b(compare|versus|vs\.?|trade[- ]?offs?|differ(ence|s)?)\b", 0.22, "comparative"),
    (r"\b(why|root cause|explain how|walk me through|derive|prove)\b", 0.20, "causal"),
    (r"\b(design|architect(ure)?|refactor|migrat(e|ion)|strategy)\b", 0.22, "architectural"),
    (
        r"\b(synthes(is|ize)|generate (a )?dataset|training (data|pairs)|distill)\b",
        0.45,
        "synthesis",
    ),
    (
        r"\b(multi[- ]?hop|across (all|multiple)|for each .* then|chain of thought)\b",
        0.30,
        "multihop",
    ),
    (
        r"\b(adjudicat|arbitrat|which answer is (better|correct)|grade|judge)\b",
        0.35,
        "adjudication",
    ),
    (
        r"\b(implement|write) (a |the )?(full|complete|production[- ]ready|end[- ]to[- ]end)\b",
        0.28,
        "large_codegen",
    ),
    (r"\?[^?]*\?", 0.15, "multi_question"),
)

_LOCAL_PATTERNS: tuple[tuple[str, float, str], ...] = (
    (r"^\s*(what|who|when|where) (is|are|was|were)\b", -0.25, "lookup"),
    (r"\b(define|definition of|list|show me|find|search for|grep)\b", -0.20, "retrieval_only"),
    (r"\b(summar(y|ise|ize)|tl;?dr|rephrase|translate|format|rename)\b", -0.22, "transform"),
    (r"\b(typo|lint|spelling|indent)\b", -0.25, "trivial"),
)

# Tier 1a. An operator who already knows where a query belongs should not have
# to argue with a classifier about it. A bare tag anywhere in the text wins
# outright, before any scoring runs.
_TAG_OPUS = re.compile(r"(?i)(?:^|\s)(?:@opus|\[\[opus\]\]|#opus)(?:\s|$)")
_TAG_LOCAL = re.compile(r"(?i)(?:^|\s)(?:@local|\[\[local\]\]|#local)(?:\s|$)")

_LONG_QUERY_CHARS = 900
_VERY_LONG_QUERY_CHARS = 2400


def explicit_tag(query: str) -> Tier | None:
    """Tier 1a. Returns the tier an explicit tag demands, or None."""
    if _TAG_LOCAL.search(query):
        return Tier.LOCAL
    if _TAG_OPUS.search(query):
        return Tier.OPUS
    return None


def score_heuristics(query: str) -> tuple[float, list[Signal]]:
    """Deterministic lexical escalation score in [0, 1]."""
    signals: list[Signal] = []
    score = 0.0
    lowered = query.lower()

    for pattern, weight, name in _ESCALATE_PATTERNS + _LOCAL_PATTERNS:
        match = re.search(pattern, lowered)
        fired = match is not None
        if fired:
            score += weight
        signals.append(
            Signal(
                name=name,
                fired=fired,
                weight=weight,
                detail=match.group(0)[:60] if match else "",
            )
        )

    n = len(query)
    if n >= _VERY_LONG_QUERY_CHARS:
        score += 0.20
        signals.append(Signal(name="very_long_input", fired=True, weight=0.20, detail=f"{n} chars"))
    elif n >= _LONG_QUERY_CHARS:
        score += 0.10
        signals.append(Signal(name="long_input", fired=True, weight=0.10, detail=f"{n} chars"))

    return max(0.0, min(1.0, score)), signals


_ROUTER_SYSTEM = "Answer with exactly one letter. Never explain."

# One token, not a JSON object. Structured output would need 15+ tokens and a
# parse step; a single letter is one forward pass, and "L" and "O" are each a
# single token in every tokenizer this will run against.
_ROUTER_TEMPLATE = """Decide who should answer this request.

L = a small local model with retrieved context can handle it: fact lookup,
    summarization, extraction, formatting, single-file code edits.
O = it needs a frontier model: multi-hop decomposition across sources,
    architectural synthesis, generating training data, judging answers.

Request:
\"\"\"{query}\"\"\"

One letter, L or O:"""


class RoutingGate:
    """Deterministic dependency injection: nothing is constructed in here."""

    def __init__(
        self,
        *,
        settings: RoutingSettings,
        budget: RollingWindowBudget,
        audit: AuditSink,
        generator: TextGenerator | None = None,
        verifier: EnvelopeVerifier | None = None,
        slm_model: str | None = None,
        slm_max_tokens: int = 1,
        agent_id: str = "routing_gate",
    ) -> None:
        self._s = settings
        self._budget = budget
        self._audit = audit
        self._generator = generator
        self._verifier = verifier
        self._slm_model = slm_model
        self._slm_max_tokens = slm_max_tokens
        self._agent_id = agent_id

    def set_slm_model(self, model: str | None) -> None:
        """Re-point tier 3's voter at another local model.

        The gate holds the model NAME, resolved once at construction, not the
        settings object. So swapping the resident generator without calling
        this leaves the waterfall voting with a model that may no longer be
        loaded - which does not fail loudly, it just adds a cold load to the
        one stage of routing that is supposed to be cheap.
        """
        self._slm_model = model

    async def decide(
        self,
        query: str,
        *,
        retrieval: RetrievalSignal | None = None,
        envelope: GovernanceEnvelope | None = None,
        trace_id: str | None = None,
        priority: Priority = Priority.INTERACTIVE,
        output_allowance: int = 2048,
        extra_context_chars: int = 0,
    ) -> RoutingDecision:
        t0 = time.perf_counter()
        trace_id = trace_id or str(uuid.uuid4())
        retrieval = retrieval or RetrievalSignal.empty()

        # Route on the redacted text when one exists. The routing decision is
        # then made on exactly the bytes that would cross the boundary, and the
        # local SLM never sees a secret the middleware just masked. The raw
        # query is still what gets hashed and verified against.
        analysis_text = envelope.safe_text if envelope is not None else query

        heuristic_score, signals = score_heuristics(analysis_text)
        stage, want_opus, reason = self._pre_budget_verdict(
            analysis_text, heuristic_score, retrieval
        )

        slm_vote: str | None = None
        if stage is Stage.SLM_VOTE:
            slm_vote, vote_signal = await self._slm_vote(analysis_text)
            signals.append(vote_signal)
            want_opus = slm_vote == "OPUS"
            reason = f"SLM router voted {slm_vote or 'UNPARSEABLE -> LOCAL'}"

        decision = RoutingDecision(
            trace_id=trace_id,
            tier=Tier.OPUS if want_opus else Tier.LOCAL,
            decided_by=stage,
            reason=reason,
            confidence=self._decision_confidence(want_opus, heuristic_score, retrieval),
            heuristic_score=heuristic_score,
            retrieval_confidence=retrieval.confidence if retrieval.n_results else None,
            slm_vote=slm_vote,
            envelope_id=envelope.envelope_id if envelope else None,
            signals=signals,
        )

        if want_opus:
            # Invariant 1. Checked before the budget so a payload that can never
            # legally leave does not consume a reservation on its way to being
            # refused.
            veto = self._governance_veto(query, envelope)
            if veto is not None:
                decision.tier = Tier.LOCAL
                decision.decided_by = Stage.GOVERNANCE_VETO
                decision.degraded = True
                decision.reason = veto
                want_opus = False

        if want_opus:
            estimated = estimate_tokens(
                analysis_text,
                chars_per_token=self._s.chars_per_token,
                output_allowance=output_allowance,
            ) + int(extra_context_chars / max(self._s.chars_per_token, 1.0))
            verdict = self._budget.reserve(estimated, trace_id=trace_id, priority=priority)
            decision.budget = verdict
            decision.estimated_tokens = estimated
            if verdict.allowed:
                decision.reservation_id = verdict.reservation_id
            else:
                # Stage 4 veto. Downgrade, but say so loudly.
                decision.tier = Tier.LOCAL
                decision.decided_by = Stage.BUDGET_VETO
                decision.degraded = True
                decision.reason = f"escalation vetoed by budget: {verdict.reason}"

        decision.latency_ms = (time.perf_counter() - t0) * 1000.0
        decision.audit_event_id = self._emit_audit(
            query, decision, retrieval, envelope
        ).event_id
        return decision

    # ----------------------------------------------------------- internals

    def _governance_veto(
        self, query: str, envelope: GovernanceEnvelope | None
    ) -> str | None:
        """None means clear to escalate. A string is the reason it may not."""
        if not self._s.require_envelope:
            return None
        if envelope is None:
            return "no governance envelope; nothing may cross the token boundary uninspected"
        if not envelope.cleared:
            return f"governance blocked the payload: {envelope.block_reason}"
        if self._verifier is not None and not self._verifier.verify(envelope, original=query):
            return "governance envelope did not verify against this prompt"
        return None

    def _pre_budget_verdict(
        self,
        query: str,
        heuristic_score: float,
        retrieval: RetrievalSignal,
    ) -> tuple[Stage, bool, str]:
        if self._s.force_local:
            return Stage.FORCED, False, "force_local is set; Level 2 is disabled"

        # Tier 1a: an explicit tag is an instruction, not a hint. It outranks
        # every score below it - including, deliberately, a low retrieval
        # confidence that would otherwise escalate.
        tag = explicit_tag(query)
        if tag is not None:
            return Stage.EXPLICIT_TAG, tag is Tier.OPUS, f"explicit tag pinned this to {tag}"

        if heuristic_score >= self._s.heuristic_hard_escalate:
            return (
                Stage.HEURISTIC,
                True,
                f"heuristic score {heuristic_score:.2f} >= {self._s.heuristic_hard_escalate:.2f}",
            )

        if retrieval.n_results and retrieval.confidence < self._s.escalation_threshold:
            return (
                Stage.RETRIEVAL_CONFIDENCE,
                True,
                f"retrieval confidence {retrieval.confidence:.3f} < "
                f"{self._s.escalation_threshold:.2f}",
            )

        if heuristic_score <= self._s.heuristic_hard_local:
            return (
                Stage.HEURISTIC,
                False,
                f"heuristic score {heuristic_score:.2f} <= {self._s.heuristic_hard_local:.2f}",
            )

        if not self._s.slm_vote_enabled or self._generator is None:
            return (
                Stage.HEURISTIC,
                False,
                "tier 3 unavailable; defaulting to LOCAL (fail-safe)",
            )

        return Stage.SLM_VOTE, False, "tiers 1 and 2 inconclusive; deferring to the resident model"

    async def _slm_vote(self, query: str) -> tuple[str | None, Signal]:
        """Tier 3. Asks the model already pinned in VRAM for one token.

        `model=self._slm_model` resolves to the generator, so this reuses the
        resident weights and never triggers a load. Passing a different model
        name here loads a second model; do that deliberately or not at all.
        """
        if self._generator is None:
            return None, Signal(name="slm_vote", fired=False, weight=0.0, detail="no generator")
        prompt = _ROUTER_TEMPLATE.format(query=query[:4000])
        try:
            completion = await self._generator.generate(
                prompt,
                model=self._slm_model,
                max_tokens=self._slm_max_tokens,
                system=_ROUTER_SYSTEM,
            )
        except LocalInferenceError as exc:
            return None, Signal(
                name="slm_vote", fired=False, weight=0.0, detail=f"failed: {exc}"[:120]
            )

        # First meaningful character decides. A model that ignored the
        # instruction and started a sentence yields neither L nor O, which is
        # unparseable, which fails safe to LOCAL.
        first = next((c for c in completion.text.strip().upper() if c.isalpha()), "")
        vote = {"O": "OPUS", "L": "LOCAL"}.get(first)
        return vote, Signal(
            name="slm_vote",
            fired=vote == "OPUS",
            weight=0.0,
            detail=f"{vote or 'unparseable'} ({completion.latency_ms:.0f} ms)",
        )

    @staticmethod
    def _decision_confidence(
        want_opus: bool, heuristic_score: float, retrieval: RetrievalSignal
    ) -> float:
        if want_opus:
            return round(max(heuristic_score, 1.0 - retrieval.confidence), 4)
        if retrieval.n_results:
            return round(retrieval.confidence, 4)
        return round(1.0 - heuristic_score, 4)

    def _emit_audit(
        self,
        query: str,
        decision: RoutingDecision,
        retrieval: RetrievalSignal,
        envelope: GovernanceEnvelope | None,
    ) -> AuditRecord:
        flags: list[str] = [f"tier:{decision.tier}", f"stage:{decision.decided_by}"]
        if envelope is not None:
            flags.extend(envelope.compliance_flags())
        if decision.degraded:
            flags.append("degraded_local")
        if decision.budget and not decision.budget.allowed:
            flags.extend(decision.budget.compliance_flags)

        return self._audit.emit(
            AuditRecord(
                trace_id=decision.trace_id,
                stage="routing_decision",
                agent_id=self._agent_id,
                tier=str(decision.tier),
                model_id=self._slm_model if decision.slm_vote else None,
                prompt_hash=hash_prompt(query),
                prompt_chars=len(query),
                pii_status=envelope.pii_status if envelope else PIIStatus.NOT_EVALUATED,
                guardrail_verdict=(
                    envelope.guardrail_verdict if envelope else GuardrailVerdict.NOT_EVALUATED
                ),
                token_source=TokenSource.LOCAL_OLLAMA if decision.slm_vote else TokenSource.NONE,
                latency_ms=decision.latency_ms,
                confidence=decision.confidence,
                decision_reason=decision.reason,
                retrieval_sources=list(retrieval.sources),
                compliance_flags=flags,
                extra={
                    "heuristic_score": decision.heuristic_score,
                    "retrieval_confidence": decision.retrieval_confidence,
                    "slm_vote": decision.slm_vote,
                    "estimated_tokens": decision.estimated_tokens,
                    "reservation_id": decision.reservation_id,
                    "envelope_id": decision.envelope_id,
                    "fired_signals": [s.name for s in decision.signals if s.fired],
                },
            )
        )

    # --------------------------------------------------------- settlement

    def settle(self, decision: RoutingDecision, *, tokens_in: int, tokens_out: int) -> None:
        """Convert a reservation into measured spend. Idempotent per reservation."""
        if decision.reservation_id:
            self._budget.commit(decision.reservation_id, tokens_in=tokens_in, tokens_out=tokens_out)
            decision.reservation_id = None

    def abort(self, decision: RoutingDecision, *, reason: str = "unused") -> None:
        """Hand the reserved capacity back - the cloud call never happened."""
        if decision.reservation_id:
            self._budget.release(decision.reservation_id, reason=reason)
            decision.reservation_id = None

    @asynccontextmanager
    async def escalation(self, decision: RoutingDecision) -> AsyncIterator[_Settlement]:
        """Guarantees a reservation is settled even if the cloud call raises.

        async with gate.escalation(decision) as spend:
            result = await opus.complete(...)
            spend.record(result.tokens_in, result.tokens_out)
        """
        settlement = _Settlement()
        try:
            yield settlement
        except BaseException:
            self.abort(decision, reason="exception during escalation")
            raise
        else:
            if settlement.recorded:
                self.settle(
                    decision,
                    tokens_in=settlement.tokens_in,
                    tokens_out=settlement.tokens_out,
                )
            else:
                self.abort(decision, reason="no spend recorded")


@dataclass(slots=True)
class _Settlement:
    tokens_in: int = 0
    tokens_out: int = 0
    recorded: bool = False

    def record(self, tokens_in: int, tokens_out: int) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.recorded = True
