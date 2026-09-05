from __future__ import annotations

import pytest

from hybrid_rag.config import EvalSettings, PromotionSettings
from hybrid_rag.eval.eval_harness import EvalCase, EvalHarness, GroundednessScorer, Response
from hybrid_rag.eval.promotion_gate import (
    GgufArtifact,
    PromotionError,
    PromotionGate,
)

CONTEXT = [
    "The rolling window budget takes a reservation before the cloud call and "
    "settles it with measured spend afterwards.",
    "Stage four outranks the router: a veto is recorded as degraded_local.",
]
GROUNDED = "Stage four outranks the router and a veto is recorded as degraded_local."
INVENTED = "It replicates every audit event to a Kafka cluster spanning three regions."

CASES = [EvalCase(case_id="a", query="q1"), EvalCase(case_id="b", query="q2")]


class FixedResponder:
    def __init__(self, label: str, text: str, latency_ms: float) -> None:
        self._label = label
        self._text = text
        self._latency = latency_ms

    @property
    def label(self) -> str:
        return self._label

    async def respond(self, case: EvalCase) -> Response:
        del case
        return Response(
            text=self._text,
            context_chunks=CONTEXT,
            tokens_in=100,
            tokens_out=40,
            latency_ms=self._latency,
        )


class RecordingRunner:
    """Stands in for the ollama CLI. Nothing shells out during tests."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self._fail = fail

    def __call__(self, argv: list[str]) -> str:
        self.calls.append(argv)
        if self._fail:
            raise PromotionError("ollama exited 1: model not found")
        return "success"


@pytest.fixture
def gate(embedder):
    def _gate(runner: RecordingRunner | None = None, **overrides: object) -> PromotionGate:
        eval_settings = EvalSettings()
        harness = EvalHarness(
            settings=eval_settings, scorer=GroundednessScorer(embedder, eval_settings)
        )
        return PromotionGate(
            settings=PromotionSettings(**overrides),  # type: ignore[arg-type]
            harness=harness,
            runner=runner or RecordingRunner(),
        )

    return _gate


# ------------------------------------------------------------------ verdicts


async def test_a_matching_candidate_is_eligible(gate) -> None:
    verdict = await gate().evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    assert verdict.eligible
    assert verdict.promoted
    assert all(c.passed for c in verdict.criteria)


async def test_an_ungrounded_candidate_fails_parity(gate) -> None:
    verdict = await gate().evaluate(
        candidate=FixedResponder("candidate", INVENTED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    assert not verdict.eligible
    parity = next(c for c in verdict.criteria if c.name == "groundedness_parity")
    assert not parity.passed
    assert parity.observed == 0.0


async def test_an_accurate_but_slow_candidate_fails_the_sla(gate) -> None:
    """Accuracy does not buy latency. Gated on p50 per the Module 5 spec."""
    verdict = await gate().evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 3000.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    assert not verdict.eligible
    latency = next(c for c in verdict.criteria if c.name == "latency_p50_ms")
    assert not latency.passed
    assert latency.threshold == 1200.0


async def test_instruction_precision_is_an_absolute_floor(gate) -> None:
    """A baseline that also broke the rule does not excuse breaking the rule."""
    cases = [EvalCase(case_id="a", query="q", must_not_include=["degraded_local"])]
    verdict = await gate().evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=cases,
    )
    instr = next(c for c in verdict.criteria if c.name == "instruction_precision")
    assert not instr.passed
    assert not verdict.eligible


async def test_a_worthless_baseline_cannot_make_anything_eligible(gate) -> None:
    """Parity against zero is undefined; refusing beats dividing by zero."""
    verdict = await gate().evaluate(
        candidate=FixedResponder("candidate", INVENTED, 300.0),
        baseline=FixedResponder("opus-baseline", INVENTED, 4000.0),
        cases=CASES,
    )
    parity = next(c for c in verdict.criteria if c.name == "groundedness_parity")
    assert not parity.passed
    assert "undefined" in parity.detail


async def test_an_empty_suite_refuses_to_promote(gate) -> None:
    verdict = await gate().evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=[],
    )
    assert not verdict.eligible
    assert "empty suite" in verdict.reason


async def test_the_threshold_is_configurable(gate) -> None:
    strict = gate(min_groundedness_parity=1.01)
    verdict = await strict.evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    assert not verdict.eligible


# ---------------------------------------------------------------- promotion


async def test_promotion_is_a_dry_run_unless_apply_is_passed(gate) -> None:
    runner = RecordingRunner()
    g = gate(runner)
    verdict = await g.evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    verdict = await g.promote(verdict, model_name="cand-v1")

    assert verdict.eligible
    assert verdict.applied is False
    assert runner.calls == [], "an eval run must not silently change what the host serves"
    assert "dry run" in verdict.reason


async def test_applying_repoints_the_serving_alias(gate) -> None:
    runner = RecordingRunner()
    g = gate(runner)
    verdict = await g.evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    verdict = await g.promote(verdict, model_name="cand-v1", apply=True)

    assert verdict.applied is True
    assert runner.calls == [["ollama", "cp", "cand-v1", "hybrid-rag-serving"]]


async def test_an_ineligible_candidate_is_never_applied(gate) -> None:
    runner = RecordingRunner()
    g = gate(runner)
    verdict = await g.evaluate(
        candidate=FixedResponder("candidate", INVENTED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    verdict = await g.promote(verdict, model_name="cand-v1", apply=True)

    assert verdict.applied is False
    assert runner.calls == []


async def test_a_failing_alias_update_raises_rather_than_lying(gate) -> None:
    g = gate(RecordingRunner(fail=True))
    verdict = await g.evaluate(
        candidate=FixedResponder("candidate", GROUNDED, 300.0),
        baseline=FixedResponder("opus-baseline", GROUNDED, 4000.0),
        cases=CASES,
    )
    with pytest.raises(PromotionError, match="failed to update alias"):
        await g.promote(verdict, model_name="cand-v1", apply=True)


# --------------------------------------------------------------- GGUF intake


def test_discover_finds_the_single_gguf(tmp_path) -> None:
    (tmp_path / "q4_k_m.gguf").write_bytes(b"stub")
    (tmp_path / "Modelfile").write_text("FROM ./q4_k_m.gguf\n", encoding="utf-8")

    artifact = GgufArtifact.discover(tmp_path)
    assert artifact.gguf_path.name == "q4_k_m.gguf"
    assert artifact.modelfile_path is not None


def test_discover_refuses_an_ambiguous_directory(tmp_path) -> None:
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "b.gguf").write_bytes(b"x")
    with pytest.raises(PromotionError, match="ambiguous"):
        GgufArtifact.discover(tmp_path)


def test_discover_reports_an_empty_directory(tmp_path) -> None:
    with pytest.raises(PromotionError, match=r"no .gguf found"):
        GgufArtifact.discover(tmp_path)


async def test_registering_writes_a_modelfile_when_colab_omitted_one(gate, tmp_path) -> None:
    (tmp_path / "q4_k_m.gguf").write_bytes(b"stub")
    artifact = GgufArtifact.discover(tmp_path)
    assert artifact.modelfile_path is None

    runner = RecordingRunner()
    await gate(runner).register_gguf(artifact, model_name="cand-v1")

    generated = tmp_path / "Modelfile.generated"
    assert generated.exists()
    assert "FROM ./q4_k_m.gguf" in generated.read_text(encoding="utf-8")
    assert runner.calls[0][:3] == ["ollama", "create", "cand-v1"]


async def test_registering_is_not_promoting(gate, tmp_path) -> None:
    """A registered candidate is loadable. Nothing points at it yet."""
    (tmp_path / "q4_k_m.gguf").write_bytes(b"stub")
    runner = RecordingRunner()
    await gate(runner).register_gguf(GgufArtifact.discover(tmp_path), model_name="cand-v1")

    assert all("cp" not in argv for argv in runner.calls)


async def test_a_failed_registration_surfaces(gate, tmp_path) -> None:
    (tmp_path / "q4_k_m.gguf").write_bytes(b"stub")
    with pytest.raises(PromotionError):
        await gate(RecordingRunner(fail=True)).register_gguf(
            GgufArtifact.discover(tmp_path), model_name="cand-v1"
        )
