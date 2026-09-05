"""Module 5c - the promotion gate.

A fine-tuned adapter is a liability until something proves it is an asset. This
is that something: it registers a GGUF from `scripts/train_colab.py`, runs it
through the same eval suite as the incumbent, compares it to the recorded Opus
baseline, and re-points the serving alias only if it clears every criterion.

Three criteria, and all three have to pass:

  groundedness parity  >= 85% of the Opus baseline. Relative, not absolute -
                       what matters is how much of the frontier model's
                       reliability the small one retained.
  latency p95          <= 1200 ms. A candidate that is accurate and slow fails.
                       p95, not p50, because the tail is what users feel.
  instruction floor    absolute, not relative. Ignoring a negative constraint
                       is not a quality gradient; it is a broken answer, and a
                       baseline that also ignored it does not excuse it.

The verdict is computed automatically. Applying it is not: `promote()` will not
touch Ollama unless `apply=True` is passed explicitly. Importing this module,
or running the evaluation, cannot mutate what your machine serves.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from hybrid_rag.config import PromotionSettings
from hybrid_rag.eval.eval_harness import EvalCase, EvalHarness, EvalReport, Responder


class PromotionError(RuntimeError):
    """Registration or alias update failed. Nothing was changed."""


class CriterionResult(BaseModel):
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.observed:.4g} vs {self.threshold:.4g} - {self.detail}"


class PromotionVerdict(BaseModel):
    candidate: str
    baseline: str
    promoted: bool = False
    applied: bool = False
    criteria: list[CriterionResult] = Field(default_factory=list)
    candidate_report: EvalReport | None = None
    baseline_report: EvalReport | None = None
    reason: str = ""

    @property
    def eligible(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    def summary(self) -> str:
        head = f"{self.candidate} vs {self.baseline}: " + (
            "ELIGIBLE" if self.eligible else "REJECTED"
        )
        return "\n".join([head, *(c.line() for c in self.criteria), self.reason]).strip()


@dataclass(slots=True, frozen=True)
class GgufArtifact:
    """A GGUF plus the Modelfile that describes how to serve it."""

    gguf_path: Path
    modelfile_path: Path | None = None

    @classmethod
    def discover(cls, directory: Path) -> GgufArtifact:
        """Find the single .gguf in a Colab download directory."""
        candidates = sorted(directory.glob("*.gguf"))
        if not candidates:
            raise PromotionError(f"no .gguf found in {directory}")
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            raise PromotionError(f"ambiguous: {len(candidates)} GGUF files in {directory}: {names}")
        modelfile = directory / "Modelfile"
        return cls(
            gguf_path=candidates[0],
            modelfile_path=modelfile if modelfile.exists() else None,
        )


class PromotionGate:
    """Evaluate a candidate, then decide. Applying the decision is separate."""

    def __init__(
        self,
        *,
        settings: PromotionSettings,
        harness: EvalHarness,
        runner: object | None = None,
    ) -> None:
        self._s = settings
        self._harness = harness
        # Injected so tests never shell out. Signature: (list[str]) -> str.
        self._runner = runner if callable(runner) else _run_ollama

    # ------------------------------------------------------------ registration

    async def register_gguf(self, artifact: GgufArtifact, *, model_name: str) -> str:
        """`ollama create` the candidate under its own name.

        Registering is not promoting. The candidate becomes loadable so it can
        be evaluated; nothing points at it until `promote(apply=True)` runs.
        """
        modelfile = artifact.modelfile_path
        if modelfile is None:
            modelfile = artifact.gguf_path.parent / "Modelfile.generated"
            modelfile.write_text(
                f"FROM ./{artifact.gguf_path.name}\n"
                "PARAMETER temperature 0\n"
                "PARAMETER num_ctx 4096\n"
                "PARAMETER seed 1337\n",
                encoding="utf-8",
            )
        try:
            return await asyncio.to_thread(
                self._runner,
                [self._s.ollama_binary, "create", model_name, "-f", str(modelfile)],
            )
        except PromotionError:
            raise
        except Exception as exc:
            raise PromotionError(f"ollama create failed: {exc}") from exc

    # -------------------------------------------------------------- evaluation

    async def evaluate(
        self,
        *,
        candidate: Responder,
        baseline: Responder,
        cases: Sequence[EvalCase],
    ) -> PromotionVerdict:
        """Run both sides and grade the candidate. Spends no cloud tokens."""
        if not cases:
            return PromotionVerdict(
                candidate=candidate.label,
                baseline=baseline.label,
                reason="no evaluation cases; refusing to promote on an empty suite",
            )

        candidate_report = await self._harness.run(candidate, cases)
        baseline_report = await self._harness.run(baseline, cases)
        criteria = self._grade(candidate_report, baseline_report)

        verdict = PromotionVerdict(
            candidate=candidate.label,
            baseline=baseline.label,
            criteria=criteria,
            candidate_report=candidate_report,
            baseline_report=baseline_report,
        )
        verdict.promoted = verdict.eligible
        verdict.reason = (
            "all criteria met; eligible to serve"
            if verdict.eligible
            else "; ".join(c.detail for c in criteria if not c.passed)
        )
        return verdict

    def _grade(self, candidate: EvalReport, baseline: EvalReport) -> list[CriterionResult]:
        base_grounded = baseline.groundedness_mean
        if base_grounded <= 0.0:
            parity = 0.0
            parity_detail = "baseline groundedness is zero; parity is undefined, so it fails"
        else:
            parity = candidate.groundedness_mean / base_grounded
            parity_detail = (
                f"candidate {candidate.groundedness_mean:.3f} / "
                f"baseline {base_grounded:.3f} = {parity:.1%} of Opus parity"
            )

        return [
            CriterionResult(
                name="groundedness_parity",
                passed=parity >= self._s.min_groundedness_parity,
                observed=round(parity, 4),
                threshold=self._s.min_groundedness_parity,
                detail=parity_detail,
            ),
            CriterionResult(
                name="latency_p95_ms",
                passed=candidate.latency_p95_ms <= self._s.max_latency_p95_ms,
                observed=candidate.latency_p95_ms,
                threshold=self._s.max_latency_p95_ms,
                detail=(
                    f"p95 {candidate.latency_p95_ms:.0f} ms against a "
                    f"{self._s.max_latency_p95_ms:.0f} ms SLA"
                ),
            ),
            CriterionResult(
                name="instruction_precision",
                passed=(
                    candidate.instruction_precision_mean >= self._s.min_instruction_precision
                ),
                observed=candidate.instruction_precision_mean,
                threshold=self._s.min_instruction_precision,
                detail=(
                    f"followed {candidate.instruction_precision_mean:.1%} of constraints; "
                    "an absolute floor, because ignoring a negative constraint is "
                    "not a quality gradient"
                ),
            ),
        ]

    # -------------------------------------------------------------- promotion

    async def promote(
        self, verdict: PromotionVerdict, *, model_name: str, apply: bool = False
    ) -> PromotionVerdict:
        """Re-point the serving alias. Does nothing unless `apply=True`.

        The gate decides automatically; the mutation is opt-in. An eval run that
        silently changed what your machine serves would be a very unpleasant
        surprise on a laptop you also work on.
        """
        if not verdict.eligible:
            verdict.applied = False
            return verdict
        if not apply:
            verdict.applied = False
            verdict.reason = (
                f"{verdict.reason} (dry run - pass apply=True to move "
                f"{self._s.serving_alias!r} to {model_name!r})"
            )
            return verdict

        try:
            await asyncio.to_thread(
                self._runner,
                [self._s.ollama_binary, "cp", model_name, self._s.serving_alias],
            )
        except Exception as exc:
            raise PromotionError(f"failed to update alias {self._s.serving_alias}: {exc}") from exc

        verdict.applied = True
        verdict.reason = f"{self._s.serving_alias} now points at {model_name}"
        return verdict


def _run_ollama(argv: list[str]) -> str:
    """Default runner. Injected away in tests so nothing shells out."""
    completed = subprocess.run(
        argv, capture_output=True, text=True, check=False, timeout=600
    )
    if completed.returncode != 0:
        raise PromotionError(
            f"{' '.join(argv)} exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )
    return completed.stdout.strip()
