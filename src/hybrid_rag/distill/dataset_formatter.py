"""Module 4a - dataset curation. Runs locally; training does not.

This is the half of the distillation pipeline that stays on the workstation.
It reads Level 2 completions, validates them, strips anything that must not
leave the machine, and writes ShareGPT-format JSONL ready to upload to Colab.

The governance point is easy to miss and expensive to get wrong: **a training
set is an egress channel**. Every sample here is destined for a Google-hosted
notebook and then baked into model weights, which is the least reversible form
of data disclosure available. So the formatter runs the same egress guardrails
as a live cloud call, and a sample carrying a credential is dropped rather than
redacted - a masked secret in a training set teaches the model the shape of
your secrets.

Curation is deliberately aggressive. On a Claude Pro budget you will collect
hundreds of samples, not millions, and every one of them cost a message from a
rolling window. Quality per sample is the only lever available.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from pydantic import BaseModel, Field, field_validator

from hybrid_rag.governance.guardrails import Direction, Guardrail, RegexGuardrail
from hybrid_rag.governance.pii import PIIScrubber
from hybrid_rag.rag.local_rag import Triplet

_ROLES = ("system", "human", "gpt")


class Turn(BaseModel):
    """One ShareGPT turn. `from` is the wire name; `role` is the Python name."""

    role: str = Field(alias="from")
    value: str

    model_config = {"populate_by_name": True}

    @field_validator("role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        if v not in _ROLES:
            raise ValueError(f"role must be one of {_ROLES}, got {v!r}")
        return v


class Sample(BaseModel):
    """One ShareGPT conversation."""

    conversations: list[Turn]
    # Provenance. Kept in the file so a bad adapter can be traced to its inputs.
    source_model: str = "claude-opus-5"
    trace_id: str | None = None

    @property
    def fingerprint(self) -> str:
        """Content hash over the human turn only - the same question answered
        twice is one training signal, not two."""
        human = next((t.value for t in self.conversations if t.role == "human"), "")
        return hashlib.sha256(human.strip().lower().encode()).hexdigest()

    @property
    def total_chars(self) -> int:
        return sum(len(t.value) for t in self.conversations)

    def to_jsonl(self) -> bytes:
        return orjson.dumps(self.model_dump(by_alias=True))


class RejectionReason(BaseModel):
    fingerprint: str
    reason: str
    detail: str = ""


@dataclass(slots=True)
class CurationReport:
    seen: int = 0
    kept: int = 0
    rejections: list[RejectionReason] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return self.seen - self.kept

    def counts_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rejections:
            out[r.reason] = out.get(r.reason, 0) + 1
        return out

    def summary(self) -> str:
        by_reason = ", ".join(f"{k}={v}" for k, v in sorted(self.counts_by_reason().items()))
        return (
            f"{self.kept}/{self.seen} kept ({self.dropped} dropped)"
            + (f" - {by_reason}" if by_reason else "")
        )


@dataclass(slots=True, frozen=True)
class CurationPolicy:
    """Thresholds, in one place, so a rejected corpus can be explained."""

    min_answer_chars: int = 120
    max_answer_chars: int = 24_000
    min_question_chars: int = 12
    # Opus refusing or hedging is not a training signal worth 1000 gradient steps.
    refusal_markers: tuple[str, ...] = (
        "i cannot help with",
        "i can't help with",
        "i'm not able to",
        "as an ai language model",
    )
    drop_on_pii: bool = True


class DatasetFormatter:
    """Validate, screen and de-duplicate Level 2 completions."""

    def __init__(
        self,
        *,
        scrubber: PIIScrubber | None = None,
        guardrail: Guardrail | None = None,
        policy: CurationPolicy | None = None,
    ) -> None:
        self._scrubber = scrubber or PIIScrubber()
        self._guardrail = guardrail or RegexGuardrail()
        self._policy = policy or CurationPolicy()

    async def curate(
        self, samples: Iterable[Sample], *, report: CurationReport | None = None
    ) -> tuple[list[Sample], CurationReport]:
        report = report or CurationReport()
        seen: set[str] = set()
        kept: list[Sample] = []

        for sample in samples:
            report.seen += 1
            fingerprint = sample.fingerprint
            reason = self._reject_reason(sample)

            if reason is None and fingerprint in seen:
                reason = ("duplicate", "same question already in the set")

            if reason is None:
                verdict = await self._screen(sample)
                if verdict is not None:
                    reason = verdict

            if reason is not None:
                report.rejections.append(
                    RejectionReason(fingerprint=fingerprint, reason=reason[0], detail=reason[1])
                )
                continue

            seen.add(fingerprint)
            kept.append(sample)
            report.kept += 1

        return kept, report

    def _reject_reason(self, sample: Sample) -> tuple[str, str] | None:
        roles = [t.role for t in sample.conversations]
        if "human" not in roles or "gpt" not in roles:
            return ("incomplete", f"roles present: {roles}")

        question = next(t.value for t in sample.conversations if t.role == "human")
        answer = next(t.value for t in sample.conversations if t.role == "gpt")

        if len(question.strip()) < self._policy.min_question_chars:
            return ("question_too_short", f"{len(question)} chars")
        if len(answer.strip()) < self._policy.min_answer_chars:
            return ("answer_too_short", f"{len(answer)} chars")
        if len(answer) > self._policy.max_answer_chars:
            return ("answer_too_long", f"{len(answer)} chars")

        lowered = answer[:400].lower()
        for marker in self._policy.refusal_markers:
            if marker in lowered:
                return ("refusal", marker)
        return None

    async def _screen(self, sample: Sample) -> tuple[str, str] | None:
        """Egress screening. A training set leaves the machine permanently."""
        blob = "\n".join(t.value for t in sample.conversations)

        scrubbed = self._scrubber.scrub(blob)
        if self._policy.drop_on_pii and scrubbed.blocked_types:
            # Dropped, not redacted: a masked credential still teaches the model
            # what your credentials look like, and weights cannot be un-trained.
            return ("credential", ", ".join(scrubbed.blocked_types))

        guard = await self._guardrail.check(blob, Direction.EGRESS)
        if guard.denied:
            return ("guardrail", ", ".join(sorted({f.rule for f in guard.findings})))
        return None


# --------------------------------------------------------------------------
# Readers, writers, splitting
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[Sample]:
    """Parse a ShareGPT JSONL file, skipping rows that are not valid samples."""
    if not path.exists():
        return
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            yield Sample.model_validate(orjson.loads(line))
        except (orjson.JSONDecodeError, ValueError):
            continue


def write_jsonl(path: Path, samples: Sequence[Sample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for sample in samples:
            fh.write(sample.to_jsonl() + b"\n")
    return len(samples)


def split(
    samples: Sequence[Sample], *, val_fraction: float = 0.1, seed: int = 1337
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic train/validation split. The seed is pinned so a re-run
    produces the same split and two adapter runs stay comparable."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 9 else 0
    return shuffled[cut:], shuffled[:cut]


def sample_from_triplet(triplet: Triplet, system_prompt: str) -> Sample:
    """Turn a mined (Query, Positive, Negative) triplet into a grounded sample.

    The negative is included in the prompt on purpose. Training only on the
    right passage teaches the model to trust whatever it is handed; showing it
    a plausible-but-wrong passage alongside teaches it to discriminate, which is
    the behaviour that actually matters in a retrieval pipeline.
    """
    context = (
        f"[1] {triplet.positive}\n\n[2] {triplet.negative}\n\n"
        f"Question: {triplet.query}"
    )
    return Sample(
        conversations=[
            Turn(role="system", value=system_prompt),
            Turn(role="human", value=context),
            Turn(role="gpt", value=""),
        ],
        source_model="local-mined-triplet",
    )
