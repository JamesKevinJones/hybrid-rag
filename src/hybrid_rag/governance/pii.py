"""Module 2a - PII detection and redaction.

Two detectors behind one Protocol:

* `RegexPIIDetector` is the default. Zero dependencies, deterministic, and it
  catches the categories that actually leak in a RAG pipeline - credentials,
  card numbers, emails, phone numbers, national IDs. Determinism matters here
  more than recall: a governance control whose output changes between runs
  cannot be audited.
* `PresidioPIIDetector` is the optional upgrade (extra: `governance`). It adds
  NER-backed entities regex cannot see - person names, locations, organisations
  - at the cost of a spaCy model.

`CompositePIIDetector` runs both and merges overlapping spans.

Redaction is placeholder-based (`<EMAIL_ADDRESS_1>`) and the reverse map stays
in memory on the result object. It is never logged and never serialised into
the ledger - `ScrubResult.reverse_map` is explicitly excluded from dumps.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from hybrid_rag.governance.audit import PIIStatus


class PIIEntity(BaseModel):
    """One detected span. Offsets are into the original text."""

    entity_type: str
    start: int
    end: int
    score: float
    detector: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: PIIEntity) -> bool:
        return self.start < other.end and other.start < self.end


class ScrubResult(BaseModel):
    """Outcome of one scrub. `reverse_map` is in-memory only."""

    original_length: int
    safe_text: str
    status: PIIStatus
    entities: list[PIIEntity] = Field(default_factory=list)
    # Only the types that triggered a hard block, not every type detected.
    blocked_types: list[str] = Field(default_factory=list)
    # Placeholder -> original substring. Excluded from every serialisation so a
    # careless `model_dump()` into the ledger cannot leak what was redacted.
    reverse_map: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)

    @property
    def counts(self) -> dict[str, int]:
        """Entity-type histogram. This is what is safe to put in an audit record."""
        out: dict[str, int] = defaultdict(int)
        for entity in self.entities:
            out[entity.entity_type] += 1
        return dict(out)

    def restore(self, text: str) -> str:
        """Put the originals back after a round trip through a cloud model."""
        for placeholder, original in self.reverse_map.items():
            text = text.replace(placeholder, original)
        return text


@runtime_checkable
class PIIDetector(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, text: str) -> list[PIIEntity]: ...


# --------------------------------------------------------------------------
# Regex detector
#
# Ordered most-specific first. Every pattern is anchored on word boundaries so
# a hash or a UUID does not get shredded into fake findings.
# --------------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, str, float], ...] = (
    # Credentials first - these are the ones that actually cost money if they leak.
    ("ANTHROPIC_API_KEY", r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b", 1.0),
    ("OPENAI_API_KEY", r"\bsk-(?:proj-)?[A-Za-z0-9]{20,}\b", 0.95),
    ("GITHUB_TOKEN", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", 1.0),
    ("AWS_ACCESS_KEY", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 1.0),
    ("SLACK_TOKEN", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", 1.0),
    (
        "PRIVATE_KEY",
        # Bounded body, not `[\s\S]+?`. Unbounded, an unterminated BEGIN marker
        # makes the engine rescan to end-of-input from every start position -
        # measured quadratic: 400 markers took 31 ms, 1600 took 416 ms. This
        # pattern runs over ingested corpus text, which is the least trusted
        # input in the system. A real PEM body is under 4 KB.
        r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]{1,4000}?-----END "
        r"(?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
        1.0,
    ),
    ("JWT", r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", 0.9),
    # Personal identifiers.
    ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", 0.95),
    ("US_SSN", r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", 0.9),
    (
        "PHONE_NUMBER",
        r"(?<![\d.\-])(?:\+?1[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}(?![\d.\-])",
        0.7,
    ),
    ("CREDIT_CARD", r"\b(?:\d[ \-]?){13,19}\b", 0.6),
    ("IP_ADDRESS", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b", 0.6),
    ("IBAN", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", 0.7),
)

_COMPILED: tuple[tuple[str, re.Pattern[str], float], ...] = tuple(
    (name, re.compile(pattern), score) for name, pattern, score in _PATTERNS
)


def _luhn_valid(digits: str) -> bool:
    """Checksum a candidate card number. Without this, every order id is a false positive."""
    stripped = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(stripped) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(stripped)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


class RegexPIIDetector:
    """Deterministic, dependency-free, and fast enough to run on every request."""

    name = "regex"

    def detect(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []
        for entity_type, pattern, score in _COMPILED:
            for match in pattern.finditer(text):
                if entity_type == "CREDIT_CARD" and not _luhn_valid(match.group(0)):
                    continue
                found.append(
                    PIIEntity(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        score=score,
                        detector=self.name,
                    )
                )
        return found


class PresidioPIIDetector:
    """NER-backed detection for names, locations and organisations.

    Optional on purpose. It pulls spaCy and a language model, which is a large
    install for a laptop that is already budgeting VRAM, and the regex detector
    covers the categories with real blast radius on its own.
    """

    name = "presidio"

    def __init__(self, language: str = "en", score_threshold: float = 0.5) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "presidio-analyzer is not installed (extra: governance)"
            ) from exc
        self._engine = AnalyzerEngine()
        self._language = language
        self._threshold = score_threshold

    def detect(self, text: str) -> list[PIIEntity]:
        results = self._engine.analyze(
            text=text, language=self._language, score_threshold=self._threshold
        )
        return [
            PIIEntity(
                entity_type=str(r.entity_type),
                start=int(r.start),
                end=int(r.end),
                score=float(r.score),
                detector=self.name,
            )
            for r in results
        ]


class CompositePIIDetector:
    """Union of several detectors with overlapping spans collapsed."""

    name = "composite"

    def __init__(self, detectors: Sequence[PIIDetector]) -> None:
        self._detectors = list(detectors)

    def detect(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []
        for detector in self._detectors:
            found.extend(detector.detect(text))
        return merge_spans(found)


def merge_spans(entities: Sequence[PIIEntity]) -> list[PIIEntity]:
    """Collapse overlaps, keeping the longest span and breaking ties on score.

    A credit card and an IBAN pattern will both fire on the same digits. Two
    redactions of one span produce corrupt text, so exactly one has to win.
    """
    ordered = sorted(entities, key=lambda e: (e.start, -e.length, -e.score))
    kept: list[PIIEntity] = []
    for entity in ordered:
        clash = next((k for k in kept if k.overlaps(entity)), None)
        if clash is None:
            kept.append(entity)
            continue
        if (entity.length, entity.score) > (clash.length, clash.score):
            kept[kept.index(clash)] = entity
    return sorted(kept, key=lambda e: e.start)


class PIIScrubber:
    """Detect, redact, classify. The only PII entry point the middleware uses."""

    def __init__(
        self,
        detector: PIIDetector | None = None,
        *,
        block_types: frozenset[str] = frozenset(
            {
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "GITHUB_TOKEN",
                "AWS_ACCESS_KEY",
                "SLACK_TOKEN",
                "PRIVATE_KEY",
            }
        ),
    ) -> None:
        self._detector = detector or RegexPIIDetector()
        # Redacting a leaked credential is not enough. If a live key reached the
        # prompt at all, the request is refused outright and the key is rotated
        # by a human - masking it and continuing normalises the incident.
        self._block_types = block_types

    def scrub(self, text: str) -> ScrubResult:
        entities = merge_spans(self._detector.detect(text))
        if not entities:
            return ScrubResult(
                original_length=len(text), safe_text=text, status=PIIStatus.CLEAN
            )

        counters: dict[str, int] = defaultdict(int)
        reverse: dict[str, str] = {}
        pieces: list[str] = []
        cursor = 0
        for entity in entities:
            counters[entity.entity_type] += 1
            placeholder = f"<{entity.entity_type}_{counters[entity.entity_type]}>"
            reverse[placeholder] = text[entity.start : entity.end]
            pieces.append(text[cursor : entity.start])
            pieces.append(placeholder)
            cursor = entity.end
        pieces.append(text[cursor:])

        blocked = sorted({e.entity_type for e in entities if e.entity_type in self._block_types})
        return ScrubResult(
            original_length=len(text),
            safe_text="".join(pieces),
            status=PIIStatus.BLOCKED if blocked else PIIStatus.SCRUBBED,
            entities=entities,
            blocked_types=blocked,
            reverse_map=reverse,
        )
