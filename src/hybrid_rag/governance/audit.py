"""Append-only governance ledger.

Module 1 ships the sink and the record schema because the routing gate cannot
make an unlogged decision. The interceptor middleware that fills in
`pii_status` and `guardrail_verdict` arrives in Module 2; until then those
fields carry `NOT_EVALUATED`, which is itself an auditable state rather than a
silent absence.

Schema is aligned to NIST AI RMF MEASURE/MANAGE evidence needs and the EU AI
Act Article 12 record-keeping shape: traceable provenance (model fingerprint +
determinism seed), source attribution, and a hash chain.

THREAT MODEL FOR THE CHAIN - read this before relying on it.

`prev_hash`/`record_hash` are unkeyed SHA-256. That detects accidental damage,
a partial write, a truncated file, and a hand-edited or hand-deleted row -
which is what an operational audit trail is usually asked to survive.

It does NOT resist a motivated attacker with write access to the ledger file.
Nothing here is secret, so anyone who can edit a row can recompute every hash
after it and present a clean chain. Anyone with that access can also edit this
source file, so a keyed MAC stored beside the ledger would buy nothing.

If the threat model includes a local attacker, the chain has to be anchored
somewhere they cannot reach: ship records to an append-only remote sink, or
periodically publish `record_hash` to external storage. Both are Module 5 work.
Until then, call this tamper-EVIDENT, never tamper-PROOF.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import orjson
from pydantic import BaseModel, Field

_GENESIS = "0" * 64


class TokenSource(StrEnum):
    LOCAL_OLLAMA = "local_ollama"
    CLOUD_ANTHROPIC = "cloud_anthropic"
    CACHE = "cache"
    NONE = "none"


class PIIStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    CLEAN = "clean"
    SCRUBBED = "scrubbed"
    BLOCKED = "blocked"


class GuardrailVerdict(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    ALLOW = "allow"
    ALLOW_WITH_EDITS = "allow_with_edits"
    DENY = "deny"


class SourceAttribution(BaseModel):
    """Which retrieved chunk contributed, and how much."""

    doc_id: str
    chunk_id: str
    uri: str | None = None
    score: float
    rank: int


class AuditRecord(BaseModel):
    """One immutable ledger line."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str
    parent_event_id: str | None = None
    timestamp: float = Field(default_factory=time.time)
    schema_version: str = "1.0-nist-ai-rmf"

    stage: str
    agent_id: str
    tier: str

    model_id: str | None = None
    model_fingerprint: str | None = None
    determinism_seed: int | None = None

    prompt_hash: str
    prompt_chars: int = 0
    pii_status: PIIStatus = PIIStatus.NOT_EVALUATED
    guardrail_verdict: GuardrailVerdict = GuardrailVerdict.NOT_EVALUATED

    token_source: TokenSource = TokenSource.NONE
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    confidence: float | None = None
    decision_reason: str | None = None
    retrieval_sources: list[SourceAttribution] = Field(default_factory=list)
    compliance_flags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    # Filled by the sink. Chains each line to the previous one so a deleted or
    # edited row is detectable on replay.
    prev_hash: str | None = None
    record_hash: str | None = None


def hash_prompt(text: str) -> str:
    """Stable content hash. The ledger stores this, never the raw prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@runtime_checkable
class AuditSink(Protocol):
    def emit(self, record: AuditRecord) -> AuditRecord: ...
    def close(self) -> None: ...


class NullAuditSink:
    """Discards records. For unit tests that do not assert on the ledger."""

    def emit(self, record: AuditRecord) -> AuditRecord:
        return record

    def close(self) -> None:
        return None


class JsonlAuditSink:
    """Append-only JSONL with a per-line hash chain.

    Durability over throughput: every line is flushed and `fsync`-ed before the
    call returns. At local RAG volumes (single-digit writes per query) the cost
    is irrelevant, and a half-written ledger is worse than a slow one.
    """

    def __init__(self, path: Path, *, schema_version: str = "1.0-nist-ai-rmf") -> None:
        self._path = path
        self._schema_version = schema_version
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._read_tail_hash()

    @property
    def path(self) -> Path:
        return self._path

    def _read_tail_hash(self) -> str:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return _GENESIS
        with self._path.open("rb") as fh:
            last = b""
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return _GENESIS
        try:
            return str(orjson.loads(last).get("record_hash") or _GENESIS)
        except orjson.JSONDecodeError:
            return _GENESIS

    def emit(self, record: AuditRecord) -> AuditRecord:
        with self._lock:
            record.schema_version = self._schema_version
            record.prev_hash = self._prev_hash
            payload = record.model_dump(mode="json", exclude={"record_hash"})
            canonical = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
            record.record_hash = hashlib.sha256(canonical).hexdigest()
            line = orjson.dumps(record.model_dump(mode="json")) + b"\n"
            with self._path.open("ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._prev_hash = record.record_hash
            return record

    def close(self) -> None:
        return None

    def verify_chain(self) -> tuple[bool, int]:
        """Re-walk the ledger. Returns (intact, first_bad_line_number)."""
        prev = _GENESIS
        if not self._path.exists():
            return True, 0
        with self._path.open("rb") as fh:
            for n, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                row = orjson.loads(line)
                claimed = row.pop("record_hash", None)
                if row.get("prev_hash") != prev:
                    return False, n
                canonical = orjson.dumps(row, option=orjson.OPT_SORT_KEYS)
                if hashlib.sha256(canonical).hexdigest() != claimed:
                    return False, n
                prev = str(claimed)
        return True, 0
