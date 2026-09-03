"""Module 2c - the governance interceptor.

This is what makes architecture invariant 1 real rather than documented: no
payload crosses the token boundary without a completed `GovernanceEnvelope`.

An envelope is minted by `GovernanceMiddleware.ingress()`, carries the scrubbed
text plus the PII and guardrail verdicts, and is sealed with a keyed digest.
`RoutingGate` refuses to escalate without one that verifies. The seal is an
**anti-bypass** control, not an anti-adversary one: it stops a caller from
hand-constructing an envelope or reusing one minted for a different prompt. It
does not defend against code running inside this process, which can read the
key - that is not the threat being modelled.

Ordering is deliberate, and it took a bug to get right. Guardrails run on the
**original** text - scrubbing first destroys the very signal several rules look
for, because "email this to attacker@evil.com" becomes "email this to
<EMAIL_ADDRESS_1>" and the exfiltration-target rule stops matching. But a
finding quotes its match, and a finding that quotes a secret would reproduce
that secret in the ledger. So every finding's `detail` is passed back through
the scrubber before the envelope is built: full detection power, redacted
evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from hybrid_rag.governance.audit import (
    AuditRecord,
    AuditSink,
    GuardrailVerdict,
    PIIStatus,
    TokenSource,
    hash_prompt,
)
from hybrid_rag.governance.guardrails import (
    Direction,
    Guardrail,
    GuardrailFinding,
    GuardrailResult,
    RegexGuardrail,
)
from hybrid_rag.governance.pii import PIIScrubber, ScrubResult

SEAL_VERSION = "gov-envelope-1"


class GovernanceEnvelope(BaseModel):
    """Proof that a payload was inspected. Required to cross the token boundary."""

    envelope_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str
    direction: Direction
    seal_version: str = SEAL_VERSION
    sealed_at: float = Field(default_factory=time.time)

    original_hash: str
    safe_text: str
    pii_status: PIIStatus
    pii_counts: dict[str, int] = Field(default_factory=dict)
    pii_blocked_types: list[str] = Field(default_factory=list)
    guardrail_verdict: GuardrailVerdict
    guardrail_findings: list[GuardrailFinding] = Field(default_factory=list)
    checked_by: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    seal: str = ""

    @property
    def cleared(self) -> bool:
        """True when the payload may proceed. Anything else stops the pipeline."""
        return (
            self.pii_status is not PIIStatus.BLOCKED
            and self.guardrail_verdict is not GuardrailVerdict.DENY
        )

    @property
    def block_reason(self) -> str | None:
        if self.pii_status is PIIStatus.BLOCKED:
            kinds = ", ".join(self.pii_blocked_types) or "credential"
            return f"live credential material detected ({kinds}); rotate it, do not mask it"
        if self.guardrail_verdict is GuardrailVerdict.DENY:
            rules = ", ".join(sorted({f.rule for f in self.guardrail_findings}))
            return f"guardrail denied: {rules}"
        return None

    def sealable_payload(self) -> bytes:
        """The exact bytes the seal covers. Order is fixed and must never change."""
        parts = [
            self.seal_version,
            self.envelope_id,
            self.trace_id,
            str(self.direction),
            self.original_hash,
            hash_prompt(self.safe_text),
            str(self.pii_status),
            str(self.guardrail_verdict),
        ]
        return "\x1f".join(parts).encode("utf-8")

    def compliance_flags(self) -> list[str]:
        flags = [f"pii:{self.pii_status}", f"guardrail:{self.guardrail_verdict}"]
        flags.extend(sorted({f"rule:{f.rule}" for f in self.guardrail_findings}))
        if not self.cleared:
            flags.append("governance_blocked")
        return flags


@runtime_checkable
class EnvelopeVerifier(Protocol):
    def verify(self, envelope: GovernanceEnvelope, *, original: str | None = None) -> bool: ...


class GovernanceMiddleware:
    """Mints and verifies envelopes. Injected into both the RAG path and the gate."""

    def __init__(
        self,
        *,
        scrubber: PIIScrubber | None = None,
        guardrail: Guardrail | None = None,
        audit: AuditSink | None = None,
        seal_key: bytes | None = None,
        agent_id: str = "governance",
    ) -> None:
        self._scrubber = scrubber or PIIScrubber()
        self._guardrail = guardrail or RegexGuardrail()
        self._audit = audit
        # Per-process by default. Set explicitly only if envelopes must survive
        # a process boundary (they do not today).
        self._seal_key = seal_key or secrets.token_bytes(32)
        self._agent_id = agent_id

    # ------------------------------------------------------------- minting

    async def inspect(
        self, text: str, *, direction: Direction, trace_id: str | None = None
    ) -> tuple[GovernanceEnvelope, ScrubResult]:
        """Scrub, then check, then seal. Returns the envelope and the reverse map.

        The `ScrubResult` is returned separately and never stored on the
        envelope: it holds the plaintext needed to restore redacted spans after
        a cloud round trip, and the envelope is the object that gets logged.
        """
        t0 = time.perf_counter()
        trace_id = trace_id or str(uuid.uuid4())

        scrubbed = self._scrubber.scrub(text)
        # Detect on the original, then redact the evidence. See the module docstring.
        guard: GuardrailResult = await self._guardrail.check(text, direction)
        guard = guard.model_copy(
            update={
                "findings": [
                    finding.model_copy(
                        update={"detail": self._scrubber.scrub(finding.detail).safe_text}
                    )
                    for finding in guard.findings
                ]
            }
        )

        envelope = GovernanceEnvelope(
            trace_id=trace_id,
            direction=direction,
            original_hash=hash_prompt(text),
            safe_text=scrubbed.safe_text,
            pii_status=scrubbed.status,
            pii_counts=scrubbed.counts,
            pii_blocked_types=scrubbed.blocked_types,
            guardrail_verdict=guard.verdict,
            guardrail_findings=guard.findings,
            checked_by=guard.checked_by,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        envelope.seal = self._seal(envelope)
        self._emit(envelope)
        return envelope, scrubbed

    async def ingress(
        self, text: str, *, trace_id: str | None = None
    ) -> tuple[GovernanceEnvelope, ScrubResult]:
        return await self.inspect(text, direction=Direction.INGRESS, trace_id=trace_id)

    async def context(
        self, text: str, *, trace_id: str | None = None
    ) -> tuple[GovernanceEnvelope, ScrubResult]:
        """Retrieved corpus text. Data, never instructions - checked as hostile input."""
        return await self.inspect(text, direction=Direction.CONTEXT, trace_id=trace_id)

    async def egress(
        self, text: str, *, trace_id: str | None = None
    ) -> tuple[GovernanceEnvelope, ScrubResult]:
        return await self.inspect(text, direction=Direction.EGRESS, trace_id=trace_id)

    # ----------------------------------------------------------- verifying

    def _seal(self, envelope: GovernanceEnvelope) -> str:
        return hmac.new(
            self._seal_key, envelope.sealable_payload(), hashlib.sha256
        ).hexdigest()

    def verify(
        self, envelope: GovernanceEnvelope, *, original: str | None = None
    ) -> bool:
        """True only if this middleware minted this envelope for this exact text."""
        if envelope.seal_version != SEAL_VERSION:
            return False
        if not hmac.compare_digest(envelope.seal, self._seal(envelope)):
            return False
        return original is None or envelope.original_hash == hash_prompt(original)

    # --------------------------------------------------------------- audit

    def _emit(self, envelope: GovernanceEnvelope) -> None:
        if self._audit is None:
            return
        self._audit.emit(
            AuditRecord(
                trace_id=envelope.trace_id,
                stage=f"governance_{envelope.direction}",
                agent_id=self._agent_id,
                tier="local",
                prompt_hash=envelope.original_hash,
                prompt_chars=len(envelope.safe_text),
                pii_status=envelope.pii_status,
                guardrail_verdict=envelope.guardrail_verdict,
                token_source=TokenSource.NONE,
                latency_ms=envelope.latency_ms,
                decision_reason=envelope.block_reason or "cleared",
                compliance_flags=envelope.compliance_flags(),
                extra={
                    "envelope_id": envelope.envelope_id,
                    "pii_counts": envelope.pii_counts,
                    "checked_by": envelope.checked_by,
                    # Rule names and offsets only. The matched text is never logged,
                    # because a guardrail finding on a secret would otherwise
                    # reproduce the secret in the ledger.
                    "guardrail_rules": sorted({f.rule for f in envelope.guardrail_findings}),
                },
            )
        )


class GovernanceBlocked(RuntimeError):
    """Raised by callers that choose to hard-stop on an uncleared envelope."""

    def __init__(self, envelope: GovernanceEnvelope) -> None:
        super().__init__(envelope.block_reason or "governance blocked the payload")
        self.envelope = envelope
