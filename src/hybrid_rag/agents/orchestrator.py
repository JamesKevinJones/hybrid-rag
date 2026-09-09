"""Module 5 - the execution orchestrator. Level 2 is no longer stubbed.

This is the file where the whole system finally spends money, so it is the file
where every guard has to actually hold. The order below is the contract:

    1. ingress governance   scrub, screen, seal - or refuse outright
    2. local retrieval      on the REDACTED text, never the raw query
    3. context screening    every retrieved chunk is untrusted input
    4. routing gate         3-tier waterfall, then governance and budget vetoes
    5a. local generation    resident 3B, grounded on the surviving chunks
    5b. Level 2 call        only with a reservation held open around it
    6. egress governance    on the RESPONSE, before the user ever sees it
    7. distillation         successful completions become training data
    8. audit                every branch, including the ones that spent nothing

Two invariants are worth stating because they are easy to break later:

**A vetoed request makes zero cloud calls.** Not a cancelled call, not a call
whose result is discarded - the client is never touched. `_execute_frontier` is
unreachable unless a reservation is held, and the reservation is released on
every exit path including exceptions.

**The reverse map never leaves this module.** PII is restored into the answer
returned to the user, because they are the person it belongs to. It is not
restored into the audit ledger, and it is not restored into the training set.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field

from hybrid_rag.config import DistillSettings, Settings
from hybrid_rag.distill.dataset_formatter import (
    DatasetFormatter,
    Sample,
    Turn,
)
from hybrid_rag.governance.audit import (
    AuditRecord,
    AuditSink,
    GuardrailVerdict,
    PIIStatus,
    SourceAttribution,
    TokenSource,
    hash_prompt,
)
from hybrid_rag.governance.guardrails import Direction
from hybrid_rag.governance.middleware import GovernanceEnvelope, GovernanceMiddleware
from hybrid_rag.governance.pii import PIIScrubber, ScrubResult
from hybrid_rag.llm.ollama_client import LocalInferenceError, TextGenerator
from hybrid_rag.llm.opus_client import FrontierClient, OpusUnavailable
from hybrid_rag.rag.local_rag import LocalRAGEngine, RetrievalResult
from hybrid_rag.routing.budget import Priority
from hybrid_rag.routing.routing_gate import RetrievalSignal, RoutingDecision, RoutingGate, Tier

_log = logging.getLogger(__name__)

_LOCAL_SYSTEM = (
    "You are a grounded technical assistant. Answer only from the provided "
    "context. If the context does not contain the answer, say so plainly."
)

_OPUS_SYSTEM = (
    "You are the frontier tier of a local-first retrieval system. The local "
    "model could not answer this with confidence. Decompose the question, "
    "reason across the supplied context, and answer precisely. If the context "
    "is insufficient, say what is missing rather than inventing it."
)

# Any string that reaches `decision_reason` is written to the governance
# ledger, and rule 4 says nothing raw goes there. Exception text from a
# FrontierClient, a local model or an unexpected crash is raw by definition -
# and the scrub has to happen HERE rather than inside one client, because the
# Protocol lets any implementation raise anything it likes. A test proved that:
# scrubbing only inside AnthropicOpusClient left a stub client leaking freely.
_REASON_SCRUBBER = PIIScrubber(block_types=frozenset())


def _safe_reason(text: str, *, limit: int = 300) -> str:
    return _REASON_SCRUBBER.scrub(text).safe_text[:limit]


_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


class Stage(StrEnum):
    """The pipeline phases a caller may want to display while they run.

    Emitted by `execute()` through an optional callback so a UI can show what
    is *actually* happening rather than animating a plausible-looking sequence
    on a timer. A progress indicator that is not driven by real events is a
    fabricated measurement wearing a spinner.
    """

    GOVERNANCE = "governance"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"


StageCallback = Callable[[Stage], None]


class Outcome(StrEnum):
    LOCAL = "local"
    LOCAL_DEGRADED = "local_degraded"
    OPUS = "opus"
    BLOCKED = "blocked"
    FAILED = "failed"


class ExecutionResult(BaseModel):
    """Everything a caller or an auditor needs about one request."""

    trace_id: str
    outcome: Outcome
    answer: str = ""
    reason: str = ""
    # The ledger row this execution produced. It is the chain hash, so it is
    # also the handle a user needs to find this exact turn in the audit trail
    # and check that it has not been rewritten since.
    audit_hash: str | None = None

    retrieval_confidence: float = 0.0
    routing_stage: str | None = None
    degraded_reason: str | None = None
    # Distinguishes "Level 2 was unreachable" from "Level 2 was not needed" and
    # from "Level 2 was refused". An operator reading the ledger on an
    # air-gapped host needs to see connectivity, not infer it from prose.
    frontier_unavailable: bool = False

    # Governance outcome, carried so a caller can display it without re-running
    # the scan. Counts only: the entities themselves stay inside the scrubber,
    # per rule 10.
    pii_status: PIIStatus = PIIStatus.NOT_EVALUATED
    pii_entities: int = 0

    local_tokens_in: int = 0
    local_tokens_out: int = 0
    cloud_tokens_in: int = 0
    cloud_tokens_out: int = 0
    latency_ms: float = 0.0
    # Model time only. `latency_ms` includes governance, retrieval and rerank,
    # so dividing tokens by it understates throughput by roughly the retrieval
    # cost - which for a short answer is most of the wall clock.
    generation_ms: float = 0.0

    sources: list[SourceAttribution] = Field(default_factory=list)
    quarantined_chunks: int = 0
    distilled: bool = False

    # `computed_field`, not a bare property. A plain `@property` is invisible to
    # `model_dump()`, so the moment this model is serialised over HTTP the two
    # most load-bearing facts in it - did this cost money, and where did the
    # tokens come from - silently disappear. A web client would then have to
    # re-derive them from the token counts, which is exactly the "compute a
    # plausible number in the renderer" that rule 14 forbids.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def spent_cloud_tokens(self) -> bool:
        return (self.cloud_tokens_in + self.cloud_tokens_out) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def token_source(self) -> TokenSource:
        if self.spent_cloud_tokens:
            return TokenSource.CLOUD_ANTHROPIC
        if self.local_tokens_out:
            return TokenSource.LOCAL_OLLAMA
        return TokenSource.NONE


@dataclass(slots=True)
class _Context:
    """Retrieval output after screening. Poisoned chunks never reach a prompt."""

    text: str
    sources: list[SourceAttribution]
    quarantined: int


class AgentOrchestrator:
    """Wires Modules 1-4 into one execution path. Constructed by the container."""

    def __init__(
        self,
        *,
        settings: Settings,
        governance: GovernanceMiddleware,
        rag: LocalRAGEngine,
        gate: RoutingGate,
        local: TextGenerator,
        frontier: FrontierClient,
        audit: AuditSink,
        formatter: DatasetFormatter | None = None,
        agent_id: str = "orchestrator",
    ) -> None:
        self._settings = settings
        self._governance = governance
        self._rag = rag
        self._gate = gate
        self._local = local
        self._frontier = frontier
        self._audit = audit
        self._formatter = formatter or DatasetFormatter()
        self._agent_id = agent_id

    # ------------------------------------------------------------------ api

    async def execute(
        self,
        query: str,
        *,
        trace_id: str | None = None,
        priority: Priority = Priority.INTERACTIVE,
        on_stage: StageCallback | None = None,
    ) -> ExecutionResult:
        """Run one request end to end. Never raises; failures become results.

        `on_stage` is called as each phase begins. It is fired inside the
        pipeline rather than estimated outside it, so a UI driven by it shows
        the stage the request is really in.

        A callback that raises is logged and ignored. It has to be: this
        function turns every exception into `Outcome.FAILED`, so letting a
        display bug out would report a broken spinner as a failed request -
        with the renderer's traceback standing where the routing reason should
        be. Presentation cannot be allowed to change the outcome it is
        presenting.
        """
        trace_id = trace_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        decision: RoutingDecision | None = None

        def stage(phase: Stage) -> None:
            if on_stage is None:
                return
            try:
                on_stage(phase)
            except Exception:
                _log.exception("stage callback failed at %s; continuing", phase)

        try:
            stage(Stage.GOVERNANCE)
            envelope, scrub = await self._governance.ingress(query, trace_id=trace_id)
            if not envelope.cleared:
                return self._finish(
                    ExecutionResult(
                        trace_id=trace_id,
                        outcome=Outcome.BLOCKED,
                        reason=envelope.block_reason or "governance refused the request",
                    ),
                    t0,
                    envelope,
                )

            stage(Stage.RETRIEVAL)
            retrieval = await self._rag.query(envelope.safe_text, trace_id=trace_id)
            context = await self._screen_context(retrieval, trace_id)

            decision = await self._gate.decide(
                query,
                retrieval=RetrievalSignal(
                    confidence=retrieval.confidence,
                    n_results=len(retrieval.results),
                    top_score=retrieval.top_score,
                    sources=tuple(context.sources),
                ),
                envelope=envelope,
                trace_id=trace_id,
                priority=priority,
            )

            result = ExecutionResult(
                trace_id=trace_id,
                outcome=Outcome.LOCAL,
                retrieval_confidence=retrieval.confidence,
                routing_stage=str(decision.decided_by),
                sources=context.sources,
                quarantined_chunks=context.quarantined,
            )

            stage(Stage.GENERATION)
            if decision.tier is Tier.OPUS:
                await self._execute_frontier(result, decision, envelope, scrub, context)
            else:
                await self._execute_local(result, decision, envelope, scrub, context)

            return self._finish(result, t0, envelope)

        except Exception as exc:
            return self._finish(
                ExecutionResult(
                    trace_id=trace_id,
                    outcome=Outcome.FAILED,
                    reason=_safe_reason(f"{type(exc).__name__}: {exc}"),
                ),
                t0,
                None,
            )
        finally:
            # Belt and braces. `escalation()` releases on its own exit paths;
            # this catches a reservation stranded by an exception raised between
            # the gate returning and the context manager opening.
            if decision is not None and decision.reservation_id is not None:
                self._gate.abort(decision, reason="orchestrator exit with reservation open")

    # ------------------------------------------------------------ pipeline

    async def _screen_context(self, retrieval: RetrievalResult, trace_id: str) -> _Context:
        """Every retrieved chunk is untrusted. Poisoned ones are excluded.

        The MCP tool returns quarantined passages to its caller so a human can
        see the corpus is compromised. Here the audience is a language model,
        so they are dropped from the prompt entirely - surfacing them would be
        handing the attacker exactly the channel they were reaching for.
        """
        usable: list[str] = []
        sources: list[SourceAttribution] = []
        quarantined = 0

        for i, scored in enumerate(retrieval.results):
            envelope, _ = await self._governance.inspect(
                scored.chunk.text, direction=Direction.CONTEXT, trace_id=trace_id
            )
            if not envelope.cleared:
                quarantined += 1
                continue
            usable.append(f"[{len(usable) + 1}] {envelope.safe_text}")
            attribution = scored.to_attribution()
            attribution.rank = i
            sources.append(attribution)

        return _Context(text="\n\n".join(usable), sources=sources, quarantined=quarantined)

    async def _execute_local(
        self,
        result: ExecutionResult,
        decision: RoutingDecision,
        envelope: GovernanceEnvelope,
        scrub: ScrubResult,
        context: _Context,
    ) -> None:
        """Level 0 answer. Also the destination of every veto and degradation."""
        result.outcome = Outcome.LOCAL_DEGRADED if decision.degraded else Outcome.LOCAL
        if decision.degraded:
            result.degraded_reason = decision.reason
        result.reason = decision.reason

        prompt = _PROMPT.format(
            context=context.text or "(no local context retrieved)",
            question=envelope.safe_text,
        )
        gen_t0 = time.perf_counter()
        try:
            completion = await self._local.generate(prompt, system=_LOCAL_SYSTEM)
        except LocalInferenceError as exc:
            result.generation_ms = (time.perf_counter() - gen_t0) * 1000.0
            result.outcome = Outcome.FAILED
            result.reason = _safe_reason(f"local generation failed: {exc}")
            return
        result.generation_ms = (time.perf_counter() - gen_t0) * 1000.0

        result.local_tokens_in = completion.tokens_in
        result.local_tokens_out = completion.tokens_out

        # The local answer is screened on the way out, exactly as the Level 2
        # answer is. This used to be frontier-only, on the reasoning that a
        # local model is our own and the ingress screen had already vetted the
        # question. Both halves of that were wrong.
        #
        # The ingress screen is a CLASSIFIER, and llama-guard3 assigns one
        # category to a whole prompt. Once any category is non-blocking
        # (decision 31), a request can be steered into the exempt label and
        # carry the rest of its content through with it - measured: "explain
        # how to hotwire a car" denies as S2, and the same request behind a
        # copyright-flavoured clause clears as S8, 3 times out of 3. Screening
        # here closes the harm channel rather than the label, and it holds for
        # whatever the next classifier gets wrong too.
        #
        # It also covers the case the frontier screen was written for: a model
        # echoing an injected instruction back, or reproducing a secret it read
        # out of a retrieved chunk. A 3B is not less likely to do that.
        outbound, _ = await self._governance.inspect(
            completion.text, direction=Direction.EGRESS, trace_id=result.trace_id
        )
        if not outbound.cleared:
            result.outcome = Outcome.BLOCKED
            result.reason = _safe_reason(
                f"local response refused on egress: {outbound.block_reason}"
            )
            return

        # Restore from the EGRESS-scrubbed text, not the raw completion, so a
        # residual secret the screen redacted cannot be reinstated on the way
        # to the user. Same ordering as the frontier path.
        result.answer = scrub.restore(outbound.safe_text)

    async def _execute_frontier(
        self,
        result: ExecutionResult,
        decision: RoutingDecision,
        envelope: GovernanceEnvelope,
        scrub: ScrubResult,
        context: _Context,
    ) -> None:
        """Level 2. Reached only with a live reservation in hand."""
        if not self._frontier.available:
            # Hand the capacity straight back rather than holding it while we
            # do something that was never going to need it.
            self._gate.abort(decision, reason="frontier tier unavailable")
            result.frontier_unavailable = True
            decision.degraded = True
            decision.reason = "Level 2 unavailable (no credentials); answered locally"
            await self._execute_local(result, decision, envelope, scrub, context)
            return

        prompt = _PROMPT.format(
            context=context.text or "(no local context retrieved)",
            question=envelope.safe_text,
        )

        # Last check before the bytes leave the machine. The ingress envelope
        # covered the question; this covers the question PLUS the retrieved
        # context that was concatenated onto it.
        outbound, _ = await self._governance.inspect(
            prompt, direction=Direction.EGRESS, trace_id=result.trace_id
        )
        if not outbound.cleared:
            self._gate.abort(decision, reason="egress governance refused the payload")
            decision.degraded = True
            decision.reason = _safe_reason(f"egress refused: {outbound.block_reason}")
            await self._execute_local(result, decision, envelope, scrub, context)
            return

        gen_t0 = time.perf_counter()
        try:
            async with self._gate.escalation(decision) as spend:
                completion = await self._frontier.complete(
                    outbound.safe_text,
                    system=_OPUS_SYSTEM,
                    max_tokens=self._settings.anthropic.max_output_tokens,
                )
                spend.record(completion.tokens_in, completion.tokens_out)
        except OpusUnavailable as exc:
            # Every Level 2 failure lands here: no key, bad key, DNS failure on
            # an air-gapped host, refused handshake, 429, timeout. All of them
            # degrade; none of them raise.
            result.frontier_unavailable = True
            decision.degraded = True
            decision.reason = _safe_reason(f"Level 2 call failed ({exc}); answered locally")
            await self._execute_local(result, decision, envelope, scrub, context)
            return

        result.generation_ms = (time.perf_counter() - gen_t0) * 1000.0
        result.cloud_tokens_in = completion.tokens_in
        result.cloud_tokens_out = completion.tokens_out

        # The response is screened too. A frontier model that echoes an
        # injected instruction back, or reproduces a secret from context, must
        # not reach the user unexamined.
        inbound, _ = await self._governance.inspect(
            completion.text, direction=Direction.EGRESS, trace_id=result.trace_id
        )
        if not inbound.cleared:
            result.outcome = Outcome.BLOCKED
            result.reason = _safe_reason(
                f"Level 2 response refused on egress: {inbound.block_reason}"
            )
            return

        result.outcome = Outcome.OPUS
        result.reason = decision.reason
        result.answer = scrub.restore(inbound.safe_text)

        # Harvest. The SCRUBBED text goes into the training set, never the
        # restored answer - the reverse map exists for the user, not the model.
        result.distilled = await self._harvest(
            question=envelope.safe_text, answer=inbound.safe_text
        )

    # --------------------------------------------------------- distillation

    async def _harvest(self, *, question: str, answer: str) -> bool:
        """Append a curated sample. Returns whether it survived curation."""
        distill: DistillSettings = self._settings.distill
        if not distill.enabled:
            return False

        sample = Sample(
            conversations=[
                Turn(role="system", value=distill.system_prompt),
                Turn(role="human", value=question),
                Turn(role="gpt", value=answer),
            ],
            source_model=self._frontier.model,
        )
        kept, _ = await self._formatter.curate([sample])
        if not kept:
            return False

        distill.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with distill.dataset_path.open("ab") as fh:
            fh.write(kept[0].to_jsonl() + b"\n")
        return True

    # ---------------------------------------------------------------- audit

    def _finish(
        self, result: ExecutionResult, t0: float, envelope: GovernanceEnvelope | None
    ) -> ExecutionResult:
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        if envelope is not None:
            result.pii_status = envelope.pii_status
            result.pii_entities = sum(envelope.pii_counts.values())

        flags = [f"outcome:{result.outcome}"]
        if result.degraded_reason:
            flags.append("degraded_local")
        if result.frontier_unavailable:
            flags.append("frontier_unavailable")
        if result.quarantined_chunks:
            flags.append("context_quarantined")
        if result.distilled:
            flags.append("harvested_for_distillation")
        if envelope is not None:
            flags.extend(envelope.compliance_flags())

        emitted = self._audit.emit(
            AuditRecord(
                trace_id=result.trace_id,
                stage="orchestration",
                agent_id=self._agent_id,
                tier="opus" if result.spent_cloud_tokens else "local",
                model_id=(
                    self._frontier.model
                    if result.spent_cloud_tokens
                    else self._settings.ollama.generator_model
                ),
                determinism_seed=self._settings.ollama.seed,
                prompt_hash=envelope.original_hash if envelope else hash_prompt(result.trace_id),
                pii_status=(
                    envelope.pii_status if envelope else PIIStatus.NOT_EVALUATED
                ),
                guardrail_verdict=(
                    envelope.guardrail_verdict if envelope else GuardrailVerdict.NOT_EVALUATED
                ),
                token_source=result.token_source,
                tokens_in=result.cloud_tokens_in or result.local_tokens_in,
                tokens_out=result.cloud_tokens_out or result.local_tokens_out,
                latency_ms=result.latency_ms,
                confidence=result.retrieval_confidence,
                decision_reason=result.reason,
                retrieval_sources=result.sources,
                compliance_flags=flags,
                extra={
                    "routing_stage": result.routing_stage,
                    "quarantined_chunks": result.quarantined_chunks,
                    "distilled": result.distilled,
                    "degraded_reason": result.degraded_reason,
                    "frontier_unavailable": result.frontier_unavailable,
                },
            )
        )
        result.audit_hash = emitted.record_hash
        return result
