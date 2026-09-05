"""Composition root.

The only module allowed to call constructors of concrete backends. Every other
module receives its collaborators through `__init__`, which is what makes the
routing gate testable without Ollama and the RAG engine testable without a disk
index. If you find yourself importing `OllamaClient` anywhere else, the
dependency belongs here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from hybrid_rag.agents.orchestrator import AgentOrchestrator
from hybrid_rag.config import Settings, get_settings
from hybrid_rag.distill.dataset_formatter import DatasetFormatter
from hybrid_rag.eval.eval_harness import EvalHarness, GroundednessScorer
from hybrid_rag.eval.promotion_gate import PromotionGate
from hybrid_rag.governance.audit import AuditSink, JsonlAuditSink
from hybrid_rag.governance.guardrails import GuardrailChain, RegexGuardrail
from hybrid_rag.governance.middleware import GovernanceMiddleware
from hybrid_rag.governance.pii import PIIScrubber
from hybrid_rag.llm.ollama_client import OllamaClient
from hybrid_rag.llm.opus_client import (
    AnthropicOpusClient,
    FrontierClient,
    UnavailableFrontierClient,
)
from hybrid_rag.rag.local_rag import (
    FlashRankReranker,
    LocalRAGEngine,
    NullReranker,
    Reranker,
    SemanticChunker,
    VectorStore,
    VectorStoreError,
    build_vector_store,
)
from hybrid_rag.routing.budget import RollingWindowBudget
from hybrid_rag.routing.routing_gate import RoutingGate


@dataclass(slots=True)
class Container:
    settings: Settings
    audit: AuditSink
    governance: GovernanceMiddleware
    ollama: OllamaClient
    store: VectorStore
    rag: LocalRAGEngine
    budget: RollingWindowBudget
    gate: RoutingGate
    frontier: FrontierClient
    orchestrator: AgentOrchestrator
    harness: EvalHarness
    promotion: PromotionGate

    async def __aenter__(self) -> Container:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.rag.aclose()
        await self.ollama.aclose()
        await self.frontier.aclose()
        self.audit.close()


def _build_reranker(settings: Settings) -> Reranker:
    if not settings.retrieval.rerank_enabled:
        return NullReranker()
    try:
        return FlashRankReranker(
            settings.retrieval.rerank_model,
            cache_dir=settings.data_dir / "models",
        )
    except VectorStoreError:
        # Reranking is an optimisation, not a requirement. Degrade, do not crash.
        return NullReranker()


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    settings.ensure_dirs()

    audit = JsonlAuditSink(settings.audit.path, schema_version=settings.audit.schema_version)
    ollama = OllamaClient(settings.ollama, runtime=settings.runtime)
    # Llama-Guard is deliberately NOT chained in by default: the model is not
    # pulled by `doctor`, and a guardrail that silently fails open on every
    # request is worse than one that was never claimed. Add it explicitly once
    # `ollama pull llama-guard3:1b` has run.
    governance = GovernanceMiddleware(
        scrubber=PIIScrubber(),
        guardrail=GuardrailChain([RegexGuardrail()]),
        audit=audit,
    )
    store = build_vector_store(settings.vector)
    rag = LocalRAGEngine(
        settings=settings.retrieval,
        embedder=ollama,
        store=store,
        chunker=SemanticChunker(settings.retrieval, embedder=ollama),
        reranker=_build_reranker(settings),
        audit=audit,
        runtime=settings.runtime,
    )
    budget = RollingWindowBudget(settings.budget)
    gate = RoutingGate(
        settings=settings.routing,
        budget=budget,
        audit=audit,
        generator=ollama,
        verifier=governance,
        slm_model=settings.ollama.generator_model,
        slm_max_tokens=settings.ollama.router_max_tokens,
    )
    # The null object rather than None: the degradation path is then the same
    # code in development as in production, instead of a branch that only runs
    # when something is misconfigured.
    live = AnthropicOpusClient(settings.anthropic)
    frontier: FrontierClient = (
        live if live.available else UnavailableFrontierClient("no ANTHROPIC_API_KEY configured")
    )
    formatter = DatasetFormatter()
    orchestrator = AgentOrchestrator(
        settings=settings,
        governance=governance,
        rag=rag,
        gate=gate,
        local=ollama,
        frontier=frontier,
        audit=audit,
        formatter=formatter,
    )
    harness = EvalHarness(
        settings=settings.eval,
        scorer=GroundednessScorer(ollama, settings.eval),
    )
    promotion = PromotionGate(settings=settings.promotion, harness=harness)

    return Container(
        settings=settings,
        audit=audit,
        governance=governance,
        ollama=ollama,
        store=store,
        rag=rag,
        budget=budget,
        gate=gate,
        frontier=frontier,
        orchestrator=orchestrator,
        harness=harness,
        promotion=promotion,
    )
