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
from hybrid_rag.governance.guardrails import (
    Guardrail,
    GuardrailChain,
    LlamaGuardGuardrail,
    RegexGuardrail,
)
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

    def switch_generator(self, name: str) -> str:
        """Point the resident generator at another profile. Returns the model id.

        Accepts a profile key (`code`) or the model it maps to
        (`qwen2.5-coder:3b`), and NOTHING else. An arbitrary model name is
        refused on purpose: the co-residency envelope in `OllamaSettings` was
        measured against ~3B Q4 weights, and a 7B typed at a prompt does not
        fail - it silently spills into shared system memory and costs an order
        of magnitude on a single-channel bus. Adding a profile is a config
        change, which is reviewable; typing one at a REPL is not.

        The gate is re-pointed alongside the client because it holds the
        resolved name rather than the settings object.
        """
        ollama = self.settings.ollama
        profile = name if name in ollama.generator_profiles else None
        if profile is None:
            profile = next(
                (key for key, model in ollama.generator_profiles.items() if model == name),
                None,
            )
        if profile is None:
            known = ", ".join(
                f"{key} ({model})" for key, model in sorted(ollama.generator_profiles.items())
            )
            raise ValueError(f"unknown generator {name!r}; known profiles: {known}")

        ollama.generator_profile = profile
        # An override would win over the profile we just set, so clear it.
        ollama.generator_model_override = None
        resolved = ollama.generator_model
        self.gate.set_slm_model(resolved)
        return resolved

    def set_generation_params(
        self,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        keep_alive: int | str | None = None,
    ) -> dict[str, object]:
        """Retune the resident generator. Returns the settings now in force.

        `OllamaClient` holds the settings OBJECT and reads these per request, so
        a change here applies to the next call with no reconstruction.

        Two of these carry consequences worth stating rather than hiding:

        - `temperature` is pinned at 0.0 by default because the audit ledger
          records it beside the model digest so a decision can be replayed
          bit-for-bit. Raising it trades that reproducibility for variety, and
          the ledger will show the temperature that was actually used.
        - `num_ctx` is the fastest way to leave the VRAM envelope. 4096 on a 3B
          Q4 costs roughly 0.4 GB of KV cache; the cost scales with the context,
          and past the envelope Ollama spills into system memory rather than
          failing, which on a single-channel bus is about 10x slower.

        This mutates process-global settings. Acceptable because the server is
        single-user and loopback-only; it would be a race behind a shared one.
        """
        ollama = self.settings.ollama
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(f"temperature must be between 0.0 and 2.0, got {temperature}")
            ollama.temperature = temperature
        if num_ctx is not None:
            if not 256 <= num_ctx <= 131_072:
                raise ValueError(f"num_ctx must be between 256 and 131072, got {num_ctx}")
            ollama.num_ctx = num_ctx
        if keep_alive is not None:
            ollama.keep_alive = keep_alive
        return {
            "temperature": ollama.temperature,
            "num_ctx": ollama.num_ctx,
            "keep_alive": ollama.keep_alive,
            "seed": ollama.seed,
        }

    def add_generator_profile(self, name: str, model: str) -> str:
        """Register a new generator profile. Returns the model it maps to.

        `switch_generator` deliberately refuses any model that is not already a
        profile, which protects the measured co-residency envelope from a name
        typed at a prompt. This is the sanctioned way to widen that set, so it
        validates the shape of the name and that the model exists - and then
        allows it, reporting the fit rather than refusing on size.

        Refusing a large model outright would be the wrong call: whether a
        spill into system memory is acceptable is the operator's judgement, not
        this function's. What is not acceptable is making that trade silently,
        so callers get the assessment alongside.
        """
        cleaned = name.strip().lower()
        if not cleaned or not cleaned.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"profile name {name!r} must be alphanumeric, with - or _ separators"
            )
        if not model.strip():
            raise ValueError("a profile needs a model id")
        self.settings.ollama.generator_profiles[cleaned] = model.strip()
        return model.strip()

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
    # Two ORTHOGONAL layers, not one behind the other. Regex catches prompt
    # injection and secret exfiltration; Llama Guard catches harmful content.
    # Measurement showed Llama Guard flags none of the former - see its
    # docstring - so neither substitutes for the other.
    layers: list[Guardrail] = [RegexGuardrail()]
    if settings.ollama.guard_enabled:
        layers.append(
            LlamaGuardGuardrail(
                ollama,
                model=settings.ollama.guard_model,
                blocking_hazards=settings.ollama.blocking_hazards,
            )
        )
    governance = GovernanceMiddleware(
        scrubber=PIIScrubber(),
        guardrail=GuardrailChain(layers),
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
