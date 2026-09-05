"""Typed, environment-driven configuration.

Single source of truth for every tunable in the system. Nothing else in the
codebase reads `os.environ` directly - modules receive a settings object (or a
narrow slice of one) through their constructor, so tests inject a literal
`Settings(...)` instead of monkeypatching the environment.

Environment binding uses a `HYBRID_RAG_` prefix and `__` as the nesting
delimiter, e.g. `HYBRID_RAG_ROUTING__ESCALATION_THRESHOLD=0.85`.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class VectorBackend(StrEnum):
    QDRANT = "qdrant"
    CHROMA = "chroma"


class OllamaSettings(BaseModel):
    """Level 0 inference plane, sized for a 5.2 GB usable VRAM budget.

    The hard number: 6 GB card minus roughly 0.8 GB held by the Windows desktop
    compositor. Whatever is left has to hold the weights AND the KV cache.

    **One model, pinned.** An earlier build ran a 1.5B router alongside the 3B
    generator. Ollama evicts to fit, so every query that touched both paid a
    2-4 second cold load over PCIe - on a single-channel bus that reload is the
    dominant cost of the whole request, dwarfing the inference it enables. The
    router is gone; tier 3 of the waterfall asks the generator that is already
    resident. `keep_alive: -1` pins it so nothing evicts it between queries.

    Embeddings run on a 45 MB MiniLM rather than the 274 MB nomic model: 384
    dimensions instead of 768 halves the index in RAM, and at this corpus size
    the cross-encoder re-rank is what carries retrieval quality anyway.
    """

    base_url: str = "http://localhost:11434"
    # The single resident model. Tier 3 routing and generation share it.
    generator_model: str = "llama3.2:3b-instruct-q4_K_M"
    embed_model: str = "all-minilm:l6-v2"
    embed_dim: int = 384
    # 4096 tokens of KV cache on a 3B Q4 is roughly 0.4 GB. Raising this is the
    # fastest way to push the model into shared system memory and lose 10x.
    num_ctx: int = 4096
    # -1 pins the model in VRAM indefinitely. This is the single most important
    # latency setting in the file; a duration string here reintroduces eviction.
    keep_alive: int | str = -1
    request_timeout_s: float = 120.0
    # Pinned for reproducibility. The audit ledger records this alongside the
    # model digest so a decision can be replayed bit-for-bit.
    seed: int = 1337
    temperature: float = 0.0
    max_retries: int = 2
    # Tier 3 emits exactly one token: "L" or "O". Anything longer is the model
    # explaining itself, which costs latency and buys nothing.
    router_max_tokens: int = 1
    # Optional second guardrail layer. Loading it evicts the generator on this
    # hardware, so it is off unless explicitly wired in container.py.
    guard_model: str = "llama-guard3:1b"


class RuntimeSettings(BaseModel):
    """Host-level concurrency limits.

    Single-channel DDR gives roughly 22 GB/s, about half a dual-channel rig.
    Fanning out embedding batches or ONNX re-ranks across every core does not
    go faster - the cores queue on memory, latency variance explodes, and the
    Ollama server starts competing with its own client for bandwidth. Two
    workers is the measured sweet spot; it is a semaphore, not a thread pool,
    so it also bounds in-flight HTTP requests to the model server.
    """

    max_workers: int = 2


class VectorSettings(BaseModel):
    backend: VectorBackend = VectorBackend.QDRANT
    persist_dir: Path = REPO_ROOT / "data" / "index"
    # The embedding model is part of the collection's identity: a 384-dim index
    # cannot answer a 768-dim query, and silently mixing them returns nonsense
    # rather than an error. Changing embed_model means re-ingesting.
    collection: str = "hybrid_rag_minilm_384"
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    # Keep vectors and the HNSW graph on disk instead of resident in RAM.
    # Honest caveat: qdrant-client's embedded mode does brute-force search and
    # ignores both of these. They are correct, and they take effect unchanged
    # the moment this points at a real Qdrant server - which is the upgrade
    # path when the corpus outgrows a laptop.
    on_disk: bool = True
    quantize_int8: bool = True
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128


class RetrievalSettings(BaseModel):
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    # Semantic split fires when adjacent-window cosine similarity drops below
    # this. Set to 0.0 to fall back to pure fixed-window chunking.
    semantic_split_threshold: float = 0.62
    fetch_k: int = 24
    top_k: int = 6
    rerank_enabled: bool = True
    # FlashRank's smallest cross-encoder: ~4 MB versus ~34 MB for MiniLM-L-12.
    # Both are CPU-only ONNX and neither touches VRAM. See decision 15 for the
    # measured quality cost of the smaller one.
    rerank_model: str = "ms-marco-TinyBERT-L-2-v2"
    # The score above which a passage counts as relevant. This is a property of
    # the CROSS-ENCODER, not a universal constant, and it must be re-measured
    # whenever rerank_model changes. Measured for TinyBERT-L-2 on this corpus:
    # junk queries top out at 0.0007, the weakest genuine match scores 0.096.
    # 0.05 sits in that gap with two orders of magnitude of headroom either side.
    # MiniLM-L-12 needs roughly 0.5 instead - its scores are far less bimodal.
    relevance_floor: float = 0.05
    # Confidence calibration weights. Must sum to 1.0. See calibrate_confidence()
    # for why there are two terms and not three.
    w_top_score: float = 0.60
    w_support: float = 0.40

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> RetrievalSettings:
        total = self.w_top_score + self.w_support
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"confidence weights must sum to 1.0, got {total:.6f} "
                f"({self.w_top_score} + {self.w_support})"
            )
        return self


class RoutingSettings(BaseModel):
    """Level 0 -> Level 2 escalation policy."""

    # Below this calibrated retrieval confidence the local answer is not
    # trustworthy and the query is escalated. Spec-mandated default.
    escalation_threshold: float = 0.78
    # Ask the local SLM for a LOCAL/OPUS vote. Costs ~200 ms locally, zero cloud
    # tokens. Disable for latency-critical paths.
    slm_vote_enabled: bool = True
    # A heuristic score at or above this escalates without consulting the SLM.
    heuristic_hard_escalate: float = 0.75
    # A heuristic score at or below this stays local without consulting the SLM.
    heuristic_hard_local: float = 0.15
    # Global kill switch: pin every request to Level 0 regardless of signals.
    force_local: bool = False
    # Architecture invariant 1: no payload crosses the token boundary without a
    # verified GovernanceEnvelope. Turning this off is a deliberate, auditable
    # downgrade - it is not a performance knob.
    require_envelope: bool = True
    # Rough chars-per-token divisor used for pre-flight budget reservation.
    chars_per_token: float = 3.6


class BudgetSettings(BaseModel):
    """Claude Pro rolling-window accounting.

    Anthropic tunes the published Pro limits over time, so these are inputs,
    not constants. Set them from what your own account actually shows and the
    gate enforces whatever you declare. `reserve_fraction` keeps headroom so an
    unattended batch job cannot consume the window you wanted for interactive
    work.
    """

    window_hours: float = 5.0
    max_messages_per_window: int = 45
    max_tokens_per_window: int = 900_000
    weekly_message_cap: int = 900
    reserve_fraction: float = 0.20
    ledger_path: Path = REPO_ROOT / "data" / "audit" / "budget_ledger.jsonl"
    # Longest a reservation may sit uncommitted before the sweeper reclaims it.
    reservation_ttl_s: float = 900.0


class AuditSettings(BaseModel):
    path: Path = REPO_ROOT / "data" / "audit" / "governance_audit.jsonl"
    schema_version: str = "1.0-nist-ai-rmf"
    # Store only the hash of prompts, never the raw text.
    hash_prompts_only: bool = True
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4318/v1/traces"


class AnthropicSettings(BaseModel):
    model: str = "claude-opus-5"
    max_output_tokens: int = 8192
    temperature: float = 0.0
    api_key: SecretStr | None = None
    request_timeout_s: float = 600.0


class DistillSettings(BaseModel):
    """Where Level 2 completions are harvested to."""

    enabled: bool = True
    dataset_path: Path = REPO_ROOT / "data" / "datasets" / "distillation_train.jsonl"
    # Baked into every harvested sample so the distilled model learns the role
    # it will actually be asked to play.
    system_prompt: str = (
        "You are a grounded technical assistant. Answer only from the provided "
        "context. If the context does not contain the answer, say so plainly."
    )


class EvalSettings(BaseModel):
    """Local evaluation. No paid LLM-as-a-judge anywhere in this pipeline.

    Groundedness is scored by embedding each answer sentence and each context
    chunk locally, then asking whether any chunk supports it. A cloud judge
    would be more accurate and would also make the eval cost more than the
    thing it evaluates - which on a Claude Pro budget is self-defeating.
    """

    cases_path: Path = REPO_ROOT / "data" / "datasets" / "eval_cases.jsonl"
    baseline_path: Path = REPO_ROOT / "data" / "datasets" / "opus_baseline.jsonl"
    reports_dir: Path = REPO_ROOT / "data" / "datasets" / "eval_reports"
    # Cosine similarity between an answer sentence and a context chunk above
    # which the sentence counts as supported. Measured on all-minilm:l6-v2.
    support_similarity: float = 0.62
    # Content-word overlap that also counts as support. Catches verbatim
    # quotation, which embeddings score oddly when the sentence is very short.
    support_overlap: float = 0.55
    # Sentences shorter than this are boilerplate ("Sure.", "In summary:") and
    # are excluded rather than counted as unsupported.
    min_scored_sentence_chars: int = 25


class PromotionSettings(BaseModel):
    """The gate a fine-tuned candidate has to clear before it serves traffic."""

    # Fraction of the Opus baseline's groundedness the candidate must reach.
    min_groundedness_parity: float = 0.85
    # Hard SLA. A model that is accurate and slow still fails.
    max_latency_p95_ms: float = 1200.0
    # Instruction-following may not regress at all; format and negative
    # constraints are the difference between a usable answer and a broken one.
    min_instruction_precision: float = 0.90
    # The alias local callers point at. Promotion re-points this, nothing else.
    serving_alias: str = "hybrid-rag-serving"
    ollama_binary: str = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HYBRID_RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = REPO_ROOT / "data"
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    distill: DistillSettings = Field(default_factory=DistillSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    promotion: PromotionSettings = Field(default_factory=PromotionSettings)

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.vector.persist_dir,
            self.audit.path.parent,
            self.budget.ledger_path.parent,
            self.distill.dataset_path.parent,
            self.eval.reports_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call `get_settings.cache_clear()` in tests."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
