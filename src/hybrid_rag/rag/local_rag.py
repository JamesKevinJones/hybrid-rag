"""Module 1b - the local vector engine.

Everything here runs on the workstation and costs nothing but electricity.
The backend is a protocol, not a class: Qdrant is the default because its
embedded mode is a single on-disk directory with no server process, but Chroma
drops in behind the same interface, and so will anything else that can store a
vector and return neighbours.

The output that matters to the rest of the system is
`RetrievalResult.confidence` - a calibrated scalar in [0, 1] that the routing
gate compares against `escalation_threshold` (0.78). Its formula is documented
on `calibrate_confidence()` and is deliberately explainable; a number that
decides whether you spend cloud quota should not come out of a black box.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from hybrid_rag.config import (
    RetrievalSettings,
    RuntimeSettings,
    VectorBackend,
    VectorSettings,
)
from hybrid_rag.governance.audit import (
    AuditRecord,
    AuditSink,
    SourceAttribution,
    TokenSource,
    hash_prompt,
)
from hybrid_rag.llm.ollama_client import EmbeddingBackend

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}")
_WHITESPACE = re.compile(r"\s+")
_NAMESPACE = uuid.UUID("6f0f3f3e-2a1b-4f8f-9c3a-4a1d2b8e7c55")


class VectorStoreError(RuntimeError):
    """The index could not be opened, written to, or queried."""


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


class Document(BaseModel):
    doc_id: str
    text: str
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def point_id(self) -> str:
        """Deterministic UUID - re-ingesting the same chunk overwrites, never duplicates."""
        return str(uuid.uuid5(_NAMESPACE, self.chunk_id))


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float  # normalized to [0, 1]
    raw_score: float
    rank: int
    reranked: bool = False

    def to_attribution(self) -> SourceAttribution:
        return SourceAttribution(
            doc_id=self.chunk.doc_id,
            chunk_id=self.chunk.chunk_id,
            uri=self.chunk.uri,
            score=self.score,
            rank=self.rank,
        )


class RetrievalResult(BaseModel):
    query: str
    results: list[ScoredChunk]
    confidence: float
    top_score: float
    margin: float
    coverage: float
    embed_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    reranked: bool = False

    @property
    def context(self) -> str:
        """Chunks joined with source markers, ready to drop into a grounded prompt."""
        return "\n\n".join(
            f"[{i + 1}] ({sc.chunk.uri or sc.chunk.doc_id}) {sc.chunk.text}"
            for i, sc in enumerate(self.results)
        )

    @property
    def attributions(self) -> list[SourceAttribution]:
        return [sc.to_attribution() for sc in self.results]


@dataclass(slots=True, frozen=True)
class Triplet:
    """(Query, Positive, Negative) mined locally for the distillation pipeline."""

    query: str
    positive: str
    negative: str
    positive_score: float
    negative_score: float
    doc_id: str


# --------------------------------------------------------------------------
# Token estimation and chunking
# --------------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """Tokenizer-free length estimate.

    Deliberately not tiktoken: the exact count does not matter for chunk sizing
    and a hard dependency on a tokenizer that does not match the local model's
    vocabulary would be false precision. Words x 1.3 tracks BPE closely enough
    for English prose and code comments alike.
    """
    words = len(_WHITESPACE.split(text.strip())) if text.strip() else 0
    return int(words * 1.3) + 1


@runtime_checkable
class Chunker(Protocol):
    async def split(self, document: Document) -> list[Chunk]: ...


class SemanticChunker:
    """Sentence-aligned windows, optionally split at semantic discontinuities.

    Fixed-width chunking cuts arguments in half; pure semantic chunking is
    quadratic in embedding calls. This does the cheap thing first - pack whole
    sentences into a token budget - then, when `semantic_split_threshold > 0`,
    embeds the sentence sequence once and forces a boundary wherever adjacent
    sentence similarity falls off a cliff.
    """

    def __init__(
        self,
        settings: RetrievalSettings,
        *,
        embedder: EmbeddingBackend | None = None,
        max_sentences_for_semantic: int = 400,
    ) -> None:
        self._s = settings
        self._embedder = embedder
        self._max_sentences = max_sentences_for_semantic

    async def split(self, document: Document) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(document.text) if s.strip()]
        if not sentences:
            return []

        breakpoints: set[int] = set()
        use_semantic = (
            self._embedder is not None
            and self._s.semantic_split_threshold > 0.0
            and 1 < len(sentences) <= self._max_sentences
        )
        if use_semantic:
            breakpoints = await self._semantic_breakpoints(sentences)

        chunks: list[Chunk] = []
        window: list[str] = []
        window_tokens = 0
        ordinal = 0

        def flush() -> None:
            nonlocal window, window_tokens, ordinal
            if not window:
                return
            text = " ".join(window)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{ordinal:04d}",
                    doc_id=document.doc_id,
                    text=text,
                    ordinal=ordinal,
                    uri=document.uri,
                    metadata=dict(document.metadata),
                )
            )
            ordinal += 1
            # Carry the tail forward as overlap so a boundary never orphans context.
            overlap: list[str] = []
            budget = self._s.chunk_overlap_tokens
            for sentence in reversed(window):
                cost = approx_tokens(sentence)
                if cost > budget:
                    break
                overlap.insert(0, sentence)
                budget -= cost
            window = overlap
            window_tokens = sum(approx_tokens(s) for s in window)

        for i, sentence in enumerate(sentences):
            cost = approx_tokens(sentence)
            if window and (window_tokens + cost > self._s.chunk_tokens or i in breakpoints):
                flush()
            window.append(sentence)
            window_tokens += cost
        flush()
        # The final flush leaves the overlap tail unemitted by design - it is
        # already contained in the chunk before it.
        return chunks

    async def _semantic_breakpoints(self, sentences: list[str]) -> set[int]:
        assert self._embedder is not None
        embeddings = await self._embedder.embed(sentences)
        matrix = np.asarray(embeddings.vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit = matrix / norms
        sims = np.sum(unit[:-1] * unit[1:], axis=1)
        return {i + 1 for i, s in enumerate(sims) if float(s) < self._s.semantic_split_threshold}


# --------------------------------------------------------------------------
# Vector store abstraction
# --------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    async def ensure_collection(self, dim: int) -> None: ...
    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int: ...
    async def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]: ...
    async def count(self) -> int: ...
    async def delete_collection(self) -> None: ...
    async def aclose(self) -> None: ...


def _payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "ordinal": chunk.ordinal,
        "uri": chunk.uri,
        "metadata": chunk.metadata,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(payload.get("chunk_id", "")),
        doc_id=str(payload.get("doc_id", "")),
        text=str(payload.get("text", "")),
        ordinal=int(payload.get("ordinal", 0)),
        uri=payload.get("uri"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _cosine_to_unit(similarity: float) -> float:
    """Cosine similarity lives in [-1, 1]; every score in this system lives in [0, 1]."""
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


class QdrantVectorStore:
    """Embedded Qdrant. One directory on disk, no daemon, no port."""

    def __init__(self, settings: VectorSettings) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError("qdrant-client is not installed") from exc
        settings.persist_dir.mkdir(parents=True, exist_ok=True)
        self._s = settings
        self._client = QdrantClient(path=str(settings.persist_dir))

    async def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import (
            Distance,
            HnswConfigDiff,
            ScalarQuantization,
            ScalarQuantizationConfig,
            ScalarType,
            VectorParams,
        )

        distance = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclid": Distance.EUCLID,
        }[self._s.distance]

        def _run() -> None:
            if self._client.collection_exists(self._s.collection):
                self._assert_dim(dim)
                return
            self._client.create_collection(
                collection_name=self._s.collection,
                vectors_config=VectorParams(
                    size=dim, distance=distance, on_disk=self._s.on_disk
                ),
                hnsw_config=HnswConfigDiff(
                    m=self._s.hnsw_m,
                    ef_construct=self._s.hnsw_ef_construct,
                    on_disk=self._s.on_disk,
                ),
                # int8 scalar quantization: a quarter of the resident footprint
                # for a few points of recall, recovered by the cross-encoder
                # re-rank downstream. Embedded mode ignores it; a real Qdrant
                # server honours it, and this is written for both.
                quantization_config=(
                    ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8, always_ram=False
                        )
                    )
                    if self._s.quantize_int8
                    else None
                ),
            )

        await asyncio.to_thread(_run)

    def _assert_dim(self, dim: int) -> None:
        """Fail loudly on an embedding-model change.

        A 384-dim query against a 768-dim index does not error on its own - it
        returns confident nonsense, which is the worst failure mode available.
        """
        try:
            info = self._client.get_collection(self._s.collection)
            params = info.config.params.vectors
            existing = getattr(params, "size", None)
        except Exception:
            return
        if existing is not None and int(existing) != dim:
            raise VectorStoreError(
                f"collection {self._s.collection!r} holds {existing}-dim vectors but the "
                f"embedder produces {dim}. The embedding model changed: point "
                f"vector.collection at a new name and re-ingest, or delete "
                f"{self._s.persist_dir}."
            )

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=c.point_id, vector=list(v), payload=_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        if not points:
            return 0
        await asyncio.to_thread(
            self._client.upsert, collection_name=self._s.collection, points=points, wait=True
        )
        return len(points)

    async def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        def _run() -> list[Any]:
            # An index that was never built is an empty index, not an error.
            # Raising here would propagate out of an MCP tool call as a stack
            # trace; returning nothing yields confidence 0.0, which escalates -
            # the correct fail-safe for "we have no local evidence".
            if not self._client.collection_exists(self._s.collection):
                return []
            response = self._client.query_points(
                collection_name=self._s.collection,
                query=list(vector),
                limit=k,
                with_payload=True,
            )
            return list(response.points)

        hits = await asyncio.to_thread(_run)
        out: list[ScoredChunk] = []
        for rank, hit in enumerate(hits):
            payload = dict(hit.payload or {})
            raw = float(hit.score)
            out.append(
                ScoredChunk(
                    chunk=_chunk_from_payload(payload),
                    score=_cosine_to_unit(raw) if self._s.distance == "cosine" else raw,
                    raw_score=raw,
                    rank=rank,
                )
            )
        return out

    async def count(self) -> int:
        def _run() -> int:
            if not self._client.collection_exists(self._s.collection):
                return 0
            return int(self._client.count(self._s.collection, exact=True).count)

        return await asyncio.to_thread(_run)

    async def delete_collection(self) -> None:
        await asyncio.to_thread(self._client.delete_collection, self._s.collection)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)


class ChromaVectorStore:
    """Alternate backend behind the identical protocol. Requires the `chroma` extra."""

    def __init__(self, settings: VectorSettings) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError("chromadb is not installed (extra: chroma)") from exc
        settings.persist_dir.mkdir(parents=True, exist_ok=True)
        self._s = settings
        self._client = chromadb.PersistentClient(path=str(settings.persist_dir))
        self._collection: Any = None

    async def ensure_collection(self, dim: int) -> None:
        del dim

        def _run() -> None:
            self._collection = self._client.get_or_create_collection(
                name=self._s.collection,
                metadata={"hnsw:space": self._s.distance},
            )

        await asyncio.to_thread(_run)

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if self._collection is None:
            await self.ensure_collection(len(vectors[0]))

        def _run() -> None:
            self._collection.upsert(
                ids=[c.point_id for c in chunks],
                embeddings=[list(v) for v in vectors],
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "ordinal": c.ordinal,
                        "uri": c.uri or "",
                    }
                    for c in chunks
                ],
            )

        await asyncio.to_thread(_run)
        return len(chunks)

    async def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        if self._collection is None:
            await self.ensure_collection(len(vector))

        def _run() -> dict[str, Any]:
            return dict(
                self._collection.query(
                    query_embeddings=[list(vector)],
                    n_results=k,
                    include=["documents", "metadatas", "distances"],
                )
            )

        res = await asyncio.to_thread(_run)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[ScoredChunk] = []
        for rank, (text, meta, dist) in enumerate(zip(docs, metas, dists, strict=False)):
            similarity = 1.0 - float(dist)
            payload = dict(meta or {})
            payload["text"] = text
            out.append(
                ScoredChunk(
                    chunk=_chunk_from_payload(payload),
                    score=_cosine_to_unit(similarity),
                    raw_score=float(dist),
                    rank=rank,
                )
            )
        return out

    async def count(self) -> int:
        if self._collection is None:
            await self.ensure_collection(0)
        return int(await asyncio.to_thread(self._collection.count))

    async def delete_collection(self) -> None:
        await asyncio.to_thread(self._client.delete_collection, self._s.collection)
        self._collection = None

    async def aclose(self) -> None:
        return None


def build_vector_store(settings: VectorSettings) -> VectorStore:
    """Factory. The only place a backend class name appears outside its own module."""
    if settings.backend is VectorBackend.QDRANT:
        return QdrantVectorStore(settings)
    if settings.backend is VectorBackend.CHROMA:
        return ChromaVectorStore(settings)
    raise VectorStoreError(f"unknown vector backend: {settings.backend}")


# --------------------------------------------------------------------------
# Re-ranking
# --------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class NullReranker:
    """Identity re-ranker. Keeps the pipeline shape identical when reranking is off."""

    async def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        del query
        return list(candidates)


def _to_unit(score: float) -> float:
    """Squash an unbounded cross-encoder logit into [0, 1]. Already-bounded scores pass through."""
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


class FlashRankReranker:
    """CPU-only ONNX cross-encoder. Roughly 50 MB, no GPU contention with Ollama."""

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        try:
            from flashrank import Ranker
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise VectorStoreError("flashrank is not installed (extra: rerank)") from exc
        self._ranker = Ranker(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    async def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        from flashrank import RerankRequest

        passages = [{"id": i, "text": c.chunk.text, "meta": {}} for i, c in enumerate(candidates)]

        def _run() -> list[dict[str, Any]]:
            return list(self._ranker.rerank(RerankRequest(query=query, passages=passages)))

        ranked = await asyncio.to_thread(_run)
        out: list[ScoredChunk] = []
        for rank, row in enumerate(ranked):
            original = candidates[int(row["id"])]
            raw = float(row["score"])
            out.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=_to_unit(raw),
                    raw_score=raw,
                    rank=rank,
                    reranked=True,
                )
            )
        return out


# --------------------------------------------------------------------------
# Confidence calibration
# --------------------------------------------------------------------------


def calibrate_confidence(
    scored: Sequence[ScoredChunk],
    settings: RetrievalSettings,
    *,
    relevance_floor: float | None = None,
) -> tuple[float, float, float, float]:
    """Turn a ranked list into one number the routing gate can threshold on.

    Three orthogonal questions, weighted:

      top_score   Is the best match actually good?          (w_top_score)
      margin      Is it clearly better than the runner-up?  (w_margin)
      coverage    Did enough passages clear the floor?      (w_coverage)

    A single strong hit with nothing behind it scores lower than a strong hit
    backed by corroborating passages - which is the behaviour you want, because
    the failure mode of a local RAG stack is confident retrieval of one
    plausible-looking chunk. Returns (confidence, top, margin, coverage).
    """
    if not scored:
        return 0.0, 0.0, 0.0, 0.0

    floor = settings.relevance_floor if relevance_floor is None else relevance_floor
    top = float(scored[0].score)
    second = float(scored[1].score) if len(scored) > 1 else 0.0
    margin = (top - second) / top if top > 1e-9 else 0.0
    margin = max(0.0, min(1.0, margin))
    above_floor = sum(1 for s in scored if s.score >= floor)
    coverage = min(1.0, above_floor / max(1, min(settings.top_k, len(scored))))

    # A margin over noise is noise. When the top score is itself below the
    # relevance floor, the ratio between it and the runner-up is numerically
    # large and semantically empty - measured live, a nonsense query scored
    # top=0.002 with margin=0.871 and rode that to a 0.35 confidence. Margin
    # only counts as evidence when there is something to have a margin over.
    effective_margin = margin if top >= floor else 0.0
    confidence = settings.w_top_score * top + settings.w_support * max(
        effective_margin, coverage
    )
    return round(max(0.0, min(1.0, confidence)), 4), top, round(margin, 4), round(coverage, 4)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class LocalRAGEngine:
    """Level 0 retrieval. Constructed once, injected everywhere."""

    def __init__(
        self,
        *,
        settings: RetrievalSettings,
        embedder: EmbeddingBackend,
        store: VectorStore,
        chunker: Chunker | None = None,
        reranker: Reranker | None = None,
        audit: AuditSink | None = None,
        runtime: RuntimeSettings | None = None,
        agent_id: str = "local_rag",
    ) -> None:
        self._s = settings
        # Bounds the CPU-side work - ONNX re-ranking and embedding batches -
        # for the same reason the Ollama client is bounded: on a single-channel
        # host, parallel work contends for memory bandwidth instead of using
        # idle cores.
        self._gate = asyncio.Semaphore((runtime or RuntimeSettings()).max_workers)
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or SemanticChunker(settings, embedder=embedder)
        self._reranker = reranker or NullReranker()
        self._audit = audit
        self._agent_id = agent_id

    async def ingest(self, documents: Iterable[Document], *, batch_size: int = 32) -> int:
        """Chunk, embed and upsert. Returns the number of chunks written."""
        await self._store.ensure_collection(self._embedder.dim)
        written = 0
        for document in documents:
            chunks = await self._chunker.split(document)
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                async with self._gate:
                    embeddings = await self._embedder.embed([c.text for c in batch])
                written += await self._store.upsert(batch, embeddings.vectors)
        return written

    async def ingest_paths(self, paths: Iterable[Path], *, encoding: str = "utf-8") -> int:
        """Convenience loader for a local corpus directory."""
        documents: list[Document] = []
        for path in paths:
            text = path.read_text(encoding=encoding, errors="replace")
            documents.append(
                Document(
                    doc_id=hashlib.sha256(str(path).encode()).hexdigest()[:16],
                    text=text,
                    uri=path.as_posix(),
                    metadata={"filename": path.name, "bytes": path.stat().st_size},
                )
            )
        return await self.ingest(documents)

    async def query(
        self,
        query: str,
        *,
        top_k: int | None = None,
        fetch_k: int | None = None,
        trace_id: str | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self._s.top_k
        fetch_k = fetch_k or max(self._s.fetch_k, top_k)

        t0 = time.perf_counter()
        embedded = await self._embedder.embed([query])
        embed_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        candidates = await self._store.search(embedded.vectors[0], fetch_k)
        search_ms = (time.perf_counter() - t1) * 1000.0

        rerank_ms = 0.0
        reranked = False
        if self._s.rerank_enabled and candidates and not isinstance(self._reranker, NullReranker):
            t2 = time.perf_counter()
            async with self._gate:
                candidates = await self._reranker.rerank(query, candidates)
            rerank_ms = (time.perf_counter() - t2) * 1000.0
            reranked = True

        final = candidates[:top_k]
        for rank, chunk in enumerate(final):
            chunk.rank = rank

        confidence, top, margin, coverage = calibrate_confidence(final, self._s)
        result = RetrievalResult(
            query=query,
            results=final,
            confidence=confidence,
            top_score=top,
            margin=margin,
            coverage=coverage,
            embed_latency_ms=embed_ms,
            search_latency_ms=search_ms,
            rerank_latency_ms=rerank_ms,
            reranked=reranked,
        )
        self._emit_audit(query, result, embedded.model, embedded.fingerprint, trace_id)
        return result

    async def mine_triplets(
        self,
        queries: Sequence[str],
        *,
        positive_floor: float = 0.72,
        negative_ceiling: float = 0.45,
        per_query: int = 1,
    ) -> list[Triplet]:
        """Harvest (Query, Positive, Negative) triplets for the distillation set.

        Negatives are mined from the same retrieval call rather than sampled at
        random: a passage the retriever *almost* liked is a far harder negative
        than an unrelated document, and hard negatives are what actually move
        an embedding or a distilled reader.
        """
        triplets: list[Triplet] = []
        for query in queries:
            result = await self.query(query, top_k=self._s.fetch_k)
            chosen_positives = [s for s in result.results if s.score >= positive_floor][:per_query]
            positive_ids = {s.chunk.chunk_id for s in chosen_positives}
            # Ordered by descending score, so the hardest negatives come first.
            # A passage cannot be both, no matter how the thresholds are set.
            negatives = [
                s
                for s in result.results
                if s.score <= negative_ceiling and s.chunk.chunk_id not in positive_ids
            ]
            for pos, neg in zip(chosen_positives, negatives, strict=False):
                if neg.score >= pos.score:
                    continue
                triplets.append(
                    Triplet(
                        query=query,
                        positive=pos.chunk.text,
                        negative=neg.chunk.text,
                        positive_score=pos.score,
                        negative_score=neg.score,
                        doc_id=pos.chunk.doc_id,
                    )
                )
        return triplets

    async def count(self) -> int:
        return await self._store.count()

    async def aclose(self) -> None:
        await self._store.aclose()

    def _emit_audit(
        self,
        query: str,
        result: RetrievalResult,
        model: str,
        fingerprint: str | None,
        trace_id: str | None,
    ) -> None:
        if self._audit is None:
            return
        self._audit.emit(
            AuditRecord(
                trace_id=trace_id or str(uuid.uuid4()),
                stage="local_retrieval",
                agent_id=self._agent_id,
                tier="local",
                model_id=model,
                model_fingerprint=fingerprint,
                prompt_hash=hash_prompt(query),
                prompt_chars=len(query),
                token_source=TokenSource.LOCAL_OLLAMA,
                latency_ms=result.embed_latency_ms
                + result.search_latency_ms
                + result.rerank_latency_ms,
                confidence=result.confidence,
                decision_reason=(
                    f"top={result.top_score:.3f} margin={result.margin:.3f} "
                    f"coverage={result.coverage:.3f}"
                ),
                retrieval_sources=result.attributions,
                compliance_flags=["reranked"] if result.reranked else [],
                extra={
                    "n_results": len(result.results),
                    "embed_ms": round(result.embed_latency_ms, 2),
                    "search_ms": round(result.search_latency_ms, 2),
                    "rerank_ms": round(result.rerank_latency_ms, 2),
                },
            )
        )
