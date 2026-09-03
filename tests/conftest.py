"""Shared test doubles.

Every stub here implements one of the Protocols the production code depends on,
which is the payoff of constructor injection: the whole Module 1 surface is
exercised with no Ollama daemon, no on-disk index and no network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import pytest

from hybrid_rag.llm.ollama_client import Completion, Embeddings
from hybrid_rag.rag.local_rag import Chunk, ScoredChunk, _cosine_to_unit


class StubEmbedder:
    """Deterministic hash embeddings. Same text always yields the same vector."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str], *, model: str | None = None) -> Embeddings:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.lower().encode()).digest()
            raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self._dim)]
            norm = sum(v * v for v in raw) ** 0.5 or 1.0
            vectors.append([v / norm for v in raw])
        return Embeddings(
            vectors=vectors,
            model=model or "stub-embed",
            fingerprint="stub-digest",
            dim=self._dim,
        )


class InMemoryVectorStore:
    """Brute-force cosine search. Correct, obviously so, and fast enough for tests."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.chunks: dict[str, Chunk] = {}
        self.dim: int | None = None

    async def ensure_collection(self, dim: int) -> None:
        self.dim = dim

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.vectors[chunk.point_id] = list(vector)
            self.chunks[chunk.point_id] = chunk
        return len(chunks)

    async def search(self, vector: Sequence[float], k: int) -> list[ScoredChunk]:
        query = list(vector)
        qn = sum(v * v for v in query) ** 0.5 or 1.0
        scored: list[tuple[float, str]] = []
        for pid, stored in self.vectors.items():
            sn = sum(v * v for v in stored) ** 0.5 or 1.0
            dot = sum(a * b for a, b in zip(query, stored, strict=True))
            scored.append((dot / (qn * sn), pid))
        scored.sort(reverse=True)
        return [
            ScoredChunk(
                chunk=self.chunks[pid],
                score=_cosine_to_unit(sim),
                raw_score=sim,
                rank=rank,
            )
            for rank, (sim, pid) in enumerate(scored[:k])
        ]

    async def count(self) -> int:
        return len(self.chunks)

    async def delete_collection(self) -> None:
        self.vectors.clear()
        self.chunks.clear()

    async def aclose(self) -> None:
        return None


class StubGenerator:
    """Scripted local model. `script` maps a substring of the prompt to a reply.

    The default is a bare "L" because tier 3 of the routing waterfall now asks
    the resident generator for a single token, not a JSON object.
    """

    def __init__(self, default: str = "L") -> None:
        self.default = default
        self.script: dict[str, str] = {}
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self.raises: Exception | None = None

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> Completion:
        del json_mode, system
        self.prompts.append(prompt)
        self.calls.append({"model": model, "max_tokens": max_tokens})
        if self.raises is not None:
            raise self.raises
        text = self.default
        for needle, reply in self.script.items():
            if needle.lower() in prompt.lower():
                text = reply
                break
        return Completion(
            text=text,
            model=model or "stub-router",
            fingerprint="stub-digest",
            tokens_in=len(prompt) // 4,
            tokens_out=16,
            latency_ms=1.0,
            seed=1337,
        )


class RecordingAuditSink:
    """Keeps records in memory so tests can assert on what was logged."""

    def __init__(self) -> None:
        self.records: list[Any] = []

    def emit(self, record: Any) -> Any:
        self.records.append(record)
        return record

    def close(self) -> None:
        return None


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def generator() -> StubGenerator:
    return StubGenerator()


@pytest.fixture
def audit() -> RecordingAuditSink:
    return RecordingAuditSink()
