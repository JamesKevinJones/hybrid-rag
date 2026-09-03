"""Exercises the real embedded Qdrant adapter.

Still no daemon and no network - qdrant-client's `path=` mode is a local
on-disk index. This is the only place the concrete backend is tested; the unit
suite stays on the in-memory double.
"""

from __future__ import annotations

import pytest

from hybrid_rag.config import RetrievalSettings, VectorSettings
from hybrid_rag.rag.local_rag import (
    Document,
    LocalRAGEngine,
    QdrantVectorStore,
    build_vector_store,
)

DOCS = [
    Document(
        doc_id="rmf",
        uri="corpus/rmf.md",
        text=(
            "Traceable provenance means every model output can be tied back to the "
            "inputs, the model version and the retrieval sources that produced it."
        ),
    ),
    Document(
        doc_id="budget",
        uri="corpus/budget.md",
        text=(
            "A rolling window limits how many requests may be issued in a fixed "
            "period. Reservations are taken before the call and settled after it."
        ),
    ),
]


@pytest.fixture
def vector_settings(tmp_path) -> VectorSettings:
    return VectorSettings(persist_dir=tmp_path / "index", collection="itest")


async def test_qdrant_roundtrip_is_persistent_and_idempotent(
    vector_settings, embedder
) -> None:
    settings = RetrievalSettings(chunk_tokens=64, chunk_overlap_tokens=8, rerank_enabled=False)

    store = build_vector_store(vector_settings)
    assert isinstance(store, QdrantVectorStore)
    engine = LocalRAGEngine(settings=settings, embedder=embedder, store=store)

    first = await engine.ingest(DOCS)
    assert first > 0
    count_after_first = await engine.count()

    # Re-ingesting identical content must overwrite by deterministic point id.
    await engine.ingest(DOCS)
    assert await engine.count() == count_after_first

    result = await engine.query("what does traceable provenance mean")
    assert result.results
    assert result.results[0].chunk.doc_id in {"rmf", "budget"}
    assert 0.0 <= result.confidence <= 1.0
    await engine.aclose()

    # Reopen the same directory: the index survived the process boundary.
    reopened = build_vector_store(vector_settings)
    engine2 = LocalRAGEngine(settings=settings, embedder=embedder, store=reopened)
    assert await engine2.count() == count_after_first
    await engine2.aclose()
