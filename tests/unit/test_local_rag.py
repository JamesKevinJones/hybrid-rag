from __future__ import annotations

import pytest

from hybrid_rag.config import RetrievalSettings
from hybrid_rag.rag.local_rag import (
    Chunk,
    Document,
    LocalRAGEngine,
    ScoredChunk,
    SemanticChunker,
    calibrate_confidence,
)

CORPUS = [
    Document(
        doc_id="nist",
        uri="corpus/nist.md",
        text=(
            "The NIST AI Risk Management Framework organizes trustworthiness into four "
            "functions. Govern establishes policy and accountability. Map builds context "
            "for the system. Measure quantifies risk with metrics and evaluation. Manage "
            "allocates resources against prioritized risk. Traceable provenance is a "
            "recurring requirement across all four."
        ),
    ),
    Document(
        doc_id="qlora",
        uri="corpus/qlora.md",
        text=(
            "QLoRA fine-tunes a quantized base model by backpropagating through frozen "
            "4-bit weights into low rank adapters. Memory savings come from the NF4 data "
            "type and double quantization. Adapter rank controls capacity. Paged "
            "optimizers absorb gradient checkpointing spikes on small cards."
        ),
    ),
]


def _settings(**overrides: object) -> RetrievalSettings:
    base = {
        "chunk_tokens": 48,
        "chunk_overlap_tokens": 12,
        "semantic_split_threshold": 0.0,
        "fetch_k": 8,
        "top_k": 3,
        "rerank_enabled": False,
    }
    base.update(overrides)
    return RetrievalSettings(**base)  # type: ignore[arg-type]


async def test_chunker_respects_token_budget_and_overlaps(embedder) -> None:
    settings = _settings()
    chunker = SemanticChunker(settings, embedder=embedder)
    chunks = await chunker.split(CORPUS[0])

    assert len(chunks) > 1, "a 48-token budget must split this document"
    assert all(c.doc_id == "nist" for c in chunks)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.uri == "corpus/nist.md" for c in chunks)
    # Overlap: the tail of chunk n reappears at the head of chunk n+1.
    first_tail = chunks[0].text.split()[-3:]
    assert " ".join(first_tail) in chunks[1].text


async def test_semantic_breakpoints_are_used_when_enabled(embedder) -> None:
    chunker = SemanticChunker(_settings(semantic_split_threshold=0.99), embedder=embedder)
    chunks = await chunker.split(CORPUS[1])
    # A 0.99 threshold forces a break at nearly every sentence boundary.
    assert len(chunks) >= 3


async def test_point_id_is_stable_across_runs() -> None:
    a = Chunk(chunk_id="doc::0001", doc_id="doc", text="x", ordinal=1)
    b = Chunk(chunk_id="doc::0001", doc_id="doc", text="different text", ordinal=1)
    assert a.point_id == b.point_id, "re-ingest must overwrite, not duplicate"


async def test_ingest_then_query_returns_grounded_context(embedder, store, audit) -> None:
    engine = LocalRAGEngine(settings=_settings(), embedder=embedder, store=store, audit=audit)
    written = await engine.ingest(CORPUS)
    assert written == await engine.count() > 0

    result = await engine.query("What does the Measure function quantify?")
    assert result.results
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.results) <= 3
    assert [sc.rank for sc in result.results] == list(range(len(result.results)))
    assert "[1]" in result.context
    assert result.attributions[0].chunk_id == result.results[0].chunk.chunk_id

    # Retrieval is audited even though it never touched the cloud.
    assert audit.records[-1].stage == "local_retrieval"
    assert audit.records[-1].token_source == "local_ollama"
    assert audit.records[-1].retrieval_sources


async def test_query_on_empty_index_is_zero_confidence(embedder, store) -> None:
    engine = LocalRAGEngine(settings=_settings(), embedder=embedder, store=store)
    result = await engine.query("anything at all")
    assert result.results == []
    assert result.confidence == 0.0


def _scored(scores: list[float]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text="t", ordinal=i),
            score=s,
            raw_score=s,
            rank=i,
        )
        for i, s in enumerate(scores)
    ]


@pytest.mark.parametrize(
    ("scores", "expect_above_threshold"),
    [
        # One decisive hit with weak runners-up: margin carries it.
        ([0.90, 0.01, 0.004], True),
        # A tight cluster of strong hits: coverage carries it. The first version
        # of the formula scored this LOW, which a live corpus proved wrong -
        # three cross-encoder scores above 0.9 means three passages that really
        # do answer the question.
        ([0.92, 0.91, 0.90], True),
        # Weak and undifferentiated: neither term rescues it.
        ([0.51, 0.50, 0.49], False),
        # Nothing retrieved at all.
        ([], False),
    ],
)
def test_confidence_tracks_retrieval_quality(scores, expect_above_threshold) -> None:
    confidence, _, _, _ = calibrate_confidence(_scored(scores), _settings())
    assert (confidence >= 0.78) is expect_above_threshold


def test_confidence_is_monotonic_in_top_score() -> None:
    settings = _settings()
    low, *_ = calibrate_confidence(_scored([0.60, 0.30]), settings)
    high, *_ = calibrate_confidence(_scored([0.95, 0.30]), settings)
    assert high > low


async def test_mine_triplets_pairs_hard_negatives(embedder, store) -> None:
    engine = LocalRAGEngine(settings=_settings(), embedder=embedder, store=store)
    await engine.ingest(CORPUS)
    # Deliberately overlapping thresholds: every passage qualifies as both.
    # The miner must still never emit a triplet whose positive is its negative.
    triplets = await engine.mine_triplets(
        ["adapter rank and quantization"], positive_floor=0.0, negative_ceiling=1.0
    )
    assert triplets
    assert triplets[0].positive != triplets[0].negative
    assert triplets[0].positive_score > triplets[0].negative_score


def test_confidence_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"must sum to 1.0"):
        RetrievalSettings(w_top_score=0.5, w_support=0.9)


def test_margin_and_coverage_are_alternatives_not_requirements() -> None:
    """The bug this encodes: summing them let each cancel the other out."""
    settings = _settings()
    decisive, _, margin, coverage = calibrate_confidence(_scored([0.90, 0.01, 0.004]), settings)
    assert margin > 0.8 and coverage < 0.4, "one strong hit, no corroboration"

    corroborated, _, margin2, coverage2 = calibrate_confidence(
        _scored([0.90, 0.89, 0.88]), settings
    )
    assert margin2 < 0.1 and coverage2 == 1.0, "corroboration, no margin"

    # Opposite evidence shapes, both legitimate, both clear the bar.
    assert decisive >= 0.78
    assert corroborated >= 0.78
