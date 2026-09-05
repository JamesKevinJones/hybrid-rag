"""Re-derive the routing thresholds against an external corpus.

Run this after changing `rerank_model`, `embed_model`, or the confidence
formula. All three change the score distribution, and a stale threshold does
not fail loudly - it silently routes wrong, in whichever direction the drift
happened to go.

    uv run python scripts/fetch_calibration_corpus.py
    uv run python scripts/calibrate_thresholds.py

The corpus is Python PEPs, deliberately unrelated to this repo. Calibrating
against the project's own documentation is circular: the corpus is *about* the
router being tested, every query finds a suspiciously good match, and the
numbers flatter themselves.

The labels below are the only judgement call in the file, and they are the
honest part - a query is ANSWERABLE if the fetched PEPs actually contain the
answer, and UNANSWERABLE if they do not. Getting a label wrong moves the
threshold, so they are grouped with a note on why each one is where it is.
"""

from __future__ import annotations

import asyncio
import os
import statistics
from pathlib import Path

# Calibration runs against its own collection so it never disturbs a working index.
os.environ.setdefault("HYBRID_RAG_VECTOR__COLLECTION", "calib_minilm_384")

from hybrid_rag.config import get_settings  # noqa: E402
from hybrid_rag.container import build_container  # noqa: E402

# The PEPs in the corpus cover these directly.
ANSWERABLE = [
    "What is the maximum line length recommended for Python code?",
    "How do I annotate a function that accepts a callable?",
    "What is the assignment expression operator for?",
    "Which fields belong in the project table of pyproject.toml?",
    "How often is a new Python version released?",
    "Which standard library modules are being removed?",
    "How does the event loop schedule callbacks?",
    "How should I name a private helper method?",
    "What is the difference between a coroutine and a future?",
    "How do I publish a package to PyPI?",
]

# Two kinds of negative, deliberately mixed. Pure nonsense is easy; the
# interesting ones are plausible Python questions these particular PEPs do NOT
# answer, because that is what a real corpus gap looks like.
UNANSWERABLE = [
    "recipe for sourdough bread with rye starter",
    "purple monkey dishwasher tax code",
    "how do I change a car tyre in the rain",
    "quarterly sales figures for the Lisbon office",
    "What are the trade-offs of static typing in a dynamic language?",  # spec, not philosophy
    "Why was the walrus operator controversial?",  # PEP 572 is the spec, not the debate
    "How do I configure mypy strict mode?",  # a tool, not a PEP
    "What does the GIL do to threaded code?",  # not in this selection
]


async def main() -> int:
    settings = get_settings()
    container = build_container(settings)
    async with container:
        indexed = await container.rag.count()
        if not indexed:
            print("No calibration index. Run scripts/fetch_calibration_corpus.py, then ingest:")
            print("  uv run hybrid-rag ingest ./data/calibration")
            return 1
        await container.rag.query("warmup")

        scores: dict[str, list[float]] = {"answerable": [], "unanswerable": []}
        for label, queries in (("answerable", ANSWERABLE), ("unanswerable", UNANSWERABLE)):
            for query in queries:
                result = await container.rag.query(query)
                scores[label].append(result.confidence)

    lo = min(scores["answerable"])
    hi = max(scores["unanswerable"])

    print(f"corpus: {indexed} chunks")
    print(f"reranker: {settings.retrieval.rerank_model}")
    print(f"embedder: {settings.ollama.embed_model} ({settings.ollama.embed_dim}d)")
    print(f"relevance_floor: {settings.retrieval.relevance_floor}")
    print()
    print(f"  answerable   n={len(scores['answerable']):2d} "
          f"min={lo:.4f} median={statistics.median(scores['answerable']):.4f}")
    print(f"  unanswerable n={len(scores['unanswerable']):2d} "
          f"max={hi:.4f} median={statistics.median(scores['unanswerable']):.4f}")
    print()

    if hi >= lo:
        print(f"NO SEPARATION: unanswerable reaches {hi:.4f}, answerable bottoms at {lo:.4f}.")
        print("No threshold classifies this set correctly. The reranker is not")
        print("discriminating - check that it is actually running.")
        return 2

    current = settings.routing.escalation_threshold
    midpoint = round((lo + hi) / 2, 3)
    print(f"clean separation: ({hi:.4f}, {lo:.4f}) - width {lo - hi:.4f}")
    print(f"midpoint of the gap: {midpoint}")
    print(f"currently configured: {current}")
    print()

    for threshold in sorted({0.78, 0.6, 0.45, 0.35, 0.25, midpoint, current}):
        false_escalations = sum(1 for c in scores["answerable"] if c < threshold)
        false_locals = sum(1 for c in scores["unanswerable"] if c >= threshold)
        verdict = "OK" if not (false_escalations or false_locals) else "  "
        print(
            f"  {verdict} threshold {threshold:<6.3f} "
            f"false escalations {false_escalations}/{len(ANSWERABLE)}  "
            f"false locals {false_locals}/{len(UNANSWERABLE)}"
        )

    print()
    print("Direction matters, and it is easy to get backwards:")
    print("  conf >= threshold -> LOCAL      conf < threshold -> ESCALATE")
    print("  LOWERING the threshold reduces false escalations (saves money).")
    print("  RAISING  it reduces false locals (avoids weak answers).")
    print("A false escalation spends one of 25 messages in a 5-hour window; a")
    print("false local returns a worse answer for free. Pick inside the gap with")
    print("margin on both sides, and re-run this whenever the reranker changes.")
    return 0


if __name__ == "__main__":
    if not (Path("data") / "calibration").exists():
        print("Fetch the corpus first: uv run python scripts/fetch_calibration_corpus.py")
        raise SystemExit(1)
    raise SystemExit(asyncio.run(main()))
