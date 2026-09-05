"""Fetch an external corpus for threshold calibration.

Why this exists: the routing thresholds were originally measured against this
project's own documentation. That is circular - the corpus is *about* the
router being tested, so every query has a suspiciously good match and the
numbers flatter themselves. Calibration has to happen against text the system
was not written about.

Python Enhancement Proposals are a good fit. They are public, genuinely
technical, written by many hands over decades, and completely unrelated to this
repo. They also contain exactly the shapes that break naive PII detection -
version numbers, dates, identifiers - which makes them a fair test.

This is the ONLY network access anywhere in the project, it is one-time, and it
is not part of the runtime. `data/` is gitignored, so the corpus is fetched
rather than vendored; re-run this to reproduce a calibration.

    uv run python scripts/fetch_calibration_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

RAW = "https://raw.githubusercontent.com/python/peps/main/peps/{name}.rst"

# A spread of styles: a style guide, a philosophy text, a dense type-system
# spec, a contentious syntax proposal, a packaging standard, a concurrency
# design, a release process and a deprecation notice.
PEPS = [
    "pep-0008",  # Style Guide for Python Code
    "pep-0020",  # The Zen of Python
    "pep-0484",  # Type Hints
    "pep-0572",  # Assignment Expressions
    "pep-0621",  # Storing project metadata in pyproject.toml
    "pep-3156",  # Asynchronous IO Support Rebooted
    "pep-0602",  # Annual Release Cycle for Python
    "pep-0594",  # Removing dead batteries from the standard library
]

TARGET = Path(__file__).resolve().parents[1] / "data" / "calibration"


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    fetched = 0
    total_bytes = 0

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for name in PEPS:
            destination = TARGET / f"{name}.md"
            if destination.exists():
                print(f"  skip   {name} (already present)")
                fetched += 1
                continue
            try:
                response = client.get(RAW.format(name=name))
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"  FAILED {name}: {exc}", file=sys.stderr)
                continue
            destination.write_text(response.text, encoding="utf-8")
            total_bytes += len(response.text)
            fetched += 1
            print(f"  got    {name}  {len(response.text) / 1024:.0f} KB")

    print(f"\n{fetched}/{len(PEPS)} documents in {TARGET}  ({total_bytes / 1024:.0f} KB new)")
    if fetched < len(PEPS):
        print("Some fetches failed; calibration on a partial corpus is still valid,")
        print("but record which documents were present when you quote the numbers.")
    return 0 if fetched else 1


if __name__ == "__main__":
    raise SystemExit(main())
