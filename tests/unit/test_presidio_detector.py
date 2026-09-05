"""Module 2a - the optional Presidio NER detector, exercised for the first time.

Written in Module 2, never once run until now. These tests skip cleanly when
the `governance` extra is absent, which is the default install.

The headline result is a trade, not a win. Presidio genuinely finds entities
regex cannot - people, places, organisations - and it genuinely mangles
technical prose, because a spaCy NER model trained on news text reads "Q3" as a
place name. That is why it stays opt-in.
"""

from __future__ import annotations

import pytest

from hybrid_rag.governance.pii import (
    CompositePIIDetector,
    PIIScrubber,
    RegexPIIDetector,
)

presidio = pytest.importorskip(
    "presidio_analyzer", reason="needs the `governance` extra (uv sync --extra governance)"
)

from hybrid_rag.governance.pii import PresidioPIIDetector  # noqa: E402

PROSE = "Maria Alvarez in Lisbon emailed a@b.com about the contract."
TECHNICAL = (
    "In Q3 the p95 latency for the S1 path dropped after the TinyBERT swap; "
    "see ADR 14 and the Ollama keep_alive setting."
)


@pytest.fixture(scope="module")
def detector() -> PresidioPIIDetector:
    return PresidioPIIDetector()


def test_it_constructs_at_all(detector) -> None:
    """It never had. `AnalyzerEngine()` needs a spaCy model that is not a
    declared dependency, so this could have been broken since it was written."""
    assert detector.name == "presidio"


def test_it_finds_entities_regex_cannot(detector) -> None:
    found = {e.entity_type for e in detector.detect(PROSE)}
    assert "PERSON" in found
    assert "LOCATION" in found

    regex_found = {e.entity_type for e in RegexPIIDetector().detect(PROSE)}
    assert "PERSON" not in regex_found, "this is the whole reason the extra exists"


def test_it_over_redacts_technical_text(detector) -> None:
    """The measured cost, and the reason it is not the default.

    "Q3" comes back as an entity - DATE_TIME in this sentence, LOCATION in a
    slightly different one, since the NER is context-dependent. Neither is PII
    in any sense that matters here, and redacting it removes exactly the token
    a technical answer needs. A corpus of engineering documents is full of
    quarters, versions and identifiers that read like named entities to a model
    trained on news prose.
    """
    found = [(e.entity_type, TECHNICAL[e.start : e.end]) for e in detector.detect(TECHNICAL)]
    flagged = {text for _, text in found}

    assert found, f"expected Presidio to flag something in technical prose, got {found}"
    assert "Q3" in flagged, f"the canonical false positive; got {found}"


def test_the_regex_detector_leaves_technical_text_alone() -> None:
    """The contrast that justifies the default."""
    assert RegexPIIDetector().detect(TECHNICAL) == []


def test_composite_merges_both_without_double_redacting(detector) -> None:
    composite = CompositePIIDetector([RegexPIIDetector(), detector])
    result = PIIScrubber(detector=composite).scrub(PROSE)

    assert "a@b.com" not in result.safe_text
    assert "Maria Alvarez" not in result.safe_text
    assert result.safe_text.count("<EMAIL_ADDRESS_1>") == 1, "one span, one placeholder"
    assert result.restore(result.safe_text) == PROSE


def test_composite_still_blocks_credentials(detector) -> None:
    """Adding NER must not dilute the hard block the regex layer provides."""
    composite = CompositePIIDetector([RegexPIIDetector(), detector])
    result = PIIScrubber(detector=composite).scrub(
        "Maria deployed with sk-ant-api03-AAAABBBBCCCCDDDDEEEE today."
    )
    assert result.blocked_types == ["ANTHROPIC_API_KEY"]
