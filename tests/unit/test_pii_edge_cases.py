"""Module 2a - PII masking edge cases.

The happy path is covered in test_governance.py. This file is the awkward
half: boundaries, adjacency, collisions, repeats and adversarial input. Every
one of these is a way to corrupt the output text while still reporting a clean
scrub, which is the worst failure mode a redactor has - it looks like it worked.
"""

from __future__ import annotations

import time

import pytest

from hybrid_rag.governance.audit import PIIStatus
from hybrid_rag.governance.pii import (
    PIIEntity,
    PIIScrubber,
    RegexPIIDetector,
    merge_spans,
)

MAIL_A = "a@b.com"
MAIL_B = "c@d.org"
FAKE_KEY = "sk-ant-api03-AAAABBBBCCCCDDDDEEEE"


# ------------------------------------------------------------- boundaries


def test_an_entity_at_the_very_start_is_masked() -> None:
    result = PIIScrubber().scrub(f"{MAIL_A} is the owner")
    assert result.safe_text == "<EMAIL_ADDRESS_1> is the owner"
    assert result.restore(result.safe_text) == f"{MAIL_A} is the owner"


def test_an_entity_at_the_very_end_is_masked() -> None:
    result = PIIScrubber().scrub(f"the owner is {MAIL_A}")
    assert result.safe_text.endswith("<EMAIL_ADDRESS_1>")


def test_a_string_that_is_only_an_entity() -> None:
    result = PIIScrubber().scrub(MAIL_A)
    assert result.safe_text == "<EMAIL_ADDRESS_1>"
    assert result.restore(result.safe_text) == MAIL_A


def test_an_empty_string_is_clean() -> None:
    result = PIIScrubber().scrub("")
    assert result.status is PIIStatus.CLEAN
    assert result.safe_text == ""
    assert result.counts == {}


def test_whitespace_only_is_clean() -> None:
    assert PIIScrubber().scrub("   \n\t  ").status is PIIStatus.CLEAN


# -------------------------------------------------------------- adjacency


def test_adjacent_entities_with_no_separator() -> None:
    result = PIIScrubber().scrub(f"{MAIL_A},{MAIL_B}")
    assert result.safe_text == "<EMAIL_ADDRESS_1>,<EMAIL_ADDRESS_2>"
    assert result.restore(result.safe_text) == f"{MAIL_A},{MAIL_B}"


def test_the_same_value_twice_gets_two_placeholders_and_still_restores() -> None:
    result = PIIScrubber().scrub(f"{MAIL_A} and again {MAIL_A}")
    assert result.counts["EMAIL_ADDRESS"] == 2
    assert "<EMAIL_ADDRESS_1>" in result.safe_text
    assert "<EMAIL_ADDRESS_2>" in result.safe_text
    assert result.restore(result.safe_text) == f"{MAIL_A} and again {MAIL_A}"


def test_interleaved_types_keep_independent_counters() -> None:
    result = PIIScrubber().scrub(f"{MAIL_A} 10.0.0.1 {MAIL_B} 10.0.0.2")
    assert result.counts == {"EMAIL_ADDRESS": 2, "IP_ADDRESS": 2}
    assert "<EMAIL_ADDRESS_2>" in result.safe_text
    assert "<IP_ADDRESS_2>" in result.safe_text


# --------------------------------------------------------------- collisions


def test_text_that_already_contains_a_placeholder_still_round_trips() -> None:
    """Someone pasting our own output back in must not corrupt the restore."""
    original = f"previously <EMAIL_ADDRESS_1> now {MAIL_A}"
    result = PIIScrubber().scrub(original)

    assert result.counts["EMAIL_ADDRESS"] == 1
    restored = result.restore(result.safe_text)
    assert MAIL_A in restored
    assert restored.count("@") == 1, "the literal placeholder must not gain an address"


def test_restore_is_a_no_op_when_the_model_dropped_the_placeholder() -> None:
    result = PIIScrubber().scrub(f"contact {MAIL_A}")
    assert result.restore("the model answered without any placeholder") == (
        "the model answered without any placeholder"
    )


def test_restore_on_clean_text_returns_it_unchanged() -> None:
    result = PIIScrubber().scrub("nothing sensitive here at all")
    assert result.restore("some answer") == "some answer"


# ------------------------------------------------------------- unicode / eol


def test_offsets_survive_non_ascii_prefixes() -> None:
    """Python string indices are code points; a naive byte-offset scrub corrupts this."""
    original = f"連絡先 - 🎯 {MAIL_A} まで"
    result = PIIScrubber().scrub(original)

    assert MAIL_A not in result.safe_text
    assert result.safe_text.startswith("連絡先 - 🎯 ")
    assert result.safe_text.endswith(" まで")
    assert result.restore(result.safe_text) == original


def test_crlf_line_endings_are_preserved() -> None:
    original = f"line one\r\ncontact {MAIL_A}\r\nline three"
    result = PIIScrubber().scrub(original)
    assert "\r\n" in result.safe_text
    assert result.restore(result.safe_text) == original


def test_an_email_inside_a_url_is_still_found() -> None:
    result = PIIScrubber().scrub(f"https://example.com/u/{MAIL_A}/profile")
    assert MAIL_A not in result.safe_text


# ------------------------------------------------------------ span merging


def test_a_longer_span_beats_a_shorter_overlapping_one() -> None:
    long_span = PIIEntity(entity_type="CREDIT_CARD", start=0, end=19, score=0.6, detector="x")
    short_span = PIIEntity(entity_type="IBAN", start=2, end=14, score=0.99, detector="y")
    assert [e.entity_type for e in merge_spans([long_span, short_span])] == ["CREDIT_CARD"]


def test_equal_length_overlaps_break_the_tie_on_score() -> None:
    low = PIIEntity(entity_type="A", start=0, end=10, score=0.2, detector="x")
    high = PIIEntity(entity_type="B", start=0, end=10, score=0.9, detector="y")
    assert [e.entity_type for e in merge_spans([low, high])] == ["B"]


def test_touching_but_not_overlapping_spans_both_survive() -> None:
    first = PIIEntity(entity_type="A", start=0, end=5, score=0.9, detector="x")
    second = PIIEntity(entity_type="B", start=5, end=10, score=0.9, detector="x")
    assert len(merge_spans([first, second])) == 2


def test_merging_an_empty_list_is_empty() -> None:
    assert merge_spans([]) == []


def test_merged_spans_come_back_in_document_order() -> None:
    spans = [
        PIIEntity(entity_type="C", start=20, end=25, score=0.9, detector="x"),
        PIIEntity(entity_type="A", start=0, end=5, score=0.9, detector="x"),
        PIIEntity(entity_type="B", start=10, end=15, score=0.9, detector="x"),
    ]
    assert [e.start for e in merge_spans(spans)] == [0, 10, 20]


# ------------------------------------------------------- false positives


@pytest.mark.parametrize(
    "text",
    [
        "order 4111111111111112 shipped",  # fails Luhn
        "commit 0a1b2c3d4e5f6a7b8c9d0e1f",  # a git sha is not a key
        "version 1.2.3.4 released",  # not an IP octet range... but shaped like one
    ],
)
def test_lookalikes_that_should_not_fire_as_credentials(text: str) -> None:
    types = {e.entity_type for e in RegexPIIDetector().detect(text)}
    assert "CREDIT_CARD" not in types
    assert "ANTHROPIC_API_KEY" not in types
    assert "AWS_ACCESS_KEY" not in types


def test_a_phone_shaped_substring_of_a_longer_number_is_not_matched() -> None:
    types = {e.entity_type for e in RegexPIIDetector().detect("id 15551234567890123")}
    assert "PHONE_NUMBER" not in types


# ------------------------------------------------------------- blocking


def test_a_credential_blocks_even_when_other_pii_is_present() -> None:
    result = PIIScrubber().scrub(f"mail {MAIL_A} using {FAKE_KEY}")
    assert result.status is PIIStatus.BLOCKED
    assert result.blocked_types == ["ANTHROPIC_API_KEY"]
    assert "EMAIL_ADDRESS" not in result.blocked_types


def test_the_block_list_is_configurable() -> None:
    permissive = PIIScrubber(block_types=frozenset())
    result = permissive.scrub(f"key {FAKE_KEY}")
    assert result.status is PIIStatus.SCRUBBED
    assert result.blocked_types == []
    assert FAKE_KEY not in result.safe_text, "still masked, just not refused"


def test_blocked_text_is_still_fully_masked() -> None:
    result = PIIScrubber().scrub(f"{FAKE_KEY} and {MAIL_A}")
    assert FAKE_KEY not in result.safe_text
    assert MAIL_A not in result.safe_text


# ------------------------------------------------------------- robustness


def test_a_pathological_private_key_marker_does_not_hang() -> None:
    """Regression: an unbounded body made this quadratic on ingested text."""
    hostile = "-----BEGIN PRIVATE KEY-----" * 1600 + "A" * 2000
    t0 = time.perf_counter()
    RegexPIIDetector().detect(hostile)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 250, f"regex took {elapsed_ms:.0f} ms on 45 KB of adversarial input"


def test_a_large_clean_document_scrubs_quickly() -> None:
    document = ("The rolling window budget reserves capacity before each call. " * 2000)
    t0 = time.perf_counter()
    result = PIIScrubber().scrub(document)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert result.status is PIIStatus.CLEAN
    assert elapsed_ms < 500


def test_hundreds_of_entities_all_get_distinct_placeholders() -> None:
    text = " ".join(f"user{i}@example.com" for i in range(200))
    result = PIIScrubber().scrub(text)

    assert result.counts["EMAIL_ADDRESS"] == 200
    assert "<EMAIL_ADDRESS_200>" in result.safe_text
    assert len(result.reverse_map) == 200
    assert result.restore(result.safe_text) == text


def test_a_second_scrub_of_scrubbed_text_numbers_above_the_existing_placeholder() -> None:
    """Regression: the counter used to restart at 1 and collide with a literal."""
    first = PIIScrubber().scrub(f"owner {MAIL_A}")
    assert "<EMAIL_ADDRESS_1>" in first.safe_text

    second = PIIScrubber().scrub(f"{first.safe_text} and also {MAIL_B}")
    assert "<EMAIL_ADDRESS_2>" in second.safe_text
    assert second.reverse_map == {"<EMAIL_ADDRESS_2>": MAIL_B}

    restored = second.restore(second.safe_text)
    assert restored.count("@") == 1, "the pre-existing placeholder must stay a placeholder"
    assert "<EMAIL_ADDRESS_1>" in restored
