"""Module 2 - ledger integrity under corruption.

Each test damages the file a different way and asserts `verify_chain` notices,
and names the right line. The threat model is stated at the top of audit.py:
this is unkeyed SHA-256, so it detects accidents, partial writes and casual
edits - not an attacker who recomputes the chain. These tests cover exactly
that scope and no more, because a test that implied otherwise would be the
overclaim the module docstring exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_rag.governance.audit import AuditRecord, JsonlAuditSink, hash_prompt


def _record(n: int) -> AuditRecord:
    return AuditRecord(
        trace_id=f"t{n}",
        stage="routing_decision",
        agent_id="test",
        tier="local",
        prompt_hash=hash_prompt(f"query {n}"),
        decision_reason=f"reason {n}",
    )


@pytest.fixture
def ledger(tmp_path: Path):
    def _write(count: int) -> Path:
        path = tmp_path / "audit.jsonl"
        sink = JsonlAuditSink(path)
        for i in range(count):
            sink.emit(_record(i))
        return path

    return _write


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _rewrite(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- intact


def test_a_missing_file_is_vacuously_intact(tmp_path: Path) -> None:
    intact, bad = JsonlAuditSink(tmp_path / "never-written.jsonl").verify_chain()
    assert intact and bad == 0


def test_an_empty_file_is_intact(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.touch()
    intact, bad = JsonlAuditSink(path).verify_chain()
    assert intact and bad == 0


def test_a_healthy_chain_verifies(ledger) -> None:
    intact, bad = JsonlAuditSink(ledger(20)).verify_chain()
    assert intact and bad == 0


def test_blank_lines_between_records_are_tolerated(ledger) -> None:
    path = ledger(3)
    lines = _lines(path)
    _rewrite(path, [lines[0], "", lines[1], "   ", lines[2]])
    intact, _ = JsonlAuditSink(path).verify_chain()
    assert intact, "whitespace is not tampering"


def test_appending_after_a_reopen_extends_the_same_chain(ledger) -> None:
    path = ledger(3)
    reopened = JsonlAuditSink(path)
    reopened.emit(_record(99))

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert intact and bad == 0
    assert len(_lines(path)) == 4


# ---------------------------------------------------------------- corruption


def test_an_edited_field_is_caught_at_its_own_line(ledger) -> None:
    path = ledger(5)
    lines = _lines(path)
    row = json.loads(lines[2])
    row["tier"] = "opus"  # rewrite history: claim a local decision went to Opus
    lines[2] = json.dumps(row)
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 3, "1-indexed line number of the first bad row"


def test_a_deleted_middle_line_breaks_the_chain(ledger) -> None:
    """The property a per-record signature does NOT give you."""
    path = ledger(5)
    lines = _lines(path)
    del lines[2]
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 3, "the line that now sits where the deleted one linked from"


def test_a_deleted_first_line_breaks_the_chain(ledger) -> None:
    path = ledger(4)
    lines = _lines(path)
    _rewrite(path, lines[1:])

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 1


def test_reordered_lines_are_caught(ledger) -> None:
    path = ledger(4)
    lines = _lines(path)
    lines[1], lines[2] = lines[2], lines[1]
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 2


def test_a_duplicated_line_is_caught(ledger) -> None:
    path = ledger(3)
    lines = _lines(path)
    _rewrite(path, [lines[0], lines[1], lines[1], lines[2]])

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 3


def test_a_forged_record_appended_by_hand_is_caught(ledger) -> None:
    """A plausible-looking row whose prev_hash does not match its predecessor."""
    path = ledger(3)
    forged = _record(999).model_dump(mode="json")
    forged["prev_hash"] = "0" * 64
    forged["record_hash"] = "f" * 64
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(forged) + "\n")

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 4


def test_a_truncated_final_line_is_caught(ledger) -> None:
    """A crash mid-fsync leaves half a row. That is damage, and it must show."""
    path = ledger(3)
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw[: len(raw) - 40], encoding="utf-8")

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 3, "a half-written row is a break, not an exception for the caller"


def test_a_non_json_line_in_the_middle_is_caught(ledger) -> None:
    path = ledger(3)
    lines = _lines(path)
    lines[1] = "this is not json at all"
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 2


def test_a_json_scalar_where_a_record_belongs_is_caught(ledger) -> None:
    path = ledger(3)
    lines = _lines(path)
    lines[1] = '"a bare string is valid JSON but is not a record"'
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 2


def test_stripping_the_record_hash_is_caught(ledger) -> None:
    path = ledger(3)
    lines = _lines(path)
    row = json.loads(lines[1])
    row.pop("record_hash")
    lines[1] = json.dumps(row)
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 2


def test_a_corrupt_tail_does_not_stop_the_sink_from_writing(tmp_path: Path) -> None:
    """A damaged ledger must still accept new records - losing new evidence to
    protect old evidence is the wrong trade."""
    path = tmp_path / "audit.jsonl"
    path.write_text("garbage that is not json\n", encoding="utf-8")

    sink = JsonlAuditSink(path)
    written = sink.emit(_record(1))
    assert written.record_hash
    assert len(_lines(path)) == 2


def test_the_first_break_is_reported_not_the_last(ledger) -> None:
    path = ledger(6)
    lines = _lines(path)
    for i in (1, 3, 5):
        row = json.loads(lines[i])
        row["tier"] = "opus"
        lines[i] = json.dumps(row)
    _rewrite(path, lines)

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 2


def test_the_schema_version_travels_with_each_record(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    sink = JsonlAuditSink(path, schema_version="2.0-test")
    sink.emit(_record(1))
    assert json.loads(_lines(path)[0])["schema_version"] == "2.0-test"


def test_two_sinks_on_one_file_produce_a_detectable_fork(tmp_path: Path) -> None:
    """Not supported, and the point is that it is not silently supported."""
    path = tmp_path / "audit.jsonl"
    first = JsonlAuditSink(path)
    second = JsonlAuditSink(path)

    first.emit(_record(1))
    second.emit(_record(2))  # second's cached tail hash is now stale

    intact, bad = JsonlAuditSink(path).verify_chain()
    assert not intact
    assert bad == 2
