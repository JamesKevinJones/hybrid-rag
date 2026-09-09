"""Claude Pro rolling-window budget ledger.

The routing gate can *want* to escalate. This module decides whether it may.
It is deliberately pessimistic: a reservation is taken before the cloud call
and only converted to real spend afterwards, so a crashed or hung request
cannot leave the window silently overdrawn.

Two windows are enforced simultaneously:

* the short rolling window (`window_hours`, default 5) - messages and tokens
* the weekly message cap - a 7-day trailing count

`reserve_fraction` carves off headroom that only `Priority.INTERACTIVE`
callers may touch, so an unattended distillation batch cannot eat the quota
you wanted for a live session.

Durability model is the same as the audit ledger: append-only JSONL, replayed
on construction. No database, no daemon, survives a hard kill.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

import orjson
from pydantic import BaseModel, Field, computed_field

from hybrid_rag.config import BudgetSettings

_WEEK_SECONDS = 7 * 24 * 3600.0


class Priority(StrEnum):
    """Who is asking. Determines whether the reserve pool is reachable."""

    INTERACTIVE = "interactive"
    BATCH = "batch"


class Verdict(StrEnum):
    ALLOW = "allow"
    DENY_WINDOW_MESSAGES = "deny_window_messages"
    DENY_WINDOW_TOKENS = "deny_window_tokens"
    DENY_RESERVE_POOL = "deny_reserve_pool"
    DENY_WEEKLY_CAP = "deny_weekly_cap"


class BudgetSnapshot(BaseModel):
    now: float
    window_started_at: float
    messages_used: int
    messages_limit: int
    tokens_used: int
    tokens_limit: int
    weekly_messages_used: int
    weekly_messages_limit: int
    open_reservations: int
    seconds_to_window_reset: float

    # `computed_field` so these survive `model_dump()`. They are the only two
    # numbers a budget display actually reads, and a plain property drops them
    # from every JSON response without a word.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def messages_remaining(self) -> int:
        return max(0, self.messages_limit - self.messages_used)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens_remaining(self) -> int:
        return max(0, self.tokens_limit - self.tokens_used)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def weekly_messages_remaining(self) -> int:
        return max(0, self.weekly_messages_limit - self.weekly_messages_used)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exhausted(self) -> bool:
        """No further escalation can be reserved right now.

        Mirrors `RollingWindowBudget.reserve()` exactly - all three caps it
        enforces, not just the rolling window. A version of this that only
        checked the 5-hour window reported capacity while the weekly cap was
        vetoing every escalation, which is worse than not showing a status at
        all: it looks authoritative and is wrong.
        """
        return (
            self.messages_remaining == 0
            or self.tokens_remaining == 0
            or self.weekly_messages_remaining == 0
        )


class BudgetDecision(BaseModel):
    allowed: bool
    verdict: Verdict
    reason: str
    reservation_id: str | None = None
    estimated_tokens: int = 0
    snapshot: BudgetSnapshot
    compliance_flags: list[str] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class _Entry:
    kind: str  # "reserve" | "commit" | "release"
    entry_id: str
    ts: float
    messages: int
    tokens: int
    trace_id: str | None = None
    ref: str | None = None  # reservation this commit/release settles


class RollingWindowBudget:
    """Authoritative spend ledger. Injected into the routing gate."""

    def __init__(self, settings: BudgetSettings, *, clock: object | None = None) -> None:
        self._s = settings
        self._now = clock if callable(clock) else time.time
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []
        self._open: dict[str, _Entry] = {}
        self._s.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._replay()

    # ---------------------------------------------------------------- replay

    def _replay(self) -> None:
        if not self._s.ledger_path.exists():
            return
        settled: set[str] = set()
        with self._s.ledger_path.open("rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                entry = _Entry(
                    kind=str(row.get("kind", "")),
                    entry_id=str(row.get("entry_id", "")),
                    ts=float(row.get("ts", 0.0)),
                    messages=int(row.get("messages", 0)),
                    tokens=int(row.get("tokens", 0)),
                    trace_id=row.get("trace_id"),
                    ref=row.get("ref"),
                )
                self._entries.append(entry)
                if entry.kind in {"commit", "release"} and entry.ref:
                    settled.add(entry.ref)
        for entry in self._entries:
            if entry.kind == "reserve" and entry.entry_id not in settled:
                self._open[entry.entry_id] = entry
        self._prune(self._now())

    def _append(self, entry: _Entry) -> None:
        payload = {
            "kind": entry.kind,
            "entry_id": entry.entry_id,
            "ts": entry.ts,
            "messages": entry.messages,
            "tokens": entry.tokens,
            "trace_id": entry.trace_id,
            "ref": entry.ref,
        }
        with self._s.ledger_path.open("ab") as fh:
            fh.write(orjson.dumps(payload) + b"\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._entries.append(entry)

    # ------------------------------------------------------------ accounting

    def _prune(self, now: float) -> None:
        """Drop entries older than the weekly horizon and expire stale reservations."""
        horizon = now - _WEEK_SECONDS
        self._entries = [e for e in self._entries if e.ts >= horizon]
        expired = [rid for rid, e in self._open.items() if now - e.ts > self._s.reservation_ttl_s]
        for rid in expired:
            entry = self._open.pop(rid)
            self._append(
                _Entry(
                    kind="release",
                    entry_id=str(uuid.uuid4()),
                    ts=now,
                    messages=0,
                    tokens=0,
                    trace_id=entry.trace_id,
                    ref=rid,
                )
            )

    def _window_usage(self, now: float) -> tuple[int, int]:
        """(messages, tokens) inside the short rolling window, reservations included."""
        start = now - self._s.window_hours * 3600.0
        messages = 0
        tokens = 0
        settled: set[str] = {e.ref for e in self._entries if e.kind == "release" and e.ref}
        committed: dict[str, _Entry] = {
            e.ref: e for e in self._entries if e.kind == "commit" and e.ref
        }
        for e in self._entries:
            if e.ts < start:
                continue
            if e.kind == "reserve":
                if e.entry_id in settled:
                    continue
                if e.entry_id in committed:
                    actual = committed[e.entry_id]
                    messages += actual.messages
                    tokens += actual.tokens
                else:
                    # Still in flight - charge the estimate.
                    messages += e.messages
                    tokens += e.tokens
        return messages, tokens

    def _weekly_messages(self, now: float) -> int:
        start = now - _WEEK_SECONDS
        return sum(1 for e in self._entries if e.kind == "commit" and e.ts >= start)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_locked(self._now())

    def _snapshot_locked(self, now: float) -> BudgetSnapshot:
        self._prune(now)
        messages, tokens = self._window_usage(now)
        window_start = now - self._s.window_hours * 3600.0
        oldest = min(
            (e.ts for e in self._entries if e.ts >= window_start and e.kind == "reserve"),
            default=now,
        )
        return BudgetSnapshot(
            now=now,
            window_started_at=oldest,
            messages_used=messages,
            messages_limit=self._s.max_messages_per_window,
            tokens_used=tokens,
            tokens_limit=self._s.max_tokens_per_window,
            weekly_messages_used=self._weekly_messages(now),
            weekly_messages_limit=self._s.weekly_message_cap,
            open_reservations=len(self._open),
            seconds_to_window_reset=max(0.0, oldest + self._s.window_hours * 3600.0 - now),
        )

    # --------------------------------------------------------------- the API

    def reserve(
        self,
        estimated_tokens: int,
        *,
        trace_id: str,
        priority: Priority = Priority.INTERACTIVE,
        messages: int = 1,
    ) -> BudgetDecision:
        """Attempt to claim window capacity for one cloud call.

        Returns a decision carrying `reservation_id` when allowed. The caller
        MUST follow with `commit()` or `release()`; anything left open past
        `reservation_ttl_s` is reclaimed by the next sweep.
        """
        with self._lock:
            now = self._now()
            snap = self._snapshot_locked(now)

            usable_messages = self._s.max_messages_per_window
            usable_tokens = self._s.max_tokens_per_window
            if priority is Priority.BATCH:
                factor = 1.0 - self._s.reserve_fraction
                usable_messages = int(usable_messages * factor)
                usable_tokens = int(usable_tokens * factor)

            if snap.weekly_messages_used + messages > self._s.weekly_message_cap:
                return BudgetDecision(
                    allowed=False,
                    verdict=Verdict.DENY_WEEKLY_CAP,
                    reason=(
                        f"weekly cap {self._s.weekly_message_cap} would be exceeded "
                        f"({snap.weekly_messages_used} used)"
                    ),
                    estimated_tokens=estimated_tokens,
                    snapshot=snap,
                    compliance_flags=["budget_denied", "weekly_cap"],
                )

            if snap.messages_used + messages > usable_messages:
                verdict = (
                    Verdict.DENY_RESERVE_POOL
                    if priority is Priority.BATCH
                    and snap.messages_used + messages <= self._s.max_messages_per_window
                    else Verdict.DENY_WINDOW_MESSAGES
                )
                return BudgetDecision(
                    allowed=False,
                    verdict=verdict,
                    reason=(
                        f"{snap.messages_used}/{usable_messages} messages used in the "
                        f"{self._s.window_hours}h window; resets in "
                        f"{snap.seconds_to_window_reset / 60:.0f} min"
                    ),
                    estimated_tokens=estimated_tokens,
                    snapshot=snap,
                    compliance_flags=["budget_denied", str(verdict)],
                )

            if snap.tokens_used + estimated_tokens > usable_tokens:
                verdict = (
                    Verdict.DENY_RESERVE_POOL
                    if priority is Priority.BATCH
                    and snap.tokens_used + estimated_tokens <= self._s.max_tokens_per_window
                    else Verdict.DENY_WINDOW_TOKENS
                )
                return BudgetDecision(
                    allowed=False,
                    verdict=verdict,
                    reason=(
                        f"{snap.tokens_used}+{estimated_tokens} would exceed "
                        f"{usable_tokens} tokens in the window"
                    ),
                    estimated_tokens=estimated_tokens,
                    snapshot=snap,
                    compliance_flags=["budget_denied", str(verdict)],
                )

            entry = _Entry(
                kind="reserve",
                entry_id=str(uuid.uuid4()),
                ts=now,
                messages=messages,
                tokens=estimated_tokens,
                trace_id=trace_id,
            )
            self._append(entry)
            self._open[entry.entry_id] = entry
            return BudgetDecision(
                allowed=True,
                verdict=Verdict.ALLOW,
                reason="within window",
                reservation_id=entry.entry_id,
                estimated_tokens=estimated_tokens,
                snapshot=self._snapshot_locked(now),
            )

    def commit(self, reservation_id: str, *, tokens_in: int, tokens_out: int) -> None:
        """Replace the estimate with measured spend."""
        with self._lock:
            entry = self._open.pop(reservation_id, None)
            self._append(
                _Entry(
                    kind="commit",
                    entry_id=str(uuid.uuid4()),
                    ts=self._now(),
                    messages=1,
                    tokens=tokens_in + tokens_out,
                    trace_id=entry.trace_id if entry else None,
                    ref=reservation_id,
                )
            )

    def release(self, reservation_id: str, *, reason: str = "unused") -> None:
        """Give the capacity back - the call never happened."""
        with self._lock:
            entry = self._open.pop(reservation_id, None)
            self._append(
                _Entry(
                    kind="release",
                    entry_id=str(uuid.uuid4()),
                    ts=self._now(),
                    messages=0,
                    tokens=0,
                    trace_id=entry.trace_id if entry else None,
                    ref=reservation_id,
                )
            )
            del reason


def estimate_tokens(text: str, *, chars_per_token: float, output_allowance: int) -> int:
    """Cheap pre-flight estimate. Deliberately over-counts rather than under."""
    prompt_tokens = int(len(text) / max(chars_per_token, 1.0)) + 1
    return prompt_tokens + output_allowance
