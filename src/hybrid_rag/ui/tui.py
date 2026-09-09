"""The interactive terminal UI.

    uv run hybrid-rag

A chat loop over `AgentOrchestrator.execute`, plus a status line that reports
what the machine is actually doing. It adds no capability: every answer goes
through the same governance, routing and budget path as `hybrid-rag ask`.

**Every number on screen is measured, cited, or absent.** That rule is the
reason this file is longer than a REPL needs to be. This project has a
documented habit of plausible-sounding figures turning out to be invented
(decisions 15, 23, 24, 27), and a status panel is the easiest place in a
codebase to launder a guess into something that looks instrumented. So:

- Live values come from the running system - `ollama ps` for residency, the
  budget ledger for remaining capacity, the execution result for latency,
  throughput and confidence.
- Recorded values are shown with the decision that recorded them, so a reader
  can tell a benchmark from a live reading.
- Anything that could not be measured says so. "not measured" is a valid
  status; a placeholder that looks like a reading is not.

The spinner follows the same rule. It advances on `Stage` events fired inside
the pipeline, not on a timer, so it can never claim to be reranking while it
is really waiting on the guard model.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hybrid_rag.agents.orchestrator import ExecutionResult, Outcome, Stage
from hybrid_rag.config import Settings
from hybrid_rag.container import Container, build_container
from hybrid_rag.governance.audit import PIIStatus
from hybrid_rag.ui.residency import ResidentModel, probe_residency, residency_for

__all__ = ["HybridRagTui", "run_tui"]

# --------------------------------------------------------------------------
# Palette. Terracotta on whatever the terminal's own background is - the TUI
# never paints a background, so it inherits the user's theme instead of
# fighting it.
# --------------------------------------------------------------------------

BRAND: Final = "#d97757"
BRAND_DIM: Final = "#a85a41"
MUTED: Final = "grey58"
OK: Final = "green"
WARN: Final = "yellow"
BAD: Final = "red"

VERSION: Final = "0.1.0"

# The 06. STAR emblem. Rendered from block glyphs rather than an image because
# a terminal is the one place where pixel art is native.
STAR: Final = """\
        ▄
     ▄  █  ▄
       ███
   ▄ ███████ ▄
       ███
     ▄  █  ▄
        ▀"""

# Legacy Windows consoles still run cp437/cp1252, where the block glyphs above
# raise UnicodeEncodeError on write. Losing the logo is acceptable; crashing on
# startup because of decoration is not.
STAR_ASCII: Final = """\
        *
     *  |  *
       ###
   * ####### *
       ###
     *  |  *
        '"""

SPINNER_FRAMES: Final = ("✢", "✳", "*", "✻", "✽")
SPINNER_FRAMES_ASCII: Final = ("+", "x", "*", "x", "+")

STAGE_LABELS: Final[dict[Stage, str]] = {
    Stage.GOVERNANCE: "Scrubbing PII & screening for prompt injection...",
    Stage.RETRIEVAL: "Querying the local index & reranking via TinyBERT...",
    Stage.GENERATION: "Generating locally (or routing to Opus)...",
}

COMMANDS: Final[dict[str, str]] = {
    "/help": "list these commands",
    "/doctor": "hardware and residency health table",
    "/budget": "remaining rolling-window Claude Pro capacity",
    "/audit": "tail the last 5 governance ledger entries",
    "/model": "hot-swap the resident generator (/model code)",
    "/clear": "clear the screen and reprint the banner",
    "/exit": "leave (quit and Ctrl-D also work)",
}


# --------------------------------------------------------------------------
# Terminal capability
# --------------------------------------------------------------------------


def supports_unicode(console: Console) -> bool:
    """Whether the console can encode the block glyphs, asked rather than assumed."""
    encoding = console.encoding or "ascii"
    try:
        STAR.encode(encoding)
        "".join(SPINNER_FRAMES).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


# --------------------------------------------------------------------------
# Spinner
# --------------------------------------------------------------------------


class StagePulse:
    """A spinner whose label is set by real pipeline events.

    `Live` re-renders this on its own clock, so the glyph animates, but the
    text only changes when `set_stage` is called from an orchestrator callback.
    """

    def __init__(self, *, unicode_ok: bool) -> None:
        self._frames = SPINNER_FRAMES if unicode_ok else SPINNER_FRAMES_ASCII
        self._label = "Starting..."
        self._t0 = time.monotonic()

    def set_stage(self, stage: Stage) -> None:
        self._label = STAGE_LABELS.get(stage, str(stage))

    def __rich__(self) -> Text:
        elapsed = time.monotonic() - self._t0
        frame = self._frames[int(elapsed / 0.12) % len(self._frames)]
        text = Text(f" {frame} ", style=BRAND)
        text.append(self._label, style=MUTED)
        text.append(f"  {elapsed:.1f}s", style="grey37")
        return text


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------


class SlashCompleter(Completer):
    """Completes slash commands and `/model` arguments, and nothing else.

    Deliberately silent on ordinary prose: a dropdown that fires on every word
    of a question is noise, and the questions here are natural language.
    """

    def __init__(self, profiles: Iterable[str]) -> None:
        self._profiles = sorted(profiles)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterator[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        head, _, tail = text.partition(" ")
        if not _:
            for name, description in COMMANDS.items():
                if name.startswith(head):
                    yield Completion(name, start_position=-len(head), display_meta=description)
            return
        if head == "/model":
            for profile in self._profiles:
                if profile.startswith(tail):
                    yield Completion(profile, start_position=-len(tail))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_banner(*, unicode_ok: bool) -> RenderableType:
    star = Text(STAR if unicode_ok else STAR_ASCII, style=BRAND)
    heading = Text()
    heading.append(f"HYBRID-RAG v{VERSION}", style=f"bold {BRAND}")
    heading.append("  Multi-Agent Local Framework", style="bold white")
    subtitle = Text(
        "Offline-First  .  NIST AI RMF Governed  .  3B Co-Resident",
        style=MUTED,
    )
    return Group(star, Text(), heading, subtitle)


def render_hud(
    settings: Settings,
    *,
    budget_line: str,
    budget_style: str,
    frontier_available: bool,
    residency: list[ResidentModel] | None,
) -> RenderableType:
    """The session status panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column()

    generator = settings.ollama.generator_model
    gen_text, gen_resident = residency_for(residency, generator)
    gen_style = OK if gen_resident else MUTED
    table.add_row(
        "Model",
        Text.assemble(
            (generator, f"bold {BRAND}"), "  ", (gen_text, gen_style)
        ),
    )

    rerank = settings.retrieval.rerank_model
    table.add_row(
        "Reranker",
        Text.assemble(
            (rerank if settings.retrieval.rerank_enabled else f"{rerank} (disabled)", "white"),
            ("  21 ms p50, recorded in decision 15", MUTED),
        ),
    )

    if settings.ollama.guard_enabled:
        guard = Text.assemble(
            (settings.ollama.guard_model, "white"),
            (" + ingress regex", "white"),
            ("  38 ms p50, recorded in decision 24", MUTED),
        )
    else:
        guard = Text.assemble(
            ("ingress regex only", "white"),
            (f"  {settings.ollama.guard_model} disabled", WARN),
        )
    table.add_row("Guardrails", guard)

    escalation = Text.assemble(
        (f"{settings.anthropic.model} above conf {settings.routing.escalation_threshold}", "white"),
        "  ",
        (budget_line, budget_style),
    )
    table.add_row("Escalation", escalation)

    if frontier_available:
        airgap = Text.assemble(
            ("LEVEL 2 ARMED", f"bold {WARN}"),
            ("  credentials present; escalation can spend cloud tokens", MUTED),
        )
    else:
        airgap = Text.assemble(
            ("OFFLINE ARMED", f"bold {OK}"),
            ("  no credentials; every escalation degrades to local", MUTED),
        )
    table.add_row("Air-gap", airgap)

    return Panel(
        table,
        border_style=BRAND_DIM,
        title=Text("session", style=MUTED),
        title_align="left",
        padding=(0, 1),
    )


def render_badge(result: ExecutionResult, *, unicode_ok: bool) -> Text:
    """The one-line receipt under an answer. Every field is from `result`."""
    sep = "  •  " if unicode_ok else "  |  "

    tier_label, tier_style = {
        Outcome.LOCAL: ("Local", OK),
        Outcome.LOCAL_DEGRADED: ("Local (degraded)", WARN),
        Outcome.OPUS: ("Opus 5", BRAND),
        Outcome.BLOCKED: ("Blocked", BAD),
        Outcome.FAILED: ("Failed", BAD),
    }[result.outcome]

    fields: list[tuple[str, str]] = [(tier_label, tier_style)]

    tokens_out = result.cloud_tokens_out or result.local_tokens_out
    if tokens_out and result.generation_ms > 0:
        fields.append((f"{tokens_out / (result.generation_ms / 1000.0):.0f} tok/s", MUTED))

    fields.append((f"conf {result.retrieval_confidence:.2f}", MUTED))

    pii = {
        PIIStatus.CLEAN: ("PII clean", MUTED),
        PIIStatus.SCRUBBED: (f"PII {result.pii_entities} redacted", WARN),
        PIIStatus.BLOCKED: ("PII blocked", BAD),
        PIIStatus.NOT_EVALUATED: ("PII not evaluated", MUTED),
    }[result.pii_status]
    fields.append(pii)

    fields.append((f"{result.latency_ms:.0f} ms", MUTED))
    if result.generation_ms > 0:
        fields.append((f"gen {result.generation_ms:.0f} ms", MUTED))

    if result.quarantined_chunks:
        fields.append((f"{result.quarantined_chunks} quarantined", BAD))

    if result.audit_hash:
        fields.append((f"ledger {result.audit_hash[:4]}..{result.audit_hash[-2:]}", MUTED))

    badge = Text("  ")
    for index, (value, style) in enumerate(fields):
        if index:
            badge.append(sep, style="grey37")
        badge.append(value, style=style)
    return badge


# --------------------------------------------------------------------------
# The application
# --------------------------------------------------------------------------


def terminal_hint(exc: BaseException) -> str | None:
    """Turn a terminal-attachment failure into something a reader can act on.

    prompt_toolkit needs a real Windows console. Under MinTTY - which is what
    Git Bash gives you - it raises `NoConsoleScreenBufferError` and the default
    behaviour is a forty-line traceback through its Win32 bindings, which reads
    like the program is broken rather than like the terminal is wrong.

    Matched by class name rather than imported: the exception only exists in
    prompt_toolkit's Windows backend, and importing it unconditionally would
    break this module on a POSIX host for the sake of an error message.
    """
    if type(exc).__name__ != "NoConsoleScreenBufferError":
        return None
    return (
        "This terminal is not a Windows console, so the interactive UI cannot "
        "attach to it.\n"
        "  Two causes seen here: MinTTY (Git Bash), and any shell whose output "
        "is redirected or piped.\n"
        "  An interactive PowerShell, Windows Terminal or cmd.exe window works.\n"
        '  For a non-interactive answer, use: hybrid-rag ask "your question"'
    )


class HybridRagTui:
    """Owns the console, the prompt session and one container."""

    def __init__(
        self,
        container: Container,
        *,
        console: Console | None = None,
        history: History | None = None,
    ) -> None:
        self._container = container
        self._settings = container.settings
        self._console = console or Console()
        self._unicode = supports_unicode(self._console)
        # In-memory by default. A question typed here can contain exactly the
        # personal data the scrubber exists to keep off disk, and rule 4 keeps
        # raw prompt text out of the ledger for that reason - writing the same
        # text to a plaintext history file beside it would undo that. Opt in
        # with --history-file when the convenience is worth it.
        self._history = history or InMemoryHistory()
        self._session: PromptSession[str] | None = None

    def _prompt_session(self) -> PromptSession[str]:
        """Built on first use, not in `__init__`.

        `PromptSession` binds to a real terminal at construction. Deferring it
        means the whole command surface - every renderer and every slash
        command - can be exercised by a test with a plain `Console`, with no
        pty and no console screen buffer.
        """
        if self._session is None:
            self._session = PromptSession(
                history=self._history,
                completer=SlashCompleter(self._settings.ollama.generator_profiles),
                complete_while_typing=True,
            )
        return self._session

    # ---------------------------------------------------------------- chrome

    @property
    def console(self) -> Console:
        return self._console

    def _budget_line(self) -> tuple[str, str]:
        snap = self._container.budget.snapshot()
        line = f"budget {snap.messages_used}/{snap.messages_limit} msgs"
        if snap.messages_remaining == 0 or snap.tokens_remaining == 0:
            return f"{line} - exhausted, every request stays local", WARN
        return line, MUTED

    def print_banner(self) -> None:
        self._console.print()
        self._console.print(render_banner(unicode_ok=self._unicode))
        self._console.print()
        self.print_hud()
        self._console.print(
            Text("  /help for commands, /exit to leave", style="grey37")
        )
        self._console.print()

    def print_hud(self) -> None:
        line, style = self._budget_line()
        self._console.print(
            render_hud(
                self._settings,
                budget_line=line,
                budget_style=style,
                frontier_available=self._container.frontier.available,
                residency=probe_residency(self._settings.promotion.ollama_binary),
            )
        )

    # -------------------------------------------------------------- commands

    def _cmd_help(self) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=BRAND, no_wrap=True)
        table.add_column(style=MUTED)
        for name, description in COMMANDS.items():
            table.add_row(name, description)
        self._console.print(table)
        self._console.print(
            Text(
                "  Anything else is a question. It runs the same pipeline as "
                "`hybrid-rag ask`.",
                style="grey37",
            )
        )

    async def _cmd_doctor(self) -> None:
        settings = self._settings
        ollama_up = await self._container.ollama.health()
        chunks = await self._container.rag.count()
        residency = probe_residency(settings.promotion.ollama_binary)

        table = Table(title="doctor", show_header=False, border_style=BRAND_DIM)
        table.add_column(style=MUTED, no_wrap=True)
        table.add_column()
        table.add_row(
            "ollama", Text("up", style=OK) if ollama_up else Text("unreachable", style=BAD)
        )
        table.add_row("ollama url", settings.ollama.base_url)
        table.add_row(
            "generator",
            f"{settings.ollama.generator_model} ({settings.ollama.generator_profile})",
        )
        table.add_row("keep_alive", str(settings.ollama.keep_alive))
        table.add_row("num_ctx", str(settings.ollama.num_ctx))
        table.add_row("embeddings", f"{settings.ollama.embed_model} ({settings.ollama.embed_dim}d)")
        table.add_row("reranker", settings.retrieval.rerank_model)
        table.add_row("max workers", str(settings.runtime.max_workers))
        table.add_row(
            "vector backend", f"{settings.vector.backend} @ {settings.vector.persist_dir}"
        )
        table.add_row("indexed chunks", str(chunks))
        table.add_row("escalation threshold", str(settings.routing.escalation_threshold))
        table.add_row("relevance floor", str(settings.retrieval.relevance_floor))

        if residency is None:
            table.add_row(
                "VRAM residency",
                Text("not measured - `ollama ps` did not answer", style=MUTED),
            )
        elif not residency:
            table.add_row("VRAM residency", Text("nothing resident", style=MUTED))
        else:
            for index, model in enumerate(residency):
                style = OK if model.processor.endswith("GPU") else WARN
                table.add_row(
                    "VRAM residency" if index == 0 else "",
                    Text(f"{model.name}  {model.size}  {model.processor}", style=style),
                )
        self._console.print(table)

    def _cmd_budget(self) -> None:
        snap = self._container.budget.snapshot()
        settings = self._settings.budget
        table = Table(title="budget window", show_header=False, border_style=BRAND_DIM)
        table.add_column(style=MUTED, no_wrap=True)
        table.add_column()
        table.add_row("window", f"{settings.window_hours} h")
        table.add_row(
            "messages",
            f"{snap.messages_used}/{snap.messages_limit} used, "
            f"{snap.messages_remaining} left",
        )
        table.add_row(
            "tokens",
            f"{snap.tokens_used:,}/{snap.tokens_limit:,} used, "
            f"{snap.tokens_remaining:,} left",
        )
        table.add_row(
            "weekly", f"{snap.weekly_messages_used}/{snap.weekly_messages_limit}"
        )
        table.add_row("open reservations", str(snap.open_reservations))
        table.add_row("resets in", f"{snap.seconds_to_window_reset / 60.0:.1f} min")
        self._console.print(table)
        self._console.print(
            Text(
                "  These caps are deliberate under-estimates of the real Pro "
                "limits, not the published ones. See docs/DECISIONS.md.",
                style="grey37",
            )
        )

    def _cmd_audit(self, count: int = 5) -> None:
        path = self._settings.audit.path
        if not path.exists():
            self._console.print(Text(f"  no ledger yet at {path}", style=MUTED))
            return
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            self._console.print(Text(f"  ledger at {path} is empty", style=MUTED))
            return

        table = Table(title=f"last {min(count, len(lines))} of {len(lines)} ledger records",
                      border_style=BRAND_DIM)
        table.add_column("stage", style=MUTED)
        table.add_column("tier")
        table.add_column("tokens", justify="right", style=MUTED)
        table.add_column("conf", justify="right", style=MUTED)
        table.add_column("hash", style="grey37")
        for line in lines[-count:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                table.add_row("(unparseable)", "", "", "", "")
                continue
            tier = str(row.get("tier", "?"))
            digest = str(row.get("record_hash") or "")
            table.add_row(
                str(row.get("stage", "?")),
                Text(tier, style=BRAND if tier == "opus" else OK),
                f"{row.get('tokens_in', 0)}+{row.get('tokens_out', 0)}",
                f"{float(row.get('confidence') or 0.0):.2f}",
                f"{digest[:8]}.." if digest else "-",
            )
        self._console.print(table)
        self._console.print(
            Text("  `hybrid-rag verify-ledger` re-walks the whole chain.", style="grey37")
        )

    def _cmd_model(self, argument: str) -> None:
        profiles = self._settings.ollama.generator_profiles
        if not argument:
            table = Table.grid(padding=(0, 2))
            table.add_column(style=BRAND, no_wrap=True)
            table.add_column(style=MUTED)
            current = self._settings.ollama.generator_profile
            for name, model in sorted(profiles.items()):
                marker = " (current)" if name == current else ""
                table.add_row(name, f"{model}{marker}")
            self._console.print(table)
            return
        try:
            resolved = self._container.switch_generator(argument)
        except ValueError as exc:
            self._console.print(Text(f"  {exc}", style=BAD))
            return
        self._console.print(
            Text.assemble(
                ("  generator -> ", MUTED),
                (resolved, f"bold {BRAND}"),
                (
                    "  (loads on next query; the previous model stays resident "
                    "until Ollama evicts it)",
                    "grey37",
                ),
            )
        )

    # ------------------------------------------------------------------ loop

    async def handle(self, line: str) -> bool:
        """Process one line. Returns False when the session should end."""
        text = line.strip()
        if not text:
            return True

        if text.lower() in {"exit", "quit", "/exit", "/quit"}:
            return False

        if text.startswith("/"):
            command, _, argument = text.partition(" ")
            command = command.lower()
            if command == "/help":
                self._cmd_help()
            elif command == "/doctor":
                await self._cmd_doctor()
            elif command == "/budget":
                self._cmd_budget()
            elif command == "/audit":
                self._cmd_audit()
            elif command == "/model":
                self._cmd_model(argument.strip())
            elif command == "/clear":
                self._console.clear()
                self.print_banner()
            else:
                self._console.print(
                    Text(f"  unknown command {command}. /help lists them.", style=WARN)
                )
            self._console.print()
            return True

        await self.ask(text)
        return True

    async def ask(self, question: str) -> ExecutionResult:
        """Run one question through the full pipeline and render the result."""
        pulse = StagePulse(unicode_ok=self._unicode)
        with Live(pulse, console=self._console, refresh_per_second=12, transient=True):
            result = await self._container.orchestrator.execute(
                question, on_stage=pulse.set_stage
            )

        if result.answer:
            self._console.print(Markdown(result.answer, code_theme="ansi_dark"))
        else:
            # An outcome with no answer is a refusal or a failure, and the
            # reason is the whole content of the turn.
            self._console.print(
                Text(f"  {result.reason or 'no answer produced'}", style=WARN)
            )
        self._console.print(render_badge(result, unicode_ok=self._unicode))
        if result.degraded_reason:
            self._console.print(Text(f"  {result.degraded_reason}", style="grey37"))
        self._console.print()
        return result

    async def run(self) -> None:
        self.print_banner()
        prompt = "✦ hybrid-rag > " if self._unicode else "> hybrid-rag > "
        while True:
            try:
                with patch_stdout():
                    line = await self._prompt_session().prompt_async(prompt)
            except KeyboardInterrupt:
                # Ctrl-C abandons the line, like a shell. Ctrl-D leaves.
                continue
            except EOFError:
                break
            except Exception as exc:
                # Ordered last on purpose: EOFError is an Exception, and
                # catching it here instead would break Ctrl-D.
                hint = terminal_hint(exc)
                if hint is None:
                    raise
                self._console.print(Text(hint, style=WARN))
                return
            try:
                if not await self.handle(line):
                    break
            except Exception as exc:  # a bad turn must not end the session
                self._console.print(Text(f"  {type(exc).__name__}: {exc}", style=BAD))
                self._console.print()
        self._console.print(Text("  bye", style=MUTED))


def resolve_history(path: str | None) -> History:
    """Opt-in persistent history.

    Off unless a path is given, because a prompt can hold the personal data the
    scrubber is there to contain, and this file is plaintext.
    """
    if not path:
        return InMemoryHistory()
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(target))


def run_tui(container: Container | None = None, *, history_path: str | None = None) -> None:
    """Blocking entry point used by the CLI."""

    async def _main() -> None:
        owned = container is None
        active = container or build_container()
        try:
            await HybridRagTui(active, history=resolve_history(history_path)).run()
        finally:
            if owned:
                await active.aclose()

    asyncio.run(_main())
