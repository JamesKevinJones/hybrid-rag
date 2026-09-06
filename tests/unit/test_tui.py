"""Module 6 - the terminal UI.

The interesting assertions here are not "does it render". They are the two ways
a status panel goes wrong:

1. It shows a number nobody measured.
2. It shows a stale number as though it were live.

So the tests below check that an unmeasurable value renders as unmeasured, that
the badge is derived from the result object rather than assembled from
constants, and that a hot-swapped generator actually moves everything that
holds the model name - not just the one the display reads.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
from rich.console import Console

from hybrid_rag.agents.orchestrator import ExecutionResult, Outcome, Stage
from hybrid_rag.config import Settings
from hybrid_rag.container import Container
from hybrid_rag.governance.audit import PIIStatus
from hybrid_rag.governance.guardrails import GuardrailChain, RegexGuardrail
from hybrid_rag.governance.middleware import GovernanceMiddleware
from hybrid_rag.governance.pii import PIIScrubber
from hybrid_rag.routing.budget import RollingWindowBudget
from hybrid_rag.routing.routing_gate import RoutingGate
from hybrid_rag.ui import tui

# `ollama ps` as it actually prints. Both SIZE and PROCESSOR are two tokens,
# which is the detail an earlier parser got wrong - it reported "100%" and
# dropped the word that says whether the model is on the GPU at all.
PS_TABLE = """NAME                           ID              SIZE      PROCESSOR    CONTEXT    UNTIL
qwen2.5-coder:3b               f72c60cabf62    2.2 GB    100% GPU     4096       4 minutes from now
llama-guard3:1b                494147e06bf9    1.5 GB    100% GPU     4096       4 minutes from now
"""


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ollama", "ps"], returncode=returncode, stdout=stdout, stderr=""
    )


def _console() -> Console:
    return Console(width=100, force_terminal=True, record=True)


def _render(renderable: Any) -> str:
    console = _console()
    console.print(renderable)
    return console.export_text()


# --------------------------------------------------------------------------
# Residency probe
# --------------------------------------------------------------------------


def test_ps_parse_keeps_both_words_of_the_processor(monkeypatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(PS_TABLE))
    models = tui.probe_residency()
    assert models is not None
    assert [m.name for m in models] == ["qwen2.5-coder:3b", "llama-guard3:1b"]
    assert models[0].size == "2.2 GB"
    assert models[0].processor == "100% GPU", "'100%' alone loses the only part anyone reads"


@pytest.mark.parametrize(
    ("failure", "why"),
    [
        (OSError("no such binary"), "ollama is not installed"),
        (subprocess.TimeoutExpired(cmd="ollama ps", timeout=5.0), "the daemon hung"),
    ],
)
def test_probe_returns_none_when_it_could_not_ask(monkeypatch, failure, why) -> None:
    def _boom(*_: Any, **__: Any) -> Any:
        raise failure

    monkeypatch.setattr(subprocess, "run", _boom)
    assert tui.probe_residency() is None, why


def test_nothing_resident_is_not_the_same_as_could_not_measure(monkeypatch) -> None:
    """The distinction the whole probe exists to preserve."""
    header = "NAME    ID    SIZE    PROCESSOR    CONTEXT    UNTIL\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(header))
    assert tui.probe_residency() == []

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("", returncode=1))
    assert tui.probe_residency() is None


def test_unmeasured_residency_says_so_rather_than_guessing() -> None:
    text, _ = tui._residency_cell(None, "qwen2.5-coder:3b")
    assert "not measured" in text

    text, _ = tui._residency_cell([], "qwen2.5-coder:3b")
    assert "not resident" in text

    resident = [tui.ResidentModel("qwen2.5-coder:3b", "2.2 GB", "100% GPU")]
    text, _ = tui._residency_cell(resident, "qwen2.5-coder:3b")
    assert "100% GPU" in text and "2.2 GB" in text


# --------------------------------------------------------------------------
# Terminal capability
# --------------------------------------------------------------------------


def test_block_glyphs_are_checked_against_the_console_encoding() -> None:
    assert tui.supports_unicode(Console(file=None, width=80)) in {True, False}

    class Cp437(Console):
        @property
        def encoding(self) -> str:
            return "cp437"

    assert tui.supports_unicode(Cp437()) is False, "legacy consoles must fall back, not crash"


def test_ascii_banner_is_encodable_on_a_legacy_console() -> None:
    tui.STAR_ASCII.encode("cp437")
    "".join(tui.SPINNER_FRAMES_ASCII).encode("cp437")


# --------------------------------------------------------------------------
# Badge
# --------------------------------------------------------------------------


def test_badge_reports_the_result_it_was_given() -> None:
    result = ExecutionResult(
        trace_id="t",
        outcome=Outcome.OPUS,
        retrieval_confidence=0.31,
        cloud_tokens_out=210,
        generation_ms=6000.0,
        latency_ms=6900.0,
        pii_status=PIIStatus.SCRUBBED,
        pii_entities=2,
        quarantined_chunks=1,
        audit_hash="a8f9bb31c1",
    )
    text = tui.render_badge(result, unicode_ok=True).plain

    assert "Opus 5" in text
    assert "35 tok/s" in text, "210 tokens over 6.0 s of MODEL time"
    assert "conf 0.31" in text
    assert "PII 2 redacted" in text
    assert "6900 ms" in text
    assert "gen 6000 ms" in text
    assert "1 quarantined" in text
    assert "ledger a8f9..c1" in text


def test_badge_omits_throughput_when_generation_was_not_timed() -> None:
    """No denominator, no number. Not a zero, not a dash - absent."""
    result = ExecutionResult(
        trace_id="t", outcome=Outcome.BLOCKED, pii_status=PIIStatus.BLOCKED, latency_ms=12.0
    )
    text = tui.render_badge(result, unicode_ok=True).plain
    assert "tok/s" not in text
    assert "PII blocked" in text
    assert "Blocked" in text


def test_badge_has_an_ascii_separator_for_legacy_consoles() -> None:
    result = ExecutionResult(trace_id="t", outcome=Outcome.LOCAL, latency_ms=1.0)
    tui.render_badge(result, unicode_ok=False).plain.encode("cp437")


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------


def test_hud_distinguishes_offline_from_armed() -> None:
    settings = Settings()
    offline = _render(
        tui.render_hud(
            settings,
            budget_line="budget 0/25 msgs",
            budget_style="grey58",
            frontier_available=False,
            residency=[],
        )
    )
    assert "OFFLINE ARMED" in offline
    assert "every escalation degrades to local" in offline

    armed = _render(
        tui.render_hud(
            settings,
            budget_line="budget 0/25 msgs",
            budget_style="grey58",
            frontier_available=True,
            residency=[],
        )
    )
    assert "LEVEL 2 ARMED" in armed
    assert "can spend cloud tokens" in armed


def test_hud_cites_the_source_of_recorded_latencies() -> None:
    """A benchmark shown next to live values has to say it is a benchmark."""
    text = _render(
        tui.render_hud(
            Settings(),
            budget_line="budget 0/25 msgs",
            budget_style="grey58",
            frontier_available=False,
            residency=None,
        )
    )
    assert "recorded in decision 15" in text
    assert "recorded in decision 24" in text
    assert "not measured" in text, "residency was unavailable and must not be invented"


def test_hud_reads_the_live_threshold_not_a_literal() -> None:
    settings = Settings()
    settings.routing.escalation_threshold = 0.42
    text = _render(
        tui.render_hud(
            settings,
            budget_line="budget 0/25 msgs",
            budget_style="grey58",
            frontier_available=False,
            residency=[],
        )
    )
    assert "0.42" in text


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------


def _complete(text: str) -> list[str]:
    from prompt_toolkit.document import Document

    completer = tui.SlashCompleter(["general", "code"])
    return [c.text for c in completer.get_completions(Document(text, len(text)), None)]  # type: ignore[arg-type]


def test_completion_fires_on_slash_commands_only() -> None:
    assert "/doctor" in _complete("/d")
    assert _complete("how does the budget") == [], "prose must not raise a dropdown"


def test_model_argument_completes_to_known_profiles() -> None:
    assert _complete("/model ") == ["code", "general"]
    assert _complete("/model c") == ["code"]


# --------------------------------------------------------------------------
# Commands, against a stub container
# --------------------------------------------------------------------------


class StubOrchestrator:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result
        self.stages: list[Stage] = []
        self.queries: list[str] = []

    async def execute(self, query: str, **kwargs: Any) -> ExecutionResult:
        self.queries.append(query)
        on_stage = kwargs.get("on_stage")
        if on_stage is not None:
            for phase in (Stage.GOVERNANCE, Stage.RETRIEVAL, Stage.GENERATION):
                on_stage(phase)
                self.stages.append(phase)
        return self._result


@pytest.fixture
def container(tmp_path) -> Container:
    settings = Settings()
    settings.audit.path = tmp_path / "audit.jsonl"
    settings.budget.ledger_path = tmp_path / "budget.jsonl"
    budget = RollingWindowBudget(settings.budget)
    governance = GovernanceMiddleware(
        scrubber=PIIScrubber(), guardrail=GuardrailChain([RegexGuardrail()])
    )
    gate = RoutingGate(
        settings=settings.routing,
        budget=budget,
        audit=None,  # type: ignore[arg-type]
        slm_model=settings.ollama.generator_model,
    )
    answer = ExecutionResult(
        trace_id="t",
        outcome=Outcome.LOCAL,
        answer="The gate holds a veto.",
        retrieval_confidence=0.87,
        local_tokens_out=40,
        generation_ms=1000.0,
        latency_ms=1200.0,
        pii_status=PIIStatus.CLEAN,
        audit_hash="deadbeefcafe",
    )
    return Container(
        settings=settings,
        audit=None,  # type: ignore[arg-type]
        governance=governance,
        ollama=None,  # type: ignore[arg-type]
        store=None,  # type: ignore[arg-type]
        rag=None,  # type: ignore[arg-type]
        budget=budget,
        gate=gate,
        frontier=None,  # type: ignore[arg-type]
        orchestrator=StubOrchestrator(answer),  # type: ignore[arg-type]
        harness=None,  # type: ignore[arg-type]
        promotion=None,  # type: ignore[arg-type]
    )


@pytest.fixture
def app(container: Container) -> tui.HybridRagTui:
    return tui.HybridRagTui(container, console=_console())


def test_constructing_the_app_does_not_need_a_terminal(container: Container) -> None:
    """PromptSession binds to a real tty; deferring it is what makes this file possible."""
    app = tui.HybridRagTui(container, console=_console())
    assert app._session is None


async def test_exit_words_end_the_session(app: tui.HybridRagTui) -> None:
    for word in ("exit", "quit", "/exit", "/quit", "EXIT"):
        assert await app.handle(word) is False
    assert await app.handle("") is True


async def test_unknown_command_is_reported_not_answered(app: tui.HybridRagTui) -> None:
    await app.handle("/nope")
    text = app.console.export_text()
    assert "unknown command /nope" in text


async def test_help_lists_every_command(app: tui.HybridRagTui) -> None:
    await app.handle("/help")
    text = app.console.export_text()
    for name in tui.COMMANDS:
        assert name in text


async def test_budget_shows_the_live_window(app: tui.HybridRagTui) -> None:
    await app.handle("/budget")
    text = app.console.export_text()
    assert "0/25 used" in text
    assert "under-estimates" in text, "the caps must not read as the published limits"


async def test_audit_reports_an_absent_ledger_rather_than_an_empty_table(
    app: tui.HybridRagTui,
) -> None:
    await app.handle("/audit")
    assert "no ledger yet" in app.console.export_text()


async def test_audit_tails_real_records(app: tui.HybridRagTui, container: Container) -> None:
    path = container.settings.audit.path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        '{"stage":"orchestration","tier":"local","tokens_in":10,"tokens_out":20,'
        '"confidence":0.91,"record_hash":"abcdef1234"}',
        "{ this is not json",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")

    await app.handle("/audit")
    text = app.console.export_text()
    assert "orchestration" in text
    assert "abcdef12" in text
    assert "unparseable" in text, "a corrupt row must be shown, not skipped"


async def test_model_swap_moves_the_router_too(app: tui.HybridRagTui, container: Container) -> None:
    """The gate holds the resolved name, so a swap that only touches settings is half a swap."""
    assert container.settings.ollama.generator_profile == "general"

    await app.handle("/model code")

    assert container.settings.ollama.generator_model == "qwen2.5-coder:3b"
    assert container.gate._slm_model == "qwen2.5-coder:3b"


async def test_model_swap_refuses_an_unknown_model(
    app: tui.HybridRagTui, container: Container
) -> None:
    """Typing a 7B at a prompt would silently break the measured VRAM envelope."""
    await app.handle("/model llama3.1:70b")

    text = app.console.export_text()
    assert "unknown generator" in text
    assert container.settings.ollama.generator_profile == "general", "unchanged"


async def test_bare_model_lists_profiles(app: tui.HybridRagTui) -> None:
    await app.handle("/model")
    text = app.console.export_text()
    assert "qwen2.5-coder:3b" in text
    assert "(current)" in text


async def test_a_question_runs_the_pipeline_and_prints_a_badge(
    app: tui.HybridRagTui, container: Container
) -> None:
    await app.handle("How does the budget veto work?")

    orchestrator = container.orchestrator
    assert isinstance(orchestrator, StubOrchestrator)
    assert orchestrator.queries == ["How does the budget veto work?"]
    text = app.console.export_text()
    assert "The gate holds a veto." in text
    assert "conf 0.87" in text
    assert "ledger dead..fe" in text


async def test_the_spinner_is_driven_by_pipeline_events(
    app: tui.HybridRagTui, container: Container
) -> None:
    """Not a timer. A progress indicator that is not event-driven is a fabrication."""
    await app.ask("anything")

    orchestrator = container.orchestrator
    assert isinstance(orchestrator, StubOrchestrator)
    assert orchestrator.stages == [Stage.GOVERNANCE, Stage.RETRIEVAL, Stage.GENERATION]


def test_stage_labels_cover_every_stage() -> None:
    assert set(tui.STAGE_LABELS) == set(Stage)


async def test_a_refusal_prints_its_reason_instead_of_an_empty_answer(
    container: Container,
) -> None:
    blocked = ExecutionResult(
        trace_id="t",
        outcome=Outcome.BLOCKED,
        answer="",
        reason="live credential material detected (ANTHROPIC_API_KEY)",
        pii_status=PIIStatus.BLOCKED,
        latency_ms=8.0,
    )
    container.orchestrator = StubOrchestrator(blocked)  # type: ignore[assignment]
    app = tui.HybridRagTui(container, console=_console())

    await app.handle("here is my key")

    text = app.console.export_text()
    assert "live credential material detected" in text
    assert "PII blocked" in text


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


def test_history_is_in_memory_unless_a_path_is_given(tmp_path) -> None:
    """A prompt can hold exactly the data the scrubber exists to keep off disk."""
    from prompt_toolkit.history import FileHistory, InMemoryHistory

    assert isinstance(tui.resolve_history(None), InMemoryHistory)
    assert isinstance(tui.resolve_history(""), InMemoryHistory)
    assert isinstance(tui.resolve_history(str(tmp_path / "h" / "history")), FileHistory)


# --------------------------------------------------------------------------
# Terminal attachment
# --------------------------------------------------------------------------


def test_console_attachment_failure_becomes_an_actionable_message() -> None:
    """A forty-line Win32 traceback reads like the program is broken."""

    class NoConsoleScreenBufferError(Exception):
        pass

    hint = tui.terminal_hint(NoConsoleScreenBufferError("Found xterm-256color"))
    assert hint is not None
    assert "MinTTY" in hint
    assert "hybrid-rag ask" in hint, "it has to name the thing that does work"


def test_unrelated_exceptions_are_not_swallowed() -> None:
    assert tui.terminal_hint(ValueError("something else")) is None
    assert tui.terminal_hint(EOFError()) is None, "Ctrl-D must keep reaching its own handler"


async def test_a_broken_spinner_cannot_fail_a_request(container: Container) -> None:
    """`execute` turns every exception into FAILED, so a raising callback would
    report a renderer bug as a failed request. Presentation cannot change the
    outcome it is presenting."""
    from hybrid_rag.agents.orchestrator import AgentOrchestrator
    from hybrid_rag.distill.dataset_formatter import DatasetFormatter
    from hybrid_rag.governance.audit import NullAuditSink
    from hybrid_rag.llm.opus_client import UnavailableFrontierClient
    from hybrid_rag.rag.local_rag import LocalRAGEngine, NullReranker, SemanticChunker
    from tests.conftest import InMemoryVectorStore, StubEmbedder, StubGenerator

    settings = container.settings
    embedder = StubEmbedder()
    generator = StubGenerator(default="an answer")
    rag = LocalRAGEngine(
        settings=settings.retrieval,
        embedder=embedder,
        store=InMemoryVectorStore(),
        chunker=SemanticChunker(settings.retrieval, embedder=embedder),
        reranker=NullReranker(),
        audit=NullAuditSink(),
        runtime=settings.runtime,
    )
    # A gate of its own: the shared fixture's has `audit=None`, which is fine
    # for the renderers but not for a real `decide()`.
    gate = RoutingGate(
        settings=settings.routing,
        budget=container.budget,
        audit=NullAuditSink(),
        generator=generator,
        verifier=container.governance,
        slm_model=settings.ollama.generator_model,
    )
    orchestrator = AgentOrchestrator(
        settings=settings,
        governance=container.governance,
        rag=rag,
        gate=gate,
        local=generator,
        frontier=UnavailableFrontierClient("offline"),
        audit=NullAuditSink(),
        formatter=DatasetFormatter(),
    )

    def explode(_: Stage) -> None:
        raise RuntimeError("the renderer is broken")

    result = await orchestrator.execute("a question", on_stage=explode)

    assert result.outcome is Outcome.LOCAL, result.reason
    assert result.answer == "an answer"
    assert "the renderer is broken" not in result.reason
