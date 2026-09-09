"""Operator CLI.

Thin. Every command is a few lines of wiring over the composition root - the
logic lives in the modules, not here.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from hybrid_rag.config import get_settings
from hybrid_rag.container import build_container
from hybrid_rag.eval.eval_harness import LocalResponder, RecordedResponder, load_cases
from hybrid_rag.eval.promotion_gate import GgufArtifact
from hybrid_rag.governance.audit import JsonlAuditSink
from hybrid_rag.governance.guardrails import Direction
from hybrid_rag.routing.routing_gate import RetrievalSignal
from hybrid_rag.scaffolder import (
    ProjectKind,
    build_scaffold,
    validate_project_name,
    write_scaffold,
)
from hybrid_rag.ui.tui import run_tui
from hybrid_rag.web.app import build_app

app = typer.Typer(
    add_completion=False,
    help="Local-first tiered RAG. Run with no arguments for the interactive UI.",
    invoke_without_command=True,
)
console = Console()


def configure_logging() -> None:
    """Claim the root logger before a dependency configures it for us.

    FlashRank calls `logging.basicConfig(level=INFO)` when its ranker is built,
    which turns on every INFO logger in the process - including httpx, which
    then prints a line per Ollama request straight through whatever is on
    screen. `basicConfig` is a no-op once the root logger already has a
    handler, so claiming it first at the entry point is enough to stop that,
    and it leaves an explicit place to raise the level when debugging.

    An application may configure logging; a library may not. This is the
    application.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)


@app.callback()
def main(
    ctx: typer.Context,
    history_file: Annotated[
        str | None,
        typer.Option(
            "--history-file",
            help=(
                "Persist prompt history to this path. Off by default: a prompt "
                "can hold the personal data the scrubber exists to contain, and "
                "the file is plaintext."
            ),
        ),
    ] = None,
) -> None:
    """Launch the TUI when no subcommand is given."""
    configure_logging()
    if ctx.invoked_subcommand is None:
        run_tui(history_path=history_file)


@app.command()
def chat(
    history_file: Annotated[
        str | None, typer.Option(help="Persist prompt history to this path.")
    ] = None,
) -> None:
    """Open the interactive UI explicitly. Same thing as running with no arguments."""
    run_tui(history_path=history_file)


@app.command()
def doctor() -> None:
    """Check that the local plane is actually up before anything else is run."""
    settings = get_settings()

    async def _run() -> None:
        container = build_container(settings)
        async with container:
            ok = await container.ollama.health()
            count = await container.rag.count()
            snap = container.budget.snapshot()

        table = Table(title="hybrid-rag doctor", show_header=False)
        table.add_row("ollama", "[green]up[/]" if ok else "[red]unreachable[/]")
        table.add_row("ollama url", settings.ollama.base_url)
        table.add_row("resident model", settings.ollama.generator_model)
        table.add_row("keep_alive", str(settings.ollama.keep_alive))
        table.add_row(
            "embeddings", f"{settings.ollama.embed_model} ({settings.ollama.embed_dim}d)"
        )
        table.add_row("reranker", settings.retrieval.rerank_model)
        table.add_row("max workers", str(settings.runtime.max_workers))
        table.add_row(
            "vector backend", f"{settings.vector.backend} @ {settings.vector.persist_dir}"
        )
        table.add_row("indexed chunks", str(count))
        table.add_row("escalation threshold", f"{settings.routing.escalation_threshold}")
        table.add_row(
            "budget window",
            f"{snap.messages_used}/{snap.messages_limit} msgs, "
            f"{snap.tokens_used}/{snap.tokens_limit} tokens",
        )
        console.print(table)

    asyncio.run(_run())


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to index.")],
    glob: Annotated[
        str, typer.Option(help="Pattern used when PATH is a directory.")
    ] = "**/*.md",
) -> None:
    """Chunk, embed and index a local corpus. Never leaves the machine."""

    async def _run() -> None:
        container = build_container()
        async with container:
            paths = [path] if path.is_file() else sorted(p for p in path.glob(glob) if p.is_file())
            if not paths:
                console.print(f"[yellow]no files matched {path}/{glob}[/]")
                raise typer.Exit(1)
            written = await container.rag.ingest_paths(paths)
            console.print(f"indexed [green]{written}[/] chunks from {len(paths)} files")

    asyncio.run(_run())


def _prompt_for_kind() -> ProjectKind:
    """Ask, when `--kind` was not given. Accepts a number or the name."""
    kinds = list(ProjectKind)
    console.print("Project type:")
    for index, kind in enumerate(kinds, start=1):
        console.print(f"  [bold]{index}[/]. {kind.label} [dim]({kind.value})[/]")
    while True:
        answer = typer.prompt("Choose", default="1").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(kinds):
            return kinds[int(answer) - 1]
        for kind in kinds:
            if answer == kind.value:
                return kind
        console.print("[yellow]pick a number from the list, or its short name[/]")


@app.command()
def scaffold(
    name: Annotated[str, typer.Argument(help="Project name, e.g. my-app.")],
    kind: Annotated[
        ProjectKind | None, typer.Option(help="Skip the prompt and use this type.")
    ] = None,
    path: Annotated[
        Path, typer.Option(help="Parent directory the project is created inside.")
    ] = Path(),
    ingest: Annotated[
        bool, typer.Option(help="Index the new ARCHITECTURE.md so RAG can answer about it.")
    ] = True,
    force: Annotated[bool, typer.Option(help="Overwrite existing files.")] = False,
) -> None:
    """Generate a modular, running project skeleton and index its contract.

    The generated project passes its own `pytest` with no edits. That is a
    deliberate constraint: `.github/copilot-instructions.md` forbids agents
    from emitting `pass` and `# TODO` placeholders, and a scaffolder that
    emitted them would be teaching the opposite of the rule it ships beside.
    """
    chosen = kind or _prompt_for_kind()

    # Validation, layout and writing share one handler. A rejected name is a
    # user error with a one-line answer, not a stack trace - and the name is
    # validated BEFORE it is joined to a path, so a traversing name never
    # reaches `resolve()` at all.
    try:
        project = validate_project_name(name)
        plan = build_scaffold(project, chosen, (path / project).resolve())
        written = write_scaffold(plan, force=force)
    except (FileExistsError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc

    console.print(f"scaffolded [green]{plan.kind.label}[/] at {plan.root}")
    for target in written:
        console.print(f"  [dim]{target.relative_to(plan.root)}[/]")

    if not ingest:
        console.print("[dim]not indexed (--no-ingest)[/]")
        return

    async def _index() -> int:
        container = build_container()
        async with container:
            return await container.rag.ingest_paths(
                [plan.architecture_path, plan.root / "README.md"]
            )

    try:
        chunks = asyncio.run(_index())
    except Exception as exc:  # the project is already on disk; indexing is the extra
        console.print(f"[yellow]indexing skipped: {type(exc).__name__}: {exc}[/]")
        console.print(
            "[dim]the project was still created. Index it later with "
            f"`hybrid-rag ingest {plan.root}`[/]"
        )
        return

    console.print(f"indexed [green]{chunks}[/] chunks from ARCHITECTURE.md and README.md")
    console.print(
        "[dim]queries ABOUT this project now retrieve instead of scoring 0.00. "
        "Unrelated queries still score near zero, which is the signal the "
        "router escalates on - see decision 27.[/]"
    )


@app.command()
def route(query: str) -> None:
    """Run the full Level 0 pipeline and print the routing decision without spending anything."""

    async def _run() -> None:
        container = build_container()
        async with container:
            # Invariant 1: inspect before anything else touches the text.
            envelope, _ = await container.governance.ingress(query)
            if not envelope.cleared:
                console.print(f"[red]blocked by governance[/]: {envelope.block_reason}")
                raise typer.Exit(2)

            # Retrieval runs on the redacted text, so a masked secret never
            # reaches the embedding model either.
            retrieval = await container.rag.query(envelope.safe_text)
            decision = await container.gate.decide(
                query,
                retrieval=RetrievalSignal(
                    confidence=retrieval.confidence,
                    n_results=len(retrieval.results),
                    top_score=retrieval.top_score,
                    sources=tuple(retrieval.attributions),
                ),
                envelope=envelope,
            )
            # Nothing was actually sent to Opus, so hand the capacity straight back.
            container.gate.abort(decision, reason="dry run")

        table = Table(title="routing decision", show_header=False)
        table.add_row("tier", f"[bold]{decision.tier}[/]")
        table.add_row("decided by", str(decision.decided_by))
        table.add_row("reason", decision.reason)
        table.add_row("decision confidence", f"{decision.confidence:.3f}")
        table.add_row("retrieval confidence", f"{retrieval.confidence:.3f}")
        table.add_row("heuristic score", f"{decision.heuristic_score:.3f}")
        table.add_row("slm vote", decision.slm_vote or "-")
        table.add_row("degraded", str(decision.degraded))
        table.add_row("pii", f"{envelope.pii_status} {envelope.pii_counts or ''}".strip())
        table.add_row("guardrail", str(envelope.guardrail_verdict))
        table.add_row(
            "fired signals", ", ".join(s.name for s in decision.signals if s.fired) or "-"
        )
        console.print(table)

    asyncio.run(_run())


@app.command()
def scan(
    text: Annotated[str, typer.Argument(help="Text to inspect.")],
    direction: Annotated[
        Direction, typer.Option(help="ingress (user), context (retrieved), egress (outbound).")
    ] = Direction.INGRESS,
) -> None:
    """Run the governance middleware over a string without routing anything."""

    async def _run() -> None:
        container = build_container()
        async with container:
            envelope, scrubbed = await container.governance.inspect(text, direction=direction)

        table = Table(title=f"governance scan ({direction})", show_header=False)
        table.add_row("cleared", "[green]yes[/]" if envelope.cleared else "[red]no[/]")
        table.add_row("pii status", str(envelope.pii_status))
        table.add_row("pii entities", str(envelope.pii_counts or "-"))
        table.add_row("guardrail", str(envelope.guardrail_verdict))
        table.add_row(
            "rules fired",
            ", ".join(sorted({f.rule for f in envelope.guardrail_findings})) or "-",
        )
        table.add_row("reason", envelope.block_reason or "cleared")
        table.add_row("safe text", scrubbed.safe_text[:300])
        console.print(table)
        if not envelope.cleared:
            raise typer.Exit(2)

    asyncio.run(_run())


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="The question to answer.")],
) -> None:
    """Run the full Module 5 pipeline. May spend cloud tokens if the gate escalates."""

    async def _run() -> None:
        container = build_container()
        async with container:
            result = await container.orchestrator.execute(query)

        table = Table(title="execution", show_header=False)
        table.add_row("outcome", f"[bold]{result.outcome}[/]")
        table.add_row("reason", result.reason)
        table.add_row("retrieval confidence", f"{result.retrieval_confidence:.3f}")
        table.add_row("routing stage", result.routing_stage or "-")
        table.add_row("quarantined chunks", str(result.quarantined_chunks))
        table.add_row(
            "tokens",
            f"local {result.local_tokens_in}+{result.local_tokens_out} | "
            f"cloud {result.cloud_tokens_in}+{result.cloud_tokens_out}",
        )
        table.add_row("latency", f"{result.latency_ms:.0f} ms")
        table.add_row("harvested", str(result.distilled))
        console.print(table)
        console.print(result.answer or "[dim](no answer emitted)[/]")

    asyncio.run(_run())


@app.command()
def evaluate(
    model: Annotated[
        str | None, typer.Option(help="Candidate model. Defaults to the resident one.")
    ] = None,
) -> None:
    """Score the local model against the recorded Opus baseline. Spends nothing."""

    async def _run() -> None:
        container = build_container()
        settings = container.settings
        cases = load_cases(settings.eval.cases_path)
        if not cases:
            console.print(f"[yellow]no eval cases at {settings.eval.cases_path}[/]")
            raise typer.Exit(1)

        async with container:
            candidate = LocalResponder(
                rag=container.rag, generator=container.ollama, model=model
            )
            report = await container.harness.run(candidate, cases)
            baseline = RecordedResponder(settings.eval.baseline_path)
            baseline_report = await container.harness.run(baseline, cases)

        for r in (report, baseline_report):
            console.print(r.summary())
        report.save(settings.eval.reports_dir / f"{report.label}.json")

    asyncio.run(_run())


@app.command()
def promote(
    gguf_dir: Annotated[Path, typer.Argument(help="Directory holding the Colab GGUF.")],
    name: Annotated[str, typer.Option(help="Name to register the candidate under.")],
    apply: Annotated[
        bool, typer.Option(help="Actually re-point the serving alias if eligible.")
    ] = False,
) -> None:
    """Register a GGUF, evaluate it, and re-point the alias only if it qualifies."""

    async def _run() -> None:
        container = build_container()
        settings = container.settings
        cases = load_cases(settings.eval.cases_path)

        async with container:
            artifact = GgufArtifact.discover(gguf_dir)
            await container.promotion.register_gguf(artifact, model_name=name)
            verdict = await container.promotion.evaluate(
                candidate=LocalResponder(
                    rag=container.rag, generator=container.ollama, model=name, label=name
                ),
                baseline=RecordedResponder(settings.eval.baseline_path),
                cases=cases,
            )
            verdict = await container.promotion.promote(verdict, model_name=name, apply=apply)

        console.print(verdict.summary())
        if not verdict.eligible:
            raise typer.Exit(2)

    asyncio.run(_run())


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address. Loopback unless forced.")] = None,
    port: Annotated[int | None, typer.Option(help="Port to listen on.")] = None,
    public: Annotated[
        bool,
        typer.Option(
            "--public",
            help="Permit a non-loopback bind. This server is UNAUTHENTICATED.",
        ),
    ] = False,
) -> None:
    """Serve the web UI and its API on localhost.

    Holds one container for the process lifetime, which means it holds the
    embedded Qdrant lock: the TUI and every other `hybrid-rag` command will
    refuse to start while this runs, and vice versa. That is embedded mode
    working as designed - see decision 33.
    """
    import uvicorn

    settings = get_settings()
    try:
        web = settings.web.model_copy(
            update={
                "host": host or settings.web.host,
                "port": port or settings.web.port,
                "allow_public_bind": public or settings.web.allow_public_bind,
            }
        )
        # model_copy skips validation, so re-validate deliberately rather than
        # letting --host 0.0.0.0 slip past the loopback guard.
        web = type(web).model_validate(web.model_dump())
    except ValidationError as exc:
        console.print(f"[red]{exc.errors()[0]['msg']}[/]")
        raise typer.Exit(2) from exc

    settings.web = web

    if not web.static_dir.is_dir():
        console.print(
            f"[yellow]no compiled UI at {web.static_dir}[/] - serving the API only. "
            "Build it with `cd web; npm run build`, or run `npm run dev` on :3000."
        )

    try:
        application = build_app(settings)
    except RuntimeError as exc:
        # Embedded Qdrant takes an EXCLUSIVE, non-blocking lock. Anything else
        # holding it - the TUI, an MCP server, another `hybrid-rag` - makes
        # this fail instantly, and a traceback buries the one useful sentence.
        if "already accessed by another instance" in str(exc):
            console.print(
                f"[red]the vector index at {settings.vector.persist_dir} is already open.[/]"
            )
            console.print(
                "[dim]Embedded Qdrant is single-process. Close the TUI, any MCP "
                "server and any other hybrid-rag command, then try again.[/]"
            )
            raise typer.Exit(2) from exc
        raise

    if web.allow_public_bind and web.host not in {"127.0.0.1", "localhost", "::1"}:
        console.print(
            f"[yellow]binding {web.host} - this server has NO authentication and can "
            "read the governance ledger. Put something in front of it.[/]"
        )

    console.print(f"hybrid-rag on [bold]http://{web.host}:{web.port}[/]")
    uvicorn.run(application, host=web.host, port=web.port, log_level="warning")


@app.command("verify-ledger")
def verify_ledger() -> None:
    """Re-walk the audit hash chain and report the first tampered line, if any."""
    settings = get_settings()
    sink = JsonlAuditSink(settings.audit.path, schema_version=settings.audit.schema_version)
    intact, bad_line = sink.verify_chain()
    if intact:
        console.print(f"[green]ledger intact[/] {settings.audit.path}")
    else:
        console.print(f"[red]ledger broken at line {bad_line}[/] {settings.audit.path}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
