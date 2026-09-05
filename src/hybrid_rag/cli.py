"""Operator CLI.

Thin. Every command is a few lines of wiring over the composition root - the
logic lives in the modules, not here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hybrid_rag.config import get_settings
from hybrid_rag.container import build_container
from hybrid_rag.eval.eval_harness import LocalResponder, RecordedResponder, load_cases
from hybrid_rag.eval.promotion_gate import GgufArtifact
from hybrid_rag.governance.audit import JsonlAuditSink
from hybrid_rag.governance.guardrails import Direction
from hybrid_rag.routing.routing_gate import RetrievalSignal

app = typer.Typer(add_completion=False, help="Local-first tiered RAG operator commands.")
console = Console()


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
