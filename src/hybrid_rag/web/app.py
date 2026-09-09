"""Starlette app over the existing container.

    uv run hybrid-rag serve

Starlette rather than FastAPI: `starlette` and `uvicorn` already arrive with
`fastmcp`, while FastAPI is not installed and would be a new dependency bought
for routing sugar four endpoints do not need.

Four things in here are less obvious than they look.

**Lifespan owns the container.** It is built once in `to_thread` (~3 s cold:
Qdrant open plus the FlashRank ONNX load) and closed in a `finally`. The MCP
server never closes its container, which leaks the Qdrant lock until the
process is killed; that is the bug this file exists not to repeat.

**Blocking work is offloaded.** `build_container`, `verify_chain` and the
budget snapshot all block - fsync per ledger record, a full file re-walk, a
threading lock. On an event loop serving SSE that stalls every other request.

**SSE is hand-rolled on `StreamingResponse`.** `sse_starlette` is present
transitively but is not ours to depend on, and the framing is four lines.

**The static mount is optional.** A missing `web/out` is a normal state - the
frontend may simply not be built yet - so it degrades to a page explaining how
to build it rather than refusing to start. A server that will not boot because
a UI is missing cannot serve the API that would tell you why.
"""

from __future__ import annotations

import asyncio
import html
import json
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from hybrid_rag.agents.orchestrator import ExecutionResult, Stage
from hybrid_rag.config import Settings, get_settings
from hybrid_rag.container import Container, build_container
from hybrid_rag.governance.audit import JsonlAuditSink
from hybrid_rag.hardware import (
    Assessment,
    GpuInfo,
    InstalledModel,
    assess_model_with,
    installed_models,
    probe_gpu,
)
from hybrid_rag.ui.residency import probe_residency, residency_for

__all__ = ["build_app"]

# A sentinel is cleaner than a second queue or a flag: the producer puts this on
# the queue when the pipeline is done, and the consumer stops draining.
_DONE = object()

_NO_BUILD_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>hybrid-rag - UI not built</title>
<style>
  body {{ background:#f4f1ea; color:#0b0b0b; font-family:ui-monospace,monospace;
          margin:0; display:grid; place-items:center; min-height:100vh; padding:24px; }}
  main {{ background:#fff; border:3px solid #0b0b0b; box-shadow:6px 6px 0 0 #0b0b0b;
          padding:28px 32px; max-width:620px; }}
  h1 {{ margin:0 0 12px; font-size:20px; letter-spacing:-0.02em; }}
  code {{ background:#ddf247; padding:1px 5px; }}
  p {{ line-height:1.6; }}
</style>
<main>
  <h1>The API is up. The UI is not built.</h1>
  <p>Expected a compiled frontend at <code>{static_dir}</code>.</p>
  <p>Build it:</p>
  <p><code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code></p>
  <p>Or run the dev server on :3000 and leave this one serving the API:</p>
  <p><code>cd web &amp;&amp; npm run dev</code></p>
  <p>Either way <code>/api/health</code> is answering right now.</p>
</main>
"""


def _container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


# --------------------------------------------------------------------------
# Request helpers
# --------------------------------------------------------------------------


async def _json_body(request: Request) -> dict[str, Any]:
    """Parse a JSON body, refusing anything that skips the CORS preflight.

    `request.json()` will happily parse a body sent as `text/plain`, and a
    text/plain POST is a CORS *simple request* - no preflight, so the origin
    allowlist is never consulted and the request runs. Measured before this
    existed: a POST carrying `Origin: https://evil.example.com` executed the
    whole pipeline and returned 200.

    CORS decides who may READ a response. It does not decide who may cause one.
    Demanding `application/json` makes the request non-simple, which forces the
    preflight the allowlist then rejects. Every mutating route goes through
    here, so none of them can regain that hole by being written later.

    Raises `HTTPException` rather than returning an error response, so a
    caller cannot forget to check for one - the previous `dict | JSONResponse`
    return required an `isinstance` guard at every one of the four call
    sites, and the guard being load-bearing rather than enforced was the
    actual gap. `_error_response` below turns the exception back into the
    same JSON shape at the edge.
    """
    content_type = request.headers.get("content-type", "")
    if content_type.split(";")[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="content-type must be application/json")
    try:
        parsed = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="expected a JSON body") from None
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object")
    return parsed


async def _error_response(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)  # registered only for HTTPException below
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


async def health(request: Request) -> JSONResponse:
    """What the machine is actually running. Unmeasured values say so."""
    container = _container(request)
    settings: Settings = container.settings

    ollama_up = await container.ollama.health()
    chunks = await container.rag.count()
    models = await asyncio.to_thread(probe_residency, settings.promotion.ollama_binary)
    generator = settings.ollama.generator_model
    residency_text, generator_resident = residency_for(models, generator)

    return JSONResponse(
        {
            "ollama_up": ollama_up,
            "ollama_url": settings.ollama.base_url,
            "generator": generator,
            "generator_profile": settings.ollama.generator_profile,
            "generator_resident": generator_resident,
            "residency": residency_text,
            # None means "could not ask", which is not the same as "nothing
            # loaded" and is rendered differently. See ui/residency.py.
            "resident_models": (
                None
                if models is None
                else [
                    {"name": m.name, "size": m.size, "processor": m.processor} for m in models
                ]
            ),
            "embed_model": settings.ollama.embed_model,
            "embed_dim": settings.ollama.embed_dim,
            "rerank_model": settings.retrieval.rerank_model,
            "rerank_enabled": settings.retrieval.rerank_enabled,
            "guard_model": settings.ollama.guard_model,
            "guard_enabled": settings.ollama.guard_enabled,
            "indexed_chunks": chunks,
            "escalation_threshold": settings.routing.escalation_threshold,
            "frontier_available": container.frontier.available,
            "frontier_model": settings.anthropic.model,
        }
    )


async def budget(request: Request) -> JSONResponse:
    """Remaining capacity in the rolling window.

    `exhausted` comes straight off `BudgetSnapshot` now - it is a
    `@computed_field` there covering all three caps `reserve()` enforces
    (rolling window messages, rolling window tokens, weekly messages), not
    hand-derived here from only two of them.
    """
    container = _container(request)
    snap = await asyncio.to_thread(container.budget.snapshot)
    payload: dict[str, Any] = snap.model_dump(mode="json")
    payload["window_hours"] = container.settings.budget.window_hours
    return JSONResponse(payload)


def _read_ledger(sink: JsonlAuditSink, limit: int) -> dict[str, Any]:
    """Blocking. Called through `to_thread`.

    Single pass with a bounded `deque`, not `read_text().splitlines()` -
    the ledger is append-only and only grows, and materialising the whole
    file to keep its last `limit` lines was a second full read on top of
    `verify_chain()`'s own walk, on every `/api/ledger` call.
    """
    intact, bad_line = sink.verify_chain()
    total = 0
    tail: deque[str] = deque(maxlen=limit)
    if sink.path.exists():
        with sink.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                tail.append(line)

    rows: list[dict[str, Any]] = []
    for line in tail:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"stage": "(unparseable)", "tier": "?"})
    return {
        "path": str(sink.path),
        "intact": intact,
        "first_bad_line": bad_line,
        "records": total,
        "rows": rows,
    }


async def ledger(request: Request) -> JSONResponse:
    """The last N audit records, and whether the hash chain still reconciles.

    Safe to serve: `AuditRecord` carries `prompt_hash` and never prompt text
    (rule 4), and guardrail finding evidence is scrubbed before it is stored.
    A test asserts no raw prompt survives this endpoint.
    """
    container = _container(request)
    try:
        limit = int(request.query_params.get("limit", "5"))
    except ValueError:
        limit = 5
    limit = max(1, min(limit, 100))

    settings = container.settings
    sink = JsonlAuditSink(settings.audit.path, schema_version=settings.audit.schema_version)
    payload = await asyncio.to_thread(_read_ledger, sink, limit)
    return JSONResponse(payload)


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def ask(request: Request) -> Response:
    """Run one question and stream the pipeline stages, then the result.

    The stage events come from `on_stage`, which the orchestrator fires from
    inside the pipeline. That is what makes the progress indicator honest: it
    cannot claim to be reranking while it is really waiting on the guard model.

    `execute()` never raises - failures come back as an `ExecutionResult` with
    `outcome: failed` - so the stream always terminates with a `result` event.
    """
    container = _container(request)

    body = await _json_body(request)
    query = str(body.get("query", "")).strip()
    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    queue: asyncio.Queue[Any] = asyncio.Queue()

    def on_stage(stage: Stage) -> None:
        # `put_nowait`, NOT `call_soon_threadsafe`. The orchestrator fires this
        # inline from `execute()`, which is already running on this event loop,
        # so a direct put preserves ordering. Scheduling it instead defers every
        # stage behind the result: the first version of this used
        # `call_soon_threadsafe` and the client received the answer with no
        # stages at all, because the stream had already seen the sentinel and
        # closed by the time the callbacks ran.
        queue.put_nowait(stage)

    async def run() -> None:
        try:
            result = await container.orchestrator.execute(query, on_stage=on_stage)
            await queue.put(result)
        finally:
            await queue.put(_DONE)

    async def stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, Stage):
                    yield _sse("stage", {"stage": str(item)})
                elif isinstance(item, ExecutionResult):
                    yield _sse("result", item.model_dump(mode="json"))
        finally:
            # A client that disconnects mid-answer must not leave the pipeline
            # running with a budget reservation open.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Models and capability
# --------------------------------------------------------------------------


def _companions_mb(container: Container, listed: list[InstalledModel] | None) -> int:
    """VRAM the guard and embedder hold, so the fit check asks the real question.

    "Does the generator fit on an empty card" is not the constraint this project
    lives under. "Does the whole co-resident set fit" is.

    Takes the installed-model list rather than measuring it: every caller
    already has one from its own `ollama list` call this request, and asking
    twice was a second subprocess spawn for information already in hand.
    """
    settings = container.settings
    wanted = {settings.ollama.embed_model}
    if settings.ollama.guard_enabled:
        wanted.add(settings.ollama.guard_model)
    return sum(m.size_mb for m in (listed or []) if m.name in wanted)


def _params(container: Container) -> dict[str, Any]:
    ollama = container.settings.ollama
    return {
        "temperature": ollama.temperature,
        "num_ctx": ollama.num_ctx,
        "keep_alive": ollama.keep_alive,
        "seed": ollama.seed,
    }


def _assessment_json(a: Assessment) -> dict[str, Any]:
    """`asdict` handles every plain field; `usable_mb` on `GpuInfo` is a
    `@property`, which `asdict` silently drops - the same trap `spent_cloud_tokens`
    hit as a pydantic `@property` before it was promoted to `@computed_field`.
    Restore it explicitly rather than hand-writing the other dozen fields
    around it, which is what let this trap go unnoticed the first time.
    """
    payload = asdict(a)
    if a.gpu is not None:
        payload["gpu"]["usable_mb"] = a.gpu.usable_mb
    return payload


def _gpu_json(gpu: GpuInfo | None) -> dict[str, Any] | None:
    if gpu is None:
        return None
    payload = asdict(gpu)
    payload["usable_mb"] = gpu.usable_mb  # `@property` - `asdict` drops it
    return payload


async def models(request: Request) -> JSONResponse:
    """Profiles, what Ollama has installed, and the card they have to fit on."""
    container = _container(request)
    settings = container.settings
    listed = await asyncio.to_thread(installed_models, settings.promotion.ollama_binary)
    gpu = await asyncio.to_thread(probe_gpu)

    return JSONResponse(
        {
            "current_profile": settings.ollama.generator_profile,
            "current_model": settings.ollama.generator_model,
            "profiles": dict(settings.ollama.generator_profiles),
            "params": _params(container),
            # None means Ollama could not be asked, which is not an empty list.
            "installed": (
                None
                if listed is None
                else [
                    {"name": m.name, "digest": m.digest, "size_mb": m.size_mb} for m in listed
                ]
            ),
            "gpu": _gpu_json(gpu),
            "companions_mb": _companions_mb(container, listed),
        }
    )


async def select_model(request: Request) -> JSONResponse:
    """Switch the resident generator to another profile."""
    container = _container(request)
    body = await _json_body(request)
    name = str(body.get("profile", "")).strip()
    if not name:
        return JSONResponse({"error": "profile is required"}, status_code=400)
    try:
        resolved = container.switch_generator(name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(
        {
            "current_profile": container.settings.ollama.generator_profile,
            "current_model": resolved,
            "note": (
                "The previous model stays resident until Ollama evicts it, so both "
                "may occupy VRAM briefly."
            ),
        }
    )


async def set_params(request: Request) -> JSONResponse:
    """Retune temperature, context length and keep_alive."""
    container = _container(request)
    body = await _json_body(request)
    try:
        temperature = body.get("temperature")
        num_ctx = body.get("num_ctx")
        updated = container.set_generation_params(
            temperature=float(temperature) if temperature is not None else None,
            num_ctx=int(num_ctx) if num_ctx is not None else None,
            keep_alive=body.get("keep_alive"),
        )
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(updated)


async def add_profile(request: Request) -> JSONResponse:
    """Register a new generator profile, and report whether it fits."""
    container = _container(request)
    body = await _json_body(request)
    name = str(body.get("name", "")).strip()
    model = str(body.get("model", "")).strip()
    if not name or not model:
        return JSONResponse({"error": "name and model are required"}, status_code=400)
    try:
        resolved = container.add_generator_profile(name, model)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    binary = container.settings.promotion.ollama_binary
    # One measurement of each, reused for both the companion footprint and the
    # fit check below - `assess_model` on its own would spawn `nvidia-smi` and
    # `ollama list` a second time for facts already in hand.
    listed = await asyncio.to_thread(installed_models, binary)
    gpu = await asyncio.to_thread(probe_gpu)
    companions = _companions_mb(container, listed)
    assessment = await asyncio.to_thread(
        assess_model_with,
        resolved,
        num_ctx=container.settings.ollama.num_ctx,
        companions_mb=companions,
        binary=binary,
        gpu=gpu,
        listed=listed,
    )
    return JSONResponse(
        {
            "profiles": dict(container.settings.ollama.generator_profiles),
            "added": {"name": name.lower(), "model": resolved},
            "assessment": _assessment_json(assessment),
        }
    )


async def requirements(request: Request) -> JSONResponse:
    """Can this machine serve `model` alongside the guard and the embedder?

    Weights and GPU capacity are measured. The KV cache is an ESTIMATE scaled
    from one datapoint, and the payload says so - see `hardware.py`. A capability
    check that presents a guess as a measurement is worse than no check, because
    it sounds authoritative.
    """
    container = _container(request)
    model = request.query_params.get("model", "").strip()
    if not model:
        model = container.settings.ollama.generator_model
    try:
        num_ctx = int(request.query_params.get("num_ctx", container.settings.ollama.num_ctx))
    except ValueError:
        num_ctx = container.settings.ollama.num_ctx
    num_ctx = max(256, min(num_ctx, 131_072))

    binary = container.settings.promotion.ollama_binary
    listed = await asyncio.to_thread(installed_models, binary)
    gpu = await asyncio.to_thread(probe_gpu)
    companions = _companions_mb(container, listed)
    assessment = await asyncio.to_thread(
        assess_model_with,
        model,
        num_ctx=num_ctx,
        companions_mb=companions,
        binary=binary,
        gpu=gpu,
        listed=listed,
    )
    payload = _assessment_json(assessment)
    payload["num_ctx"] = num_ctx
    payload["kv_is_estimate"] = True
    return JSONResponse(payload)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_app(settings: Settings | None = None) -> Starlette:
    """Assemble the ASGI app. The container is built during lifespan startup."""
    resolved = settings or get_settings()
    web = resolved.web

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        container = await asyncio.to_thread(build_container, resolved)
        app.state.container = container
        try:
            yield
        finally:
            # `finally`, not a bare trailing statement: a failure anywhere in
            # the served lifetime must still release the Qdrant lock, or the
            # next `hybrid-rag` command on this machine fails to start.
            await container.aclose()

    routes: list[Route | Mount] = [
        Route("/api/health", health, methods=["GET"]),
        Route("/api/budget", budget, methods=["GET"]),
        Route("/api/ledger", ledger, methods=["GET"]),
        Route("/api/ask", ask, methods=["POST"]),
        Route("/api/models", models, methods=["GET"]),
        Route("/api/models/select", select_model, methods=["POST"]),
        Route("/api/models/params", set_params, methods=["POST"]),
        Route("/api/models/profiles", add_profile, methods=["POST"]),
        Route("/api/requirements", requirements, methods=["GET"]),
    ]

    static_dir = web.static_dir
    if static_dir.is_dir():
        routes.append(Mount("/", app=StaticFiles(directory=static_dir, html=True), name="ui"))
    else:

        async def not_built(request: Request) -> HTMLResponse:
            # `static_dir` comes from settings rather than the network, so this
            # is not reachable by a caller - but it is still a path being
            # interpolated into HTML, and escaping it costs nothing.
            return HTMLResponse(
                _NO_BUILD_PAGE.format(static_dir=html.escape(str(static_dir))),
                status_code=200,
            )

        routes.append(Route("/", not_built, methods=["GET"]))

    # DNS-rebinding guard. Without it an attacker points `evil.com` at
    # 127.0.0.1 with a short TTL, at which point their page is SAME-ORIGIN with
    # this server and CORS stops applying entirely - they can then read
    # /api/ledger and every answer, not merely trigger them. Measured before
    # this existed: `Host: evil.example.com` was served a 200.
    #
    # `allow_public_bind` disables it, because an operator who has deliberately
    # bound a public interface is reaching this by some hostname we cannot know.
    # They have already been warned that it is unauthenticated.
    allowed_hosts = ["*"] if web.allow_public_bind else ["127.0.0.1", "localhost", "[::1]"]

    return Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[
            Middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts),
            Middleware(
                CORSMiddleware,
                # Exact origins. A wildcard on a server that can read the
                # governance ledger is not a shortcut worth taking.
                allow_origins=web.cors_origins,
                allow_methods=["GET", "POST"],
                allow_headers=["content-type"],
            )
        ],
        # Turns `_json_body`'s `raise HTTPException(...)` back into the same
        # `{"error": ...}` JSON shape every route returned by hand before.
        exception_handlers={HTTPException: _error_response},
    )
