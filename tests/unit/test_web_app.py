"""The local web surface, driven through a real ASGI transport.

The container is stubbed so these run with no Ollama, no index and no network -
but the routes, the middleware, the SSE framing and the lifespan are the real
ones. Same shape as `test_mcp_server.py`, which drives FastMCP in-process.

Three of these assertions are load-bearing rather than descriptive:

1. **No raw prompt text leaves `/api/ledger`.** Rule 4 says the ledger stores
   hashes only; this is the endpoint that would expose a regression in that.
2. **The stage events arrive in pipeline order**, because a progress indicator
   driven by anything other than real events is a fabricated measurement.
3. **The `computed_field` values are present in the JSON.** They were plain
   `@property` until this module existed, which meant `model_dump()` silently
   dropped the two facts the whole project is about.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from hybrid_rag.agents.orchestrator import ExecutionResult, Outcome, Stage
from hybrid_rag.config import Settings, WebSettings
from hybrid_rag.governance.audit import PIIStatus
from hybrid_rag.routing.budget import BudgetSnapshot
from hybrid_rag.web.app import build_app

SECRET_PROMPT = "my password is hunter2 and my email is kevin@example.com"


class StubOllama:
    async def health(self) -> bool:
        return True


class StubRag:
    async def count(self) -> int:
        return 42


class StubBudget:
    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            now=0.0,
            window_started_at=0.0,
            messages_used=3,
            messages_limit=25,
            tokens_used=1000,
            tokens_limit=400_000,
            weekly_messages_used=3,
            weekly_messages_limit=250,
            open_reservations=0,
            seconds_to_window_reset=600.0,
        )


class StubFrontier:
    available = False


class StubOrchestrator:
    """Fires the real Stage sequence, then returns a canned result."""

    def __init__(self, result: ExecutionResult) -> None:
        self.result = result
        self.queries: list[str] = []

    async def execute(self, query: str, **kwargs: Any) -> ExecutionResult:
        self.queries.append(query)
        on_stage = kwargs.get("on_stage")
        if on_stage is not None:
            for stage in (Stage.GOVERNANCE, Stage.RETRIEVAL, Stage.GENERATION):
                on_stage(stage)
        return self.result


class StubContainer:
    def __init__(self, settings: Settings, result: ExecutionResult) -> None:
        self.settings = settings
        self.ollama = StubOllama()
        self.rag = StubRag()
        self.budget = StubBudget()
        self.frontier = StubFrontier()
        self.orchestrator = StubOrchestrator(result)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def _result(**overrides: Any) -> ExecutionResult:
    base: dict[str, Any] = {
        "trace_id": "trace-1",
        "outcome": Outcome.LOCAL,
        "answer": "The budget ledger reserves before the call and settles after.",
        "reason": "retrieval confidence cleared the threshold",
        "audit_hash": "a8f9bb31c1d2e3f4",
        "retrieval_confidence": 0.94,
        "local_tokens_in": 500,
        "local_tokens_out": 120,
        "latency_ms": 412.0,
        "generation_ms": 340.0,
        "pii_status": PIIStatus.CLEAN,
    }
    base.update(overrides)
    return ExecutionResult(**base)


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.audit.path = tmp_path / "audit.jsonl"
    s.web = WebSettings(static_dir=tmp_path / "no-such-build")
    return s


def _app(settings: Settings, result: ExecutionResult | None = None) -> tuple[Any, StubContainer]:
    """The real app with a stub container injected.

    `httpx.ASGITransport` does not run the lifespan protocol, so the real
    lifespan never fires here and the state is set directly. Lifespan gets its
    own test below, driven through the ASGI protocol by hand - it is where the
    Qdrant lock is released, so it is worth exercising rather than assuming.
    """
    container = StubContainer(settings, result or _result())
    app = build_app(settings)
    app.state.container = container
    return app, container


async def _client(app: Any) -> AsyncIterator[httpx.AsyncClient]:
    # A loopback base_url, because `TrustedHostMiddleware` now checks the Host
    # header and httpx derives it from here. The old `http://test` was rejected
    # by the DNS-rebinding guard - which is the guard working, not a defect.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        yield client


# ------------------------------------------------------------------ settings


def test_a_non_loopback_bind_is_refused() -> None:
    """The server is unauthenticated and can read the ledger."""
    with pytest.raises(ValueError, match="refusing to bind"):
        WebSettings(host="0.0.0.0")


def test_a_public_bind_is_allowed_only_when_asked_for_deliberately() -> None:
    web = WebSettings(host="0.0.0.0", allow_public_bind=True)
    assert web.host == "0.0.0.0"


def test_cors_is_an_exact_allowlist_not_a_wildcard() -> None:
    assert "*" not in WebSettings().cors_origins


# -------------------------------------------------------------------- routes


async def test_health_reports_what_it_could_measure(settings: Settings) -> None:
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ollama_up"] is True
    assert body["indexed_chunks"] == 42
    assert body["escalation_threshold"] == settings.routing.escalation_threshold
    assert body["frontier_available"] is False


async def test_budget_exposes_the_computed_remaining_fields(settings: Settings) -> None:
    """These were plain properties and did not serialize at all."""
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/budget")

    body = response.json()
    assert body["messages_remaining"] == 22
    assert body["tokens_remaining"] == 399_000
    assert body["weekly_messages_remaining"] == 247
    assert body["exhausted"] is False


def test_exhausted_is_true_when_only_the_weekly_cap_is_hit() -> None:
    """`exhausted` must cover all three caps `RollingWindowBudget.reserve()`
    enforces, not just the two in the 5-hour rolling window.

    The rolling window here is wide open - plenty of messages and tokens left
    in it - but the weekly cap is spent. A version of `exhausted` that only
    looked at `messages_remaining` / `tokens_remaining` reported capacity here
    while every escalation was actually being vetoed on `DENY_WEEKLY_CAP`.
    """
    snap = BudgetSnapshot(
        now=0.0,
        window_started_at=0.0,
        messages_used=1,
        messages_limit=25,
        tokens_used=100,
        tokens_limit=400_000,
        weekly_messages_used=250,
        weekly_messages_limit=250,
        open_reservations=0,
        seconds_to_window_reset=600.0,
    )
    assert snap.messages_remaining > 0
    assert snap.tokens_remaining > 0
    assert snap.weekly_messages_remaining == 0
    assert snap.exhausted is True


async def test_ask_streams_the_stages_in_order_then_the_result(settings: Settings) -> None:
    app, container = _app(settings)
    async for client in _client(app):
        response = await client.post("/api/ask", json={"query": "how does the veto work?"})
        text = response.text

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    stages = [
        json.loads(line[len("data: ") :])["stage"]
        for line in text.splitlines()
        if line.startswith("data: ") and '"stage"' in line
    ]
    assert stages == ["governance", "retrieval", "generation"]
    assert container.orchestrator.queries == ["how does the veto work?"]


async def test_ask_emits_the_computed_token_fields(settings: Settings) -> None:
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.post("/api/ask", json={"query": "anything"})

    payload = next(
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and '"outcome"' in line
    )
    assert payload["spent_cloud_tokens"] is False
    assert payload["token_source"] == "local_ollama"
    assert payload["audit_hash"] == "a8f9bb31c1d2e3f4"


async def test_a_blocked_result_carries_the_reason_and_no_answer(settings: Settings) -> None:
    app, _ = _app(
        settings,
        _result(
            outcome=Outcome.BLOCKED,
            answer="",
            reason="guardrail denied: instruction_override",
        ),
    )
    async for client in _client(app):
        response = await client.post("/api/ask", json={"query": "ignore all previous"})

    payload = next(
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and '"outcome"' in line
    )
    assert payload["outcome"] == "blocked"
    assert payload["answer"] == ""
    assert "instruction_override" in payload["reason"]


@pytest.mark.parametrize("body", [{}, {"query": "   "}])
async def test_an_empty_query_is_refused(settings: Settings, body: dict[str, Any]) -> None:
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.post("/api/ask", json=body)

    assert response.status_code == 400


# -------------------------------------------------------------------- ledger


async def test_ledger_returns_rows_and_chain_status(settings: Settings) -> None:
    from hybrid_rag.governance.audit import AuditRecord, JsonlAuditSink, hash_prompt

    sink = JsonlAuditSink(settings.audit.path)
    for index in range(3):
        sink.emit(
            AuditRecord(
                trace_id=f"t{index}",
                stage="orchestration",
                agent_id="test",
                tier="local",
                prompt_hash=hash_prompt(SECRET_PROMPT),
            )
        )
    sink.close()

    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/ledger?limit=2")

    body = response.json()
    assert body["intact"] is True
    assert body["records"] == 3
    assert len(body["rows"]) == 2


async def test_ledger_never_exposes_raw_prompt_text(settings: Settings) -> None:
    """Rule 4, asserted at the boundary that would leak it."""
    from hybrid_rag.governance.audit import AuditRecord, JsonlAuditSink, hash_prompt

    sink = JsonlAuditSink(settings.audit.path)
    sink.emit(
        AuditRecord(
            trace_id="t",
            stage="orchestration",
            agent_id="test",
            tier="local",
            prompt_hash=hash_prompt(SECRET_PROMPT),
        )
    )
    sink.close()

    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/ledger")

    assert "hunter2" not in response.text
    assert "kevin@example.com" not in response.text


# --------------------------------------------------------------- static mount


async def test_a_missing_ui_build_serves_an_explanation_not_a_crash(
    settings: Settings,
) -> None:
    """A frontend that has not been built yet is a normal state."""
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/")

    assert response.status_code == 200
    assert "npm run build" in response.text


async def test_a_built_ui_is_mounted_when_present(settings: Settings, tmp_path) -> None:
    build = tmp_path / "out"
    build.mkdir()
    (build / "index.html").write_text("<h1>console</h1>", encoding="utf-8")
    settings.web = WebSettings(static_dir=build)

    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/")

    assert response.status_code == 200
    assert "console" in response.text


# ------------------------------------------------------------------ lifespan
#
# `ASGITransport` skips lifespan entirely, so it is driven by hand here. This
# is where the embedded-Qdrant lock is released: a container that survives
# shutdown holds `data/index` until the process is killed, and every other
# hybrid-rag command on the machine then fails to start. The MCP server has
# exactly that bug, which is why it is tested here and not assumed.


async def _drive_lifespan(app: Any) -> list[str]:
    """Run startup then shutdown through the raw ASGI protocol."""
    sent: list[str] = []
    incoming: list[dict[str, Any]] = [
        {"type": "lifespan.startup"},
        {"type": "lifespan.shutdown"},
    ]

    async def receive() -> dict[str, Any]:
        return incoming.pop(0)

    async def send(message: dict[str, Any]) -> None:
        sent.append(message["type"])

    await app({"type": "lifespan"}, receive, send)
    return sent


async def test_lifespan_closes_the_container_on_shutdown(
    settings: Settings, monkeypatch
) -> None:
    container = StubContainer(settings, _result())
    monkeypatch.setattr("hybrid_rag.web.app.build_container", lambda _s: container)

    sent = await _drive_lifespan(build_app(settings))

    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert container.closed is True, "a leaked container holds the Qdrant lock"


async def test_lifespan_closes_the_container_even_when_serving_failed(
    settings: Settings, monkeypatch
) -> None:
    """The `finally` is the point. A crash must still release the lock."""
    container = StubContainer(settings, _result())
    monkeypatch.setattr("hybrid_rag.web.app.build_container", lambda _s: container)
    app = build_app(settings)

    async def receive() -> dict[str, Any]:
        return {"type": "lifespan.startup"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "lifespan.startup.complete":
            raise RuntimeError("the server died mid-flight")

    with pytest.raises(RuntimeError, match="died mid-flight"):
        await app({"type": "lifespan"}, receive, send)

    assert container.closed is True


# ------------------------------------------------------------------ csrf/host
#
# Both of these were found by the local security review of this module's own
# diff, and both were REPRODUCED against a running server before being fixed:
#
#   POST /api/ask, Content-Type: text/plain, Origin: https://evil.example.com
#       -> 200, and the whole pipeline ran.
#   GET /api/ledger, Host: evil.example.com
#       -> 200.
#
# CORS decides who may READ a response. It does not decide who may cause one.


async def test_a_text_plain_post_is_refused(settings: Settings) -> None:
    """A `text/plain` POST is a CORS *simple* request - no preflight, no allowlist.

    `request.json()` parses it happily, so without this check any page the user
    visits could drive the local pipeline and spend the metered window.
    """
    app, container = _app(settings)
    async for client in _client(app):
        response = await client.post(
            "/api/ask",
            content=b'{"query":"hi"}',
            headers={"content-type": "text/plain", "origin": "https://evil.example.com"},
        )

    assert response.status_code == 415
    assert container.orchestrator.queries == [], "the pipeline must not have run"


async def test_a_json_post_with_a_charset_is_accepted(settings: Settings) -> None:
    """`application/json; charset=utf-8` is still application/json."""
    app, container = _app(settings)
    async for client in _client(app):
        response = await client.post(
            "/api/ask",
            content=b'{"query":"hi"}',
            headers={"content-type": "application/json; charset=utf-8"},
        )

    assert response.status_code == 200
    assert container.orchestrator.queries == ["hi"]


@pytest.mark.parametrize("host", ["evil.example.com", "attacker.test"])
async def test_a_foreign_host_header_is_refused(settings: Settings, host: str) -> None:
    """DNS rebinding: `evil.com` -> 127.0.0.1 makes the attacker same-origin."""
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/ledger", headers={"host": host})

    assert response.status_code == 400


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "127.0.0.1:8000"])
async def test_loopback_hosts_are_accepted(settings: Settings, host: str) -> None:
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/health", headers={"host": host})

    assert response.status_code == 200


async def test_a_deliberate_public_bind_disables_the_host_guard(tmp_path) -> None:
    """An operator who bound a public interface reaches it by a name we cannot know.

    They have already been warned it is unauthenticated; the guard would only
    break the thing they explicitly asked for.
    """
    settings = Settings()
    settings.audit.path = tmp_path / "audit.jsonl"
    settings.web = WebSettings(
        host="0.0.0.0",
        allow_public_bind=True,
        static_dir=tmp_path / "none",
    )
    app, _ = _app(settings)
    async for client in _client(app):
        response = await client.get("/api/health", headers={"host": "box.local"})

    assert response.status_code == 200


# ------------------------------------------------------------------- models
#
# The stub container carries the real `Settings`, so these exercise the actual
# validators in `Container.set_generation_params` and `add_generator_profile`
# rather than a mock of them.


class ModelStubContainer(StubContainer):
    """Adds the mutation surface the model routes need."""

    def switch_generator(self, name: str) -> str:
        ollama = self.settings.ollama
        if name not in ollama.generator_profiles:
            known = ", ".join(sorted(ollama.generator_profiles))
            raise ValueError(f"unknown generator {name!r}; known profiles: {known}")
        ollama.generator_profile = name
        ollama.generator_model_override = None
        return ollama.generator_model

    def set_generation_params(self, **kwargs: Any) -> dict[str, Any]:
        from hybrid_rag.container import Container

        return Container.set_generation_params(self, **kwargs)  # type: ignore[arg-type]

    def add_generator_profile(self, name: str, model: str) -> str:
        from hybrid_rag.container import Container

        return Container.add_generator_profile(self, name, model)  # type: ignore[arg-type]


def _model_app(settings: Settings) -> tuple[Any, ModelStubContainer]:
    container = ModelStubContainer(settings, _result())
    app = build_app(settings)
    app.state.container = container
    return app, container


async def test_models_lists_profiles_and_params(settings: Settings) -> None:
    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.get("/api/models")

    body = response.json()
    assert body["current_profile"] == "general"
    assert "code" in body["profiles"]
    assert body["params"]["temperature"] == 0.0
    assert body["params"]["num_ctx"] == 4096


async def test_selecting_a_known_profile_switches_the_generator(settings: Settings) -> None:
    app, container = _model_app(settings)
    async for client in _client(app):
        response = await client.post("/api/models/select", json={"profile": "code"})

    assert response.status_code == 200
    body = response.json()
    assert body["current_model"] == "qwen2.5-coder:3b"
    assert container.settings.ollama.generator_profile == "code"
    # Exactly `SelectResult` (`lib/types.ts`) - not `ModelsInfo`. The route
    # never re-measures `installed`/`gpu`/`companions_mb`, so claiming that
    # wider type let a caller read those fields as silently-`undefined`.
    assert set(body.keys()) == {"current_profile", "current_model", "note"}


async def test_selecting_an_unknown_profile_is_refused(settings: Settings) -> None:
    """`switch_generator` refuses arbitrary names to protect the VRAM envelope."""
    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.post("/api/models/select", json={"profile": "llama3.1:70b"})

    assert response.status_code == 400
    assert "unknown generator" in response.json()["error"]


async def test_params_can_be_retuned(settings: Settings) -> None:
    app, container = _model_app(settings)
    async for client in _client(app):
        response = await client.post(
            "/api/models/params", json={"temperature": 0.7, "num_ctx": 8192}
        )

    assert response.json() == {
        "temperature": 0.7,
        "num_ctx": 8192,
        "keep_alive": -1,
        "seed": 1337,
    }
    assert container.settings.ollama.temperature == 0.7


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"temperature": 9}, "temperature must be"),
        ({"temperature": -1}, "temperature must be"),
        ({"num_ctx": 10}, "num_ctx must be"),
        ({"num_ctx": 999_999}, "num_ctx must be"),
    ],
)
async def test_out_of_range_params_are_refused(
    settings: Settings, payload: dict[str, Any], fragment: str
) -> None:
    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.post("/api/models/params", json=payload)

    assert response.status_code == 400
    assert fragment in response.json()["error"]


async def test_a_profile_can_be_added(settings: Settings) -> None:
    app, container = _model_app(settings)
    async for client in _client(app):
        response = await client.post(
            "/api/models/profiles", json={"name": "Tiny", "model": "qwen2.5:0.5b"}
        )

    body = response.json()
    assert body["added"] == {"name": "tiny", "model": "qwen2.5:0.5b"}
    assert container.settings.ollama.generator_profiles["tiny"] == "qwen2.5:0.5b"
    # It reports the fit rather than refusing on size - whether a spill into
    # system memory is acceptable is the operator's call, not the route's.
    assert "verdict" in body["assessment"]


async def test_adding_a_profile_measures_the_host_exactly_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`installed_models`/`probe_gpu` must be called once each per request.

    `add_profile` used to ask `_companions_mb` for the installed list and then
    hand the model straight to `assess_model`, which measured the GPU and the
    installed list AGAIN itself - two subprocess spawns repeated for no reason.
    `assess_model_with` takes both as arguments instead.
    """
    from hybrid_rag.hardware import GpuInfo, InstalledModel

    calls = {"listed": 0, "gpu": 0}

    def fake_installed(binary: str) -> list[InstalledModel]:
        calls["listed"] += 1
        return [InstalledModel(name="qwen2.5:0.5b", digest="abc", size_mb=500)]

    def fake_gpu() -> GpuInfo:
        calls["gpu"] += 1
        return GpuInfo(name="Test GPU", total_mb=6000, free_mb=5000, driver="1.0")

    monkeypatch.setattr("hybrid_rag.web.app.installed_models", fake_installed)
    monkeypatch.setattr("hybrid_rag.web.app.probe_gpu", fake_gpu)
    monkeypatch.setattr("hybrid_rag.hardware.model_details", lambda model, binary: None)

    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.post(
            "/api/models/profiles", json={"name": "tiny", "model": "qwen2.5:0.5b"}
        )

    assert response.status_code == 200
    assert calls == {"listed": 1, "gpu": 1}


async def test_requirements_measures_the_host_exactly_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same duplication, same fix, on `/api/requirements`."""
    from hybrid_rag.hardware import GpuInfo, InstalledModel

    calls = {"listed": 0, "gpu": 0}

    def fake_installed(binary: str) -> list[InstalledModel]:
        calls["listed"] += 1
        return [InstalledModel(name="qwen2.5:0.5b", digest="abc", size_mb=500)]

    def fake_gpu() -> GpuInfo:
        calls["gpu"] += 1
        return GpuInfo(name="Test GPU", total_mb=6000, free_mb=5000, driver="1.0")

    monkeypatch.setattr("hybrid_rag.web.app.installed_models", fake_installed)
    monkeypatch.setattr("hybrid_rag.web.app.probe_gpu", fake_gpu)
    monkeypatch.setattr("hybrid_rag.hardware.model_details", lambda model, binary: None)

    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.get("/api/requirements?model=qwen2.5:0.5b")

    assert response.status_code == 200
    assert calls == {"listed": 1, "gpu": 1}
    # `GpuInfo.usable_mb` is a `@property` - `asdict` alone drops it.
    assert response.json()["gpu"]["usable_mb"] == 6000 - 800  # COMPOSITOR_RESERVE_MB


@pytest.mark.parametrize("name", ["", "has space", "semi;colon", "../escape"])
async def test_a_malformed_profile_name_is_refused(settings: Settings, name: str) -> None:
    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.post(
            "/api/models/profiles", json={"name": name, "model": "qwen2.5:0.5b"}
        )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "path", ["/api/models/select", "/api/models/params", "/api/models/profiles"]
)
async def test_every_mutating_route_keeps_the_csrf_guard(
    settings: Settings, path: str
) -> None:
    """The guard is shared, so a route added later cannot quietly skip it."""
    app, _ = _model_app(settings)
    async for client in _client(app):
        response = await client.post(
            path,
            content=b'{"profile":"code"}',
            headers={"content-type": "text/plain", "origin": "https://evil.example.com"},
        )

    assert response.status_code == 415
