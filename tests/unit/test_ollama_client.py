"""Wire-level tests for the Level 0 transport.

These assert on the JSON that actually leaves the process. `keep_alive: -1` is
the single most important latency setting in the system - it is what keeps the
3B model pinned in VRAM between queries instead of paying a 2-4 second reload
over PCIe - and a setting that silently stops being sent is indistinguishable
from one that was never there.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from hybrid_rag.config import OllamaSettings, RuntimeSettings
from hybrid_rag.llm.ollama_client import (
    LocalInferenceError,
    NonRetryableInferenceError,
    OllamaClient,
)

BASE = "http://ollama.test"


def _settings(**overrides: object) -> OllamaSettings:
    return OllamaSettings(base_url=BASE, **overrides)  # type: ignore[arg-type]


@respx.mock
async def test_generate_pins_the_model_in_vram() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(
            200, json={"response": "L", "model": "m", "prompt_eval_count": 9, "eval_count": 1}
        )
    )
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))

    client = OllamaClient(_settings())
    await client.generate("hello", max_tokens=1)
    await client.aclose()

    sent = route.calls[0].request
    body = sent.read().decode()
    assert '"keep_alive":-1' in body.replace(" ", "")
    assert '"num_predict":1' in body.replace(" ", "")


@respx.mock
async def test_embeddings_pin_the_model_too() -> None:
    route = respx.post(f"{BASE}/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})
    )
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))

    client = OllamaClient(_settings())
    result = await client.embed(["hi"])
    await client.aclose()

    assert result.dim == 3
    assert '"keep_alive":-1' in route.calls[0].request.read().decode().replace(" ", "")


@respx.mock
async def test_default_model_is_the_single_resident_one() -> None:
    """No caller should be able to trigger an eviction by accident."""
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(200, json={"response": "L", "model": "x"})
    )
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))

    settings = _settings()
    client = OllamaClient(settings)
    await client.generate("hello")
    await client.aclose()

    body = route.calls[0].request.read().decode()
    assert settings.generator_model in body


@respx.mock
async def test_concurrency_is_capped_at_max_workers() -> None:
    """Single-channel RAM: unbounded fan-out contends for bandwidth, not cores."""
    in_flight = 0
    peak = 0

    async def _slow(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json={"response": "L", "model": "m"})

    respx.post(f"{BASE}/api/generate").mock(side_effect=_slow)
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))

    client = OllamaClient(_settings(), runtime=RuntimeSettings(max_workers=2))
    await asyncio.gather(*(client.generate(f"q{i}") for i in range(8)))
    await client.aclose()

    assert peak <= 2, f"semaphore leaked: {peak} concurrent requests"


@respx.mock
async def test_client_errors_are_not_retried() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(404, text="model not found")
    )
    client = OllamaClient(_settings())
    with pytest.raises(NonRetryableInferenceError):
        await client.generate("hello")
    await client.aclose()

    assert route.call_count == 1, "a 404 is deterministic; retrying it just burns wall time"


@respx.mock
async def test_server_errors_are_retried_then_surface() -> None:
    route = respx.post(f"{BASE}/api/generate").mock(
        return_value=httpx.Response(503, text="loading model")
    )
    client = OllamaClient(_settings(max_retries=2))
    with pytest.raises(LocalInferenceError):
        await client.generate("hello")
    await client.aclose()

    assert route.call_count == 3


@respx.mock
async def test_embed_falls_back_to_the_legacy_route() -> None:
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(404, text="no such route"))
    legacy = respx.post(f"{BASE}/api/embeddings").mock(
        return_value=httpx.Response(200, json={"embedding": [1.0, 2.0]})
    )
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))

    client = OllamaClient(_settings())
    result = await client.embed(["hi"])
    await client.aclose()

    assert legacy.called
    assert result.vectors == [[1.0, 2.0]]
