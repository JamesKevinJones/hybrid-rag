"""Level 0 inference client (Ollama).

Async, typed, retrying. Every call returns the token counts, wall latency,
resolved model name and model digest needed to write a complete audit record -
the caller never has to guess what it just spent.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from hybrid_rag.config import OllamaSettings, RuntimeSettings


class LocalInferenceError(RuntimeError):
    """Ollama was reachable but the call failed, or it was not reachable at all."""


class NonRetryableInferenceError(LocalInferenceError):
    """A 4xx from Ollama - bad model name, malformed body. Retrying only burns wall time."""


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, NonRetryableInferenceError):
        return False
    return isinstance(exc, httpx.HTTPError | LocalInferenceError)


class Completion(BaseModel):
    text: str
    model: str
    fingerprint: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    seed: int | None = None
    truncated: bool = False


class Embeddings(BaseModel):
    vectors: list[list[float]]
    model: str
    fingerprint: str | None = None
    dim: int
    latency_ms: float = 0.0


@runtime_checkable
class TextGenerator(Protocol):
    """Narrow port the routing gate depends on. Keeps the gate testable with a stub."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> Completion: ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Narrow port the RAG engine depends on."""

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: list[str], *, model: str | None = None) -> Embeddings: ...


class OllamaClient:
    """Concrete Level 0 backend. Injected, never constructed inside callers."""

    def __init__(
        self,
        settings: OllamaSettings,
        *,
        runtime: RuntimeSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings
        # Bounds in-flight requests to the model server. On a single-channel
        # host, more concurrency means more memory contention, not more speed.
        self._gate = asyncio.Semaphore((runtime or RuntimeSettings()).max_workers)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.request_timeout_s, connect=10.0),
        )
        self._fingerprints: dict[str, str] = {}
        self._fp_lock = asyncio.Lock()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def dim(self) -> int:
        return self._s.embed_dim

    async def health(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def fingerprint(self, model: str) -> str | None:
        """Content digest of the loaded weights. Recorded for provenance."""
        if model in self._fingerprints:
            return self._fingerprints[model]
        async with self._fp_lock:
            if model in self._fingerprints:
                return self._fingerprints[model]
            try:
                r = await self._client.get("/api/tags", timeout=10.0)
                r.raise_for_status()
            except httpx.HTTPError:
                return None
            for entry in r.json().get("models", []):
                name = str(entry.get("name", ""))
                digest = entry.get("digest")
                if digest:
                    self._fingerprints[name] = str(digest)
            return self._fingerprints.get(model)

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._gate:
            return await self._post_unbounded(path, payload)

    async def _post_unbounded(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._s.max_retries + 1),
            wait=wait_exponential(multiplier=0.5, max=8),
            retry=retry_if_exception(_should_retry),
            reraise=True,
        ):
            with attempt:
                try:
                    r = await self._client.post(path, json=payload)
                except httpx.HTTPError as exc:
                    last = exc
                    raise LocalInferenceError(f"{path} unreachable: {exc}") from exc
                if r.status_code >= 500:
                    last = LocalInferenceError(f"{path} -> {r.status_code}: {r.text[:200]}")
                    raise last
                if r.status_code >= 400:
                    # Client errors are deterministic; retrying only burns wall time.
                    raise NonRetryableInferenceError(f"{path} -> {r.status_code}: {r.text[:200]}")
                return dict(r.json())
        raise LocalInferenceError(str(last) if last else f"{path} failed")

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> Completion:
        # Callers that pass no model get the one pinned model. Passing a
        # different name here is what evicts it - do it deliberately or not
        # at all.
        target = model or self._s.generator_model
        options: dict[str, Any] = {
            "seed": self._s.seed,
            "temperature": self._s.temperature,
            "num_ctx": self._s.num_ctx,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": target,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": self._s.keep_alive,
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        t0 = time.perf_counter()
        data = await self._post("/api/generate", payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Completion(
            text=str(data.get("response", "")),
            model=str(data.get("model", target)),
            fingerprint=await self.fingerprint(target),
            tokens_in=int(data.get("prompt_eval_count", 0) or 0),
            tokens_out=int(data.get("eval_count", 0) or 0),
            latency_ms=latency_ms,
            seed=self._s.seed,
            truncated=str(data.get("done_reason", "")) == "length",
        )

    async def generate_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int | None = 128,
        system: str | None = None,
    ) -> tuple[dict[str, Any] | None, Completion]:
        """Constrained JSON call. Returns (parsed_or_None, raw_completion).

        A parse failure is data, not an exception: the caller decides the
        fail-safe, and for the routing gate that fail-safe is always LOCAL.
        """
        completion = await self.generate(
            prompt, model=model, json_mode=True, max_tokens=max_tokens, system=system
        )
        try:
            parsed = json.loads(completion.text)
        except (json.JSONDecodeError, TypeError):
            return None, completion
        return (parsed if isinstance(parsed, dict) else None), completion

    async def embed(self, texts: list[str], *, model: str | None = None) -> Embeddings:
        if not texts:
            return Embeddings(vectors=[], model=model or self._s.embed_model, dim=self._s.embed_dim)
        target = model or self._s.embed_model
        t0 = time.perf_counter()
        try:
            data = await self._post(
                "/api/embed",
                {"model": target, "input": texts, "keep_alive": self._s.keep_alive},
            )
            vectors = [[float(x) for x in v] for v in data.get("embeddings", [])]
        except LocalInferenceError:
            # Ollama before 0.3.x only exposes the single-input /api/embeddings route.
            vectors = []
            for text in texts:
                legacy = await self._post("/api/embeddings", {"model": target, "prompt": text})
                vectors.append([float(x) for x in legacy.get("embedding", [])])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not vectors or not vectors[0]:
            raise LocalInferenceError(f"embedding model {target} returned no vectors")
        return Embeddings(
            vectors=vectors,
            model=target,
            fingerprint=await self.fingerprint(target),
            dim=len(vectors[0]),
            latency_ms=latency_ms,
        )
