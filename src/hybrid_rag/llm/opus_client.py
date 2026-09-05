"""Level 2 transport - the only module in this repo that can spend money.

Deliberately thin. Everything that decides *whether* to call lives upstream in
the routing gate and the budget ledger; this file's job is to make the call,
report exactly what it cost, and be trivially substitutable in tests.

Two properties matter more than features here:

* `available` is checkable without making a request. The orchestrator asks
  before reserving budget, because a reservation taken for a call that cannot
  happen is capacity stolen from a call that could.
* Every failure path raises `OpusUnavailable`, never a bare provider exception.
  The caller's fail-safe is to degrade to local, and it should not have to know
  the vendor SDK's exception taxonomy to do that.
"""

from __future__ import annotations

import os
import time
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from hybrid_rag.config import AnthropicSettings
from hybrid_rag.governance.pii import PIIScrubber


class OpusUnavailable(RuntimeError):
    """No credentials, no network, or the provider refused. Degrade to local.

    Every failure mode of the Level 2 tier arrives as this one type: an absent
    key, a malformed key, a DNS failure on an air-gapped host, a refused TLS
    handshake, a 429, a 500, a timeout. The caller's response to all of them is
    identical - answer locally and record why - so making the caller
    discriminate between them would be asking it to care about a distinction
    that has no consequence.
    """


# Failure text ends up in `decision_reason`, which is written to the governance
# ledger - the artifact most likely to be handed to someone else. Rule 4 says
# nothing raw reaches the ledger, and a vendor SDK's exception string is raw by
# definition. Anthropic does not echo credentials back today; this makes that
# not matter.
_ERROR_SCRUBBER = PIIScrubber(block_types=frozenset())


def _safe_error(exc: BaseException) -> str:
    return _ERROR_SCRUBBER.scrub(f"{type(exc).__name__}: {exc}").safe_text[:300]


class OpusCompletion(BaseModel):
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@runtime_checkable
class FrontierClient(Protocol):
    """The narrow port the orchestrator depends on."""

    @property
    def available(self) -> bool: ...

    @property
    def model(self) -> str: ...

    async def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> OpusCompletion: ...

    async def aclose(self) -> None: ...


class AnthropicOpusClient:
    """Live Claude client. Constructed only by the container."""

    def __init__(self, settings: AnthropicSettings, *, client: Any | None = None) -> None:
        self._s = settings
        self._client = client
        self._owns_client = client is None
        # Explicit config wins; otherwise the ambient environment variable. The
        # key is never logged, never put in an audit record, and never returned.
        secret = settings.api_key.get_secret_value() if settings.api_key else None
        self._api_key = secret or os.environ.get("ANTHROPIC_API_KEY") or None

    @property
    def available(self) -> bool:
        """True when a call could plausibly succeed. Checked before reserving budget."""
        return self._client is not None or bool(self._api_key)

    @property
    def model(self) -> str:
        return self._s.model

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise OpusUnavailable(
                "no ANTHROPIC_API_KEY; Level 2 is unreachable and the request "
                "will be answered locally"
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise OpusUnavailable(
                "the `anthropic` package is not installed (extra: cloud)"
            ) from exc
        try:
            # Constructing the client can itself throw - a malformed key, a
            # proxy env var the SDK rejects. On an air-gapped or misconfigured
            # host that must degrade, not crash the request.
            self._client = AsyncAnthropic(
                api_key=self._api_key, timeout=self._s.request_timeout_s
            )
        except Exception as exc:
            raise OpusUnavailable(f"could not construct the client: {_safe_error(exc)}") from exc
        return self._client

    async def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> OpusCompletion:
        client = self._ensure_client()
        t0 = time.perf_counter()
        try:
            message = await client.messages.create(
                model=self._s.model,
                max_tokens=max_tokens or self._s.max_output_tokens,
                temperature=self._s.temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
        except OpusUnavailable:
            raise
        except Exception as exc:
            raise OpusUnavailable(f"Level 2 call failed: {_safe_error(exc)}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = "".join(
            getattr(block, "text", "") for block in getattr(message, "content", []) or []
        )
        usage = getattr(message, "usage", None)
        return OpusCompletion(
            text=text,
            model=str(getattr(message, "model", self._s.model)),
            tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=latency_ms,
            stop_reason=getattr(message, "stop_reason", None),
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            closer = getattr(self._client, "close", None)
            if closer is not None:
                await closer()

    async def __aenter__(self) -> AnthropicOpusClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class UnavailableFrontierClient:
    """Explicit null object for hosts with no credentials.

    Better than `None`: the orchestrator's degradation path is exercised by the
    same code in development as in production, instead of being a branch that
    only ever runs when something is misconfigured.
    """

    def __init__(self, reason: str = "no Level 2 credentials configured") -> None:
        self._reason = reason

    @property
    def available(self) -> bool:
        return False

    @property
    def model(self) -> str:
        return "unavailable"

    async def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int | None = None
    ) -> OpusCompletion:
        del prompt, system, max_tokens
        raise OpusUnavailable(self._reason)

    async def aclose(self) -> None:
        return None
