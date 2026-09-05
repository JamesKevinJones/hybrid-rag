"""Air-gap guarantee: no network, no crash, no silent success.

The requirement is that an unset key, a malformed key, a dead DNS resolver or a
refused connection all produce the SAME outcome - a local answer, a structured
governance flag, and zero cloud tokens - without a fatal exception anywhere.

These tests never touch the network. `respx` intercepts Ollama; the frontier
client is either the real `AnthropicOpusClient` with no credentials (which
short-circuits before any socket is opened) or a stub that raises the transport
errors a real air-gapped host would produce.
"""

from __future__ import annotations

import httpx
import pytest

from hybrid_rag.config import AnthropicSettings, Settings, get_settings
from hybrid_rag.llm.opus_client import (
    AnthropicOpusClient,
    OpusUnavailable,
    UnavailableFrontierClient,
)

# --------------------------------------------------------------- the client


def test_no_key_means_unavailable_without_opening_a_socket(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicOpusClient(AnthropicSettings())
    assert client.available is False


async def test_completing_without_a_key_raises_the_one_expected_type(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicOpusClient(AnthropicSettings())

    with pytest.raises(OpusUnavailable, match="no ANTHROPIC_API_KEY"):
        await client.complete("anything")
    await client.aclose()


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("[Errno 11001] getaddrinfo failed"),
        httpx.ConnectTimeout("timed out"),
        httpx.ReadTimeout("read timed out"),
        OSError("network is unreachable"),
        RuntimeError("proxy rejected the handshake"),
    ],
)
async def test_every_transport_failure_becomes_opus_unavailable(failure: Exception) -> None:
    """A caller that had to know httpx's exception tree would be a broken abstraction."""

    class ExplodingMessages:
        async def create(self, **_: object) -> object:
            raise failure

    class ExplodingClient:
        messages = ExplodingMessages()

    client = AnthropicOpusClient(AnthropicSettings(), client=ExplodingClient())
    with pytest.raises(OpusUnavailable):
        await client.complete("anything")


async def test_the_null_client_is_indistinguishable_at_the_boundary() -> None:
    null = UnavailableFrontierClient("no credentials on this host")
    assert null.available is False
    with pytest.raises(OpusUnavailable, match="no credentials"):
        await null.complete("anything")
    await null.aclose()


# ------------------------------------------------------------ .env plumbing


def test_a_bare_key_in_dotenv_reaches_the_client(tmp_path, monkeypatch) -> None:
    """The gap this closes: `env_file` maps PREFIXED names onto settings fields
    and never touches os.environ, so the ordinary `ANTHROPIC_API_KEY` that
    every other tool expects was invisible to the one code path reading it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=sk-ant-api03-DOTENVFIXTURE0000000\n", encoding="utf-8"
    )
    monkeypatch.setattr("hybrid_rag.config.REPO_ROOT", tmp_path)
    get_settings.cache_clear()
    try:
        get_settings()
        assert AnthropicOpusClient(AnthropicSettings()).available is True
    finally:
        get_settings.cache_clear()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-FROMTHEENVIRONMENT00")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-api03-FROMTHEFILE0000000\n")
    monkeypatch.setattr("hybrid_rag.config.REPO_ROOT", tmp_path)
    get_settings.cache_clear()
    try:
        get_settings()
        import os

        assert os.environ["ANTHROPIC_API_KEY"].endswith("FROMTHEENVIRONMENT00")
    finally:
        get_settings.cache_clear()


def test_a_missing_dotenv_is_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("hybrid_rag.config.REPO_ROOT", tmp_path)
    get_settings.cache_clear()
    try:
        assert isinstance(get_settings(), Settings)
    finally:
        get_settings.cache_clear()


# ----------------------------------------------------- the conservative caps


def test_the_budget_defaults_are_the_agreed_under_estimates() -> None:
    """Guessing low costs a few answers. Guessing high costs a lockout."""
    budget = Settings().budget
    assert budget.window_hours == 5.0
    assert budget.max_messages_per_window == 25
    assert budget.max_tokens_per_window == 400_000
    assert budget.weekly_message_cap == 250
