"""Prove the MCP server is reachable the way an editor actually reaches it.

    uv run python scripts/verify_mcp_clients.py
    uv run python scripts/verify_mcp_clients.py --json

`tests/unit/test_mcp_server.py` drives the same three tools through an
in-process FastMCP client. That covers the tool logic and the schemas, and it
cannot cover the thing that actually breaks in an editor: the launch command.
An in-process client never spawns an interpreter, so it cannot notice that the
configured one does not exist, that the module path only resolves from the repo
root, or that something wrote to stdout and corrupted the JSON-RPC stream.

So this speaks the wire protocol to a real subprocess: newline-delimited
JSON-RPC over stdin/stdout, `initialize`, `notifications/initialized`,
`tools/list`, `tools/call`.

It checks four things a unit test structurally cannot:

1. The interpreter named in `.vscode/mcp.json` exists on disk.
2. The module path in it resolves from a working directory that is NOT the repo
   root - the failure mode `src.hybrid_rag...` has and `hybrid_rag...` does not.
3. `tools/list` returns exactly the three frozen tool names.
4. `local_rag_search` with an empty query leaves the server alive and still
   answering, rather than taking the process down. Whether the empty query
   itself succeeds or returns a tool error is not the assertion - surviving it
   is.

Exit code 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
VSCODE_CONFIG = REPO_ROOT / ".vscode" / "mcp.json"
ANTIGRAVITY_CONFIG = REPO_ROOT / ".antigravity" / "antigravity.json"

SERVER_NAME = "hybrid-rag"
SERVER_MODULE = "hybrid_rag.mcp_servers.server"
EXPECTED_TOOLS = frozenset({"local_rag_search", "check_budget_status", "verify_audit_ledger"})

PROTOCOL_VERSION = "2025-06-18"

# The container builds a Qdrant handle and loads the FlashRank ONNX model on the
# first tool call. On a cold cache that is slow, and a timeout here would report
# a hardware fact as a protocol failure.
HANDSHAKE_TIMEOUT_S = 90.0
CALL_TIMEOUT_S = 180.0


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Report:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        return ok

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def render(self) -> str:
        width = max((len(name) for name, _, _ in self.checks), default=0)
        lines = []
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name.ljust(width)}  {detail}".rstrip())
        lines.append("")
        lines.append("OK - the server answers over stdio" if self.ok else "FAILED")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": name, "ok": ok, "detail": detail} for name, ok, detail in self.checks
            ],
        }


# --------------------------------------------------------------------------
# Config inspection
# --------------------------------------------------------------------------

_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)


def load_jsonc(path: Path) -> dict[str, Any]:
    """Parse a config that may carry `//` comments.

    VS Code reads `.vscode/mcp.json` as JSONC, and the comments in it explain a
    launch command that is easy to get subtly wrong. Stripping whole comment
    lines is enough here and avoids pulling in a JSONC dependency for one file;
    it deliberately does not touch `//` inside a string, because it only removes
    lines that are a comment start to finish.
    """
    text = _LINE_COMMENT.sub("", path.read_text(encoding="utf-8"))
    parsed: dict[str, Any] = json.loads(text)
    return parsed


def check_vscode_config(report: Report) -> list[str] | None:
    """Validate the editor launch command. Returns argv if it is usable."""
    if not report.add("vscode config present", VSCODE_CONFIG.exists(), str(VSCODE_CONFIG)):
        return None
    try:
        config = load_jsonc(VSCODE_CONFIG)
    except json.JSONDecodeError as exc:
        report.add("vscode config parses", False, str(exc))
        return None
    report.add("vscode config parses", True, "JSONC")

    server = config.get("servers", {}).get(SERVER_NAME)
    if not report.add(
        f"vscode registers '{SERVER_NAME}'",
        isinstance(server, dict),
        ", ".join(config.get("servers", {})) or "no servers",
    ):
        return None
    assert isinstance(server, dict)

    report.add("vscode transport is stdio", server.get("type") == "stdio", str(server.get("type")))

    command = str(server.get("command", ""))
    resolved = Path(command.replace("${workspaceFolder}", str(REPO_ROOT)))
    report.add("vscode interpreter exists", resolved.exists(), str(resolved))

    args = [str(a) for a in server.get("args", [])]
    module = args[1] if len(args) >= 2 and args[0] == "-m" else ""
    report.add(
        "vscode module path is importable from any cwd",
        module == SERVER_MODULE,
        f"{module or '(none)'} - `src.` prefixed paths only resolve from the repo root",
    )

    if not resolved.exists() or module != SERVER_MODULE:
        return None
    return [str(resolved), *args]


def check_antigravity_config(report: Report) -> None:
    """The Antigravity stub is unverified by design; its CONTENT still has to be right."""
    if not ANTIGRAVITY_CONFIG.exists():
        report.add("antigravity config present", False, str(ANTIGRAVITY_CONFIG))
        return
    try:
        config = json.loads(ANTIGRAVITY_CONFIG.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add("antigravity config parses", False, str(exc))
        return
    server = config.get("mcpServers", {}).get(SERVER_NAME, {})
    declared = {tool.get("name") for tool in server.get("tools", [])}
    report.add(
        "antigravity tool names match the server",
        declared == EXPECTED_TOOLS,
        ", ".join(sorted(declared)) or "none declared",
    )


# --------------------------------------------------------------------------
# JSON-RPC over stdio
# --------------------------------------------------------------------------


class StdioClient:
    """A minimal MCP client. Newline-delimited JSON-RPC, one frame per line."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._next_id = 0

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload).encode() + b"\n")
        await self._proc.stdin.drain()

    async def _read(self, timeout_s: float) -> dict[str, Any]:
        """Next JSON object on stdout, skipping anything that is not one.

        A dependency that prints a banner to stdout corrupts this stream. The
        skip keeps the check running so the report says which tool failed
        rather than dying on the first stray line - but a stray line IS the bug
        when it happens, so the line is reported.
        """
        assert self._proc.stdout is not None
        while True:
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout_s)
            if not line:
                raise ConnectionError("server closed stdout before replying")
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                parsed: dict[str, Any] = json.loads(text)
            except json.JSONDecodeError:
                print(f"  note: non-JSON on stdout: {text[:120]}", file=sys.stderr)
                continue
            return parsed

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float
    ) -> dict[str, Any]:
        self._next_id += 1
        await self._send(
            {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        )
        while True:
            message = await self._read(timeout_s)
            if message.get("id") == self._next_id:
                return message

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})


async def drain(stream: asyncio.StreamReader | None) -> None:
    """Consume stderr so a chatty dependency cannot fill the pipe and deadlock."""
    if stream is None:
        return
    while await stream.readline():
        pass


async def handshake(client: StdioClient, report: Report) -> bool:
    response = await client.request(
        "initialize",
        {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "verify_mcp_clients", "version": "1.0"},
        },
        timeout_s=HANDSHAKE_TIMEOUT_S,
    )
    if "error" in response:
        return report.add("initialize", False, json.dumps(response["error"])[:160])
    info = response.get("result", {}).get("serverInfo", {})
    name = info.get("name")
    report.add("initialize", True, f"protocol {response['result'].get('protocolVersion')}")
    ok = report.add(f"server identifies as '{SERVER_NAME}'", name == SERVER_NAME, str(name))
    await client.notify("notifications/initialized")
    return ok


async def check_tools(client: StdioClient, report: Report) -> None:
    response = await client.request("tools/list", timeout_s=HANDSHAKE_TIMEOUT_S)
    if "error" in response:
        report.add("tools/list", False, json.dumps(response["error"])[:160])
        return
    tools = response.get("result", {}).get("tools", [])
    names = {tool.get("name") for tool in tools}
    report.add(
        "tools/list returns the 3 frozen tools",
        names == EXPECTED_TOOLS,
        ", ".join(sorted(map(str, names))),
    )
    report.add(
        "every tool advertises an input schema",
        all(isinstance(tool.get("inputSchema"), dict) for tool in tools),
        f"{len(tools)} tools",
    )


async def check_empty_query(client: StdioClient, report: Report) -> None:
    """An empty query must not be able to take the server down."""
    try:
        response = await client.request(
            "tools/call",
            {"name": "local_rag_search", "arguments": {"query": ""}},
            timeout_s=CALL_TIMEOUT_S,
        )
    except (TimeoutError, ConnectionError) as exc:
        report.add("local_rag_search('') answers", False, f"{type(exc).__name__}: {exc}")
        return

    if "error" in response:
        # A protocol-level error is still a well-formed answer from a live
        # server, which is what is being checked. Say which, so a reader is not
        # left thinking an empty corpus and a crashed tool look the same.
        report.add(
            "local_rag_search('') answers",
            True,
            f"JSON-RPC error (server alive): {str(response['error'].get('message'))[:80]}",
        )
    else:
        result = response.get("result", {})
        detail = "tool error" if result.get("isError") else "ok"
        report.add("local_rag_search('') answers", True, detail)

    # The real assertion: it is still there afterwards.
    try:
        followup = await client.request("tools/list", timeout_s=HANDSHAKE_TIMEOUT_S)
        alive = "result" in followup
    except (TimeoutError, ConnectionError):
        alive = False
    report.add("server survives the empty query", alive, "tools/list answered again")


async def run_stdio_checks(argv: list[str], report: Report) -> None:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(REPO_ROOT.parent),  # deliberately NOT the repo root
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(drain(proc.stderr))
    client = StdioClient(proc)
    try:
        report.add("launches from a foreign working directory", True, str(REPO_ROOT.parent))
        if await handshake(client, report):
            await check_tools(client, report)
            await check_empty_query(client, report)
    except (TimeoutError, ConnectionError) as exc:
        report.add("stdio session", False, f"{type(exc).__name__}: {exc}")
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except TimeoutError:
                proc.kill()
        stderr_task.cancel()


async def main_async(as_json: bool) -> int:
    report = Report()
    argv = check_vscode_config(report)
    check_antigravity_config(report)
    if argv is not None:
        await run_stdio_checks(argv, report)
    else:
        report.add("stdio session", False, "skipped: the launch command is not usable")

    if as_json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(report.render())
    return 0 if report.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    return asyncio.run(main_async(args.json))


if __name__ == "__main__":
    raise SystemExit(main())
