"""The editor-facing MCP configs, and the harness that checks them.

`test_mcp_server.py` drives the tools in-process. It cannot catch a broken
launch command, because an in-process client never spawns an interpreter. These
tests cover the other half: that the checked-in configs name a command that
works, and - more importantly - that the harness which says so is capable of
saying no.

A verifier that cannot fail is the same defect as the CI job in decision 26: a
green check that inspected nothing.
"""

from __future__ import annotations

import json
import sys

import pytest

from scripts import verify_mcp_clients as vmc

WORKING = {
    "servers": {
        "hybrid-rag": {
            "type": "stdio",
            "command": "${workspaceFolder}/.venv/Scripts/python.exe",
            "args": ["-m", "hybrid_rag.mcp_servers.server"],
            "cwd": "${workspaceFolder}",
        }
    }
}


def _check(tmp_path, config: dict, name: str = "mcp.json"):
    path = tmp_path / name
    path.write_text(json.dumps(config), encoding="utf-8")
    report = vmc.Report()
    original = vmc.VSCODE_CONFIG
    vmc.VSCODE_CONFIG = path
    try:
        argv = vmc.check_vscode_config(report)
    finally:
        vmc.VSCODE_CONFIG = original
    return report, argv


def _failures(report: vmc.Report) -> list[str]:
    return [name for name, ok, _ in report.checks if not ok]


# --------------------------------------------------------------------------
# The real, checked-in configs
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the checked-in .vscode/mcp.json targets the Windows venv layout by design",
)
def test_the_checked_in_vscode_config_is_usable() -> None:
    report = vmc.Report()
    argv = vmc.check_vscode_config(report)
    assert report.ok, _failures(report)
    assert argv is not None
    assert argv[1:] == ["-m", vmc.SERVER_MODULE]


def test_the_checked_in_vscode_config_parses_as_jsonc() -> None:
    """It carries `//` comments explaining the launch command. Those must not break it."""
    raw = vmc.VSCODE_CONFIG.read_text(encoding="utf-8")
    assert "//" in raw, "the comments are load-bearing documentation"
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    assert "hybrid-rag" in vmc.load_jsonc(vmc.VSCODE_CONFIG)["servers"]


def test_antigravity_declares_the_same_three_tools() -> None:
    report = vmc.Report()
    vmc.check_antigravity_config(report)
    assert report.ok, _failures(report)


def test_the_frozen_tool_names_match_the_server() -> None:
    """Decision 28 froze these. A drift here is a stale client, not a rename to apply."""
    from hybrid_rag.mcp_servers import server as srv

    assert srv.mcp.name == vmc.SERVER_NAME
    frozen = {"local_rag_search", "check_budget_status", "verify_audit_ledger"}
    assert frozen == vmc.EXPECTED_TOOLS


# --------------------------------------------------------------------------
# The harness has to be able to fail
# --------------------------------------------------------------------------


def test_src_prefixed_module_path_is_rejected(tmp_path) -> None:
    """`src.hybrid_rag...` resolves only from the repo root, then imports twice."""
    broken = json.loads(json.dumps(WORKING))
    broken["servers"]["hybrid-rag"]["args"] = ["-m", "src.hybrid_rag.mcp_servers.server"]

    report, argv = _check(tmp_path, broken)

    assert not report.ok
    assert argv is None, "an unusable command must not be handed to the stdio session"
    assert "vscode module path is importable from any cwd" in _failures(report)


def test_missing_interpreter_is_rejected(tmp_path) -> None:
    broken = json.loads(json.dumps(WORKING))
    # Missing on every OS: `.venv/bin/python` exists on a POSIX host.
    broken["servers"]["hybrid-rag"]["command"] = "${workspaceFolder}/.venv/missing/python"

    report, argv = _check(tmp_path, broken)

    assert not report.ok
    assert argv is None
    assert "vscode interpreter exists" in _failures(report)


def test_renamed_server_is_rejected(tmp_path) -> None:
    broken = {"servers": {"LocalGovEngine": WORKING["servers"]["hybrid-rag"]}}

    report, argv = _check(tmp_path, broken)

    assert not report.ok
    assert argv is None
    assert "vscode registers 'hybrid-rag'" in _failures(report)


def test_http_transport_is_flagged(tmp_path) -> None:
    broken = json.loads(json.dumps(WORKING))
    broken["servers"]["hybrid-rag"]["type"] = "http"

    report, _ = _check(tmp_path, broken)

    assert "vscode transport is stdio" in _failures(report)


def test_absent_config_is_reported_not_skipped(tmp_path) -> None:
    report = vmc.Report()
    original = vmc.VSCODE_CONFIG
    vmc.VSCODE_CONFIG = tmp_path / "does-not-exist.json"
    try:
        argv = vmc.check_vscode_config(report)
    finally:
        vmc.VSCODE_CONFIG = original

    assert argv is None
    assert not report.ok


def test_unparseable_config_is_reported(tmp_path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{ not json at all", encoding="utf-8")
    report = vmc.Report()
    original = vmc.VSCODE_CONFIG
    vmc.VSCODE_CONFIG = path
    try:
        argv = vmc.check_vscode_config(report)
    finally:
        vmc.VSCODE_CONFIG = original

    assert argv is None
    assert "vscode config parses" in _failures(report)


def test_antigravity_tool_drift_is_caught(tmp_path) -> None:
    drifted = {
        "mcpServers": {
            "hybrid-rag": {"tools": [{"name": "search_local_knowledge"}]},
        }
    }
    path = tmp_path / "antigravity.json"
    path.write_text(json.dumps(drifted), encoding="utf-8")
    report = vmc.Report()
    original = vmc.ANTIGRAVITY_CONFIG
    vmc.ANTIGRAVITY_CONFIG = path
    try:
        vmc.check_antigravity_config(report)
    finally:
        vmc.ANTIGRAVITY_CONFIG = original

    assert not report.ok
    assert "antigravity tool names match the server" in _failures(report)


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def test_report_renders_a_verdict_either_way() -> None:
    report = vmc.Report()
    report.add("a", True, "fine")
    assert "OK" in report.render()

    report.add("b", False, "broken")
    rendered = report.render()
    assert "FAILED" in rendered
    assert "[FAIL] b" in rendered
    assert report.as_dict()["ok"] is False
