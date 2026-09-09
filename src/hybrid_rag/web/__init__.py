"""The local web surface.

Presentation and transport only. Every answer goes through the same
`AgentOrchestrator.execute()` as the CLI and the TUI, so governance, routing,
budget and egress screening apply unchanged. Rule 14 governs this package
exactly as it governs `ui/`: nothing here decides, spends or logs.
"""

from hybrid_rag.web.app import build_app

__all__ = ["build_app"]
