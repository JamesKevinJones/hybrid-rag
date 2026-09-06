"""Interactive terminal surface.

Presentation only. Nothing in this package makes a routing decision, spends a
token or writes to the ledger - it renders what the orchestrator already
decided. If a display needs a number the pipeline does not produce, the fix is
to make the pipeline produce it, not to compute a plausible one here.
"""

from hybrid_rag.ui.tui import run_tui

__all__ = ["run_tui"]
