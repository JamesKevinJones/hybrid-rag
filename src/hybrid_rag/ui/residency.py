"""What Ollama is currently holding in memory.

Shared by the terminal UI and the web server so there is one parser for
`ollama ps` rather than two that drift. Presentation only: it reads, it never
loads or evicts anything.

The important behaviour here is the three-state return. "Nothing is resident"
and "I could not find out" are different facts, and a status panel that
collapses them into an empty list starts asserting something it does not know.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

__all__ = ["ResidentModel", "probe_residency", "residency_for"]


@dataclass(frozen=True, slots=True)
class ResidentModel:
    """One row of `ollama ps`."""

    name: str
    size: str
    processor: str


def probe_residency(
    binary: str = "ollama", *, timeout_s: float = 5.0
) -> list[ResidentModel] | None:
    """Resident models, or `None` when the question could not be asked.

    `None` covers: no binary on PATH, no daemon, a timeout, a non-zero exit, or
    a table whose header we do not recognise. An empty list means Ollama
    answered and is holding nothing.
    """
    try:
        proc = subprocess.run(  # fixed argv; `binary` comes from settings, not a prompt
            [binary, "ps"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines or not lines[0].upper().startswith("NAME"):
        return None

    models: list[ResidentModel] = []
    for line in lines[1:]:
        # NAME  ID  SIZE  PROCESSOR  CONTEXT  UNTIL - whitespace-aligned, and
        # BOTH SIZE and PROCESSOR are two tokens ("2.6 GB", "100% GPU"). Taking
        # one token for the processor silently reports "100%" and loses the
        # part that says whether it is on the GPU at all, which is the only
        # part anyone reads this row for.
        parts = line.split()
        if len(parts) < 6:
            continue
        models.append(
            ResidentModel(
                name=parts[0],
                size=f"{parts[2]} {parts[3]}",
                processor=f"{parts[4]} {parts[5]}",
            )
        )
    return models


def residency_for(models: list[ResidentModel] | None, wanted: str) -> tuple[str, bool]:
    """(human status, is_resident) for one model name.

    Tag-insensitive on the second pass: `llama3.2:3b-instruct-q4_K_M` should
    still match a row reading `llama3.2:3b`, because the tag is a packaging
    detail and the question being asked is "are the weights loaded".
    """
    if models is None:
        return "residency not measured (ollama ps unavailable)", False
    for model in models:
        if model.name == wanted or model.name.split(":")[0] == wanted.split(":")[0]:
            return f"resident, {model.processor}, {model.size}", True
    return "not resident (loads on first query)", False
