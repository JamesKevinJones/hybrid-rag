"""What this machine can actually run.

Answers one question: *if I point the generator at this model, will it fit?*

Everything here is either measured off the host or labelled as an estimate.
That distinction is the whole reason the module exists rather than a lookup
table of "recommended" models - decisions 15, 23 and 27 are all cases where a
plausible number in this repo turned out to be invented, and a capability check
built on guesses is worse than no check because it sounds authoritative.

Measured:
    GPU name, total and free VRAM      `nvidia-smi`
    installed models and their sizes   `ollama list`
    parameters, quantisation, ctx      `ollama show`

Estimated, and always labelled as such:
    KV cache footprint. Scaled from ONE measured datapoint on this host - a 3B
    Q4 at 4096 context costs about 0.4 GB - by context length and parameter
    count. That is a linear approximation of something that is not linear
    across architectures, so it is reported with its own caveat rather than as
    a fact.

Nothing here is called during a request. It is a diagnostic.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Assessment",
    "GpuInfo",
    "InstalledModel",
    "ModelDetails",
    "assess_model",
    "assess_model_with",
    "installed_models",
    "model_details",
    "probe_gpu",
]

# The Windows desktop compositor holds roughly this much of the card before
# anything else loads. Measured on this host and recorded in the README; it is
# why a 6 GB card behaves like a 5.2 GB one.
COMPOSITOR_RESERVE_MB = 800

# The one measured KV datapoint everything else is scaled from.
_KV_REF_PARAMS_B = 3.2
_KV_REF_CTX = 4096
_KV_REF_MB = 400


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    total_mb: int
    free_mb: int
    driver: str

    @property
    def usable_mb(self) -> int:
        """Total minus what the desktop compositor holds."""
        return max(0, self.total_mb - COMPOSITOR_RESERVE_MB)


@dataclass(frozen=True, slots=True)
class InstalledModel:
    name: str
    digest: str
    size_mb: int


@dataclass(frozen=True, slots=True)
class ModelDetails:
    name: str
    architecture: str | None = None
    parameters: str | None = None
    quantization: str | None = None
    context_length: int | None = None

    @property
    def parameters_b(self) -> float | None:
        """`3.2B` -> 3.2. None when Ollama did not report it."""
        if not self.parameters:
            return None
        match = re.match(r"([\d.]+)\s*([BM])", self.parameters.strip(), re.IGNORECASE)
        if not match:
            return None
        value = float(match.group(1))
        return value / 1000 if match.group(2).upper() == "M" else value


Verdict = Literal["fits", "tight", "exceeds", "unknown"]


@dataclass(frozen=True, slots=True)
class Assessment:
    """A verdict, and every number that produced it."""

    model: str
    verdict: Verdict
    headline: str
    detail: str
    weights_mb: int | None
    kv_estimate_mb: int | None
    companions_mb: int
    required_mb: int | None
    usable_mb: int | None
    gpu: GpuInfo | None
    installed: bool
    details: ModelDetails | None


def _run(argv: list[str], timeout_s: float = 10.0) -> str | None:
    """Return stdout, or None if the command could not be asked."""
    try:
        proc = subprocess.run(  # fixed argv, values come from settings not prompts
            argv, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def probe_gpu() -> GpuInfo | None:
    """The NVIDIA card, or None when there is no `nvidia-smi` to ask.

    None means "could not measure", which is not the same as "no GPU" - an AMD
    or Apple machine has one and this cannot see it. Callers must render the
    difference rather than reporting 0 MB free.
    """
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    first = next((line for line in out.splitlines() if line.strip()), "")
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        return GpuInfo(
            name=parts[0], total_mb=int(parts[1]), free_mb=int(parts[2]), driver=parts[3]
        )
    except ValueError:
        return None


_SIZE = re.compile(r"([\d.]+)\s*(GB|MB|KB|B)", re.IGNORECASE)


def _size_to_mb(text: str) -> int | None:
    match = _SIZE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    factor = {"B": 1 / 1_048_576, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}[unit]
    return round(value * factor)


def installed_models(binary: str = "ollama") -> list[InstalledModel] | None:
    """What `ollama list` reports, or None when it could not be asked."""
    out = _run([binary, "list"])
    if out is None:
        return None
    lines = [line for line in out.splitlines() if line.strip()]
    if not lines or not lines[0].upper().startswith("NAME"):
        return None

    models: list[InstalledModel] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        size = _size_to_mb(f"{parts[2]} {parts[3]}")
        if size is None:
            continue
        models.append(InstalledModel(name=parts[0], digest=parts[1], size_mb=size))
    return models


def model_details(model: str, binary: str = "ollama") -> ModelDetails | None:
    """Architecture, parameter count and quantisation, from `ollama show`."""
    out = _run([binary, "show", model], timeout_s=20.0)
    if out is None:
        return None

    def field(name: str) -> str | None:
        match = re.search(rf"^\s*{re.escape(name)}\s+(.+?)\s*$", out, re.MULTILINE)
        return match.group(1).strip() if match else None

    raw_ctx = field("context length")
    try:
        ctx = int(raw_ctx) if raw_ctx else None
    except ValueError:
        ctx = None

    return ModelDetails(
        name=model,
        architecture=field("architecture"),
        parameters=field("parameters"),
        quantization=field("quantization"),
        context_length=ctx,
    )


def estimate_kv_mb(parameters_b: float | None, num_ctx: int) -> int | None:
    """KV cache footprint, ESTIMATED. None when the parameter count is unknown.

    Scaled linearly from one measured point on this host: a 3.2B Q4 at 4096
    context costs about 400 MB. KV size really scales with layers, heads and
    head dimension rather than with parameter count, and those differ per
    architecture - so treat this as an order-of-magnitude figure, which is all
    it is presented as.
    """
    if parameters_b is None or parameters_b <= 0:
        return None
    return round(_KV_REF_MB * (parameters_b / _KV_REF_PARAMS_B) * (num_ctx / _KV_REF_CTX))


def assess_model(
    model: str,
    *,
    num_ctx: int,
    companions_mb: int = 0,
    binary: str = "ollama",
) -> Assessment:
    """Can this host serve `model` alongside whatever else must stay resident?

    Measures the GPU and the installed-model list itself. A caller that has
    already measured either this request - the web layer's `/api/models` and
    `/api/requirements` routes both do - should call `assess_model_with`
    directly instead of paying for `nvidia-smi` / `ollama list` twice.
    """
    return assess_model_with(
        model,
        num_ctx=num_ctx,
        companions_mb=companions_mb,
        binary=binary,
        gpu=probe_gpu(),
        listed=installed_models(binary),
    )


def assess_model_with(
    model: str,
    *,
    num_ctx: int,
    companions_mb: int,
    binary: str,
    gpu: GpuInfo | None,
    listed: list[InstalledModel] | None,
) -> Assessment:
    """`assess_model`, given an already-measured GPU and installed-model list.

    `companions_mb` is the guard and embedder footprint, which the caller knows
    from settings. They are counted because the interesting question is not
    "does the generator fit on an empty card" but "does the whole co-resident
    set fit", which is the constraint that actually shapes this project.
    """
    entry = next((m for m in listed or [] if m.name == model), None)
    details = model_details(model, binary) if entry else None

    weights = entry.size_mb if entry else None
    kv = estimate_kv_mb(details.parameters_b if details else None, num_ctx)
    required = (weights + kv + companions_mb) if (weights is not None and kv is not None) else None
    usable = gpu.usable_mb if gpu else None

    if entry is None:
        return Assessment(
            model=model,
            verdict="unknown",
            headline="Not installed",
            detail=(
                f"`{model}` is not in `ollama list`, so its size cannot be measured. "
                f"Pull it first with `ollama pull {model}`, then re-run this check."
                if listed is not None
                else "Could not ask Ollama what is installed. Is the daemon running?"
            ),
            weights_mb=None,
            kv_estimate_mb=None,
            companions_mb=companions_mb,
            required_mb=None,
            usable_mb=usable,
            gpu=gpu,
            installed=False,
            details=None,
        )

    if required is None or usable is None:
        return Assessment(
            model=model,
            verdict="unknown",
            headline="Cannot measure this machine",
            detail=(
                "The model is installed, but "
                + (
                    "no NVIDIA GPU could be queried - `nvidia-smi` is missing or this "
                    "is not an NVIDIA card. That is not the same as having no GPU, and "
                    "this check will not guess."
                    if gpu is None
                    else "Ollama did not report a parameter count, so the KV cache "
                    "cannot be estimated."
                )
            ),
            weights_mb=weights,
            kv_estimate_mb=kv,
            companions_mb=companions_mb,
            required_mb=None,
            usable_mb=usable,
            gpu=gpu,
            installed=True,
            details=details,
        )

    headroom = usable - required
    if required <= usable * 0.85:
        verdict: Verdict = "fits"
        headline = "Fits"
        detail = (
            f"About {required} MB needed against {usable} MB usable, leaving "
            f"{headroom} MB spare. Comfortable."
        )
    elif required <= usable:
        verdict = "tight"
        headline = "Tight"
        detail = (
            f"About {required} MB needed against {usable} MB usable - only {headroom} MB "
            "spare. It should load, but a longer context or a fourth model will push it "
            "into shared system memory, where it gets roughly an order of magnitude slower."
        )
    else:
        verdict = "exceeds"
        headline = "Does not fit"
        detail = (
            f"About {required} MB needed against {usable} MB usable, over by "
            f"{-headroom} MB. Ollama will still run it by spilling into system RAM, which "
            "on a single-channel machine is the slow path rather than a failure. Try a "
            "smaller quantisation, or a lower context length."
        )

    return Assessment(
        model=model,
        verdict=verdict,
        headline=headline,
        detail=detail,
        weights_mb=weights,
        kv_estimate_mb=kv,
        companions_mb=companions_mb,
        required_mb=required,
        usable_mb=usable,
        gpu=gpu,
        installed=True,
        details=details,
    )
