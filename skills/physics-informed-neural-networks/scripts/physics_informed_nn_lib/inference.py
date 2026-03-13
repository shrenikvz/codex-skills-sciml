"""Top-level runtime orchestration for the PINNs skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import resolve_runtime_recommendations
from .environment import probe_environment, recommend_framework


class InferenceError(RuntimeError):
    """Runtime failure."""


def run_training(cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
    environment = probe_environment()
    cfg = resolve_runtime_recommendations(cfg)
    framework_choice = recommend_framework(environment, requested=cfg.get("model", {}).get("framework"))
    if framework_choice["framework"] is None:
        raise InferenceError("No supported autodiff framework is available. Install torch, jax, or tensorflow.")
    if framework_choice["framework"] != "torch":
        raise InferenceError(
            "This repository version executes the PINN runtime with the torch backend. "
            "Set `model.framework` to `torch` and install PyTorch to run the generated project."
        )
    from .training import run_torch_training

    return run_torch_training(cfg, workdir=Path(workdir).resolve(), environment=environment)
