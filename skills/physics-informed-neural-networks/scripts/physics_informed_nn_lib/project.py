"""Project scaffolding for the PINNs skill."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .analysis import inspect_problem_inputs
from .config import clone_default_config, save_config
from .io_utils import relative_or_absolute, stage_file, write_json


class ProjectError(RuntimeError):
    """Project creation failure."""


def _write_run_py(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_DIR = SCRIPT_DIR / "skill_runtime"
sys.path.insert(0, str(LIB_DIR.resolve()))

from physics_informed_nn_lib.config import load_config
from physics_informed_nn_lib.inference import run_training


def main() -> int:
    config_path = SCRIPT_DIR / "config.json"
    cfg = load_config(config_path)
    result = run_training(cfg, workdir=SCRIPT_DIR)
    print(json.dumps({"ok": True, "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def _stage_runtime_library(root: Path) -> None:
    source = Path(__file__).resolve().parent
    destination = root / "skill_runtime" / "physics_informed_nn_lib"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Physics-Informed Neural Networks Runbook

1. Review `project_summary.json` for blocking or pending questions.
2. Confirm the problem spec, observation column mapping, architecture, sampling, and loss weighting in `config.json`.
3. Run `python3 run.py`.
4. Inspect `results/diagnostics.json`, `results/residual_diagnostics.json`, and `results/evaluation_summary.json`.
5. If training stagnates, revisit coordinate scaling, collocation density, loss balancing, and architecture choice.
""",
        encoding="utf-8",
    )


def _write_generated_problem(path: Path, spec: dict[str, Any]) -> None:
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")


def create_project(
    project_dir: str,
    problem_path: str | None = None,
    observed_path: str | None = None,
    physics_text: str | None = None,
    callable_name: str | None = None,
    request_text: str | None = None,
    problem_type: str | None = None,
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    context_columns: list[str] | None = None,
    architecture: str | None = None,
    framework: str | None = None,
    sampling_strategy: str | None = None,
    loss_weighting: str | None = None,
    optimizer: str | None = None,
    plots: list[str] | None = None,
    hidden_layers: int | None = None,
    hidden_units: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ProjectError(f"Project directory already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    problem_dir = root / "problem"
    results_dir = root / "results"
    artifacts_dir = root / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    problem_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    inspection = inspect_problem_inputs(
        problem_path=problem_path,
        observed_path=observed_path,
        physics_text=physics_text,
        callable_name=callable_name,
        request_text=request_text,
        problem_type=problem_type,
        input_columns=input_columns,
        output_columns=output_columns,
        context_columns=context_columns,
        architecture=architecture,
        framework=framework,
        sampling_strategy=sampling_strategy,
        loss_weighting=loss_weighting,
        optimizer=optimizer,
        plots=plots,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    cfg = clone_default_config()
    cfg["objective"]["text"] = request_text or ""
    cfg["objective"]["problem_type"] = inspection["problem_type"]["problem_type"]
    cfg["objective"]["input_columns"] = inspection["observation_mapping"]["input_columns"]
    cfg["objective"]["output_columns"] = inspection["observation_mapping"]["output_columns"]
    cfg["objective"]["context_columns"] = inspection["observation_mapping"]["context_columns"]
    cfg["objective"]["unknown_parameters"] = list(inspection["problem_spec"].get("unknown_parameters", []))
    cfg["problem"] = inspection["problem_spec"]

    if observed_path:
        staged_observed = stage_file(observed_path, data_dir)
        cfg["objective"]["observation_path"] = relative_or_absolute(Path(staged_observed), root) if staged_observed else None

    if physics_text and not problem_path:
        generated_problem = problem_dir / "problem.json"
        _write_generated_problem(generated_problem, inspection["problem_spec"])
        cfg["problem"]["path"] = relative_or_absolute(generated_problem, root)
        cfg["problem"]["format"] = "json"
    elif problem_path:
        staged_problem = stage_file(problem_path, problem_dir)
        if not staged_problem:
            raise ProjectError("Problem path staging failed.")
        cfg["problem"]["path"] = relative_or_absolute(Path(staged_problem), root)
        cfg["problem"]["format"] = Path(problem_path).suffix.lstrip(".").lower()
        cfg["problem"]["callable"] = callable_name
    else:
        raise ProjectError("A problem specification is required.")

    hyper = inspection["default_hyperparameters"]
    cfg["model"]["framework"] = inspection["framework_recommendation"]["framework"] or "auto"
    cfg["model"]["architecture"] = inspection["architecture_recommendation"]["architecture"]
    cfg["model"]["hidden_layers"] = hyper["model"]["hidden_layers"]
    cfg["model"]["hidden_units"] = hyper["model"]["hidden_units"]
    cfg["model"]["adaptive_activation"] = bool(inspection["stabilization_recommendation"]["adaptive_activation"])
    if cfg["model"]["architecture"] == "fourier":
        cfg["model"]["fourier_features"]["enabled"] = True
    if cfg["model"]["architecture"] == "multiscale":
        cfg["model"]["multiscale"]["enabled"] = True
    if cfg["model"]["architecture"] == "transformer_operator":
        cfg["model"]["transformer"]["enabled"] = True
        cfg["model"]["transformer"]["context_dim"] = len(cfg["objective"]["context_columns"])

    cfg["sampling"]["strategy"] = inspection["sampling_recommendation"]["strategy"]
    cfg["sampling"]["interior_points"] = hyper["sampling"]["interior_points"]
    cfg["sampling"]["boundary_points"] = hyper["sampling"]["boundary_points"]
    cfg["sampling"]["initial_points"] = hyper["sampling"]["initial_points"]
    cfg["sampling"]["validation_points"] = hyper["sampling"]["validation_points"]
    if cfg["sampling"]["strategy"] == "residual_adaptive":
        cfg["sampling"]["adaptive"]["enabled"] = True

    cfg["loss"]["weighting_strategy"] = inspection["loss_weighting_recommendation"]["strategy"]
    cfg["training"]["optimizer"] = inspection["optimizer_recommendation"]["optimizer"]
    cfg["training"]["learning_rate"] = hyper["training"]["learning_rate"]
    cfg["training"]["epochs"] = hyper["training"]["epochs"]
    cfg["training"]["adam_epochs"] = hyper["training"]["adam_epochs"]
    cfg["training"]["lbfgs_steps"] = hyper["training"]["lbfgs_steps"]
    cfg["training"]["gradient_clip_norm"] = 1.0 if inspection["stabilization_recommendation"]["gradient_clipping"] else None
    cfg["training"]["domain_decomposition"]["enabled"] = bool(inspection["stabilization_recommendation"]["domain_decomposition"])
    cfg["training"]["domain_decomposition"]["num_subdomains"] = 2 if inspection["stabilization_recommendation"]["domain_decomposition"] else 1

    cfg["stabilization"]["coordinate_scaling"] = inspection["stabilization_recommendation"]["coordinate_scaling"]
    cfg["stabilization"]["residual_normalization"] = inspection["stabilization_recommendation"]["residual_normalization"]
    cfg["stabilization"]["gradient_clipping"] = inspection["stabilization_recommendation"]["gradient_clipping"]
    cfg["stabilization"]["fourier_features"] = inspection["stabilization_recommendation"]["fourier_features"]
    cfg["stabilization"]["adaptive_activation"] = inspection["stabilization_recommendation"]["adaptive_activation"]
    cfg["stabilization"]["domain_decomposition"] = inspection["stabilization_recommendation"]["domain_decomposition"]

    cfg["visualization"]["enabled"] = bool(inspection["visualization_recommendation"]["enabled"])
    cfg["visualization"]["plots"] = list(inspection["visualization_recommendation"]["plots"])

    config_path = root / "config.json"
    save_config(config_path, cfg)
    _write_run_py(root / "run.py")
    _stage_runtime_library(root)
    _write_runbook(root / "AGENT_RUNBOOK.md")

    summary = {
        "project_dir": str(root),
        "config_path": str(config_path),
        "pending_questions": inspection["pending_questions"],
        "blocking_questions": inspection["blocking_questions"],
        "environment": inspection["environment"],
        "problem_summary": inspection["problem_summary"],
        "problem_type": inspection["problem_type"],
        "observation_mapping": inspection["observation_mapping"],
        "architecture_recommendation": inspection["architecture_recommendation"],
        "framework_recommendation": inspection["framework_recommendation"],
        "sampling_recommendation": inspection["sampling_recommendation"],
        "loss_weighting_recommendation": inspection["loss_weighting_recommendation"],
        "optimizer_recommendation": inspection["optimizer_recommendation"],
        "stabilization_recommendation": inspection["stabilization_recommendation"],
        "default_hyperparameters": inspection["default_hyperparameters"],
    }
    write_json(root / "project_summary.json", summary)
    return summary

