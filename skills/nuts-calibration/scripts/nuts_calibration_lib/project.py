"""Project scaffolding for the NUTS calibration skill."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .analysis import build_equation_wrapper_source, inspect_inputs
from .config import clone_default_config, resolve_runtime_hyperparameters, save_config
from .io_utils import relative_or_absolute, stage_file, write_json
from .priors import load_prior_file, parse_parameter_bounds, parse_prior_overrides


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

from nuts_calibration_lib.config import load_config
from nuts_calibration_lib.inference import run_calibration


def main() -> int:
    config_path = SCRIPT_DIR / "config.json"
    cfg = load_config(config_path)
    result = run_calibration(cfg, workdir=SCRIPT_DIR)
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
    destination = root / "skill_runtime" / "nuts_calibration_lib"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# NUTS Calibration Runbook

1. Review `project_summary.json` for pending clarification questions about outputs, scaling, likelihood, or differentiability.
2. Confirm priors, likelihood, transforms, backend, and sampling hyperparameters in `config.json`.
3. Run `python3 run.py`.
4. Inspect `results/posterior_summary.json`, `results/sampling_diagnostics.json`, and `results/run_summary.json`.
5. If divergences occur, re-scale the data, tighten priors, increase `algorithm.warmup.target_acceptance`, or improve parameter transforms.
""",
        encoding="utf-8",
    )


def _write_command_wrapper(path: Path, runtime_hint: str, model_filename: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
PARAMS_JSON="${{1:?params json required}}"
OUTPUT_JSON="${{2:?output json required}}"
{runtime_hint} "$(dirname "$0")/{model_filename}" "$PARAMS_JSON" "$OUTPUT_JSON"
""",
        encoding="utf-8",
    )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def inspect_model_inputs(
    model_path: str | None,
    observed_path: str | None,
    command_template: str | None = None,
    equation_text: str | None = None,
    callable_name: str | None = None,
    request_text: str | None = None,
    observed_output_names: list[str] | None = None,
    user_parameter_names: list[str] | None = None,
    prior_file: str | None = None,
    prior_overrides: list[str] | None = None,
    parameter_bounds: list[str] | None = None,
    scaling_mode: str | None = None,
    likelihood_name: str | None = None,
    backend_name: str | None = None,
    plots: list[str] | None = None,
) -> dict[str, Any]:
    explicit_priors = load_prior_file(prior_file)
    explicit_priors.update(parse_prior_overrides(prior_overrides))
    bounds = parse_parameter_bounds(parameter_bounds)
    return inspect_inputs(
        model_path=model_path,
        observed_path=observed_path,
        command_template=command_template,
        equation_text=equation_text,
        callable_name=callable_name,
        request_text=request_text,
        observed_output_names=observed_output_names,
        user_parameter_names=user_parameter_names,
        explicit_priors=explicit_priors,
        parameter_bounds=bounds,
        scaling_mode=scaling_mode,
        likelihood_name=likelihood_name,
        backend_name=backend_name,
        plots=plots,
    )


def create_project(
    project_dir: str,
    model_path: str | None,
    observed_path: str,
    command_template: str | None = None,
    equation_text: str | None = None,
    callable_name: str | None = None,
    request_text: str | None = None,
    observed_output_names: list[str] | None = None,
    parameter_names: list[str] | None = None,
    prior_file: str | None = None,
    prior_overrides: list[str] | None = None,
    parameter_bounds: list[str] | None = None,
    scaling_mode: str | None = None,
    likelihood_name: str | None = None,
    backend_name: str | None = None,
    plots: list[str] | None = None,
    warmup_steps: int | None = None,
    num_samples: int | None = None,
    num_chains: int | None = None,
    target_acceptance: float | None = None,
    max_tree_depth: int | None = None,
    step_size: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(project_dir).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ProjectError(f"Project directory already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    model_dir = root / "model"
    results_dir = root / "results"
    artifacts_dir = root / "artifacts"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    inspection = inspect_model_inputs(
        model_path=model_path,
        observed_path=observed_path,
        command_template=command_template,
        equation_text=equation_text,
        callable_name=callable_name,
        request_text=request_text,
        observed_output_names=observed_output_names,
        user_parameter_names=parameter_names,
        prior_file=prior_file,
        prior_overrides=prior_overrides,
        parameter_bounds=parameter_bounds,
        scaling_mode=scaling_mode,
        likelihood_name=likelihood_name,
        backend_name=backend_name,
        plots=plots,
    )

    cfg = clone_default_config()
    cfg["objective"]["text"] = request_text or ""
    staged_observed = stage_file(observed_path, data_dir)
    if not staged_observed:
        raise ProjectError("Observed data is required.")
    cfg["objective"]["observed_path"] = relative_or_absolute(Path(staged_observed), root)
    cfg["objective"]["noise_structure"] = (
        inspection["likelihood_recommendation"]["spec"]["name"] if inspection.get("likelihood_recommendation") else "auto"
    )
    output_map = inspection["output_mapping"]
    cfg["objective"]["observed_output_names"] = output_map.get("selected", [])
    cfg["model"]["observed_output_names"] = output_map.get("selected", [])

    model_analysis = dict(inspection["model_analysis"])
    if equation_text:
        generated_equation_wrapper = model_dir / "equation_model.py"
        generated_equation_wrapper.write_text(
            build_equation_wrapper_source(equation_text, model_analysis.get("parameter_names", [])),
            encoding="utf-8",
        )
        cfg["model"]["path"] = relative_or_absolute(generated_equation_wrapper, root)
        cfg["model"]["adapter"] = "python_callable"
        cfg["model"]["callable"] = "simulate"
        cfg["model"]["call_style"] = "kwargs"
    elif model_path:
        staged_model = stage_file(model_path, model_dir)
        if not staged_model:
            raise ProjectError("Model path staging failed.")
        cfg["model"]["path"] = relative_or_absolute(Path(staged_model), root)
        cfg["model"]["adapter"] = model_analysis["adapter"]
        if model_analysis["adapter"] == "command":
            wrapper_path = model_dir / "run_model.sh"
            _write_command_wrapper(
                wrapper_path,
                model_analysis.get("runtime_hint", "bash"),
                Path(staged_model).name,
            )
            cfg["model"]["command_template"] = "bash {model_path} {params_json} {output_json}"
            cfg["model"]["path"] = relative_or_absolute(wrapper_path, root)
        else:
            cfg["model"]["callable"] = model_analysis.get("callable") or callable_name or "simulate"
            cfg["model"]["call_style"] = model_analysis.get("call_style") or "kwargs"
    elif command_template:
        cfg["model"]["adapter"] = "command"
        cfg["model"]["command_template"] = command_template
        cfg["model"]["path"] = ""
    else:
        raise ProjectError("A model source is required.")

    cfg["model"]["parameter_names"] = model_analysis.get("parameter_names", [])
    cfg["model"]["parameter_defaults"] = model_analysis.get("parameter_defaults", {})
    cfg["model"]["parameter_constraints"] = model_analysis.get("parameter_constraints", {})
    cfg["model"]["output_names"] = model_analysis.get("output_names", [])
    cfg["model"]["stochastic"] = model_analysis.get("stochastic")
    cfg["model"]["differentiable"] = inspection["differentiability_assessment"]["differentiable"]
    cfg["model"]["gradient_strategy"] = inspection["differentiability_assessment"]["gradient_strategy"]
    cfg["model"]["working_directory"] = "."

    cfg["priors"] = inspection["prior_report"]["priors"]
    cfg["likelihood"] = inspection["likelihood_recommendation"]["spec"] if inspection.get("likelihood_recommendation") else cfg["likelihood"]
    cfg["transformations"]["parameters"] = inspection["transformation_recommendation"]
    cfg["scaling"]["enabled"] = inspection["scaling_recommendation"]["enabled"]
    cfg["scaling"]["mode"] = inspection["scaling_recommendation"]["mode"]

    backend = inspection["backend_recommendation"]["backend"] or "blackjax"
    cfg["algorithm"]["backend"] = backend
    cfg["algorithm"]["name"] = "nuts"
    hyper = resolve_runtime_hyperparameters(
        cfg,
        observed_size=inspection["observed_analysis"]["size"] if inspection.get("observed_analysis") else 1,
        model_complexity=inspection["model_complexity"],
    )
    if warmup_steps is not None:
        hyper["warmup"]["num_steps"] = int(warmup_steps)
    if num_samples is not None:
        hyper["sampling"]["num_samples"] = int(num_samples)
    if num_chains is not None:
        hyper["sampling"]["num_chains"] = int(num_chains)
    if target_acceptance is not None:
        hyper["warmup"]["target_acceptance"] = float(target_acceptance)
    if max_tree_depth is not None:
        hyper["sampling"]["max_tree_depth"] = int(max_tree_depth)
    if step_size is not None:
        hyper["warmup"]["initial_step_size"] = float(step_size)
    cfg["algorithm"]["warmup"].update(hyper["warmup"])
    cfg["algorithm"]["sampling"].update(hyper["sampling"])
    cfg["posterior_predictive"].update(hyper["posterior_predictive"])

    cfg["visualization"]["enabled"] = bool(inspection["visualization_recommendation"]["enabled"])
    cfg["visualization"]["plots"] = list(inspection["visualization_recommendation"]["plots"])

    config_path = root / "config.json"
    save_config(config_path, cfg)
    _stage_runtime_library(root)
    _write_run_py(root / "run.py")
    _write_runbook(root / "AGENT_RUNBOOK.md")

    summary_payload = {
        "project_dir": str(root),
        "config_path": str(config_path),
        "pending_questions": inspection["pending_questions"],
        "environment": inspection["environment"],
        "model_analysis": model_analysis,
        "observed_analysis": inspection["observed_analysis"],
        "prior_report": inspection["prior_report"],
        "likelihood_recommendation": inspection["likelihood_recommendation"],
        "differentiability_assessment": inspection["differentiability_assessment"],
        "backend_recommendation": inspection["backend_recommendation"],
        "default_hyperparameters": {
            "warmup": cfg["algorithm"]["warmup"],
            "sampling": cfg["algorithm"]["sampling"],
        },
        "notes": [
            "NUTS requires gradients. If the model is not JAX-compatible, the runtime falls back to finite-difference gradients through a callback wrapper.",
            "External command models should verify model/run_model.sh matches the simulator CLI contract before sampling.",
        ],
    }
    write_json(root / "project_summary.json", summary_payload)
    write_json(results_dir / "prior_report.json", inspection["prior_report"])
    write_json(results_dir / "likelihood_recommendation.json", inspection["likelihood_recommendation"])
    write_json(results_dir / "model_analysis.json", model_analysis)

    return {
        "project_dir": str(root),
        "config_path": str(config_path),
        "run_script": str(root / "run.py"),
        "project_summary": str(root / "project_summary.json"),
        "pending_questions": inspection["pending_questions"],
        "backend": cfg["algorithm"]["backend"],
        "warmup": cfg["algorithm"]["warmup"],
        "sampling": cfg["algorithm"]["sampling"],
    }

