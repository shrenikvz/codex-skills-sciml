"""Project scaffolding for the ABC calibration skill."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .analysis import (
    assess_likelihood,
    build_equation_wrapper_source,
    build_prior_report,
    detect_visualization_defaults,
    infer_observed_outputs,
    inspect_model_file,
    inspect_observed_data,
    recommend_distance_metric,
    recommend_scaling_mode,
    recommend_summary_kind,
)
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
LIB_DIR = SCRIPT_DIR / \"skill_runtime\"
sys.path.insert(0, str(LIB_DIR.resolve()))

from abc_calibration_lib.config import load_config
from abc_calibration_lib.inference import run_calibration


def main() -> int:
    config_path = SCRIPT_DIR / \"config.json\"
    cfg = load_config(config_path)
    result = run_calibration(cfg, workdir=SCRIPT_DIR)
    print(json.dumps({\"ok\": True, \"result\": result}, indent=2))
    return 0


if __name__ == \"__main__\":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def _stage_runtime_library(root: Path) -> None:
    source = Path(__file__).resolve().parent
    destination = root / "skill_runtime" / "abc_calibration_lib"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)



def _write_runbook(path: Path) -> None:
    path.write_text(
        """# ABC Calibration Runbook

1. Review `config.json` and resolve any pending clarification questions recorded in `project_summary.json`.
2. Adjust priors, observed outputs, hyperparameters, and requested plots as needed.
3. Run `python3 run.py` from this folder.
4. Inspect `results/run_summary.json`, `results/posterior_summary.json`, and optional figures.
5. If the simulator is slow, increase `compute.max_workers` or reduce pilot/main budgets before rerunning.
""",
        encoding="utf-8",
    )



def _write_command_wrapper(path: Path, runtime_hint: str, model_filename: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
PARAMS_JSON=\"${{1:?params json required}}\"
OUTPUT_JSON=\"${{2:?output json required}}\"
{runtime_hint} \"$(dirname \"$0\")/{model_filename}\" \"$PARAMS_JSON\" \"$OUTPUT_JSON\"
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
    distance_metric: str | None = None,
    scaling_mode: str | None = None,
    summary_kind: str | None = None,
    likelihood_hint: str | None = None,
    plots: list[str] | None = None,
) -> dict[str, Any]:
    explicit_priors = load_prior_file(prior_file)
    explicit_priors.update(parse_prior_overrides(prior_overrides))
    bounds = parse_parameter_bounds(parameter_bounds)
    model_analysis = inspect_model_file(
        model_path,
        command_template=command_template,
        equation_text=equation_text,
        callable_name=callable_name,
        user_parameter_names=user_parameter_names,
        user_priors=explicit_priors,
    )
    prior_report = build_prior_report(
        model_analysis.get("parameter_names", []),
        model_analysis.get("parameter_defaults", {}),
        explicit_priors=explicit_priors,
        parameter_bounds=bounds,
    )
    observed_analysis = inspect_observed_data(observed_path) if observed_path else None
    output_map = infer_observed_outputs(
        model_analysis.get("output_names", []),
        observed_analysis.get("column_names", []) if observed_analysis else [],
        requested=observed_output_names,
    )
    likelihood = assess_likelihood(request_text, model_analysis, likelihood_hint)
    scaling = recommend_scaling_mode(observed_analysis["array"], scaling_preference=scaling_mode) if observed_analysis else {"enabled": False, "mode": "none"}
    summary = recommend_summary_kind(observed_analysis["array"], requested_kind=summary_kind) if observed_analysis else {"kind": "identity", "reason": "no_observed_data"}
    distance = recommend_distance_metric(observed_analysis["array"], requested_metric=distance_metric) if observed_analysis else "euclidean"
    hyper = resolve_runtime_hyperparameters(
        {
            "model": {"parameter_names": model_analysis.get("parameter_names", [])},
            "algorithm": {"two_phase": {}},
        },
        observed_size=observed_analysis.get("size") if observed_analysis else 1,
    )
    visual = detect_visualization_defaults(plots)
    return {
        "model_analysis": model_analysis,
        "observed_analysis": {
            **{k: v for k, v in (observed_analysis or {}).items() if k != "array"},
        }
        if observed_analysis
        else None,
        "prior_report": prior_report,
        "output_mapping": output_map,
        "likelihood_assessment": likelihood,
        "scaling_recommendation": scaling,
        "summary_recommendation": summary,
        "distance_recommendation": distance,
        "hyperparameter_recommendation": hyper,
        "visualization_recommendation": visual,
        "pending_questions": output_map.get("questions", []),
    }



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
    distance_metric: str | None = None,
    scaling_mode: str | None = None,
    summary_kind: str | None = None,
    likelihood_hint: str | None = None,
    plots: list[str] | None = None,
    pilot_size: int | None = None,
    main_budget: int | None = None,
    accepted_samples: int | None = None,
    epsilon_quantile: float | None = None,
    max_workers: int | str | None = None,
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
        distance_metric=distance_metric,
        scaling_mode=scaling_mode,
        summary_kind=summary_kind,
        likelihood_hint=likelihood_hint,
        plots=plots,
    )

    cfg = clone_default_config()
    cfg["objective"]["text"] = request_text or ""
    staged_observed = stage_file(observed_path, data_dir)
    if not staged_observed:
        raise ProjectError("Observed data is required.")
    cfg["objective"]["observed_path"] = relative_or_absolute(Path(staged_observed), root)
    cfg["objective"]["likelihood_hint"] = likelihood_hint or "auto"
    output_map = inspection["output_mapping"]
    cfg["objective"]["observed_output_names"] = output_map.get("selected", [])
    cfg["model"]["observed_output_names"] = output_map.get("selected", [])

    model_analysis = dict(inspection["model_analysis"])
    generated_equation_wrapper = None
    if equation_text:
        generated_equation_wrapper = model_dir / "equation_model.py"
        generated_equation_wrapper.write_text(
            build_equation_wrapper_source(equation_text, model_analysis.get("parameter_names", [])),
            encoding="utf-8",
        )
        cfg["model"]["path"] = relative_or_absolute(generated_equation_wrapper, root)
        cfg["model"]["adapter"] = "python_callable"
        cfg["model"]["callable"] = "simulate"
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
            cfg["model"]["command_template"] = f"bash {{model_path}} {{params_json}} {{output_json}}"
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
    cfg["model"]["output_names"] = model_analysis.get("output_names", [])
    cfg["model"]["stochastic"] = model_analysis.get("stochastic")
    cfg["model"]["working_directory"] = "."

    prior_report = inspection["prior_report"]
    cfg["priors"] = prior_report["priors"]

    scaling = inspection["scaling_recommendation"]
    cfg["scaling"] = scaling

    summary = inspection["summary_recommendation"]
    cfg["summary_statistics"]["kind"] = summary["kind"]

    cfg["distance"]["metric"] = inspection["distance_recommendation"]

    likelihood = inspection["likelihood_assessment"]
    hyper = inspection["hyperparameter_recommendation"]
    if pilot_size is not None:
        hyper["pilot_size"] = int(pilot_size)
    if main_budget is not None:
        hyper["main_budget"] = int(main_budget)
    if accepted_samples is not None:
        hyper["accepted_samples"] = int(accepted_samples)
    if epsilon_quantile is not None:
        hyper["epsilon_quantile"] = float(epsilon_quantile)
    cfg["algorithm"]["two_phase"].update(hyper)
    cfg["compute"]["max_workers"] = max_workers if max_workers is not None else "auto"

    visual = inspection["visualization_recommendation"]
    cfg["visualization"]["enabled"] = bool(visual["enabled"])
    cfg["visualization"]["plots"] = visual["plots"]

    cfg["output"]["results_dir"] = "results"
    cfg["output"]["artifacts_dir"] = "artifacts"

    config_path = root / "config.json"
    save_config(config_path, cfg)
    _stage_runtime_library(root)
    _write_run_py(root / "run.py")
    _write_runbook(root / "AGENT_RUNBOOK.md")

    summary_payload = {
        "project_dir": str(root),
        "config_path": str(config_path),
        "pending_questions": inspection["pending_questions"],
        "likelihood_assessment": likelihood,
        "prior_report": prior_report,
        "model_analysis": model_analysis,
        "observed_analysis": inspection["observed_analysis"],
        "hyperparameters": cfg["algorithm"]["two_phase"],
        "notes": [
            "If the likelihood is tractable, prefer MCMC or VI unless you explicitly want likelihood-free calibration.",
            "For non-Python models, verify model/run_model.sh matches the simulator's CLI contract before running.",
        ],
    }
    write_json(root / "project_summary.json", summary_payload)
    write_json(results_dir / "likelihood_assessment.json", likelihood)
    write_json(results_dir / "prior_report.json", prior_report)
    write_json(results_dir / "model_analysis.json", model_analysis)

    return {
        "project_dir": str(root),
        "config_path": str(config_path),
        "run_script": str(root / "run.py"),
        "project_summary": str(root / "project_summary.json"),
        "pending_questions": inspection["pending_questions"],
        "likelihood_assessment": likelihood,
        "priors": prior_report["summary"],
        "hyperparameters": cfg["algorithm"]["two_phase"],
    }
