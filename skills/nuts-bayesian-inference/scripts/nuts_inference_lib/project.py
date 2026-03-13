"""Project scaffolding for the NUTS Bayesian inference skill."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

from .analysis import (
    build_equation_wrapper_source,
    inspect_model_inputs,
)
from .config import clone_default_config, resolve_runtime_hyperparameters, save_config
from .environment import probe_environment
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

from nuts_inference_lib.config import load_config
from nuts_inference_lib.inference import run_inference


def main() -> int:
    config_path = SCRIPT_DIR / "config.json"
    cfg = load_config(config_path)
    result = run_inference(cfg, workdir=SCRIPT_DIR)
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
    destination = root / "skill_runtime" / "nuts_inference_lib"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# NUTS Bayesian Inference Runbook

1. Review `project_summary.json` for pending clarification questions.
2. Confirm priors, observed outputs, likelihood family, and scaling in `config.json`.
3. Run `python3 run.py`.
4. Inspect `results/diagnostics.json`, `results/posterior_summary.json`, and `results/tuned_hyperparameters.json`.
5. If divergences or poor mixing appear, raise `target_acceptance_rate`, revisit priors, or improve parameter scaling.
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
    likelihood_family: str | None = None,
    scaling_mode: str | None = None,
    gradient_strategy: str | None = None,
    plots: list[str] | None = None,
    num_warmup: int | None = None,
    num_samples: int | None = None,
    num_chains: int | None = None,
    target_acceptance: float | None = None,
    max_tree_depth: int | None = None,
    mass_matrix: str | None = None,
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
        likelihood_family=likelihood_family,
        scaling_mode=scaling_mode,
        gradient_strategy=gradient_strategy,
        plots=plots,
    )

    cfg = clone_default_config()
    cfg["objective"]["text"] = request_text or ""
    staged_observed = stage_file(observed_path, data_dir)
    if not staged_observed:
        raise ProjectError("Observed data is required.")
    cfg["objective"]["observed_path"] = relative_or_absolute(Path(staged_observed), root)
    cfg["objective"]["observed_output_names"] = inspection["output_mapping"].get("selected", [])
    cfg["model"]["observed_output_names"] = inspection["output_mapping"].get("selected", [])

    model_analysis = dict(inspection["model_analysis"])
    if equation_text:
        generated_path = model_dir / "equation_model.py"
        generated_path.write_text(
            build_equation_wrapper_source(equation_text, model_analysis.get("parameter_names", [])),
            encoding="utf-8",
        )
        cfg["model"]["path"] = relative_or_absolute(generated_path, root)
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
            _write_command_wrapper(wrapper_path, model_analysis.get("runtime_hint", "bash"), Path(staged_model).name)
            cfg["model"]["command_template"] = "bash {model_path} {params_json} {output_json}"
            cfg["model"]["path"] = relative_or_absolute(wrapper_path, root)
        else:
            cfg["model"]["callable"] = model_analysis.get("callable") or callable_name or "simulate"
            cfg["model"]["call_style"] = model_analysis.get("call_style") or "kwargs"
    else:
        raise ProjectError("A model source is required.")

    cfg["model"]["parameter_names"] = model_analysis.get("parameter_names", [])
    cfg["model"]["parameter_defaults"] = model_analysis.get("parameter_defaults", {})
    cfg["model"]["output_names"] = model_analysis.get("output_names", [])
    cfg["model"]["stochastic"] = model_analysis.get("stochastic")
    cfg["model"]["working_directory"] = "."
    cfg["model"]["gradient_strategy"] = inspection["gradient_recommendation"]["strategy"]

    cfg["priors"] = inspection["prior_report"]["priors"]
    cfg["likelihood"]["family"] = inspection["likelihood_report"]["family"]
    cfg["likelihood"]["params"].update(inspection["likelihood_report"].get("params", {}))
    cfg["scaling"] = inspection["scaling_recommendation"]

    sampler = resolve_runtime_hyperparameters(
        {
            "model": {"parameter_names": list(cfg["model"]["parameter_names"])},
            "sampler": cfg["sampler"],
            "likelihood": {"parameter_names": [name for name in cfg["priors"] if name not in cfg["model"]["parameter_names"]]},
        },
        observed_size=inspection["observed_analysis"]["size"] if inspection["observed_analysis"] else 1,
    )
    if num_warmup is not None:
        sampler["num_warmup"] = int(num_warmup)
    if num_samples is not None:
        sampler["num_samples"] = int(num_samples)
    if num_chains is not None:
        sampler["num_chains"] = int(num_chains)
    if target_acceptance is not None:
        sampler["target_acceptance_rate"] = float(target_acceptance)
    if max_tree_depth is not None:
        sampler["max_tree_depth"] = int(max_tree_depth)
    if mass_matrix is not None:
        sampler["mass_matrix"] = str(mass_matrix)
    cfg["sampler"] = sampler

    visual = inspection["visualization_recommendation"]
    cfg["visualization"]["enabled"] = bool(visual["enabled"])
    cfg["visualization"]["plots"] = visual["plots"]

    config_path = root / "config.json"
    save_config(config_path, cfg)
    _write_run_py(root / "run.py")
    _stage_runtime_library(root)
    _write_runbook(root / "AGENT_RUNBOOK.md")

    summary = {
        "project_dir": str(root),
        "config_path": str(config_path),
        "pending_questions": inspection["pending_questions"],
        "environment": probe_environment(),
        "model_analysis": inspection["model_analysis"],
        "observed_analysis": inspection["observed_analysis"],
        "prior_summary": inspection["prior_report"]["summary"],
        "likelihood_report": inspection["likelihood_report"],
        "scaling_recommendation": inspection["scaling_recommendation"],
        "gradient_recommendation": inspection["gradient_recommendation"],
        "sampler": cfg["sampler"],
    }
    write_json(root / "project_summary.json", summary)
    return summary
