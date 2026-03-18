"""Inference orchestration for the NUTS calibration skill."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import simulate_model
from .backends import BackendError, _fit_scaling, run_blackjax_nuts
from .config import resolve_runtime_hyperparameters
from .diagnostics import summarize_diagnostics
from .environment import probe_environment, recommend_backend
from .io_utils import (
    ensure_jsonable,
    load_observed_data,
    payload_to_array,
    write_json,
    write_jsonl,
    write_samples_csv,
)
from .transforms import vector_to_parameter_dict_numpy
from .visualization import (
    build_autocorrelation_plot,
    build_energy_plot,
    build_marginal_plot,
    build_pairwise_plot,
    build_posterior_predictive_plot,
    build_trace_plot,
)


class InferenceError(RuntimeError):
    """Inference failure."""


def _resolve_model_cfg(model_cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = dict(model_cfg)
    if resolved.get("path"):
        path = Path(resolved["path"])
        resolved["path"] = str((workdir / path).resolve()) if not path.is_absolute() else str(path)
    if resolved.get("working_directory"):
        wd = Path(resolved["working_directory"])
        resolved["working_directory"] = str((workdir / wd).resolve()) if not wd.is_absolute() else str(wd)
    else:
        resolved["working_directory"] = str(workdir.resolve())
    return resolved


def _posterior_summary(samples: np.ndarray, parameter_names: list[str]) -> dict[str, Any]:
    merged = samples.reshape(-1, samples.shape[-1])
    summary = {"chains": int(samples.shape[0]), "draws_per_chain": int(samples.shape[1]), "parameters": {}}
    for idx, name in enumerate(parameter_names):
        values = merged[:, idx]
        summary["parameters"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "ci_2_5": float(np.quantile(values, 0.025)),
            "ci_97_5": float(np.quantile(values, 0.975)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _posterior_records(samples: np.ndarray, parameter_names: list[str]) -> list[dict[str, Any]]:
    records = []
    for chain in range(samples.shape[0]):
        for draw in range(samples.shape[1]):
            record = {"chain": chain, "draw": draw}
            for idx, name in enumerate(parameter_names):
                record[name] = float(samples[chain, draw, idx])
            records.append(record)
    return records


def _posterior_predictive(
    samples: np.ndarray,
    cfg: dict[str, Any],
    runtime: dict[str, Any],
    observed_array: np.ndarray,
) -> dict[str, Any]:
    draws = int(cfg.get("posterior_predictive", {}).get("draws", 0) or 0)
    if draws <= 0:
        return {"enabled": False}
    merged = samples.reshape(-1, samples.shape[-1])
    if merged.size == 0:
        return {"enabled": False}
    rng = np.random.default_rng(int(cfg.get("algorithm", {}).get("random_seed", 7)) + 99)
    indices = rng.choice(merged.shape[0], size=min(draws, merged.shape[0]), replace=False)
    predictive_rows = []
    for idx in indices:
        param_row = {name: float(merged[idx, column]) for column, name in enumerate(runtime["parameter_names"])}
        payload = simulate_model(runtime["model_cfg"], param_row, workdir=runtime["workdir"])
        simulated, _ = payload_to_array(
            payload,
            output_names=runtime["selected_output_names"] or None,
            output_indices=runtime["selected_output_indices"] or None,
        )
        predictive_rows.append(np.asarray(simulated, dtype=float).reshape(-1))
    predictive = np.stack(predictive_rows, axis=0)
    obs = np.asarray(observed_array, dtype=float).reshape(-1)
    mean = np.mean(predictive, axis=0)
    lower = np.quantile(predictive, 0.025, axis=0)
    upper = np.quantile(predictive, 0.975, axis=0)
    coverage = float(np.mean((obs >= lower) & (obs <= upper))) if obs.size == mean.size else None
    consistent = bool(coverage is not None and coverage >= 0.8)
    return {
        "enabled": True,
        "draws": int(predictive.shape[0]),
        "mean": mean.tolist(),
        "ci_2_5": lower.tolist(),
        "ci_97_5": upper.tolist(),
        "coverage_95": coverage,
        "consistent_with_observed": consistent,
    }


def _write_figures(
    results_dir: Path,
    samples: np.ndarray,
    parameter_names: list[str],
    diagnostics: dict[str, Any],
    predictive: dict[str, Any],
    observed_array: np.ndarray,
    visualization_cfg: dict[str, Any],
) -> list[str]:
    if not visualization_cfg.get("enabled"):
        return []
    dpi = int(visualization_cfg.get("dpi", 140))
    plots = list(visualization_cfg.get("plots", []))
    figures_dir = results_dir / "figures"
    artifact_paths = []
    if "trace" in plots and build_trace_plot(samples, parameter_names, figures_dir / "trace.png", dpi=dpi):
        artifact_paths.append(str(figures_dir / "trace.png"))
    if "posterior_marginals" in plots and build_marginal_plot(samples, parameter_names, figures_dir / "posterior_marginals.png", dpi=dpi):
        artifact_paths.append(str(figures_dir / "posterior_marginals.png"))
    if "pairwise" in plots and build_pairwise_plot(samples, parameter_names, figures_dir / "pairwise.png", dpi=dpi):
        artifact_paths.append(str(figures_dir / "pairwise.png"))
    if "autocorrelation" in plots and build_autocorrelation_plot(samples, parameter_names, figures_dir / "autocorrelation.png", dpi=dpi):
        artifact_paths.append(str(figures_dir / "autocorrelation.png"))
    if "energy" in plots:
        energy = np.asarray(diagnostics.get("raw_info", {}).get("energy", []), dtype=float)
        if energy.size and build_energy_plot(energy, figures_dir / "energy.png", dpi=dpi):
            artifact_paths.append(str(figures_dir / "energy.png"))
    if "posterior_predictive" in plots and predictive.get("enabled"):
        if build_posterior_predictive_plot(predictive, observed_array, figures_dir / "posterior_predictive.png", dpi=dpi):
            artifact_paths.append(str(figures_dir / "posterior_predictive.png"))
    return artifact_paths


def run_calibration(cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
    parameter_names = list(cfg.get("model", {}).get("parameter_names", []))
    if not parameter_names:
        raise InferenceError("config.model.parameter_names is empty.")
    environment = probe_environment()
    backend_choice = recommend_backend(environment, requested=cfg.get("algorithm", {}).get("backend"))
    if cfg.get("algorithm", {}).get("backend") == "blackjax" and not backend_choice.get("ready"):
        raise InferenceError(f"BlackJAX backend is not available in this Python environment: {environment.get('jax', {}).get('error') or environment.get('blackjax', {}).get('error')}")
    if cfg.get("algorithm", {}).get("backend") != "blackjax":
        raise InferenceError(f"Backend {cfg['algorithm']['backend']!r} is not implemented in this skill release.")

    observed_payload = load_observed_data(str((workdir / cfg["objective"]["observed_path"]).resolve()) if not Path(cfg["objective"]["observed_path"]).is_absolute() else cfg["objective"]["observed_path"])
    observed_array = np.asarray(observed_payload["array"], dtype=float).reshape(-1)
    model_cfg = _resolve_model_cfg(cfg["model"], workdir)
    hyper = resolve_runtime_hyperparameters(cfg, observed_size=observed_payload["size"])
    scaling_state = _fit_scaling(observed_array, cfg.get("scaling", {}))
    observed_scaled = (observed_array - scaling_state["center"]) / scaling_state["scale"] if scaling_state.get("enabled") else observed_array

    default_points = {
        name: float(cfg["priors"][name]["params"].get("mean", 0.0))
        if cfg["priors"][name]["dist"] == "normal"
        else None
        for name in parameter_names
    }
    for name in parameter_names:
        if default_points[name] is None:
            from .priors import default_point

            default_points[name] = float(default_point(cfg["priors"][name]))

    runtime = {
        "workdir": Path(model_cfg.get("working_directory") or workdir).resolve(),
        "model_cfg": model_cfg,
        "parameter_names": parameter_names,
        "default_points": default_points,
        "priors": cfg["priors"],
        "transform_specs": cfg.get("transformations", {}).get("parameters", {}),
        "likelihood": cfg["likelihood"],
        "observed_scaled": np.asarray(observed_scaled, dtype=float).reshape(-1),
        "observed_array": observed_array,
        "scaling_state": scaling_state,
        "selected_output_names": list(cfg.get("model", {}).get("observed_output_names", [])),
        "selected_output_indices": list(cfg.get("model", {}).get("observed_output_indices", [])),
        "gradient_strategy": cfg.get("model", {}).get("gradient_strategy", "auto"),
        "num_chains": int(cfg["algorithm"]["sampling"]["num_chains"]),
        "warmup_steps": int(cfg["algorithm"]["warmup"]["num_steps"]),
        "num_samples": int(cfg["algorithm"]["sampling"]["num_samples"]),
        "target_acceptance": float(cfg["algorithm"]["warmup"]["target_acceptance"]),
        "max_tree_depth": int(cfg["algorithm"]["sampling"]["max_tree_depth"]),
        "mass_matrix": str(cfg["algorithm"]["warmup"]["mass_matrix"]),
        "initial_step_size": cfg["algorithm"]["warmup"]["initial_step_size"],
        "thin": int(cfg["algorithm"]["sampling"].get("thin", 1)),
        "seed": int(cfg["algorithm"].get("random_seed", 7)),
        "device_preference": cfg.get("compute", {}).get("device_preference", "auto"),
        "chain_method": cfg.get("compute", {}).get("chain_method", "auto"),
        "enable_x64": bool(cfg.get("compute", {}).get("enable_x64", False)),
        "progress_bar": bool(cfg.get("backend_options", {}).get("blackjax", {}).get("progress_bar", False)),
    }

    try:
        backend_result = run_blackjax_nuts(runtime)
    except BackendError as exc:
        raise InferenceError(str(exc)) from exc

    samples = np.asarray(backend_result["samples_constrained"], dtype=float)
    raw_info = {key: np.asarray(value) for key, value in backend_result["info"].items()}
    diagnostics = summarize_diagnostics(samples, parameter_names, raw_info, runtime["max_tree_depth"])
    diagnostics["raw_info"] = {key: ensure_jsonable(value) for key, value in raw_info.items()}
    summary = _posterior_summary(samples, parameter_names)
    predictive = _posterior_predictive(samples, cfg, runtime, observed_array) if cfg.get("posterior_predictive", {}).get("enabled", True) else {"enabled": False}

    results_dir = (workdir / cfg.get("output", {}).get("results_dir", "results")).resolve()
    artifacts_dir = (workdir / cfg.get("output", {}).get("artifacts_dir", "artifacts")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    records = _posterior_records(samples, parameter_names)
    write_samples_csv(results_dir / "posterior_samples.csv", records, field_order=["chain", "draw", *parameter_names])
    write_jsonl(results_dir / "posterior_samples.jsonl", records)
    write_json(results_dir / "posterior_summary.json", summary)
    write_json(results_dir / "sampling_diagnostics.json", {key: value for key, value in diagnostics.items() if key != "raw_info"})
    write_json(results_dir / "tuned_hyperparameters.json", {"warmup": backend_result["warmup"], "sampling": cfg["algorithm"]["sampling"]})
    write_json(results_dir / "environment.json", environment)
    write_json(results_dir / "posterior_predictive_summary.json", predictive)

    figures = _write_figures(
        results_dir,
        samples,
        parameter_names,
        diagnostics,
        predictive,
        observed_array,
        cfg.get("visualization", {}),
    )
    artifact_index = {
        "posterior_samples_csv": str(results_dir / "posterior_samples.csv"),
        "posterior_samples_jsonl": str(results_dir / "posterior_samples.jsonl"),
        "posterior_summary": str(results_dir / "posterior_summary.json"),
        "sampling_diagnostics": str(results_dir / "sampling_diagnostics.json"),
        "tuned_hyperparameters": str(results_dir / "tuned_hyperparameters.json"),
        "posterior_predictive_summary": str(results_dir / "posterior_predictive_summary.json"),
        "figures": figures,
    }
    write_json(results_dir / "artifact_index.json", artifact_index)

    run_summary = {
        "backend": backend_result["backend"],
        "execution_mode": backend_result["execution_mode"],
        "chain_method": backend_result["chain_method"],
        "devices": backend_result["devices"],
        "posterior_samples": int(samples.shape[0] * samples.shape[1]),
        "chains": int(samples.shape[0]),
        "draws_per_chain": int(samples.shape[1]),
        "acceptance_rate_mean": diagnostics["acceptance_rate"]["mean"],
        "divergences": diagnostics["divergences"]["count"],
        "warnings": diagnostics["warnings"],
        "posterior_predictive_consistent": predictive.get("consistent_with_observed"),
        "results_dir": str(results_dir),
        "artifacts_dir": str(artifacts_dir),
    }
    write_json(results_dir / "run_summary.json", run_summary)

    return {
        "results_dir": str(results_dir),
        "artifacts_dir": str(artifacts_dir),
        "posterior_samples": int(samples.shape[0] * samples.shape[1]),
        "posterior_summary": summary,
        "diagnostics": {key: value for key, value in diagnostics.items() if key != "raw_info"},
        "tuned_hyperparameters": {"warmup": backend_result["warmup"], "sampling": cfg["algorithm"]["sampling"]},
        "posterior_predictive": predictive,
        "warnings": diagnostics["warnings"],
    }

