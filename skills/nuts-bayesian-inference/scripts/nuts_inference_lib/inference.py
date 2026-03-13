"""High-level inference orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters import simulate_model
from .backends import run_backend
from .config import resolve_runtime_hyperparameters
from .diagnostics import build_diagnostics, credible_intervals, summarize_posterior
from .environment import probe_environment
from .io_utils import ensure_jsonable, load_observed_data, payload_to_array, write_json, write_jsonl, write_samples_csv
from .likelihoods import evaluate_log_likelihood
from .transforms import build_transform_specs, unconstrain_dict
from .visualization import build_requested_plots


class InferenceError(RuntimeError):
    """Inference failure."""


def _resolve_model_cfg(model_cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = dict(model_cfg)
    if resolved.get("path"):
        path = Path(resolved["path"])
        resolved["path"] = str((workdir / path).resolve()) if not path.is_absolute() else str(path)
    wd = resolved.get("working_directory")
    if wd:
        wd_path = Path(wd)
        resolved["working_directory"] = str((workdir / wd_path).resolve()) if not wd_path.is_absolute() else str(wd_path)
    else:
        resolved["working_directory"] = str(workdir.resolve())
    return resolved


def _select_observed_array(observed_payload: dict[str, Any], output_names: list[str], output_indices: list[int]) -> np.ndarray:
    array = np.asarray(observed_payload["array"], dtype=float)
    names = list(observed_payload.get("column_names") or [])
    if output_names and array.ndim >= 2 and names:
        indices = [names.index(name) for name in output_names if name in names]
        if indices:
            return np.take(array, indices, axis=1).reshape(-1)
    if output_indices:
        axis = 0 if array.ndim == 1 else 1
        return np.take(array, output_indices, axis=axis).reshape(-1)
    return array.reshape(-1)


def _fit_scaler(array: np.ndarray, scaling_cfg: dict[str, Any]) -> dict[str, Any]:
    mode = scaling_cfg.get("mode", "none")
    if not scaling_cfg.get("enabled"):
        return {"mode": "none"}
    arr = np.asarray(array, dtype=float).reshape(-1)
    if mode == "zscore":
        mean = np.mean(arr)
        std = max(float(np.std(arr)), 1e-8)
        return {"mode": "zscore", "mean": float(mean), "scale": std}
    if mode == "minmax":
        lower = float(np.min(arr))
        upper = float(np.max(arr))
        width = max(upper - lower, 1e-8)
        return {"mode": "minmax", "lower": lower, "scale": width}
    if mode == "variance":
        scale = max(float(np.var(arr)), 1e-8)
        return {"mode": "variance", "scale": scale}
    return {"mode": "none"}


def _apply_scaler(array, scaler_state: dict[str, Any], xp=np):
    mode = scaler_state.get("mode", "none")
    if mode == "none":
        return xp.asarray(array, dtype=float).reshape(-1)
    if mode == "zscore":
        return (xp.asarray(array, dtype=float).reshape(-1) - scaler_state["mean"]) / scaler_state["scale"]
    if mode == "minmax":
        return (xp.asarray(array, dtype=float).reshape(-1) - scaler_state["lower"]) / scaler_state["scale"]
    if mode == "variance":
        return xp.asarray(array, dtype=float).reshape(-1) / scaler_state["scale"]
    return xp.asarray(array, dtype=float).reshape(-1)


def _posterior_predictive(
    cfg: dict[str, Any],
    context: dict[str, Any],
    samples: np.ndarray,
) -> dict[str, Any]:
    draws = int(cfg.get("posterior_predictive", {}).get("draws", 0) or 0)
    if not cfg.get("posterior_predictive", {}).get("enabled", True) or draws <= 0:
        return {"enabled": False, "draws_attempted": 0, "draws_completed": 0}
    flat = samples.reshape(-1, samples.shape[-1])
    if flat.size == 0:
        return {"enabled": True, "draws_attempted": draws, "draws_completed": 0}
    idxs = np.linspace(0, max(flat.shape[0] - 1, 0), num=min(draws, flat.shape[0]), dtype=int)
    predictive = []
    for idx in idxs:
        params = {name: float(flat[idx, name_idx]) for name_idx, name in enumerate(context["parameter_names"])}
        model_params = {name: params[name] for name in context["model_parameter_names"]}
        payload = context["simulate_host"](model_params)
        output, _ = payload_to_array(
            payload,
            output_names=context["model_cfg"].get("observed_output_names") or None,
            output_indices=context["model_cfg"].get("observed_output_indices") or None,
        )
        predictive.append(np.asarray(output, dtype=float).reshape(-1))
    predictive_arr = np.asarray(predictive, dtype=float) if predictive else np.zeros((0, context["observed_raw"].size))
    if predictive_arr.size == 0:
        return {"enabled": True, "draws_attempted": draws, "draws_completed": 0}
    lower = np.quantile(predictive_arr, 0.05, axis=0)
    upper = np.quantile(predictive_arr, 0.95, axis=0)
    observed = context["observed_raw"].reshape(-1)
    inside = np.logical_and(observed >= lower, observed <= upper)
    return {
        "enabled": True,
        "draws_attempted": draws,
        "draws_completed": int(predictive_arr.shape[0]),
        "predictive_mean": np.mean(predictive_arr, axis=0).tolist(),
        "predictive_interval_90": {"lower": lower.tolist(), "upper": upper.tolist()},
        "fraction_observed_within_90_percent_interval": float(np.mean(inside)),
        "predictive_samples": predictive_arr,
    }


def _resolve_gradient_strategy(model_cfg: dict[str, Any]) -> str:
    strategy = (model_cfg.get("gradient_strategy") or "auto").lower()
    if strategy != "auto":
        return strategy
    path = model_cfg.get("path")
    if model_cfg.get("adapter") == "python_callable" and path:
        try:
            source = Path(path).read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            return "finite_difference"
        if any(token in source for token in ["import jax", "jax.numpy", "jnp."]):
            return "jax"
    return "finite_difference"


def run_inference(cfg: dict[str, Any], workdir: str | Path | None = None) -> dict[str, Any]:
    workdir = Path(workdir or Path.cwd()).expanduser().resolve()
    env = probe_environment()
    if not env.get("recommended_backend") and (cfg.get("sampler", {}).get("backend") or "blackjax").lower() == "blackjax":
        raise InferenceError(
            f"BlackJAX cannot run in the current environment. JAX status: {env['jax']}. BlackJAX status: {env['blackjax']}."
        )

    observed_payload = load_observed_data(str((workdir / cfg["objective"]["observed_path"]).resolve() if not Path(cfg["objective"]["observed_path"]).is_absolute() else cfg["objective"]["observed_path"]))
    observed_raw = _select_observed_array(
        observed_payload,
        cfg["objective"].get("observed_output_names", []),
        cfg["objective"].get("observed_output_indices", []),
    )
    cfg["sampler"] = resolve_runtime_hyperparameters(
        {"model": {"parameter_names": list(cfg["priors"].keys())}, "sampler": cfg["sampler"], "likelihood": {"parameter_names": []}},
        observed_size=int(observed_raw.size),
    )
    scaler_state = _fit_scaler(observed_raw, cfg.get("scaling", {}))
    observed_eval = _apply_scaler(observed_raw, scaler_state, xp=np)
    model_cfg = _resolve_model_cfg(cfg["model"], workdir)
    parameter_names = list(cfg["priors"].keys())
    model_parameter_names = list(cfg["model"].get("parameter_names", []))
    transform_specs = build_transform_specs(cfg["priors"])
    initial_point = {}
    for name in parameter_names:
        spec = cfg["priors"][name]
        params = spec["params"]
        dist = spec["dist"]
        if dist == "uniform":
            initial_point[name] = 0.5 * (params["lower"] + params["upper"])
        elif dist == "normal":
            initial_point[name] = params["mean"]
        elif dist == "lognormal":
            initial_point[name] = float(np.exp(params["mean"]))
        elif dist == "gamma":
            initial_point[name] = params["shape"] * params["scale"]
        elif dist == "beta":
            mean = params["alpha"] / (params["alpha"] + params["beta"])
            initial_point[name] = params["lower"] + (params["upper"] - params["lower"]) * mean
        elif dist == "halfnormal":
            initial_point[name] = params["scale"] * np.sqrt(2 / np.pi)
        elif dist == "student_t":
            initial_point[name] = params["loc"]
        else:
            initial_point[name] = 0.0

    context = {
        "workdir": workdir,
        "model_cfg": model_cfg,
        "likelihood_cfg": cfg["likelihood"],
        "priors": cfg["priors"],
        "parameter_names": parameter_names,
        "model_parameter_names": model_parameter_names,
        "transform_specs": transform_specs,
        "observed_raw": np.asarray(observed_raw, dtype=float).reshape(-1),
        "observed_eval": np.asarray(observed_eval, dtype=float).reshape(-1),
        "gradient_strategy": _resolve_gradient_strategy(model_cfg),
        "gradient_step": float(cfg["model"].get("gradient_step_size", 1e-4)),
        "scaler_state": scaler_state,
        "scaler_apply": lambda array, xp=np: _apply_scaler(array, scaler_state, xp=xp),
        "simulate_host": lambda params: simulate_model(model_cfg, params, workdir=workdir),
        "likelihood_fn": evaluate_log_likelihood,
        "metadata": {
            "objective": cfg.get("objective", {}),
            "model": cfg.get("model", {}),
            "likelihood": cfg.get("likelihood", {}),
            "scaling": scaler_state,
        },
        "initial_point": initial_point,
        "unconstrain_params": lambda params: unconstrain_dict(params, parameter_names, transform_specs, xp=np),
    }

    backend_result = run_backend(cfg, context)
    samples = np.asarray(backend_result["samples"], dtype=float)
    logdensity = np.asarray(backend_result["logdensity"], dtype=float)
    info = backend_result["info"]
    diagnostics = build_diagnostics(
        samples,
        parameter_names,
        np.asarray(info["energy"], dtype=float),
        np.asarray(info["is_divergent"], dtype=bool),
        np.asarray(info["acceptance_rate"], dtype=float),
        np.asarray(info["num_trajectory_expansions"], dtype=float),
        int(cfg["sampler"]["max_tree_depth"]),
    )
    posterior_summary = summarize_posterior(samples, parameter_names)
    intervals = credible_intervals(samples, parameter_names)
    predictive = _posterior_predictive(cfg, context, samples)

    results_dir = workdir / cfg.get("output", {}).get("results_dir", "results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for chain in range(samples.shape[0]):
        for draw in range(samples.shape[1]):
            row = {"chain": chain, "draw": draw, "logdensity": float(logdensity[chain, draw])}
            for idx, name in enumerate(parameter_names):
                row[name] = float(samples[chain, draw, idx])
            row["acceptance_rate"] = float(info["acceptance_rate"][chain, draw])
            row["is_divergent"] = bool(info["is_divergent"][chain, draw])
            row["tree_depth"] = int(info["num_trajectory_expansions"][chain, draw])
            records.append(row)

    write_samples_csv(results_dir / "posterior_samples.csv", records)
    write_jsonl(results_dir / "posterior_samples.jsonl", records)
    write_json(results_dir / "posterior_summary.json", posterior_summary)
    write_json(results_dir / "credible_intervals.json", intervals)
    write_json(results_dir / "diagnostics.json", diagnostics)
    write_json(results_dir / "acceptance_statistics.json", diagnostics["acceptance"])
    write_json(results_dir / "tuned_hyperparameters.json", backend_result["tuned"])
    write_json(results_dir / "prior_report.json", cfg["priors"])
    write_json(results_dir / "likelihood_report.json", cfg["likelihood"])
    predictive_summary = {k: v for k, v in predictive.items() if k != "predictive_samples"}
    write_json(results_dir / "posterior_predictive_summary.json", predictive_summary)
    plot_results = {}
    if cfg.get("visualization", {}).get("enabled"):
        plot_results = build_requested_plots(
            samples,
            parameter_names,
            np.asarray(info["energy"], dtype=float),
            np.asarray(observed_raw, dtype=float).reshape(-1),
            np.asarray(predictive.get("predictive_samples", np.asarray([])), dtype=float) if predictive.get("enabled") else None,
            cfg.get("visualization", {}).get("plots", []),
            figures_dir,
            dpi=int(cfg.get("visualization", {}).get("dpi", 140)),
        )
    artifact_index = {
        "results_dir": str(results_dir),
        "files": [
            "posterior_samples.csv",
            "posterior_samples.jsonl",
            "posterior_summary.json",
            "credible_intervals.json",
            "diagnostics.json",
            "acceptance_statistics.json",
            "tuned_hyperparameters.json",
            "prior_report.json",
            "likelihood_report.json",
            "posterior_predictive_summary.json",
        ],
        "plots": plot_results,
    }
    write_json(results_dir / "artifact_index.json", artifact_index)
    run_summary = {
        "backend": cfg["sampler"]["backend"],
        "algorithm": cfg["sampler"]["algorithm"],
        "num_chains": int(samples.shape[0]),
        "num_samples_per_chain": int(samples.shape[1]),
        "parameter_names": parameter_names,
        "results_dir": str(results_dir),
        "warnings": diagnostics["warnings"],
        "environment": ensure_jsonable(env),
        "tuned_hyperparameters": backend_result["tuned"],
    }
    write_json(results_dir / "run_summary.json", run_summary)
    return {
        "results_dir": str(results_dir),
        "posterior_samples": int(samples.shape[0] * samples.shape[1]),
        "posterior_summary": posterior_summary,
        "diagnostics": diagnostics,
        "posterior_predictive": predictive_summary,
        "tuned_hyperparameters": backend_result["tuned"],
    }
