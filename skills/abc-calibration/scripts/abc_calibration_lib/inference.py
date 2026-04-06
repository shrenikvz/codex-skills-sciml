"""ABC rejection inference engine."""

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import simulate_model
from .analysis import assess_likelihood, recommend_summary_kind
from .config import resolve_runtime_hyperparameters
from .io_utils import ensure_jsonable, load_observed_data, payload_to_array, write_json, write_jsonl, write_samples_csv
from .metrics import (
    apply_scaler,
    compute_distance,
    fit_metric_state,
    fit_scaler,
    requires_metric_state,
    safe_vector,
    summarize_array,
)
from .priors import PriorError, require_exact_prior_bounds, sample_prior_dict, summarize_priors


class InferenceError(RuntimeError):
    """Inference failure."""



def _resolve_model_cfg(model_cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
    resolved = dict(model_cfg)
    if resolved.get("path"):
        resolved["path"] = str((workdir / resolved["path"]).resolve()) if not Path(resolved["path"]).is_absolute() else resolved["path"]
    if resolved.get("working_directory"):
        wd = Path(resolved["working_directory"])
        resolved["working_directory"] = str((workdir / wd).resolve()) if not wd.is_absolute() else str(wd)
    else:
        resolved["working_directory"] = str(workdir.resolve())
    return resolved



def _select_observed_array(observed_payload: dict[str, Any], output_names: list[str], output_indices: list[int]) -> np.ndarray:
    array = np.asarray(observed_payload["array"], dtype=float)
    names = list(observed_payload.get("column_names") or [])
    if output_names and array.ndim >= 2 and names:
        indices = [names.index(name) for name in output_names if name in names]
        if indices:
            return np.take(array, indices, axis=1)
    if output_indices:
        axis = 0 if array.ndim == 1 else 1
        return np.take(array, output_indices, axis=axis)
    return array



def _evaluate_summary_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        seed = task.get("seed")
        if seed is not None:
            np.random.seed(int(seed) % (2**32 - 1))
            random.seed(int(seed))
        payload = simulate_model(task["model_cfg"], task["params"], workdir=Path(task["workdir"]))
        array, output_names = payload_to_array(
            payload,
            output_names=task.get("observed_output_names") or None,
            output_indices=task.get("observed_output_indices") or None,
        )
        summary = safe_vector(summarize_array(array, task["summary_cfg"]))
        return {
            "ok": True,
            "summary": summary.tolist(),
            "output_names": output_names,
            "raw_size": int(np.asarray(array).size),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
        }



def _evaluate_predictive_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        seed = task.get("seed")
        if seed is not None:
            np.random.seed(int(seed) % (2**32 - 1))
            random.seed(int(seed))
        payload = simulate_model(task["model_cfg"], task["params"], workdir=Path(task["workdir"]))
        array, output_names = payload_to_array(
            payload,
            output_names=task.get("observed_output_names") or None,
            output_indices=task.get("observed_output_indices") or None,
        )
        return {
            "ok": True,
            "output": np.asarray(array, dtype=float).reshape(-1).tolist(),
            "output_names": output_names,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
        }



def _run_tasks(tasks: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    if max_workers <= 1 or len(tasks) <= 1:
        return [_evaluate_summary_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_evaluate_summary_task, tasks))
    except (OSError, PermissionError):
        return [_evaluate_summary_task(task) for task in tasks]



def _run_predictive_tasks(tasks: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    if max_workers <= 1 or len(tasks) <= 1:
        return [_evaluate_predictive_task(task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_evaluate_predictive_task, tasks))
    except (OSError, PermissionError):
        return [_evaluate_predictive_task(task) for task in tasks]



def _resolve_max_workers(value: Any) -> int:
    if value in {None, "auto"}:
        cpus = os.cpu_count() or 1
        return max(1, min(8, cpus))
    return max(1, int(value))



def _posterior_summary(records: list[dict[str, Any]], parameter_names: list[str]) -> dict[str, Any]:
    if not records:
        return {"accepted_samples": 0, "parameters": {}}
    summary: dict[str, Any] = {"accepted_samples": len(records), "parameters": {}}
    for name in parameter_names:
        values = np.asarray([float(record[name]) for record in records], dtype=float)
        summary["parameters"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)),
            "ci_2_5": float(np.quantile(values, 0.025)),
            "ci_97_5": float(np.quantile(values, 0.975)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    distances = np.asarray([float(record["distance"]) for record in records], dtype=float)
    summary["distance"] = {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }
    return summary



def _build_trace_plot(records: list[dict[str, Any]], parameter_names: list[str], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not records:
        return False
    fig, axes = plt.subplots(len(parameter_names) + 1, 1, figsize=(8, 2.2 * (len(parameter_names) + 1)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    xs = np.arange(1, len(records) + 1)
    for idx, name in enumerate(parameter_names):
        axes[idx].plot(xs, [record[name] for record in records], linewidth=1.0)
        axes[idx].set_ylabel(name)
    axes[-1].plot(xs, [record["distance"] for record in records], linewidth=1.0, color="#aa5500")
    axes[-1].set_ylabel("distance")
    axes[-1].set_xlabel("accepted sample")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True



def _build_marginal_plot(records: list[dict[str, Any]], parameter_names: list[str], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not records:
        return False
    cols = min(3, len(parameter_names))
    rows = int(math.ceil(len(parameter_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.asarray(axes).reshape(-1)
    for idx, name in enumerate(parameter_names):
        values = [record[name] for record in records]
        axes[idx].hist(values, bins=20, color="#2f6f8f", alpha=0.8)
        axes[idx].set_title(name)
    for idx in range(len(parameter_names), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True



def _build_pairwise_plot(records: list[dict[str, Any]], parameter_names: list[str], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if len(parameter_names) < 2 or not records:
        return False
    names = parameter_names[: min(4, len(parameter_names))]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for row, y_name in enumerate(names):
        for col, x_name in enumerate(names):
            ax = axes[row, col]
            x = [record[x_name] for record in records]
            y = [record[y_name] for record in records]
            if row == col:
                ax.hist(x, bins=20, color="#7eaed2")
            else:
                ax.scatter(x, y, s=8, alpha=0.5, color="#1b4d6b")
            if row == n - 1:
                ax.set_xlabel(x_name)
            if col == 0:
                ax.set_ylabel(y_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True



def _build_diagnostics_plot(pilot_distances: list[float], epsilon: float, main_trace: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(pilot_distances, bins=20, color="#779977", alpha=0.8)
    axes[0].axvline(epsilon, color="#aa0000", linestyle="--", linewidth=1.5)
    axes[0].set_title("Pilot distances")
    axes[1].plot([row["distance"] for row in main_trace], linewidth=0.9, color="#224466")
    axes[1].axhline(epsilon, color="#aa0000", linestyle="--", linewidth=1.5)
    axes[1].set_title("Main-phase distances")
    axes[1].set_xlabel("proposal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True



def _build_posterior_predictive_plot(observed: np.ndarray, predictive: np.ndarray, path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if predictive.size == 0:
        return False
    obs = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predictive, dtype=float)
    mean = np.mean(pred, axis=0)
    lower = np.quantile(pred, 0.05, axis=0)
    upper = np.quantile(pred, 0.95, axis=0)
    xs = np.arange(obs.size)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(xs, lower, upper, color="#bbd4ee", alpha=0.7, label="90% predictive interval")
    ax.plot(xs, mean, color="#2f6f8f", linewidth=1.5, label="predictive mean")
    ax.plot(xs, obs, color="#111111", linewidth=1.2, label="observed")
    ax.legend(loc="best")
    ax.set_title("Posterior predictive check")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True



def _posterior_predictive(
    cfg: dict[str, Any],
    workdir: Path,
    accepted_records: list[dict[str, Any]],
    observed_array: np.ndarray,
    max_workers: int,
) -> dict[str, Any]:
    draws = int(cfg.get("posterior_predictive", {}).get("draws", 100) or 0)
    if draws <= 0 or not accepted_records:
        return {"enabled": False, "draws_attempted": 0, "draws_completed": 0}
    subset = accepted_records[: min(draws, len(accepted_records))]
    model_cfg = _resolve_model_cfg(cfg["model"], workdir)
    tasks = []
    base_seed = int(cfg.get("algorithm", {}).get("random_seed", 7)) + 100_000
    for idx, record in enumerate(subset):
        params = {name: float(record[name]) for name in cfg["model"]["parameter_names"]}
        tasks.append(
            {
                "model_cfg": model_cfg,
                "params": params,
                "observed_output_names": cfg["model"].get("observed_output_names", []),
                "observed_output_indices": cfg["model"].get("observed_output_indices", []),
                "workdir": str(workdir),
                "seed": base_seed + idx,
            }
        )
    results = _run_predictive_tasks(tasks, max_workers=max_workers)
    outputs = [np.asarray(item["output"], dtype=float) for item in results if item.get("ok")]
    if not outputs:
        return {
            "enabled": True,
            "draws_attempted": len(tasks),
            "draws_completed": 0,
            "warnings": ["All posterior predictive simulations failed."],
        }
    matrix = np.vstack(outputs)
    observed_vector = np.asarray(observed_array, dtype=float).reshape(-1)
    min_len = min(matrix.shape[1], observed_vector.size)
    matrix = matrix[:, :min_len]
    observed_vector = observed_vector[:min_len]
    lower = np.quantile(matrix, 0.05, axis=0)
    upper = np.quantile(matrix, 0.95, axis=0)
    coverage = float(np.mean((observed_vector >= lower) & (observed_vector <= upper)))
    mean_bias = float(np.mean(np.mean(matrix, axis=0) - observed_vector))
    consistent = bool(coverage >= 0.8)
    return {
        "enabled": True,
        "draws_attempted": len(tasks),
        "draws_completed": int(matrix.shape[0]),
        "coverage_90": coverage,
        "mean_bias": mean_bias,
        "consistent_with_observed": consistent,
        "predictive_mean": np.mean(matrix, axis=0).tolist(),
        "predictive_lower_90": lower.tolist(),
        "predictive_upper_90": upper.tolist(),
        "predictive_matrix": matrix.tolist(),
        "observed_vector": observed_vector.tolist(),
    }



def _make_artifact_index(results_dir: Path) -> dict[str, Any]:
    files = sorted(path.name for path in results_dir.iterdir() if path.is_file())
    return {
        "results_dir": str(results_dir),
        "files": files,
    }



def run_calibration(cfg: dict[str, Any], workdir: Path | str) -> dict[str, Any]:
    workdir = Path(workdir).expanduser().resolve()
    results_dir = workdir / cfg.get("output", {}).get("results_dir", "results")
    artifacts_dir = workdir / cfg.get("output", {}).get("artifacts_dir", "artifacts")
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    observed_path = cfg.get("objective", {}).get("observed_path")
    if not observed_path:
        raise InferenceError("config.objective.observed_path is required.")
    observed = load_observed_data(str((workdir / observed_path).resolve() if not Path(observed_path).is_absolute() else observed_path))
    observed_array = _select_observed_array(
        observed,
        cfg.get("objective", {}).get("observed_output_names", []) or cfg.get("model", {}).get("observed_output_names", []),
        cfg.get("objective", {}).get("observed_output_indices", []) or cfg.get("model", {}).get("observed_output_indices", []),
    )

    if cfg.get("summary_statistics", {}).get("kind") == "auto":
        cfg["summary_statistics"]["kind"] = recommend_summary_kind(observed_array)["kind"]

    parameter_names = list(cfg.get("model", {}).get("parameter_names", []))
    priors = dict(cfg.get("priors", {}))
    missing_priors = [name for name in parameter_names if name not in priors]
    if missing_priors:
        raise InferenceError(f"Missing priors for parameters: {missing_priors}")
    try:
        priors = require_exact_prior_bounds(parameter_names, priors)
    except PriorError as exc:
        raise InferenceError(str(exc)) from exc
    cfg["priors"] = priors

    resolve_runtime_hyperparameters(cfg, observed_size=int(np.asarray(observed_array).size))
    two_phase = cfg["algorithm"]["two_phase"]
    likelihood = assess_likelihood(cfg.get("objective", {}).get("text", ""), cfg.get("model", {}), cfg.get("objective", {}).get("likelihood_hint", "auto"))
    if (
        likelihood["recommendation"] == "likelihood_based_inference"
        and not two_phase.get("proceed_if_likelihood_available", False)
    ):
        raise InferenceError(
            "Model appears to admit a tractable likelihood. Set algorithm.two_phase.proceed_if_likelihood_available=true to force ABC."
        )

    observed_summary = safe_vector(summarize_array(observed_array, cfg["summary_statistics"]))
    scaler_state = fit_scaler(observed_summary, cfg.get("scaling", {}))
    scaled_observed = apply_scaler(observed_summary, scaler_state)
    metric_cfg = cfg.get("distance", {})
    metric_cfg["metric"] = metric_cfg.get("metric") or "euclidean"
    max_workers = _resolve_max_workers(cfg.get("compute", {}).get("max_workers"))
    batch_size = max(1, int(two_phase.get("batch_size", 32)))
    rng = np.random.default_rng(int(cfg.get("algorithm", {}).get("random_seed", 7)))
    model_cfg = _resolve_model_cfg(cfg["model"], workdir)

    pilot_size = int(two_phase["pilot_size"])
    pilot_tasks: list[dict[str, Any]] = []
    pilot_params: list[dict[str, float]] = []
    for idx in range(pilot_size):
        params = sample_prior_dict(priors, rng)
        pilot_params.append(params)
        pilot_tasks.append(
            {
                "model_cfg": model_cfg,
                "params": params,
                "summary_cfg": cfg["summary_statistics"],
                "observed_output_names": cfg["model"].get("observed_output_names", []),
                "observed_output_indices": cfg["model"].get("observed_output_indices", []),
                "workdir": str(workdir),
                "seed": int(rng.integers(0, 2**31 - 1)),
            }
        )
    pilot_results = _run_tasks(pilot_tasks, max_workers=max_workers)
    pilot_summaries = []
    pilot_trace: list[dict[str, Any]] = []
    for idx, (params, result) in enumerate(zip(pilot_params, pilot_results, strict=True), start=1):
        row = {"sample": idx, **params}
        if result.get("ok"):
            summary_vec = apply_scaler(np.asarray(result["summary"], dtype=float), scaler_state)
            pilot_summaries.append(summary_vec)
            row["status"] = "ok"
            row["summary_size"] = int(summary_vec.size)
        else:
            row["status"] = "error"
            row["error"] = result.get("error")
        pilot_trace.append(row)

    if not pilot_summaries:
        raise InferenceError("Pilot phase produced no valid simulations.")

    metric_state = fit_metric_state(metric_cfg["metric"], scaled_observed, pilot_summaries) if requires_metric_state(metric_cfg["metric"]) else {}
    valid_iter = iter(pilot_summaries)
    pilot_distances: list[float] = []
    for row in pilot_trace:
        if row["status"] == "ok":
            summary_vec = next(valid_iter)
            distance = compute_distance(metric_cfg, scaled_observed, summary_vec, metric_state=metric_state)
            row["distance"] = distance
            pilot_distances.append(distance)
        else:
            row["distance"] = None
    epsilon = float(np.quantile(np.asarray(pilot_distances, dtype=float), float(two_phase["epsilon_quantile"])))

    accepted_target = int(two_phase["accepted_samples"])
    main_budget = int(two_phase["main_budget"])
    accepted_records: list[dict[str, Any]] = []
    main_trace: list[dict[str, Any]] = []
    evaluated = 0
    while evaluated < main_budget and len(accepted_records) < accepted_target:
        batch = min(batch_size, main_budget - evaluated)
        tasks = []
        params_batch = []
        for _ in range(batch):
            params = sample_prior_dict(priors, rng)
            params_batch.append(params)
            tasks.append(
                {
                    "model_cfg": model_cfg,
                    "params": params,
                    "summary_cfg": cfg["summary_statistics"],
                    "observed_output_names": cfg["model"].get("observed_output_names", []),
                    "observed_output_indices": cfg["model"].get("observed_output_indices", []),
                    "workdir": str(workdir),
                    "seed": int(rng.integers(0, 2**31 - 1)),
                }
            )
        results = _run_tasks(tasks, max_workers=max_workers)
        for params, result in zip(params_batch, results, strict=True):
            evaluated += 1
            row = {"proposal": evaluated, **params}
            if result.get("ok"):
                summary_vec = apply_scaler(np.asarray(result["summary"], dtype=float), scaler_state)
                distance = compute_distance(metric_cfg, scaled_observed, summary_vec, metric_state=metric_state)
                accepted = bool(distance <= epsilon)
                row.update({"status": "ok", "distance": distance, "accepted": accepted})
                if accepted:
                    accepted_records.append({**params, "distance": distance, "accepted_order": len(accepted_records) + 1})
            else:
                row.update({"status": "error", "error": result.get("error"), "distance": None, "accepted": False})
            main_trace.append(row)
            if evaluated >= main_budget or len(accepted_records) >= accepted_target:
                break

    posterior = _posterior_summary(accepted_records, parameter_names)
    acceptance_rate = len(accepted_records) / max(evaluated, 1)
    predictive = _posterior_predictive(cfg, workdir, accepted_records, observed_array, max_workers=max_workers) if cfg.get("posterior_predictive", {}).get("enabled", True) else {"enabled": False}

    write_samples_csv(
        results_dir / "posterior_samples.csv",
        accepted_records,
        field_order=parameter_names + ["distance", "accepted_order"],
    )
    write_jsonl(results_dir / "posterior_samples.jsonl", accepted_records)
    write_jsonl(results_dir / "pilot_trace.jsonl", pilot_trace)
    write_jsonl(results_dir / "main_trace.jsonl", main_trace)

    write_json(results_dir / "posterior_summary.json", posterior)
    write_json(
        results_dir / "pilot_phase.json",
        {
            "pilot_size": pilot_size,
            "epsilon_quantile": float(two_phase["epsilon_quantile"]),
            "epsilon": epsilon,
            "valid_simulations": len(pilot_distances),
            "failed_simulations": sum(1 for row in pilot_trace if row["status"] == "error"),
            "distance_metric": metric_cfg["metric"],
        },
    )
    write_json(
        results_dir / "sampling_diagnostics.json",
        {
            "accepted": len(accepted_records),
            "accepted_target": accepted_target,
            "main_budget": main_budget,
            "main_evaluated": evaluated,
            "acceptance_rate": acceptance_rate,
            "epsilon": epsilon,
            "distance_metric": metric_cfg["metric"],
            "summary_kind": cfg["summary_statistics"]["kind"],
            "scaling": cfg.get("scaling", {}),
            "likelihood_assessment": likelihood,
            "status": "complete" if len(accepted_records) >= accepted_target else "budget_exhausted",
        },
    )
    write_json(results_dir / "posterior_predictive_summary.json", predictive)
    write_json(results_dir / "prior_report.json", {"priors": summarize_priors(priors)})
    write_json(results_dir / "likelihood_assessment.json", likelihood)

    plot_paths: list[str] = []
    requested_plots = list(cfg.get("visualization", {}).get("plots", [])) if cfg.get("visualization", {}).get("enabled") else []
    figure_dir = results_dir / "figures"
    if "posterior_marginals" in requested_plots:
        out = figure_dir / "posterior_marginals.png"
        if _build_marginal_plot(accepted_records, parameter_names, out):
            plot_paths.append(str(out))
    if "pairwise" in requested_plots:
        out = figure_dir / "pairwise_parameters.png"
        if _build_pairwise_plot(accepted_records, parameter_names, out):
            plot_paths.append(str(out))
    if "trace" in requested_plots:
        out = figure_dir / "accepted_trace.png"
        if _build_trace_plot(accepted_records, parameter_names, out):
            plot_paths.append(str(out))
    if "calibration_diagnostics" in requested_plots:
        out = figure_dir / "calibration_diagnostics.png"
        if _build_diagnostics_plot(pilot_distances, epsilon, [row for row in main_trace if row["status"] == "ok"], out):
            plot_paths.append(str(out))
    if "posterior_predictive" in requested_plots and predictive.get("enabled") and predictive.get("draws_completed", 0) > 0:
        out = figure_dir / "posterior_predictive.png"
        if _build_posterior_predictive_plot(
            np.asarray(predictive["observed_vector"], dtype=float),
            np.asarray(predictive["predictive_matrix"], dtype=float),
            out,
        ):
            plot_paths.append(str(out))

    run_summary = {
        "status": "complete" if len(accepted_records) >= accepted_target else "budget_exhausted",
        "posterior_samples": len(accepted_records),
        "accepted_target": accepted_target,
        "acceptance_rate": acceptance_rate,
        "epsilon": epsilon,
        "distance_metric": metric_cfg["metric"],
        "summary_kind": cfg["summary_statistics"]["kind"],
        "plots": plot_paths,
        "posterior_predictive_consistent": predictive.get("consistent_with_observed"),
    }
    write_json(results_dir / "run_summary.json", run_summary)
    write_json(results_dir / "artifact_index.json", _make_artifact_index(results_dir))

    return ensure_jsonable(
        {
            "status": run_summary["status"],
            "posterior_samples": len(accepted_records),
            "posterior_summary": posterior,
            "acceptance_rate": acceptance_rate,
            "epsilon": epsilon,
            "likelihood_assessment": likelihood,
            "posterior_predictive": {k: v for k, v in predictive.items() if k != "predictive_matrix"},
            "results_dir": str(results_dir),
            "plots": plot_paths,
        }
    )
