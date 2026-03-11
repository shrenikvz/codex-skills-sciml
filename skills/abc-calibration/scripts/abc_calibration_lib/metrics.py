"""Summary statistics, scaling, and distance metrics."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class MetricError(RuntimeError):
    """Metric or summary failure."""



def _load_python_callable(path: str, callable_name: str):
    target = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(target.stem, target)
    if spec is None or spec.loader is None:
        raise MetricError(f"Could not load Python module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise MetricError(f"Callable {callable_name!r} not found in {target}")
    return fn



def normalize_summary_config(summary_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = str(summary_cfg.get("kind", "identity")).strip().lower()
    normalized = dict(summary_cfg)
    normalized["kind"] = kind
    normalized.setdefault("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    normalized.setdefault("lag_count", 3)
    return normalized



def summarize_array(array: np.ndarray, summary_cfg: dict[str, Any]) -> np.ndarray:
    cfg = normalize_summary_config(summary_cfg)
    arr = np.asarray(array, dtype=float)
    kind = cfg["kind"]
    if kind == "identity":
        return arr.reshape(-1)
    if kind == "moments":
        flat = arr.reshape(-1)
        quantiles = np.quantile(flat, cfg.get("quantiles", [0.25, 0.5, 0.75]))
        stats = [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
        ]
        return np.asarray(stats + [float(item) for item in quantiles], dtype=float)
    if kind == "quantiles":
        flat = arr.reshape(-1)
        return np.asarray(np.quantile(flat, cfg.get("quantiles", [0.1, 0.5, 0.9])), dtype=float)
    if kind == "timeseries":
        flat = arr.reshape(-1)
        lags = max(1, int(cfg.get("lag_count", 3)))
        features = [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.median(flat)),
        ]
        quantiles = np.quantile(flat, cfg.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]))
        features.extend(float(item) for item in quantiles)
        if flat.size > 1:
            diffs = np.diff(flat)
            features.append(float(np.mean(diffs)))
            features.append(float(np.std(diffs)))
        else:
            features.extend([0.0, 0.0])
        centered = flat - float(np.mean(flat))
        denom = float(np.dot(centered, centered))
        for lag in range(1, lags + 1):
            if flat.size <= lag or denom == 0.0:
                features.append(0.0)
                continue
            numer = float(np.dot(centered[:-lag], centered[lag:]))
            features.append(numer / denom)
        return np.asarray(features, dtype=float)
    if kind == "python_callable":
        fn = _load_python_callable(cfg["path"], cfg["callable"])
        out = fn(arr)
        return np.asarray(out, dtype=float).reshape(-1)
    if kind == "command":
        command_template = cfg.get("command_template")
        if not command_template:
            raise MetricError("summary_statistics.command_template is required for command summaries.")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "summary_input.json"
            output_path = root / "summary_output.json"
            input_path.write_text(json.dumps(arr.tolist()), encoding="utf-8")
            command = command_template.format(input_json=input_path, output_json=output_path)
            completed = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                raise MetricError(f"Custom summary command failed: {completed.stderr.strip()}")
            if output_path.exists():
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            else:
                payload = json.loads(completed.stdout)
            return np.asarray(payload, dtype=float).reshape(-1)
    raise MetricError(f"Unsupported summary kind: {kind}")



def fit_scaler(observed_vector: np.ndarray, scaling_cfg: dict[str, Any]) -> dict[str, Any]:
    vector = np.asarray(observed_vector, dtype=float).reshape(-1)
    enabled = scaling_cfg.get("enabled")
    mode = str(scaling_cfg.get("mode", "none")).strip().lower()
    if enabled is False or mode in {"none", "off"}:
        return {"enabled": False, "mode": "none"}
    if mode == "auto":
        mode = "zscore"
    if mode == "zscore":
        center = vector
        scale = np.std(vector)
        if np.isscalar(scale):
            scale = np.full_like(center, float(scale) if float(scale) > 0 else 1.0)
        return {"enabled": True, "mode": "zscore", "center": center, "scale": np.asarray(scale, dtype=float)}
    if mode == "variance":
        scale = np.std(vector)
        if float(scale) <= 0:
            scale = 1.0
        return {"enabled": True, "mode": "variance", "center": np.zeros_like(vector), "scale": np.full_like(vector, float(scale))}
    if mode == "minmax":
        lower = np.min(vector)
        upper = np.max(vector)
        span = float(upper - lower)
        if span <= 0:
            span = 1.0
        return {
            "enabled": True,
            "mode": "minmax",
            "lower": np.full_like(vector, float(lower)),
            "upper": np.full_like(vector, float(upper)),
            "scale": np.full_like(vector, span),
        }
    raise MetricError(f"Unsupported scaling mode: {mode}")



def apply_scaler(vector: np.ndarray, scaler_state: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    if not scaler_state.get("enabled"):
        return arr
    mode = scaler_state.get("mode")
    if mode in {"zscore", "variance"}:
        center = np.asarray(scaler_state.get("center", np.zeros_like(arr)), dtype=float).reshape(-1)
        scale = np.asarray(scaler_state.get("scale", np.ones_like(arr)), dtype=float).reshape(-1)
        scale = np.where(scale == 0, 1.0, scale)
        return (arr - center) / scale
    if mode == "minmax":
        lower = np.asarray(scaler_state.get("lower", np.zeros_like(arr)), dtype=float).reshape(-1)
        scale = np.asarray(scaler_state.get("scale", np.ones_like(arr)), dtype=float).reshape(-1)
        scale = np.where(scale == 0, 1.0, scale)
        return (arr - lower) / scale
    raise MetricError(f"Unsupported scaler mode: {mode}")



def requires_metric_state(metric: str) -> bool:
    return metric.lower() in {"mahalanobis"}



def fit_metric_state(metric: str, observed_vector: np.ndarray, simulated_vectors: list[np.ndarray]) -> dict[str, Any]:
    metric = metric.lower()
    if metric != "mahalanobis":
        return {}
    vectors = [np.asarray(observed_vector, dtype=float).reshape(-1)]
    vectors.extend(np.asarray(item, dtype=float).reshape(-1) for item in simulated_vectors if item is not None)
    matrix = np.vstack(vectors)
    cov = np.cov(matrix, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.asarray([[float(cov)]], dtype=float)
    cov = np.asarray(cov, dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise MetricError("Mahalanobis covariance must be square.")
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)
    return {"cov_inv": inv}



def ks_statistic(observed: np.ndarray, simulated: np.ndarray) -> float:
    x = np.sort(np.asarray(observed, dtype=float).reshape(-1))
    y = np.sort(np.asarray(simulated, dtype=float).reshape(-1))
    values = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, values, side="right") / max(len(x), 1)
    cdf_y = np.searchsorted(y, values, side="right") / max(len(y), 1)
    return float(np.max(np.abs(cdf_x - cdf_y)))



def wasserstein_distance(observed: np.ndarray, simulated: np.ndarray) -> float:
    x = np.sort(np.asarray(observed, dtype=float).reshape(-1))
    y = np.sort(np.asarray(simulated, dtype=float).reshape(-1))
    n = max(len(x), len(y), 2)
    q = np.linspace(0.0, 1.0, n)
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return float(np.mean(np.abs(xq - yq)))



def compute_distance(
    metric_cfg: dict[str, Any],
    observed_vector: np.ndarray,
    simulated_vector: np.ndarray,
    metric_state: dict[str, Any] | None = None,
) -> float:
    metric = str(metric_cfg.get("metric", "euclidean")).strip().lower()
    obs = np.asarray(observed_vector, dtype=float).reshape(-1)
    sim = np.asarray(simulated_vector, dtype=float).reshape(-1)
    if obs.shape != sim.shape and metric not in {"ks", "kolmogorov_smirnov", "wasserstein"}:
        raise MetricError(f"Observed and simulated summaries must have the same shape. Got {obs.shape} vs {sim.shape}.")
    diff = sim - obs if obs.shape == sim.shape else None
    if metric in {"rmse"}:
        return float(np.sqrt(np.mean(np.square(diff))))
    if metric in {"nrmse"}:
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        denom = float(np.max(obs) - np.min(obs))
        if denom <= 0:
            denom = float(np.std(obs))
        if denom <= 0:
            denom = 1.0
        return rmse / denom
    if metric in {"euclidean", "l2"}:
        return float(np.linalg.norm(diff))
    if metric in {"mahalanobis"}:
        if not metric_state or "cov_inv" not in metric_state:
            raise MetricError("Mahalanobis distance requires fitted metric_state.cov_inv.")
        cov_inv = np.asarray(metric_state["cov_inv"], dtype=float)
        return float(np.sqrt(diff.T @ cov_inv @ diff))
    if metric in {"ks", "kolmogorov_smirnov", "kolmogorov smirnov"}:
        return ks_statistic(obs, sim)
    if metric in {"wasserstein", "earth_movers", "earth mover"}:
        return wasserstein_distance(obs, sim)
    if metric == "custom":
        if metric_cfg.get("custom_python_path") and metric_cfg.get("custom_callable"):
            fn = _load_python_callable(metric_cfg["custom_python_path"], metric_cfg["custom_callable"])
            return float(fn(obs, sim))
        command_template = metric_cfg.get("custom_command_template")
        if command_template:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                observed_path = root / "observed.json"
                simulated_path = root / "simulated.json"
                output_path = root / "distance.json"
                observed_path.write_text(json.dumps(obs.tolist()), encoding="utf-8")
                simulated_path.write_text(json.dumps(sim.tolist()), encoding="utf-8")
                command = command_template.format(
                    observed_json=observed_path,
                    simulated_json=simulated_path,
                    output_json=output_path,
                )
                completed = subprocess.run(command, shell=True, text=True, capture_output=True, check=False)
                if completed.returncode != 0:
                    raise MetricError(f"Custom distance command failed: {completed.stderr.strip()}")
                if output_path.exists():
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    return float(payload["distance"] if isinstance(payload, dict) else payload)
                return float(completed.stdout.strip())
        raise MetricError("Custom distance requires either a Python callable or a command template.")
    raise MetricError(f"Unsupported distance metric: {metric}")



def safe_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise MetricError("Summary vector contains non-finite values.")
    return arr
