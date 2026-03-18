"""Likelihood recommendation and scoring helpers."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class LikelihoodError(RuntimeError):
    """Likelihood selection or evaluation failure."""


SUPPORTED_LIKELIHOODS = [
    "gaussian",
    "student_t",
    "poisson",
    "binomial",
    "negative_binomial",
    "custom_python",
    "custom_command",
]


def normalize_likelihood_name(name: str) -> str:
    token = str(name or "auto").strip().lower()
    if token in {"normal", "gaussian"}:
        return "gaussian"
    if token in {"student-t", "student_t", "studentt"}:
        return "student_t"
    return token


def normalize_likelihood_spec(spec: dict[str, Any]) -> dict[str, Any]:
    name = normalize_likelihood_name(spec.get("name", "auto"))
    params = dict(spec.get("params", {}))
    if name == "auto":
        return {"name": "auto", "params": params}
    if name == "gaussian":
        sigma = float(params.get("sigma", 1.0))
        if sigma <= 0:
            raise LikelihoodError("Gaussian likelihood requires sigma > 0.")
        return {"name": "gaussian", "params": {"sigma": sigma}}
    if name == "student_t":
        sigma = float(params.get("sigma", 1.0))
        df = float(params.get("df", 4.0))
        if sigma <= 0 or df <= 0:
            raise LikelihoodError("Student-t likelihood requires sigma > 0 and df > 0.")
        return {"name": "student_t", "params": {"sigma": sigma, "df": df}}
    if name == "poisson":
        return {"name": "poisson", "params": {}}
    if name == "binomial":
        n_trials = int(params.get("n_trials", 1))
        if n_trials <= 0:
            raise LikelihoodError("Binomial likelihood requires n_trials > 0.")
        return {"name": "binomial", "params": {"n_trials": n_trials}}
    if name == "negative_binomial":
        dispersion = float(params.get("dispersion", 5.0))
        if dispersion <= 0:
            raise LikelihoodError("Negative binomial likelihood requires dispersion > 0.")
        return {"name": "negative_binomial", "params": {"dispersion": dispersion}}
    if name == "custom_python":
        return {
            "name": "custom_python",
            "params": params,
            "custom_python_path": spec.get("custom_python_path"),
            "custom_callable": spec.get("custom_callable"),
        }
    if name == "custom_command":
        return {
            "name": "custom_command",
            "params": params,
            "custom_command_template": spec.get("custom_command_template"),
        }
    raise LikelihoodError(f"Unsupported likelihood: {name}")


def recommend_likelihood(
    observed_array: np.ndarray,
    request_text: str | None = None,
    requested_name: str | None = None,
) -> dict[str, Any]:
    requested = normalize_likelihood_name(requested_name or "auto")
    text = (request_text or "").lower()
    finite = np.asarray(observed_array, dtype=float).reshape(-1)
    if finite.size == 0:
        raise LikelihoodError("Observed data is empty.")
    if requested != "auto":
        spec = normalize_likelihood_spec({"name": requested, "params": {}})
        return {"spec": spec, "reason": "user_preference", "needs_confirmation": False}
    if "student" in text or "heavy tail" in text or "outlier" in text:
        scale = max(float(np.std(finite)), 1e-3)
        return {
            "spec": normalize_likelihood_spec({"name": "student_t", "params": {"sigma": scale, "df": 4.0}}),
            "reason": "robust_noise_hint",
            "needs_confirmation": False,
        }
    is_integer = np.allclose(finite, np.round(finite))
    is_binary = is_integer and set(np.unique(finite)).issubset({0.0, 1.0})
    is_non_negative = bool(np.all(finite >= 0))
    if is_binary:
        return {
            "spec": normalize_likelihood_spec({"name": "binomial", "params": {"n_trials": 1}}),
            "reason": "binary_observations",
            "needs_confirmation": False,
        }
    if is_integer and is_non_negative:
        mean = float(np.mean(finite))
        variance = float(np.var(finite))
        if variance > mean + 1e-6:
            dispersion = max(1.0, mean**2 / max(variance - mean, 1e-6))
            return {
                "spec": normalize_likelihood_spec({"name": "negative_binomial", "params": {"dispersion": dispersion}}),
                "reason": "overdispersed_counts",
                "needs_confirmation": True,
            }
        return {
            "spec": normalize_likelihood_spec({"name": "poisson", "params": {}}),
            "reason": "count_observations",
            "needs_confirmation": True,
        }
    scale = max(float(np.std(finite)), 1e-3)
    return {
        "spec": normalize_likelihood_spec({"name": "gaussian", "params": {"sigma": scale}}),
        "reason": "continuous_default",
        "needs_confirmation": True,
    }


def loglikelihood_numpy(
    observed: np.ndarray,
    simulated: np.ndarray,
    spec: dict[str, Any],
    params: dict[str, float] | None = None,
) -> float:
    normalized = normalize_likelihood_spec(spec)
    observed = np.asarray(observed, dtype=float).reshape(-1)
    simulated = np.asarray(simulated, dtype=float).reshape(-1)
    if observed.shape != simulated.shape:
        raise LikelihoodError(f"Observed and simulated arrays must have matching shapes, got {observed.shape} and {simulated.shape}.")
    name = normalized["name"]
    if name == "gaussian":
        sigma = normalized["params"]["sigma"]
        resid = (observed - simulated) / sigma
        return float(-0.5 * np.sum(resid**2 + math.log(2.0 * math.pi * sigma * sigma)))
    if name == "student_t":
        sigma = normalized["params"]["sigma"]
        df = normalized["params"]["df"]
        resid = (observed - simulated) / sigma
        log_norm = (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi)
            - math.log(sigma)
        )
        return float(np.sum(log_norm - 0.5 * (df + 1.0) * np.log1p((resid**2) / df)))
    if name == "poisson":
        rate = np.clip(simulated, 1e-8, None)
        return float(np.sum(observed * np.log(rate) - rate - np.vectorize(math.lgamma)(observed + 1.0)))
    if name == "binomial":
        n_trials = normalized["params"]["n_trials"]
        prob = np.clip(simulated, 1e-8, 1.0 - 1e-8)
        coeff = np.vectorize(math.lgamma)(n_trials + 1.0) - np.vectorize(math.lgamma)(observed + 1.0) - np.vectorize(math.lgamma)(n_trials - observed + 1.0)
        return float(np.sum(coeff + observed * np.log(prob) + (n_trials - observed) * np.log1p(-prob)))
    if name == "negative_binomial":
        dispersion = normalized["params"]["dispersion"]
        mean = np.clip(simulated, 1e-8, None)
        total_count = dispersion
        probs = total_count / (total_count + mean)
        return float(
            np.sum(
                np.vectorize(math.lgamma)(observed + total_count)
                - np.vectorize(math.lgamma)(total_count)
                - np.vectorize(math.lgamma)(observed + 1.0)
                + total_count * np.log(probs)
                + observed * np.log1p(-probs)
            )
        )
    if name == "custom_python":
        return run_custom_python_likelihood(
            normalized["custom_python_path"],
            normalized["custom_callable"],
            observed,
            simulated,
            normalized,
            params or {},
        )
    if name == "custom_command":
        return run_custom_command_likelihood(
            normalized["custom_command_template"],
            observed,
            simulated,
            normalized,
            params or {},
        )
    raise LikelihoodError(f"Unsupported likelihood: {name}")


def run_custom_python_likelihood(
    python_path: str | None,
    callable_name: str | None,
    observed: np.ndarray,
    simulated: np.ndarray,
    spec: dict[str, Any],
    params: dict[str, float],
) -> float:
    if not python_path or not callable_name:
        raise LikelihoodError("Custom Python likelihood requires custom_python_path and custom_callable.")
    import importlib.util

    target = Path(python_path).expanduser().resolve()
    spec_obj = importlib.util.spec_from_file_location(target.stem, target)
    if spec_obj is None or spec_obj.loader is None:
        raise LikelihoodError(f"Could not load custom likelihood module: {target}")
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    fn = getattr(module, callable_name, None)
    if not callable(fn):
        raise LikelihoodError(f"Custom likelihood callable not found: {callable_name}")
    value = fn(observed, simulated, spec, params)
    return float(value)


def run_custom_command_likelihood(
    command_template: str | None,
    observed: np.ndarray,
    simulated: np.ndarray,
    spec: dict[str, Any],
    params: dict[str, float],
) -> float:
    if not command_template:
        raise LikelihoodError("Custom command likelihood requires custom_command_template.")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        observed_path = root / "observed.json"
        simulated_path = root / "simulated.json"
        spec_path = root / "likelihood.json"
        observed_path.write_text(json.dumps(observed.tolist()), encoding="utf-8")
        simulated_path.write_text(json.dumps(simulated.tolist()), encoding="utf-8")
        spec_path.write_text(json.dumps({"spec": spec, "params": params}), encoding="utf-8")
        command = command_template.format(
            observed_json=str(observed_path),
            simulated_json=str(simulated_path),
            likelihood_json=str(spec_path),
        )
        completed = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise LikelihoodError(completed.stderr.strip() or completed.stdout.strip() or "Custom likelihood command failed.")
        output = completed.stdout.strip()
        return float(output)

