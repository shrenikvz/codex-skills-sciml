"""Likelihood construction and evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .adapters import run_custom_likelihood
from .priors import normalize_prior_spec


class LikelihoodError(RuntimeError):
    """Likelihood construction or evaluation failure."""


def _is_integer_observed(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return np.allclose(values, np.round(values))


def infer_likelihood_family(
    observed_array: np.ndarray,
    requested_family: str | None = None,
) -> dict[str, Any]:
    family = (requested_family or "auto").strip().lower()
    if family not in {"auto", "gaussian", "student_t", "poisson", "binomial", "negative_binomial", "custom"}:
        raise LikelihoodError(f"Unsupported likelihood family: {family}")
    if family != "auto":
        return {"family": family, "questions": []}
    array = np.asarray(observed_array, dtype=float).reshape(-1)
    if array.size == 0:
        return {"family": "gaussian", "questions": []}
    if _is_integer_observed(array) and np.min(array) >= 0:
        if np.max(array) <= 1:
            return {"family": "binomial", "questions": []}
        mean = float(np.mean(array))
        var = float(np.var(array))
        if mean > 0 and var > mean * 1.5:
            return {"family": "negative_binomial", "questions": []}
        return {"family": "poisson", "questions": []}
    return {
        "family": "gaussian",
        "questions": ["If you want a heavy-tailed observation model, specify student_t instead of the Gaussian default."],
    }


def maybe_add_likelihood_priors(
    priors: dict[str, dict[str, Any]],
    likelihood_cfg: dict[str, Any],
    observed_array: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    likelihood = {"family": likelihood_cfg["family"], "params": dict(likelihood_cfg.get("params", {}))}
    out = dict(priors)
    observed_scale = float(np.std(np.asarray(observed_array, dtype=float).reshape(-1))) if np.asarray(observed_array).size else 1.0
    observed_scale = max(observed_scale, 1.0)
    family = likelihood["family"]
    if family in {"gaussian", "student_t"} and likelihood["params"].get("sigma") is None:
        name = likelihood["params"].get("sigma_parameter") or "obs_sigma"
        likelihood["params"]["sigma_parameter"] = name
        out[name] = normalize_prior_spec({"dist": "halfnormal", "params": {"scale": observed_scale}, "source": "likelihood_default"})
    if family == "negative_binomial" and likelihood["params"].get("dispersion") is None:
        name = likelihood["params"].get("dispersion_parameter") or "obs_dispersion"
        likelihood["params"]["dispersion_parameter"] = name
        out[name] = normalize_prior_spec({"dist": "gamma", "params": {"shape": 2.0, "scale": 2.0}, "source": "likelihood_default"})
    if family == "binomial" and likelihood["params"].get("total_count") is None:
        values = np.asarray(observed_array, dtype=float).reshape(-1)
        if np.max(values) <= 1:
            likelihood["params"]["total_count"] = 1
        else:
            likelihood.setdefault("questions", []).append("Binomial likelihood requires total_count when observations exceed 1.")
    return out, likelihood


def align_sigma(params: dict[str, Any], likelihood_cfg: dict[str, Any]) -> float:
    sigma_name = likelihood_cfg.get("params", {}).get("sigma_parameter")
    sigma = likelihood_cfg.get("params", {}).get("sigma")
    if sigma_name:
        sigma = params[sigma_name]
    sigma = float(sigma or 1.0)
    return max(sigma, 1e-8)


def align_dispersion(params: dict[str, Any], likelihood_cfg: dict[str, Any]) -> float:
    disp_name = likelihood_cfg.get("params", {}).get("dispersion_parameter")
    dispersion = likelihood_cfg.get("params", {}).get("dispersion")
    if disp_name:
        dispersion = params[disp_name]
    dispersion = float(dispersion or 1.0)
    return max(dispersion, 1e-8)


def gaussian_logpdf(observed, simulated, sigma, xp=np):
    resid = (observed - simulated) / sigma
    return -0.5 * xp.sum(resid**2 + xp.log(2.0 * xp.pi * sigma**2))


def student_t_logpdf(observed, simulated, sigma, df, xp=np):
    resid = (observed - simulated) / sigma
    return xp.sum(
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * (xp.log(df) + xp.log(xp.pi))
        - xp.log(sigma)
        - 0.5 * (df + 1.0) * xp.log1p((resid**2) / df)
    )


def poisson_logpmf(observed, simulated_rate, xp=np):
    rate = xp.maximum(simulated_rate, 1e-8)
    return xp.sum(observed * xp.log(rate) - rate - _gammaln(observed + 1.0, xp))


def binomial_logpmf(observed, probability, total_count: int, xp=np):
    p = xp.clip(probability, 1e-8, 1 - 1e-8)
    n = float(total_count)
    return xp.sum(
        _gammaln(n + 1.0, xp)
        - _gammaln(observed + 1.0, xp)
        - _gammaln(n - observed + 1.0, xp)
        + observed * xp.log(p)
        + (n - observed) * xp.log1p(-p)
    )


def negative_binomial_logpmf(observed, mean, dispersion, xp=np):
    mu = xp.maximum(mean, 1e-8)
    r = xp.maximum(dispersion, 1e-8)
    p = r / (r + mu)
    return xp.sum(
        _gammaln(observed + r, xp)
        - _gammaln(r, xp)
        - _gammaln(observed + 1.0, xp)
        + r * xp.log(p)
        + observed * xp.log1p(-p)
    )


def _gammaln(value, xp=np):
    vector = np.vectorize(math.lgamma)
    return xp.asarray(vector(np.asarray(value, dtype=float)))


def evaluate_log_likelihood(
    likelihood_cfg: dict[str, Any],
    params: dict[str, float],
    simulated: Any,
    observed: Any,
    metadata: dict[str, Any],
    xp=np,
    workdir=None,
) -> float:
    family = likelihood_cfg["family"]
    observed_arr = xp.asarray(observed, dtype=float).reshape(-1)
    simulated_arr = xp.asarray(simulated, dtype=float).reshape(-1)
    if family == "custom":
        if xp is not np:
            raise LikelihoodError("Custom likelihoods use host execution and are unavailable in direct JAX mode.")
        return float(run_custom_likelihood(likelihood_cfg, params, simulated_arr.tolist(), observed_arr.tolist(), metadata, workdir=workdir))
    if family == "gaussian":
        return gaussian_logpdf(observed_arr, simulated_arr, align_sigma(params, likelihood_cfg), xp=xp)
    if family == "student_t":
        return student_t_logpdf(
            observed_arr,
            simulated_arr,
            align_sigma(params, likelihood_cfg),
            float(likelihood_cfg.get("params", {}).get("df", 5.0)),
            xp=xp,
        )
    if family == "poisson":
        return poisson_logpmf(observed_arr, simulated_arr, xp=xp)
    if family == "binomial":
        total_count = int(likelihood_cfg.get("params", {}).get("total_count", 1))
        return binomial_logpmf(observed_arr, simulated_arr, total_count, xp=xp)
    if family == "negative_binomial":
        return negative_binomial_logpmf(observed_arr, simulated_arr, align_dispersion(params, likelihood_cfg), xp=xp)
    raise LikelihoodError(f"Unsupported likelihood family: {family}")
