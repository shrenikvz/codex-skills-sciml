"""Parameter transform helpers for unconstrained NUTS sampling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .priors import normalize_prior_spec, prior_support


class TransformError(RuntimeError):
    """Transformation failure."""


def _softplus_numpy(value: float | np.ndarray) -> float | np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def recommend_transform(
    parameter_name: str,
    prior_spec: dict[str, Any],
    explicit_constraint: tuple[float | None, float | None] | None = None,
) -> dict[str, Any]:
    normalized = normalize_prior_spec(prior_spec)
    lower, upper = explicit_constraint or prior_support(normalized)
    if lower is not None and upper is not None:
        return {
            "kind": "logit",
            "lower": float(lower),
            "upper": float(upper),
            "reason": "bounded_support",
        }
    if normalized["dist"] in {"lognormal", "gamma"} or (lower is not None and float(lower) > 0):
        return {
            "kind": "log",
            "lower": float(lower or 0.0),
            "upper": None,
            "reason": "strictly_positive_support",
        }
    if normalized["dist"] == "halfnormal" or (lower is not None and float(lower) == 0.0):
        return {
            "kind": "softplus",
            "lower": float(lower or 0.0),
            "upper": None,
            "reason": "non_negative_support",
        }
    return {
        "kind": "identity",
        "lower": lower,
        "upper": upper,
        "reason": f"unconstrained_{parameter_name}",
    }


def build_transform_specs(
    parameter_names: list[str],
    priors: dict[str, dict[str, Any]],
    parameter_constraints: dict[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, dict[str, Any]]:
    constraints = parameter_constraints or {}
    return {
        name: recommend_transform(name, priors[name], explicit_constraint=constraints.get(name))
        for name in parameter_names
    }


def unconstrained_to_constrained_numpy(value: float, spec: dict[str, Any]) -> float:
    kind = spec["kind"]
    lower = spec.get("lower")
    upper = spec.get("upper")
    if kind == "identity":
        return float(value)
    if kind == "log":
        shift = float(lower or 0.0)
        return float(shift + math.exp(float(value)))
    if kind == "softplus":
        shift = float(lower or 0.0)
        return float(shift + _softplus_numpy(float(value)))
    if kind == "logit":
        if lower is None or upper is None:
            raise TransformError("Logit transform requires finite lower and upper bounds.")
        sigmoid = 1.0 / (1.0 + math.exp(-float(value)))
        return float(lower + (upper - lower) * sigmoid)
    raise TransformError(f"Unsupported transform kind: {kind}")


def constrained_to_unconstrained_numpy(value: float, spec: dict[str, Any]) -> float:
    kind = spec["kind"]
    lower = spec.get("lower")
    upper = spec.get("upper")
    x = float(value)
    if kind == "identity":
        return x
    if kind == "log":
        shift = float(lower or 0.0)
        return math.log(max(x - shift, 1e-12))
    if kind == "softplus":
        shift = float(lower or 0.0)
        adjusted = max(x - shift, 1e-12)
        return math.log(math.expm1(adjusted))
    if kind == "logit":
        if lower is None or upper is None:
            raise TransformError("Logit transform requires finite lower and upper bounds.")
        unit = min(max((x - lower) / (upper - lower), 1e-8), 1.0 - 1e-8)
        return math.log(unit) - math.log1p(-unit)
    raise TransformError(f"Unsupported transform kind: {kind}")


def log_abs_det_jacobian_numpy(value: float, spec: dict[str, Any]) -> float:
    kind = spec["kind"]
    lower = spec.get("lower")
    upper = spec.get("upper")
    z = float(value)
    if kind == "identity":
        return 0.0
    if kind == "log":
        return z
    if kind == "softplus":
        return -math.log1p(math.exp(-z))
    if kind == "logit":
        if lower is None or upper is None:
            raise TransformError("Logit transform requires finite lower and upper bounds.")
        sigmoid = 1.0 / (1.0 + math.exp(-z))
        return math.log(upper - lower) + math.log(sigmoid) + math.log1p(-sigmoid)
    raise TransformError(f"Unsupported transform kind: {kind}")


def vector_to_parameter_dict_numpy(
    unconstrained: np.ndarray,
    parameter_names: list[str],
    transform_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], float]:
    params: dict[str, float] = {}
    log_det = 0.0
    for idx, name in enumerate(parameter_names):
        spec = transform_specs[name]
        value = unconstrained_to_constrained_numpy(float(unconstrained[idx]), spec)
        params[name] = value
        log_det += log_abs_det_jacobian_numpy(float(unconstrained[idx]), spec)
    return params, float(log_det)


def default_unconstrained_position(
    parameter_names: list[str],
    default_points: dict[str, float],
    transform_specs: dict[str, dict[str, Any]],
) -> np.ndarray:
    values = []
    for name in parameter_names:
        values.append(constrained_to_unconstrained_numpy(float(default_points[name]), transform_specs[name]))
    return np.asarray(values, dtype=float)

