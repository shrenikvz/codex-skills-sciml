"""Parameter transformation utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .priors import normalize_prior_spec


class TransformError(RuntimeError):
    """Transformation failure."""


def _softplus(x, xp):
    return xp.log1p(xp.exp(-xp.abs(x))) + xp.maximum(x, 0)


def _sigmoid(x, xp):
    return 1.0 / (1.0 + xp.exp(-x))


def infer_transform(spec: dict[str, Any]) -> dict[str, Any]:
    spec = normalize_prior_spec(spec)
    if spec.get("transform"):
        return {"kind": spec["transform"]}
    dist = spec["dist"]
    params = spec["params"]
    if dist in {"lognormal", "gamma"}:
        return {"kind": "log", "lower": 0.0}
    if dist == "halfnormal":
        return {"kind": "softplus", "lower": 0.0}
    if dist == "beta":
        return {"kind": "logit", "lower": params["lower"], "upper": params["upper"]}
    if dist == "uniform":
        lower = params["lower"]
        upper = params["upper"]
        if math.isfinite(lower) and math.isfinite(upper):
            return {"kind": "logit", "lower": lower, "upper": upper}
        if lower >= 0:
            return {"kind": "softplus", "lower": lower}
    return {"kind": "identity"}


def build_transform_specs(priors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: infer_transform(spec) for name, spec in priors.items()}


def forward_transform(value: float | np.ndarray, spec: dict[str, Any], xp=np):
    kind = spec.get("kind", "identity")
    if kind == "identity":
        return value
    if kind == "log":
        return xp.exp(value) + float(spec.get("lower", 0.0))
    if kind == "softplus":
        return float(spec.get("lower", 0.0)) + _softplus(value, xp)
    if kind == "logit":
        lower = float(spec.get("lower", 0.0))
        upper = float(spec.get("upper", 1.0))
        return lower + (upper - lower) * _sigmoid(value, xp)
    raise TransformError(f"Unsupported transform: {kind}")


def inverse_transform(value: float | np.ndarray, spec: dict[str, Any], xp=np):
    kind = spec.get("kind", "identity")
    if kind == "identity":
        return value
    if kind == "log":
        lower = float(spec.get("lower", 0.0))
        return xp.log(xp.maximum(value - lower, 1e-12))
    if kind == "softplus":
        lower = float(spec.get("lower", 0.0))
        shifted = xp.maximum(value - lower, 1e-12)
        return xp.log(xp.exp(shifted) - 1.0)
    if kind == "logit":
        lower = float(spec.get("lower", 0.0))
        upper = float(spec.get("upper", 1.0))
        scaled = xp.clip((value - lower) / (upper - lower), 1e-8, 1 - 1e-8)
        return xp.log(scaled) - xp.log1p(-scaled)
    raise TransformError(f"Unsupported transform: {kind}")


def log_abs_det_jacobian(value: float | np.ndarray, spec: dict[str, Any], xp=np):
    kind = spec.get("kind", "identity")
    if kind == "identity":
        return xp.zeros_like(value)
    if kind == "log":
        return value
    if kind == "softplus":
        return -_softplus(-value, xp)
    if kind == "logit":
        width = float(spec["upper"]) - float(spec["lower"])
        return xp.log(width) - _softplus(-value, xp) - _softplus(value, xp)
    raise TransformError(f"Unsupported transform: {kind}")


def constrain_array(unconstrained: Any, parameter_names: list[str], transform_specs: dict[str, dict[str, Any]], xp=np) -> dict[str, Any]:
    return {
        name: forward_transform(unconstrained[idx], transform_specs.get(name, {"kind": "identity"}), xp=xp)
        for idx, name in enumerate(parameter_names)
    }


def unconstrain_dict(params: dict[str, float], parameter_names: list[str], transform_specs: dict[str, dict[str, Any]], xp=np):
    return xp.asarray(
        [
            inverse_transform(params[name], transform_specs.get(name, {"kind": "identity"}), xp=xp)
            for name in parameter_names
        ],
        dtype=float,
    )


def jacobian_correction(unconstrained: Any, parameter_names: list[str], transform_specs: dict[str, dict[str, Any]], xp=np):
    pieces = [
        log_abs_det_jacobian(unconstrained[idx], transform_specs.get(name, {"kind": "identity"}), xp=xp)
        for idx, name in enumerate(parameter_names)
    ]
    return xp.sum(xp.asarray(pieces))
