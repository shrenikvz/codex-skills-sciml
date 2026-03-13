"""Prior construction and reporting utilities."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


class PriorError(RuntimeError):
    """Prior parsing or sampling failure."""


_PRIOR_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<spec>[A-Za-z_][A-Za-z0-9_]*\(.*\))$")
_SPEC_RE = re.compile(r"^(?P<dist>[A-Za-z_][A-Za-z0-9_]*)\((?P<body>.*)\)$")

_ALLOWED_DISTS = {
    "uniform",
    "normal",
    "gaussian",
    "lognormal",
    "gamma",
    "beta",
    "halfnormal",
    "student_t",
}

_POSITIVE_HINTS = {
    "sigma",
    "std",
    "scale",
    "rate",
    "lambda",
    "tau",
    "kappa",
    "variance",
    "var",
    "lengthscale",
    "amplitude",
    "mass",
    "concentration",
    "decay",
    "half_life",
}

_PROBABILITY_HINTS = {
    "prob",
    "probability",
    "fraction",
    "frac",
    "mix",
    "mixing",
    "weight",
    "portion",
}

_NORMAL_HINTS = {
    "mean",
    "mu",
    "offset",
    "bias",
    "intercept",
    "location",
    "loc",
}


def load_prior_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PriorError("Prior file must contain a JSON object keyed by parameter name.")
    return {str(key): normalize_prior_spec(value) for key, value in payload.items()}


def parse_prior_overrides(items: list[str] | None) -> dict[str, dict[str, Any]]:
    if not items:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        match = _PRIOR_RE.match(item.strip())
        if not match:
            raise PriorError("Invalid prior override. Use name=distribution(arg=value,...) syntax.")
        out[match.group("name")] = parse_prior_spec(match.group("spec"))
    return out


def parse_parameter_bounds(items: list[str] | None) -> dict[str, tuple[float, float]]:
    if not items:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for item in items:
        if "=" not in item or ":" not in item:
            raise PriorError("Invalid parameter bound. Use name=lower:upper.")
        name, raw_bounds = item.split("=", 1)
        lower_raw, upper_raw = raw_bounds.split(":", 1)
        lower = float(lower_raw)
        upper = float(upper_raw)
        if not lower < upper:
            raise PriorError(f"Invalid bounds for {name!r}: lower must be < upper.")
        out[name.strip()] = (lower, upper)
    return out


def positional_prior_params(dist: str, values: list[float]) -> dict[str, Any]:
    if dist == "uniform" and len(values) == 2:
        return {"lower": values[0], "upper": values[1]}
    if dist == "normal" and len(values) == 2:
        return {"mean": values[0], "std": values[1]}
    if dist == "lognormal" and len(values) == 2:
        return {"mean": values[0], "sigma": values[1]}
    if dist == "gamma" and len(values) == 2:
        return {"shape": values[0], "scale": values[1]}
    if dist == "beta" and len(values) == 2:
        return {"alpha": values[0], "beta": values[1]}
    if dist == "halfnormal" and len(values) == 1:
        return {"scale": values[0]}
    if dist == "student_t" and len(values) == 3:
        return {"df": values[0], "loc": values[1], "scale": values[2]}
    raise PriorError(f"Unsupported positional parameterization for {dist}.")


def parse_prior_spec(spec: str) -> dict[str, Any]:
    match = _SPEC_RE.match(spec.strip())
    if not match:
        raise PriorError(f"Invalid prior specification: {spec}")
    dist = match.group("dist").lower()
    body = match.group("body").strip()
    if not body:
        raise PriorError("Prior specification body is empty.")
    if "=" in body:
        params = {}
        for token in [part.strip() for part in body.split(",") if part.strip()]:
            if "=" not in token:
                raise PriorError(f"Invalid prior parameter token: {token}")
            key, value = token.split("=", 1)
            params[key.strip()] = float(value)
    else:
        params = positional_prior_params(dist, [float(part.strip()) for part in body.split(",") if part.strip()])
    return normalize_prior_spec({"dist": dist, "params": params})


def normalize_prior_spec(spec: dict[str, Any]) -> dict[str, Any]:
    dist = str(spec.get("dist", "")).strip().lower()
    if dist == "gaussian":
        dist = "normal"
    if dist not in _ALLOWED_DISTS:
        raise PriorError(f"Unsupported prior distribution: {dist}")
    params = dict(spec.get("params", {}))
    if dist == "uniform":
        lower = float(params.get("lower"))
        upper = float(params.get("upper"))
        if not lower < upper:
            raise PriorError("Uniform prior requires lower < upper.")
        params = {"lower": lower, "upper": upper}
    elif dist == "normal":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        if std <= 0:
            raise PriorError("Normal prior requires std > 0.")
        params = {"mean": mean, "std": std}
    elif dist == "lognormal":
        mean = float(params.get("mean", 0.0))
        sigma = float(params.get("sigma", params.get("std", 1.0)))
        if sigma <= 0:
            raise PriorError("Lognormal prior requires sigma > 0.")
        params = {"mean": mean, "sigma": sigma}
    elif dist == "gamma":
        shape = float(params.get("shape", 2.0))
        scale = float(params.get("scale", 1.0))
        if shape <= 0 or scale <= 0:
            raise PriorError("Gamma prior requires shape > 0 and scale > 0.")
        params = {"shape": shape, "scale": scale}
    elif dist == "beta":
        alpha = float(params.get("alpha", 2.0))
        beta = float(params.get("beta", 2.0))
        lower = float(params.get("lower", 0.0))
        upper = float(params.get("upper", 1.0))
        if alpha <= 0 or beta <= 0:
            raise PriorError("Beta prior requires alpha > 0 and beta > 0.")
        if not lower < upper:
            raise PriorError("Scaled beta prior requires lower < upper.")
        params = {"alpha": alpha, "beta": beta, "lower": lower, "upper": upper}
    elif dist == "halfnormal":
        scale = float(params.get("scale", 1.0))
        if scale <= 0:
            raise PriorError("Half-normal prior requires scale > 0.")
        params = {"scale": scale}
    elif dist == "student_t":
        df = float(params.get("df", 5.0))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        if df <= 0 or scale <= 0:
            raise PriorError("Student-t prior requires df > 0 and scale > 0.")
        params = {"df": df, "loc": loc, "scale": scale}
    normalized = {"dist": dist, "params": params}
    if "transform" in spec:
        normalized["transform"] = str(spec["transform"])
    if "source" in spec:
        normalized["source"] = str(spec["source"])
    return normalized


def recommend_prior(name: str, default: Any = None, bounds: tuple[float, float] | None = None) -> dict[str, Any]:
    token = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if bounds is not None:
        lower, upper = bounds
        if lower >= 0 and upper <= 1:
            return normalize_prior_spec(
                {"dist": "beta", "params": {"alpha": 2.0, "beta": 2.0, "lower": lower, "upper": upper}, "source": "bounds"}
            )
        if lower >= 0:
            width = max(upper - lower, 1e-3)
            mean = math.log(max(lower + 0.5 * width, 1e-6))
            return normalize_prior_spec({"dist": "lognormal", "params": {"mean": mean, "sigma": 1.0}, "source": "bounds"})
        return normalize_prior_spec({"dist": "uniform", "params": {"lower": lower, "upper": upper}, "source": "bounds"})
    if any(hint in token for hint in _PROBABILITY_HINTS):
        return normalize_prior_spec({"dist": "beta", "params": {"alpha": 2.0, "beta": 2.0}, "source": "name_heuristic"})
    if any(hint in token for hint in _POSITIVE_HINTS):
        if isinstance(default, (int, float)) and float(default) > 0:
            return normalize_prior_spec(
                {"dist": "lognormal", "params": {"mean": math.log(float(default)), "sigma": 0.5}, "source": "default_value"}
            )
        return normalize_prior_spec({"dist": "gamma", "params": {"shape": 2.0, "scale": 1.0}, "source": "name_heuristic"})
    if isinstance(default, (int, float)):
        value = float(default)
        std = max(abs(value) * 0.5, 1e-3) if value != 0 else 1.0
        return normalize_prior_spec({"dist": "normal", "params": {"mean": value, "std": std}, "source": "default_value"})
    if any(hint in token for hint in _NORMAL_HINTS):
        return normalize_prior_spec({"dist": "normal", "params": {"mean": 0.0, "std": 1.0}, "source": "name_heuristic"})
    return normalize_prior_spec({"dist": "student_t", "params": {"df": 5.0, "loc": 0.0, "scale": 2.5}, "source": "default"})


def default_point(spec: dict[str, Any]) -> float:
    spec = normalize_prior_spec(spec)
    dist = spec["dist"]
    params = spec["params"]
    if dist == "uniform":
        return 0.5 * (params["lower"] + params["upper"])
    if dist == "normal":
        return params["mean"]
    if dist == "lognormal":
        return math.exp(params["mean"])
    if dist == "gamma":
        return params["shape"] * params["scale"]
    if dist == "beta":
        mean = params["alpha"] / (params["alpha"] + params["beta"])
        return params["lower"] + (params["upper"] - params["lower"]) * mean
    if dist == "halfnormal":
        return params["scale"] * math.sqrt(2 / math.pi)
    if dist == "student_t":
        return params["loc"]
    raise PriorError(f"Unsupported prior distribution: {dist}")


def sample_prior(spec: dict[str, Any], rng: np.random.Generator) -> float:
    spec = normalize_prior_spec(spec)
    dist = spec["dist"]
    params = spec["params"]
    if dist == "uniform":
        return float(rng.uniform(params["lower"], params["upper"]))
    if dist == "normal":
        return float(rng.normal(params["mean"], params["std"]))
    if dist == "lognormal":
        return float(rng.lognormal(params["mean"], params["sigma"]))
    if dist == "gamma":
        return float(rng.gamma(params["shape"], params["scale"]))
    if dist == "beta":
        unit = float(rng.beta(params["alpha"], params["beta"]))
        return float(params["lower"] + (params["upper"] - params["lower"]) * unit)
    if dist == "halfnormal":
        return float(abs(rng.normal(0.0, params["scale"])))
    if dist == "student_t":
        return float(params["loc"] + params["scale"] * rng.standard_t(params["df"]))
    raise PriorError(f"Unsupported prior distribution: {dist}")


def sample_prior_dict(priors: dict[str, dict[str, Any]], rng: np.random.Generator) -> dict[str, float]:
    return {name: sample_prior(spec, rng) for name, spec in priors.items()}


def summarize_priors(priors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, spec in priors.items():
        normalized = normalize_prior_spec(spec)
        out[name] = {
            "dist": normalized["dist"],
            "params": normalized["params"],
            "default_point": default_point(normalized),
            "transform": normalized.get("transform"),
            "source": normalized.get("source", "user"),
        }
    return out


def build_prior_report(
    parameter_names: list[str],
    parameter_defaults: dict[str, Any],
    explicit_priors: dict[str, dict[str, Any]] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    explicit_priors = explicit_priors or {}
    parameter_bounds = parameter_bounds or {}
    priors: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for name in parameter_names:
        if name in explicit_priors:
            spec = normalize_prior_spec(explicit_priors[name])
            provenance[name] = spec.get("source", "user")
        else:
            spec = recommend_prior(name, parameter_defaults.get(name), parameter_bounds.get(name))
            provenance[name] = spec.get("source", "heuristic")
        priors[name] = spec
    return {
        "priors": priors,
        "summary": summarize_priors(priors),
        "provenance": provenance,
    }
