"""Prior construction and parsing utilities."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


class PriorError(RuntimeError):
    """Prior parsing or scoring failure."""


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
    "studentt",
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

_COUNT_HINTS = {
    "count",
    "n_",
    "trials",
    "rate",
    "events",
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


def _normalize_dist_name(name: str) -> str:
    dist = str(name).strip().lower()
    if dist == "gaussian":
        return "normal"
    if dist == "studentt":
        return "student_t"
    return dist


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
            raise PriorError(
                "Invalid prior override. Use name=distribution(arg=value,...) or name=uniform(lower,upper)."
            )
        out[match.group("name")] = parse_prior_spec(match.group("spec"))
    return out


def parse_parameter_bounds(items: list[str] | None) -> dict[str, tuple[float | None, float | None]]:
    if not items:
        return {}
    out: dict[str, tuple[float | None, float | None]] = {}
    for item in items:
        if "=" not in item or ":" not in item:
            raise PriorError("Invalid parameter bound. Use name=lower:upper, name=:upper, or name=lower:.")
        name, raw_bounds = item.split("=", 1)
        lower_raw, upper_raw = raw_bounds.split(":", 1)
        lower = float(lower_raw) if lower_raw.strip() else None
        upper = float(upper_raw) if upper_raw.strip() else None
        if lower is not None and upper is not None and not lower < upper:
            raise PriorError(f"Invalid bounds for {name!r}: lower must be < upper.")
        out[name.strip()] = (lower, upper)
    return out


def parse_prior_spec(spec: str) -> dict[str, Any]:
    match = _SPEC_RE.match(spec.strip())
    if not match:
        raise PriorError(f"Invalid prior specification: {spec}")
    dist = _normalize_dist_name(match.group("dist"))
    if dist not in {_normalize_dist_name(item) for item in _ALLOWED_DISTS}:
        raise PriorError(f"Unsupported prior distribution: {dist}")
    body = match.group("body").strip()
    if not body:
        raise PriorError("Prior specification body is empty.")
    if "=" in body:
        params: dict[str, Any] = {}
        for token in [part.strip() for part in body.split(",") if part.strip()]:
            if "=" not in token:
                raise PriorError(f"Invalid prior parameter token: {token}")
            key, value = token.split("=", 1)
            params[key.strip()] = float(value)
    else:
        values = [float(part.strip()) for part in body.split(",") if part.strip()]
        params = positional_prior_params(dist, values)
    return normalize_prior_spec({"dist": dist, "params": params})


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


def normalize_prior_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise PriorError("Prior specification must be an object.")
    dist = _normalize_dist_name(str(spec.get("dist", "")))
    if dist not in {_normalize_dist_name(item) for item in _ALLOWED_DISTS}:
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
            raise PriorError("Beta prior requires lower < upper.")
        params = {"alpha": alpha, "beta": beta, "lower": lower, "upper": upper}
    elif dist == "halfnormal":
        scale = float(params.get("scale", params.get("std", 1.0)))
        lower = float(params.get("lower", 0.0))
        if scale <= 0:
            raise PriorError("Half-normal prior requires scale > 0.")
        params = {"scale": scale, "lower": lower}
    elif dist == "student_t":
        df = float(params.get("df", 4.0))
        loc = float(params.get("loc", params.get("mean", 0.0)))
        scale = float(params.get("scale", params.get("std", 1.0)))
        if df <= 0 or scale <= 0:
            raise PriorError("Student-t prior requires df > 0 and scale > 0.")
        params = {"df": df, "loc": loc, "scale": scale}
    return {"dist": dist, "params": params}


def recommend_prior(
    name: str,
    default: Any = None,
    bounds: tuple[float | None, float | None] | None = None,
) -> dict[str, Any]:
    token = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    lower = bounds[0] if bounds else None
    upper = bounds[1] if bounds else None
    if lower is not None and upper is not None:
        if lower >= 0 and upper <= 1:
            return normalize_prior_spec({"dist": "beta", "params": {"alpha": 2.0, "beta": 2.0, "lower": lower, "upper": upper}})
        return normalize_prior_spec({"dist": "uniform", "params": {"lower": lower, "upper": upper}})
    if lower is not None and upper is None:
        if lower == 0:
            return normalize_prior_spec({"dist": "halfnormal", "params": {"scale": max(1.0, abs(float(default or 1.0))), "lower": 0.0}})
        return normalize_prior_spec({"dist": "lognormal", "params": {"mean": math.log(max(float(default or lower + 1.0), 1e-6)), "sigma": 0.7}})
    if lower is None and upper is not None:
        return normalize_prior_spec({"dist": "uniform", "params": {"lower": upper - 10.0, "upper": upper}})
    if any(hint in token for hint in _PROBABILITY_HINTS):
        return normalize_prior_spec({"dist": "beta", "params": {"alpha": 2.0, "beta": 2.0, "lower": 0.0, "upper": 1.0}})
    if any(hint in token for hint in _POSITIVE_HINTS):
        if isinstance(default, (int, float)) and float(default) > 0:
            return normalize_prior_spec({"dist": "lognormal", "params": {"mean": math.log(float(default)), "sigma": 0.5}})
        return normalize_prior_spec({"dist": "gamma", "params": {"shape": 2.0, "scale": 1.0}})
    if any(hint in token for hint in _COUNT_HINTS):
        location = max(float(default or 1.0), 1.0)
        return normalize_prior_spec({"dist": "gamma", "params": {"shape": max(location, 2.0), "scale": 1.0}})
    if isinstance(default, (int, float)):
        value = float(default)
        std = max(abs(value) * 0.5, 1e-3) if value else 1.0
        if any(hint in token for hint in _NORMAL_HINTS):
            return normalize_prior_spec({"dist": "normal", "params": {"mean": value, "std": std}})
        if abs(value) > 5:
            return normalize_prior_spec({"dist": "student_t", "params": {"df": 4.0, "loc": value, "scale": std}})
        return normalize_prior_spec({"dist": "normal", "params": {"mean": value, "std": std}})
    if any(hint in token for hint in _NORMAL_HINTS):
        return normalize_prior_spec({"dist": "normal", "params": {"mean": 0.0, "std": 1.0}})
    return normalize_prior_spec({"dist": "student_t", "params": {"df": 4.0, "loc": 0.0, "scale": 2.5}})


def prior_support(spec: dict[str, Any]) -> tuple[float | None, float | None]:
    normalized = normalize_prior_spec(spec)
    dist = normalized["dist"]
    params = normalized["params"]
    if dist == "uniform":
        return float(params["lower"]), float(params["upper"])
    if dist == "beta":
        return float(params["lower"]), float(params["upper"])
    if dist == "lognormal":
        return 0.0, None
    if dist == "gamma":
        return 0.0, None
    if dist == "halfnormal":
        return float(params.get("lower", 0.0)), None
    return None, None


def default_point(spec: dict[str, Any]) -> float:
    normalized = normalize_prior_spec(spec)
    dist = normalized["dist"]
    params = normalized["params"]
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
        return params["lower"] + params["scale"] * math.sqrt(2.0 / math.pi)
    if dist == "student_t":
        return params["loc"]
    raise PriorError(f"Unsupported prior distribution: {dist}")


def summarize_priors(priors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, spec in priors.items():
        normalized = normalize_prior_spec(spec)
        out[name] = {
            "dist": normalized["dist"],
            "params": normalized["params"],
            "default_point": default_point(normalized),
            "support": {
                "lower": prior_support(normalized)[0],
                "upper": prior_support(normalized)[1],
            },
        }
    return out


def build_prior_report(
    parameter_names: list[str],
    parameter_defaults: dict[str, Any],
    explicit_priors: dict[str, dict[str, Any]] | None = None,
    parameter_bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    explicit = explicit_priors or {}
    bounds = parameter_bounds or {}
    priors: dict[str, dict[str, Any]] = {}
    provenance: dict[str, str] = {}
    for name in parameter_names:
        if name in explicit:
            priors[name] = normalize_prior_spec(explicit[name])
            provenance[name] = "user"
            continue
        priors[name] = recommend_prior(name, default=parameter_defaults.get(name), bounds=bounds.get(name))
        provenance[name] = "heuristic"
    return {
        "priors": priors,
        "summary": summarize_priors(priors),
        "provenance": provenance,
    }

