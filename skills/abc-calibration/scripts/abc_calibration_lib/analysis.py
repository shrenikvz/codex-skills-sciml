"""Model inspection and recommendation helpers."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import load_observed_data, payload_to_array
from .priors import apply_exact_bounds, bounds_match, default_point, extract_prior_bounds, summarize_priors


class AnalysisError(RuntimeError):
    """Inspection or recommendation failure."""


_MATH_TOKENS = {
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "sqrt",
    "pi",
    "abs",
    "minimum",
    "maximum",
    "np",
    "numpy",
    "math",
}


LIKELIHOOD_AVAILABLE_HINTS = {
    "gaussian noise",
    "normal noise",
    "poisson",
    "binomial",
    "negative binomial",
    "bernoulli",
    "log likelihood",
    "likelihood",
    "closed-form",
    "closed form",
}

LIKELIHOOD_INTRACTABLE_HINTS = {
    "simulator",
    "agent-based",
    "agent based",
    "stochastic simulator",
    "likelihood-free",
    "likelihood free",
    "abc",
    "intractable",
    "black box",
    "digital twin",
}


RUNTIME_BY_SUFFIX = {
    ".py": "python3",
    ".jl": "julia",
    ".r": "Rscript",
    ".R": "Rscript",
    ".m": "octave --quiet",
    ".sh": "bash",
}



def _load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"Could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def infer_equation_parameters(expression: str) -> list[str]:
    tokens = sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression)))
    return [token for token in tokens if token not in _MATH_TOKENS and token.lower() not in _MATH_TOKENS]



def infer_adapter(model_path: str | None, command_template: str | None, equation_text: str | None) -> str:
    if equation_text:
        return "python_callable"
    if command_template:
        return "command"
    if model_path:
        suffix = Path(model_path).suffix.lower()
        if suffix == ".py":
            return "python_callable"
        return "command"
    raise AnalysisError("One of model_path, command_template, or equation_text is required.")



def inspect_python_model(
    path: str,
    callable_name: str | None = None,
    priors: dict[str, Any] | None = None,
    user_parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise AnalysisError(f"Model path does not exist: {target}")
    module = _load_python_module(target)
    public_functions = []
    for name, value in inspect.getmembers(module, inspect.isfunction):
        if value.__module__ == module.__name__ and not name.startswith("_"):
            public_functions.append(name)
    chosen = callable_name
    if not chosen:
        if len(public_functions) == 1:
            chosen = public_functions[0]
        elif "simulate" in public_functions:
            chosen = "simulate"
        elif "model" in public_functions:
            chosen = "model"
    if not chosen:
        raise AnalysisError(
            f"Could not infer callable from {target.name}. Available functions: {public_functions or 'none'}"
        )
    fn = getattr(module, chosen, None)
    if not callable(fn):
        raise AnalysisError(f"Callable {chosen!r} not found in {target}")
    signature = inspect.signature(fn)
    signature_parameter_names: list[str] = []
    defaults: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}:
            continue
        signature_parameter_names.append(name)
        if parameter.default is not inspect._empty:
            defaults[name] = parameter.default
    source = target.read_text(encoding="utf-8", errors="ignore").lower()
    stochastic = any(token in source for token in ["np.random", "random.", "rng", "stochastic"])
    call_style = "kwargs"
    if len(signature_parameter_names) == 1 and signature_parameter_names[0] in {"params", "parameters"}:
        call_style = "mapping"
    parameter_names = list(signature_parameter_names)
    if user_parameter_names:
        requested_names = [name for name in user_parameter_names]
        if call_style != "mapping":
            missing = [name for name in requested_names if name not in set(signature_parameter_names)]
            if missing:
                raise AnalysisError(
                    f"User-requested calibration parameters are not accepted by callable {chosen!r}: {missing}"
                )
        parameter_names = requested_names

    probe_point: dict[str, float] | None = None
    output_names: list[str] = []
    probe_shape: list[int] | None = None
    if all(name in defaults or (priors and name in priors) for name in parameter_names):
        probe_point = {}
        for name in parameter_names:
            if name in defaults and isinstance(defaults[name], (int, float)):
                probe_point[name] = float(defaults[name])
            elif priors and name in priors:
                probe_point[name] = float(default_point(priors[name]))
        if probe_point and len(probe_point) == len(parameter_names):
            try:
                if call_style == "mapping":
                    payload = fn(probe_point)
                else:
                    payload = fn(**probe_point)
                _, output_names = payload_to_array(payload)
                probe_shape = list(np.asarray(payload if not isinstance(payload, dict) else list(payload.values())[0]).shape)
            except Exception:
                output_names = []
                probe_shape = None

    return {
        "adapter": "python_callable",
        "path": str(target),
        "callable": chosen,
        "call_style": call_style,
        "parameter_names": parameter_names,
        "parameter_defaults": {name: defaults[name] for name in parameter_names if name in defaults},
        "available_parameter_names": signature_parameter_names,
        "public_functions": public_functions,
        "stochastic": stochastic,
        "output_names": output_names,
        "probe_point": probe_point,
        "probe_shape": probe_shape,
    }



def inspect_model_file(
    model_path: str | None,
    command_template: str | None = None,
    equation_text: str | None = None,
    callable_name: str | None = None,
    user_parameter_names: list[str] | None = None,
    user_priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = infer_adapter(model_path, command_template, equation_text)
    if equation_text:
        parameter_names = user_parameter_names or infer_equation_parameters(equation_text)
        priors = user_priors or {}
        return {
            "adapter": "python_callable",
            "path": None,
            "callable": "simulate",
            "call_style": "kwargs",
            "parameter_names": parameter_names,
            "parameter_defaults": {},
            "public_functions": ["simulate"],
            "stochastic": False,
            "output_names": [],
            "probe_point": {name: default_point(priors[name]) for name in parameter_names if name in priors} or None,
            "probe_shape": None,
            "generated_from_equation": True,
        }
    if adapter == "python_callable" and model_path:
        return inspect_python_model(
            model_path,
            callable_name=callable_name,
            priors=user_priors,
            user_parameter_names=user_parameter_names,
        )
    if model_path:
        target = Path(model_path).expanduser().resolve()
        suffix = target.suffix
        runtime = RUNTIME_BY_SUFFIX.get(suffix, "bash")
        wrapper_template = f"{runtime} {{model_path}} {{params_json}} {{output_json}}"
        return {
            "adapter": "command",
            "path": str(target),
            "callable": None,
            "call_style": None,
            "parameter_names": user_parameter_names or [],
            "parameter_defaults": {},
            "public_functions": [],
            "stochastic": None,
            "output_names": [],
            "probe_point": None,
            "probe_shape": None,
            "command_template": command_template or wrapper_template,
            "requires_wrapper_confirmation": True,
            "runtime_hint": runtime,
        }
    raise AnalysisError("Could not inspect model inputs.")



def infer_observed_outputs(model_output_names: list[str], observed_column_names: list[str], requested: list[str] | None = None) -> dict[str, Any]:
    if requested:
        return {
            "selected": [name for name in requested],
            "questions": [],
            "selected_by": "explicit",
        }
    if model_output_names and observed_column_names:
        overlap = [name for name in observed_column_names if name in set(model_output_names)]
        if overlap:
            return {
                "selected": overlap,
                "questions": [],
                "selected_by": "name_intersection",
            }
    if len(observed_column_names) == 1:
        return {
            "selected": observed_column_names,
            "questions": [],
            "selected_by": "single_observed_column",
        }
    question = "Which model outputs correspond to the observed calibration data?"
    if observed_column_names:
        question += f" Observed columns: {observed_column_names}."
    if model_output_names:
        question += f" Candidate model outputs: {model_output_names}."
    return {
        "selected": [],
        "questions": [question],
        "selected_by": "ambiguous",
    }



def assess_likelihood(
    request_text: str | None,
    model_analysis: dict[str, Any],
    likelihood_hint: str | None,
) -> dict[str, Any]:
    hint = (likelihood_hint or "auto").strip().lower()
    if hint in {"available", "tractable"}:
        return {
            "status": "likely_available",
            "recommendation": "likelihood_based_inference",
            "reasons": ["User provided an explicit likelihood availability hint."],
        }
    if hint in {"intractable", "unavailable", "likelihood_free"}:
        return {
            "status": "intractable_or_unavailable",
            "recommendation": "abc_rejection",
            "reasons": ["User explicitly marked the likelihood as intractable or unavailable."],
        }

    text = " ".join(
        part for part in [request_text or "", " ".join(model_analysis.get("public_functions", []))] if part
    ).lower()
    reasons: list[str] = []
    if any(token in text for token in LIKELIHOOD_INTRACTABLE_HINTS):
        reasons.append("Request or model description contains likelihood-free or simulator keywords.")
        status = "intractable_or_unavailable"
        recommendation = "abc_rejection"
    elif any(token in text for token in LIKELIHOOD_AVAILABLE_HINTS):
        reasons.append("Request contains closed-form likelihood or standard observation model keywords.")
        status = "likely_available"
        recommendation = "likelihood_based_inference"
    elif model_analysis.get("stochastic"):
        reasons.append("Model inspection found stochastic simulator hints without an explicit likelihood expression.")
        status = "unknown_but_likely_intractable"
        recommendation = "abc_rejection"
    else:
        reasons.append("No explicit closed-form likelihood was identified during inspection.")
        status = "unknown"
        recommendation = "abc_rejection"
    return {
        "status": status,
        "recommendation": recommendation,
        "reasons": reasons,
    }



def recommend_scaling_mode(observed_array: np.ndarray, scaling_preference: str | None = None) -> dict[str, Any]:
    if scaling_preference and scaling_preference not in {"auto", "none", "zscore", "minmax", "variance"}:
        raise AnalysisError(f"Unsupported scaling mode: {scaling_preference}")
    if scaling_preference and scaling_preference != "auto":
        return {"enabled": scaling_preference != "none", "mode": scaling_preference}
    vector = np.asarray(observed_array, dtype=float).reshape(-1)
    if vector.size <= 1:
        return {"enabled": False, "mode": "none"}
    std = float(np.std(vector))
    span = float(np.max(vector) - np.min(vector))
    magnitude = float(np.max(np.abs(vector))) if vector.size else 0.0
    if std > 0 and magnitude / max(std, 1e-12) > 10:
        return {"enabled": True, "mode": "zscore"}
    if span > 0 and (np.max(vector) > 1 or np.min(vector) < 0):
        return {"enabled": True, "mode": "minmax"}
    return {"enabled": True, "mode": "variance"}



def recommend_distance_metric(observed_array: np.ndarray, requested_metric: str | None = None) -> str:
    if requested_metric and requested_metric != "auto":
        return requested_metric
    array = np.asarray(observed_array, dtype=float)
    flat = array.reshape(-1)
    if flat.size <= 4:
        return "euclidean"
    if array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
        if flat.size >= 32:
            return "rmse"
        return "euclidean"
    return "nrmse"



def recommend_summary_kind(observed_array: np.ndarray, requested_kind: str | None = None) -> dict[str, Any]:
    if requested_kind and requested_kind != "auto":
        return {"kind": requested_kind, "reason": "explicit"}
    array = np.asarray(observed_array, dtype=float)
    if array.size > 256:
        if array.ndim == 1 or (array.ndim == 2 and 1 in array.shape):
            return {"kind": "timeseries", "reason": "high_dimensional_series"}
        return {"kind": "moments", "reason": "high_dimensional_vector"}
    return {"kind": "identity", "reason": "manageable_dimension"}



def build_prior_report(
    parameter_names: list[str],
    parameter_defaults: dict[str, Any],
    explicit_priors: dict[str, Any] | None = None,
    parameter_bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    priors: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    missing_bounds: list[str] = []
    questions: list[str] = []
    explicit_priors = explicit_priors or {}
    parameter_bounds = parameter_bounds or {}
    for name in parameter_names:
        explicit_prior = explicit_priors.get(name)
        explicit_bounds = extract_prior_bounds(explicit_prior) if explicit_prior is not None else None
        requested_bounds = parameter_bounds.get(name)
        if explicit_bounds is not None and requested_bounds is not None and not bounds_match(explicit_bounds, requested_bounds):
            raise AnalysisError(
                f"Conflicting bounds for parameter {name!r}: prior specifies {explicit_bounds}, but parameter-bound specifies {requested_bounds}."
            )
        bounds = requested_bounds or explicit_bounds
        if bounds is None:
            missing_bounds.append(name)
            questions.append(
                f"Please provide explicit prior bounds for parameter '{name}' before running ABC calibration, for example --parameter-bound {name}=LOWER:UPPER."
            )
            continue
        if explicit_prior is not None:
            priors[name] = apply_exact_bounds(explicit_prior, bounds)
            provenance[name] = "user"
            continue
        priors[name] = apply_exact_bounds({"dist": "uniform", "params": {"lower": bounds[0], "upper": bounds[1]}}, bounds)
        provenance[name] = "user_bounds"
    return {
        "priors": priors,
        "provenance": provenance,
        "summary": summarize_priors(priors),
        "missing_bounds": missing_bounds,
        "questions": questions,
        "ready": not missing_bounds,
    }



def inspect_observed_data(path: str) -> dict[str, Any]:
    observed = load_observed_data(path)
    observed["array"] = observed["array"]
    return observed



def build_equation_wrapper_source(expression: str, parameter_names: list[str]) -> str:
    if not parameter_names:
        parameter_names = infer_equation_parameters(expression)
    arg_list = ", ".join(parameter_names)
    return f'''#!/usr/bin/env python3
"""Generated equation wrapper for ABC calibration."""

from __future__ import annotations

import math
import numpy as np


def simulate({arg_list}):
    return {expression}
'''



def detect_visualization_defaults(requested: list[str] | None) -> dict[str, Any]:
    plots = requested or []
    return {
        "enabled": bool(plots),
        "plots": plots,
    }
