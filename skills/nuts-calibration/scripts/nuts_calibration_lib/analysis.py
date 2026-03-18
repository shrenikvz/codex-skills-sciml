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

from .config import infer_default_hyperparameters
from .environment import peek_environment, recommend_backend
from .io_utils import load_observed_data, payload_to_array
from .likelihoods import SUPPORTED_LIKELIHOODS, recommend_likelihood
from .priors import build_prior_report, default_point, recommend_prior
from .transforms import build_transform_specs


class AnalysisError(RuntimeError):
    """Inspection or recommendation failure."""


SUPPORTED_PLOTS = [
    "posterior_marginals",
    "pairwise",
    "trace",
    "autocorrelation",
    "energy",
    "posterior_predictive",
]

SUPPORTED_BACKENDS = ["blackjax", "numpyro", "pymc", "stan", "tensorflow_probability"]
SUPPORTED_TRANSFORMS = ["log", "logit", "softplus", "identity"]
SUPPORTED_PRIORS = ["uniform", "normal", "lognormal", "gamma", "beta", "halfnormal", "student_t"]

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
    "power",
    "jnp",
    "jax",
    "np",
    "numpy",
    "math",
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


def build_equation_wrapper_source(expression: str, parameter_names: list[str]) -> str:
    body = expression.replace("numpy.", "jnp.").replace("np.", "jnp.")
    signature = ", ".join(parameter_names) if parameter_names else "params"
    if parameter_names:
        args_block = ""
    else:
        args_block = "    locals().update(dict(params))\n"
    return f"""from __future__ import annotations
import jax.numpy as jnp

def simulate({signature}):
{args_block}    return {body}
"""


def infer_adapter(model_path: str | None, command_template: str | None, equation_text: str | None) -> str:
    if equation_text:
        return "python_callable"
    if command_template:
        return "command"
    if model_path:
        if Path(model_path).suffix.lower() == ".py":
            return "python_callable"
        return "command"
    raise AnalysisError("One of model_path, command_template, or equation_text is required.")


def inspect_python_model(path: str, callable_name: str | None = None, priors: dict[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise AnalysisError(f"Model path does not exist: {target}")
    source = target.read_text(encoding="utf-8", errors="ignore")
    lower_source = source.lower()
    try:
        module = _load_python_module(target)
    except Exception:
        module = None
    public_functions = []
    if module is not None:
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__ == module.__name__ and not name.startswith("_"):
                public_functions.append(name)
    else:
        tree = ast.parse(source)
        public_functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
    chosen = callable_name
    if not chosen:
        if len(public_functions) == 1:
            chosen = public_functions[0]
        elif "simulate" in public_functions:
            chosen = "simulate"
        elif "model" in public_functions:
            chosen = "model"
    if not chosen:
        raise AnalysisError(f"Could not infer callable from {target.name}. Available functions: {public_functions or 'none'}")
    parameter_names: list[str] = []
    defaults: dict[str, Any] = {}
    call_style = "kwargs"
    fn = getattr(module, chosen, None) if module is not None else None
    if module is not None and callable(fn):
        signature = inspect.signature(fn)
        for name, parameter in signature.parameters.items():
            if parameter.kind in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}:
                continue
            parameter_names.append(name)
            if parameter.default is not inspect._empty:
                defaults[name] = parameter.default
        if len(parameter_names) == 1 and parameter_names[0] in {"params", "parameters"}:
            call_style = "mapping"
    else:
        tree = ast.parse(source)
        function_node = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == chosen),
            None,
        )
        if function_node is None:
            raise AnalysisError(f"Callable {chosen!r} not found in {target}")
        parameter_names = [arg.arg for arg in function_node.args.args]
        default_values = list(function_node.args.defaults)
        offset = len(parameter_names) - len(default_values)
        for idx, value in enumerate(default_values):
            name = parameter_names[offset + idx]
            try:
                defaults[name] = ast.literal_eval(value)
            except Exception:
                continue
        if len(parameter_names) == 1 and parameter_names[0] in {"params", "parameters"}:
            call_style = "mapping"
    stochastic = any(token in lower_source for token in ["np.random", "random.", "rng", "stochastic"])
    jax_compatible = any(token in lower_source for token in ["import jax", "jax.numpy", "jnp.", "jax.lax", "jax.nn"])

    probe_point: dict[str, float] | None = None
    output_names: list[str] = []
    probe_shape: list[int] | None = None
    if module is not None and all(name in defaults or (priors and name in priors) for name in parameter_names):
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
        "parameter_defaults": defaults,
        "parameter_constraints": {},
        "public_functions": public_functions,
        "stochastic": stochastic,
        "output_names": output_names,
        "probe_point": probe_point,
        "probe_shape": probe_shape,
        "jax_compatible": jax_compatible,
    }


def inspect_model_file(
    model_path: str | None,
    command_template: str | None = None,
    equation_text: str | None = None,
    callable_name: str | None = None,
    user_parameter_names: list[str] | None = None,
    user_priors: dict[str, Any] | None = None,
    parameter_bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    adapter = infer_adapter(model_path, command_template, equation_text)
    if equation_text:
        parameter_names = user_parameter_names or infer_equation_parameters(equation_text)
        priors = user_priors or {name: recommend_prior(name) for name in parameter_names}
        return {
            "adapter": "python_callable",
            "path": None,
            "callable": "simulate",
            "call_style": "kwargs",
            "parameter_names": parameter_names,
            "parameter_defaults": {},
            "parameter_constraints": parameter_bounds or {},
            "public_functions": ["simulate"],
            "stochastic": False,
            "output_names": [],
            "probe_point": {name: default_point(priors[name]) for name in parameter_names},
            "probe_shape": None,
            "generated_from_equation": True,
            "jax_compatible": True,
        }
    if adapter == "python_callable" and model_path:
        info = inspect_python_model(model_path, callable_name=callable_name, priors=user_priors)
        info["parameter_constraints"] = parameter_bounds or {}
        return info
    if model_path:
        target = Path(model_path).expanduser().resolve()
        runtime = RUNTIME_BY_SUFFIX.get(target.suffix, "bash")
        wrapper_template = f"{runtime} {{model_path}} {{params_json}} {{output_json}}"
        return {
            "adapter": "command",
            "path": str(target),
            "callable": None,
            "call_style": None,
            "parameter_names": user_parameter_names or [],
            "parameter_defaults": {},
            "parameter_constraints": parameter_bounds or {},
            "public_functions": [],
            "stochastic": None,
            "output_names": [],
            "probe_point": None,
            "probe_shape": None,
            "command_template": command_template or wrapper_template,
            "requires_wrapper_confirmation": True,
            "runtime_hint": runtime,
            "jax_compatible": False,
        }
    raise AnalysisError("Could not inspect model inputs.")


def infer_observed_outputs(model_output_names: list[str], observed_column_names: list[str], requested: list[str] | None = None) -> dict[str, Any]:
    if requested:
        return {"selected": list(requested), "questions": [], "selected_by": "explicit"}
    if model_output_names and observed_column_names:
        overlap = [name for name in observed_column_names if name in set(model_output_names)]
        if overlap:
            return {"selected": overlap, "questions": [], "selected_by": "name_intersection"}
    if len(observed_column_names) == 1:
        return {"selected": observed_column_names, "questions": [], "selected_by": "single_observed_column"}
    question = "Which model outputs correspond to the observed data used for calibration?"
    if observed_column_names:
        question += f" Observed columns: {observed_column_names}."
    return {"selected": [], "questions": [question], "selected_by": "ambiguous"}


def inspect_observed_data(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = load_observed_data(path)
    array = np.asarray(payload["array"], dtype=float)
    flat = array.reshape(-1)
    payload["integer_like"] = bool(np.allclose(flat, np.round(flat)))
    payload["non_negative"] = bool(np.all(flat >= 0))
    payload["binary"] = bool(payload["integer_like"] and set(np.unique(flat)).issubset({0.0, 1.0}))
    payload["max_abs"] = float(np.max(np.abs(flat))) if flat.size else 0.0
    payload["std"] = float(np.std(flat)) if flat.size else 0.0
    payload["dynamic_range"] = float(np.max(flat) - np.min(flat)) if flat.size else 0.0
    return payload


def assess_differentiability(model_analysis: dict[str, Any], request_text: str | None = None) -> dict[str, Any]:
    if model_analysis.get("generated_from_equation"):
        return {
            "differentiable": True,
            "gradient_strategy": "jax_autodiff",
            "reason": "generated_jax_equation_wrapper",
            "needs_confirmation": False,
        }
    if model_analysis.get("adapter") == "python_callable" and model_analysis.get("jax_compatible"):
        return {
            "differentiable": True,
            "gradient_strategy": "jax_autodiff",
            "reason": "python_model_looks_jax_compatible",
            "needs_confirmation": False,
        }
    if model_analysis.get("adapter") == "python_callable":
        return {
            "differentiable": False,
            "gradient_strategy": "finite_difference",
            "reason": "python_model_not_obviously_jax_compatible",
            "needs_confirmation": True,
        }
    return {
        "differentiable": False,
        "gradient_strategy": "finite_difference",
        "reason": "external_command_model",
        "needs_confirmation": True,
    }


def recommend_scaling_mode(observed_array: np.ndarray, scaling_preference: str | None = None) -> dict[str, Any]:
    choice = str(scaling_preference or "auto").lower()
    if choice not in {"auto", "none", "zscore", "minmax", "variance"}:
        raise AnalysisError(f"Unsupported scaling mode: {scaling_preference}")
    flat = np.asarray(observed_array, dtype=float).reshape(-1)
    if choice != "auto":
        return {"enabled": choice != "none", "mode": choice, "reason": "user_preference", "needs_confirmation": False}
    if flat.size <= 1:
        return {"enabled": False, "mode": "none", "reason": "scalar_observation", "needs_confirmation": True}
    max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
    std = float(np.std(flat)) if flat.size else 0.0
    dynamic_range = float(np.max(flat) - np.min(flat)) if flat.size else 0.0
    if np.all((flat >= 0.0) & (flat <= 1.0)):
        return {"enabled": False, "mode": "none", "reason": "already_unit_scale", "needs_confirmation": True}
    if max_abs > 100.0 or (std > 0 and dynamic_range / std > 20.0):
        return {"enabled": True, "mode": "zscore", "reason": "large_scale_variation", "needs_confirmation": True}
    if np.all(flat >= 0.0) and dynamic_range > 5.0:
        return {"enabled": True, "mode": "minmax", "reason": "positive_range_scaling", "needs_confirmation": True}
    return {"enabled": True, "mode": "variance", "reason": "moderate_multioutput_variation", "needs_confirmation": True}


def detect_visualization_defaults(plots: list[str] | None) -> dict[str, Any]:
    requested = list(plots or [])
    if not requested:
        return {"enabled": False, "plots": [], "reason": "no_plots_requested"}
    unknown = [name for name in requested if name not in SUPPORTED_PLOTS]
    if unknown:
        raise AnalysisError(f"Unsupported plot names: {unknown}")
    return {"enabled": True, "plots": requested, "reason": "user_requested"}


def recommend_model_complexity(model_analysis: dict[str, Any], observed_analysis: dict[str, Any] | None) -> str:
    if model_analysis.get("adapter") == "command":
        return "external"
    if model_analysis.get("stochastic"):
        return "stochastic"
    if observed_analysis and int(observed_analysis.get("size", 0)) > 1000:
        return "timeseries"
    if len(model_analysis.get("parameter_names", [])) > 8:
        return "high_dimensional"
    return "standard"


def inspect_inputs(
    model_path: str | None,
    observed_path: str | None,
    command_template: str | None = None,
    equation_text: str | None = None,
    callable_name: str | None = None,
    request_text: str | None = None,
    observed_output_names: list[str] | None = None,
    user_parameter_names: list[str] | None = None,
    explicit_priors: dict[str, dict[str, Any]] | None = None,
    parameter_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    scaling_mode: str | None = None,
    likelihood_name: str | None = None,
    backend_name: str | None = None,
    plots: list[str] | None = None,
) -> dict[str, Any]:
    model_analysis = inspect_model_file(
        model_path,
        command_template=command_template,
        equation_text=equation_text,
        callable_name=callable_name,
        user_parameter_names=user_parameter_names,
        user_priors=explicit_priors,
        parameter_bounds=parameter_bounds,
    )
    prior_report = build_prior_report(
        model_analysis.get("parameter_names", []),
        model_analysis.get("parameter_defaults", {}),
        explicit_priors=explicit_priors,
        parameter_bounds=parameter_bounds,
    )
    observed_analysis = inspect_observed_data(observed_path) if observed_path else None
    output_map = infer_observed_outputs(
        model_analysis.get("output_names", []),
        observed_analysis.get("column_names", []) if observed_analysis else [],
        requested=observed_output_names,
    )
    differentiability = assess_differentiability(model_analysis, request_text=request_text)
    likelihood = (
        recommend_likelihood(observed_analysis["array"], request_text=request_text, requested_name=likelihood_name)
        if observed_analysis
        else None
    )
    scaling = (
        recommend_scaling_mode(observed_analysis["array"], scaling_preference=scaling_mode)
        if observed_analysis
        else {"enabled": False, "mode": "none", "reason": "no_observed_data", "needs_confirmation": False}
    )
    transform_specs = build_transform_specs(
        model_analysis.get("parameter_names", []),
        prior_report["priors"],
        parameter_constraints=parameter_bounds,
    )
    environment = peek_environment()
    backend = recommend_backend(environment, requested=backend_name)
    complexity = recommend_model_complexity(model_analysis, observed_analysis)
    hyper = infer_default_hyperparameters(
        len(model_analysis.get("parameter_names", [])),
        observed_analysis.get("size", 1) if observed_analysis else 1,
        model_complexity=complexity,
    )
    visual = detect_visualization_defaults(plots)
    pending_questions = list(output_map.get("questions", []))
    if scaling.get("needs_confirmation"):
        pending_questions.append(
            f"Should observed outputs be normalized before calibration? Recommended: {scaling['mode']}."
        )
    if differentiability.get("needs_confirmation"):
        pending_questions.append(
            "The model is not obviously JAX-differentiable. Confirm whether finite-difference gradients are acceptable."
        )
    if likelihood and likelihood.get("needs_confirmation"):
        pending_questions.append(
            f"Confirm the observation noise model. Recommended likelihood: {likelihood['spec']['name']}."
        )
    return {
        "environment": environment,
        "model_analysis": model_analysis,
        "observed_analysis": {key: value for key, value in (observed_analysis or {}).items() if key != "array"} if observed_analysis else None,
        "prior_report": prior_report,
        "output_mapping": output_map,
        "differentiability_assessment": differentiability,
        "likelihood_recommendation": likelihood,
        "scaling_recommendation": scaling,
        "transformation_recommendation": transform_specs,
        "backend_recommendation": backend,
        "default_hyperparameters": hyper,
        "visualization_recommendation": visual,
        "model_complexity": complexity,
        "pending_questions": pending_questions,
    }
