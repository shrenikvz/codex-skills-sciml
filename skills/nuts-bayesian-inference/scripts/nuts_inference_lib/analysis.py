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
from .likelihoods import infer_likelihood_family, maybe_add_likelihood_priors
from .priors import (
    build_prior_report,
    default_point,
    load_prior_file,
    parse_parameter_bounds,
    parse_prior_overrides,
    summarize_priors,
)
from .transforms import build_transform_specs


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
    "jnp",
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
    if not parameter_names:
        parameter_names = infer_equation_parameters(expression)
    arg_list = ", ".join(parameter_names)
    return f'''#!/usr/bin/env python3
"""Generated equation wrapper for NUTS Bayesian inference."""

from __future__ import annotations

import jax.numpy as np


def simulate({arg_list}):
    return {expression}
'''


def infer_adapter(model_path: str | None, command_template: str | None, equation_text: str | None) -> str:
    if equation_text:
        return "python_callable"
    if command_template:
        return "command"
    if model_path:
        return "python_callable" if Path(model_path).suffix.lower() == ".py" else "command"
    raise AnalysisError("One of model_path, command_template, or equation_text is required.")


def inspect_python_model(path: str, callable_name: str | None = None, priors: dict[str, Any] | None = None) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise AnalysisError(f"Model path does not exist: {target}")
    source_text = target.read_text(encoding="utf-8", errors="ignore")
    module = None
    module_error = None
    try:
        module = _load_python_module(target)
    except Exception as exc:  # noqa: BLE001
        module_error = str(exc)
    public_functions = []
    if module is not None:
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__ == module.__name__ and not name.startswith("_"):
                public_functions.append(name)
    else:
        tree = ast.parse(source_text)
        public_functions = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        ]
    chosen = callable_name
    if not chosen:
        for candidate in ["simulate", "model", "forward", "predict"]:
            if candidate in public_functions:
                chosen = candidate
                break
        if not chosen and len(public_functions) == 1:
            chosen = public_functions[0]
    if not chosen:
        raise AnalysisError(f"Could not infer callable from {target.name}. Available functions: {public_functions or 'none'}")
    parameter_names = []
    defaults: dict[str, Any] = {}
    if module is not None:
        fn = getattr(module, chosen, None)
        if not callable(fn):
            raise AnalysisError(f"Callable {chosen!r} not found in {target}")
        signature = inspect.signature(fn)
        for name, parameter in signature.parameters.items():
            if parameter.kind in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}:
                continue
            parameter_names.append(name)
            if parameter.default is not inspect._empty:
                defaults[name] = parameter.default
    else:
        tree = ast.parse(source_text)
        fn_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == chosen]
        if not fn_nodes:
            raise AnalysisError(f"Callable {chosen!r} not found in {target}")
        fn_node = fn_nodes[0]
        total_args = list(fn_node.args.args)
        defaults_raw = list(fn_node.args.defaults)
        defaults_offset = len(total_args) - len(defaults_raw)
        for idx, arg in enumerate(total_args):
            parameter_names.append(arg.arg)
            if idx >= defaults_offset:
                default_node = defaults_raw[idx - defaults_offset]
                if isinstance(default_node, ast.Constant):
                    defaults[arg.arg] = default_node.value
    source = source_text.lower()
    stochastic = any(token in source for token in ["np.random", "random.", "rng", "stochastic"])
    uses_jax = any(token in source for token in ["import jax", "jax.numpy", "jnp."])
    call_style = "kwargs"
    if len(parameter_names) == 1 and parameter_names[0] in {"params", "parameters"}:
        call_style = "mapping"
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
        if len(probe_point) == len(parameter_names):
            try:
                payload = fn(probe_point) if call_style == "mapping" else fn(**probe_point)
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
        "public_functions": public_functions,
        "stochastic": stochastic,
        "uses_jax": uses_jax,
        "output_names": output_names,
        "probe_point": probe_point,
        "probe_shape": probe_shape,
        "import_error": module_error,
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
        return {
            "adapter": "python_callable",
            "path": None,
            "callable": "simulate",
            "call_style": "kwargs",
            "parameter_names": parameter_names,
            "parameter_defaults": {},
            "public_functions": ["simulate"],
            "stochastic": False,
            "uses_jax": True,
            "output_names": [],
            "probe_point": {name: default_point(user_priors[name]) for name in parameter_names if user_priors and name in user_priors},
            "probe_shape": None,
            "generated_from_equation": True,
        }
    if adapter == "python_callable" and model_path:
        return inspect_python_model(model_path, callable_name=callable_name, priors=user_priors)
    if model_path:
        target = Path(model_path).expanduser().resolve()
        suffix = target.suffix
        runtime = RUNTIME_BY_SUFFIX.get(suffix, "bash")
        return {
            "adapter": "command",
            "path": str(target),
            "callable": None,
            "call_style": None,
            "parameter_names": user_parameter_names or [],
            "parameter_defaults": {},
            "public_functions": [],
            "stochastic": None,
            "uses_jax": False,
            "output_names": [],
            "probe_point": None,
            "probe_shape": None,
            "command_template": command_template or f"{runtime} {{model_path}} {{params_json}} {{output_json}}",
            "runtime_hint": runtime,
            "requires_wrapper_confirmation": True,
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
    question = "Which model outputs correspond to the observed data?"
    if observed_column_names:
        question += f" Observed columns: {observed_column_names}."
    if model_output_names:
        question += f" Candidate model outputs: {model_output_names}."
    return {"selected": [], "questions": [question], "selected_by": "ambiguous"}


def recommend_scaling_mode(observed_array: np.ndarray, scaling_preference: str | None = None, likelihood_family: str | None = None) -> dict[str, Any]:
    preference = (scaling_preference or "auto").lower()
    if preference not in {"auto", "none", "zscore", "minmax", "variance"}:
        raise AnalysisError(f"Unsupported scaling mode: {scaling_preference}")
    if likelihood_family in {"poisson", "binomial", "negative_binomial"}:
        return {"enabled": False, "mode": "none", "reason": "count_likelihood"}
    array = np.asarray(observed_array, dtype=float).reshape(-1)
    if preference != "auto":
        return {"enabled": preference != "none", "mode": preference, "reason": "user_preference"}
    if array.size <= 1:
        return {"enabled": False, "mode": "none", "reason": "scalar_output"}
    span = float(np.max(array) - np.min(array))
    std = float(np.std(array))
    if std == 0:
        return {"enabled": False, "mode": "none", "reason": "zero_variance"}
    if span > 100 * max(abs(float(np.mean(array))), 1.0):
        return {"enabled": True, "mode": "zscore", "reason": "large_dynamic_range"}
    if std > 10:
        return {"enabled": True, "mode": "variance", "reason": "large_variance"}
    return {"enabled": False, "mode": "none", "reason": "already_well_scaled"}


def recommend_gradient_strategy(model_analysis: dict[str, Any], requested_strategy: str | None = None) -> dict[str, Any]:
    strategy = (requested_strategy or "auto").lower()
    if strategy not in {"auto", "jax", "finite_difference", "surrogate"}:
        raise AnalysisError(f"Unsupported gradient strategy: {requested_strategy}")
    if strategy != "auto":
        return {"strategy": strategy, "reason": "user_preference"}
    if model_analysis.get("uses_jax") or model_analysis.get("generated_from_equation"):
        return {"strategy": "jax", "reason": "jax_native_model"}
    if model_analysis.get("adapter") == "python_callable":
        return {"strategy": "finite_difference", "reason": "python_host_model"}
    return {"strategy": "finite_difference", "reason": "external_model"}


def detect_visualization_defaults(requested: list[str] | None) -> dict[str, Any]:
    plots = requested or []
    return {"enabled": bool(plots), "plots": plots}


def inspect_model_inputs(
    model_path: str | None,
    observed_path: str | None,
    command_template: str | None = None,
    equation_text: str | None = None,
    callable_name: str | None = None,
    request_text: str | None = None,
    observed_output_names: list[str] | None = None,
    user_parameter_names: list[str] | None = None,
    prior_file: str | None = None,
    prior_overrides: list[str] | None = None,
    parameter_bounds: list[str] | None = None,
    likelihood_family: str | None = None,
    scaling_mode: str | None = None,
    gradient_strategy: str | None = None,
    plots: list[str] | None = None,
) -> dict[str, Any]:
    explicit_priors = load_prior_file(prior_file)
    explicit_priors.update(parse_prior_overrides(prior_overrides))
    bounds = parse_parameter_bounds(parameter_bounds)
    model_analysis = inspect_model_file(
        model_path,
        command_template=command_template,
        equation_text=equation_text,
        callable_name=callable_name,
        user_parameter_names=user_parameter_names,
        user_priors=explicit_priors,
    )
    prior_report = build_prior_report(
        model_analysis.get("parameter_names", []),
        model_analysis.get("parameter_defaults", {}),
        explicit_priors=explicit_priors,
        parameter_bounds=bounds,
    )
    observed_analysis = load_observed_data(observed_path) if observed_path else None
    output_map = infer_observed_outputs(
        model_analysis.get("output_names", []),
        observed_analysis.get("column_names", []) if observed_analysis else [],
        requested=observed_output_names,
    )
    likelihood_suggestion = infer_likelihood_family(
        observed_analysis["array"] if observed_analysis else np.asarray([]),
        requested_family=likelihood_family,
    )
    priors_with_likelihood, likelihood_report = maybe_add_likelihood_priors(
        prior_report["priors"],
        {"family": likelihood_suggestion["family"], "params": {}},
        observed_analysis["array"] if observed_analysis else np.asarray([]),
    )
    scaling = recommend_scaling_mode(
        observed_analysis["array"] if observed_analysis is not None else np.asarray([]),
        scaling_preference=scaling_mode,
        likelihood_family=likelihood_report["family"],
    )
    transform_specs = build_transform_specs(priors_with_likelihood)
    gradient = recommend_gradient_strategy(model_analysis, requested_strategy=gradient_strategy)
    visual = detect_visualization_defaults(plots)
    pending_questions = list(output_map.get("questions", []))
    pending_questions.extend(likelihood_suggestion.get("questions", []))
    if scaling_mode in {None, "", "auto"}:
        pending_questions.append("Do you want normalization or standardization applied before likelihood evaluation? If not specified, the recommended default will be used.")
    if not visual["plots"]:
        pending_questions.append("Which visualizations do you want, if any: posterior marginals, pairwise, trace, autocorrelation, energy, posterior predictive?")
    return {
        "model_analysis": model_analysis,
        "observed_analysis": {k: v for k, v in (observed_analysis or {}).items() if k != "array"} if observed_analysis else None,
        "prior_report": {
            "priors": priors_with_likelihood,
            "summary": summarize_priors(priors_with_likelihood),
            "provenance": {
                **prior_report["provenance"],
                **{
                    name: priors_with_likelihood[name].get("source", "likelihood_default")
                    for name in priors_with_likelihood
                    if name not in prior_report["provenance"]
                },
            },
        },
        "output_mapping": output_map,
        "likelihood_report": likelihood_report,
        "scaling_recommendation": scaling,
        "transform_report": transform_specs,
        "gradient_recommendation": gradient,
        "visualization_recommendation": visual,
        "pending_questions": pending_questions,
        "request_text": request_text or "",
    }
