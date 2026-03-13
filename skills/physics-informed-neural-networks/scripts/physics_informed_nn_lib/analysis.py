"""Problem inspection and recommendation helpers for PINNs."""

from __future__ import annotations

import math
from typing import Any

from .config import infer_default_counts
from .environment import peek_environment, recommend_framework
from .io_utils import load_observed_data
from .problem_spec import ProblemSpecError, load_problem_spec, summarize_problem


class AnalysisError(RuntimeError):
    """Inspection or recommendation failure."""


SUPPORTED_ARCHITECTURES = [
    "mlp",
    "fourier",
    "resnet",
    "multiscale",
    "coordinate",
    "transformer_operator",
]

SUPPORTED_FRAMEWORKS = ["torch", "jax", "tensorflow"]
SUPPORTED_SAMPLING = ["uniform", "latin_hypercube", "sobol", "adaptive", "residual_adaptive"]
SUPPORTED_WEIGHTING = ["fixed", "dynamic_balance", "gradient_norm", "uncertainty", "adaptive_residual"]
SUPPORTED_OPTIMIZERS = ["adam", "lbfgs", "hybrid", "curriculum", "domain_decomposition"]
SUPPORTED_PLOTS = [
    "solution_field",
    "time_evolution",
    "residual_heatmap",
    "loss_curves",
    "analytical_comparison",
    "uncertainty_bands",
]


def _normalize_choice(value: str | None, supported: list[str], field_name: str) -> str:
    choice = str(value or "auto").lower()
    if choice != "auto" and choice not in supported:
        raise AnalysisError(f"Unsupported {field_name}: {value}")
    return choice


def infer_observation_mapping(
    observed_info: dict[str, Any] | None,
    independent_variables: list[str],
    dependent_variables: list[str],
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    context_columns: list[str] | None = None,
) -> dict[str, Any]:
    if not observed_info:
        return {
            "input_columns": [],
            "output_columns": [],
            "context_columns": [],
            "questions": [],
            "selected_by": "no_data",
        }
    if observed_info.get("table"):
        available = list(observed_info.get("column_names", []))
        inputs = list(input_columns or [])
        outputs = list(output_columns or [])
        contexts = list(context_columns or [])
        if not inputs:
            inputs = [name for name in independent_variables if name in available]
        if not outputs:
            outputs = [name for name in dependent_variables if name in available]
        if not contexts:
            contexts = [name for name in available if name not in set(inputs) and name not in set(outputs)]
        questions: list[str] = []
        if not inputs:
            questions.append(
                f"Which observation columns are model inputs or coordinates? Available columns: {available}."
            )
        if not outputs:
            questions.append(
                f"Which observation columns are targets for the dependent variables? Available columns: {available}."
            )
        return {
            "input_columns": inputs,
            "output_columns": outputs,
            "context_columns": contexts,
            "questions": questions,
            "selected_by": "name_match" if not questions else "ambiguous",
        }
    if input_columns or output_columns:
        return {
            "input_columns": list(input_columns or []),
            "output_columns": list(output_columns or []),
            "context_columns": list(context_columns or []),
            "questions": [],
            "selected_by": "explicit_non_tabular",
        }
    return {
        "input_columns": [],
        "output_columns": [],
        "context_columns": [],
        "questions": ["Observed data is not tabular; specify input and output column semantics explicitly."],
        "selected_by": "non_tabular",
    }


def detect_problem_type(
    spec: dict[str, Any],
    observed_info: dict[str, Any] | None,
    request_text: str | None = None,
    explicit_problem_type: str | None = None,
    observation_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    choice = str(explicit_problem_type or "auto").lower()
    if choice not in {"auto", "forward", "inverse", "data_assisted", "operator_learning"}:
        raise AnalysisError(f"Unsupported problem type: {explicit_problem_type}")
    if choice != "auto":
        return {"problem_type": choice, "reason": "user_preference"}
    text = (request_text or "").lower()
    mapping = observation_mapping or {}
    if "operator" in text or mapping.get("context_columns"):
        return {"problem_type": "operator_learning", "reason": "conditioning_context"}
    if spec.get("unknown_parameters"):
        return {"problem_type": "inverse", "reason": "trainable_physics_parameters"}
    if observed_info is not None:
        return {"problem_type": "data_assisted", "reason": "observations_present"}
    return {"problem_type": "forward", "reason": "physics_only"}


def recommend_architecture(
    spec: dict[str, Any],
    problem_type: str,
    requested: str | None = None,
    has_data: bool = False,
) -> dict[str, Any]:
    choice = _normalize_choice(requested, SUPPORTED_ARCHITECTURES, "architecture")
    if choice != "auto":
        return {"architecture": choice, "reason": "user_preference"}
    dim = max(1, len(spec.get("independent_variables", [])))
    fields = max(1, len(spec.get("dependent_variables", [])))
    text = " ".join(
        [
            str(spec.get("description", "")),
            *(item.get("expression", "") for item in spec.get("equations", [])),
            *(item.get("expression", "") for item in spec.get("constitutive_relations", [])),
        ]
    ).lower()
    if problem_type == "operator_learning":
        return {"architecture": "transformer_operator", "reason": "operator_learning_context"}
    if any(token in text for token in ["stiff", "multi-scale", "multiscale", "high-frequency", "oscillat"]):
        return {"architecture": "multiscale", "reason": "stiff_or_multiscale_problem"}
    if dim >= 3 or has_data and fields > 1:
        return {"architecture": "resnet", "reason": "higher_dimensional_or_coupled_system"}
    if dim <= 2:
        return {"architecture": "fourier", "reason": "low_dimensional_pde_default"}
    return {"architecture": "mlp", "reason": "simple_default"}


def recommend_sampling_strategy(spec: dict[str, Any], requested: str | None = None) -> dict[str, Any]:
    choice = _normalize_choice(requested, SUPPORTED_SAMPLING, "sampling strategy")
    if choice != "auto":
        return {"strategy": choice, "reason": "user_preference"}
    dim = max(1, len(spec.get("independent_variables", [])))
    if dim <= 2:
        return {"strategy": "sobol", "reason": "low_dimensional_domain"}
    if dim <= 5:
        return {"strategy": "latin_hypercube", "reason": "moderate_dimensional_domain"}
    return {"strategy": "uniform", "reason": "high_dimensional_domain"}


def recommend_loss_weighting(
    spec: dict[str, Any],
    problem_type: str,
    observed_info: dict[str, Any] | None,
    requested: str | None = None,
) -> dict[str, Any]:
    choice = _normalize_choice(requested, SUPPORTED_WEIGHTING, "loss weighting")
    active_terms = ["pde"]
    if spec.get("boundary_conditions"):
        active_terms.append("bc")
    if spec.get("initial_conditions"):
        active_terms.append("ic")
    if spec.get("algebraic_constraints"):
        active_terms.append("algebraic")
    if observed_info is not None:
        active_terms.append("data")
    if choice != "auto":
        return {"strategy": choice, "reason": "user_preference", "active_terms": active_terms, "needs_confirmation": False}
    if problem_type in {"inverse", "data_assisted"} and "data" in active_terms:
        default = "gradient_norm"
        reason = "mixed_physics_and_data_terms"
    elif len(active_terms) >= 4:
        default = "dynamic_balance"
        reason = "many_active_loss_terms"
    else:
        default = "fixed"
        reason = "simple_loss_structure"
    return {"strategy": default, "reason": reason, "active_terms": active_terms, "needs_confirmation": True}


def recommend_optimizer(problem_type: str, requested: str | None = None) -> dict[str, Any]:
    choice = _normalize_choice(requested, SUPPORTED_OPTIMIZERS, "optimizer")
    if choice != "auto":
        return {"optimizer": choice, "reason": "user_preference"}
    if problem_type in {"inverse", "data_assisted"}:
        return {"optimizer": "hybrid", "reason": "mixed_parameter_and_solution_training"}
    return {"optimizer": "hybrid", "reason": "robust_default"}


def recommend_stabilization(
    spec: dict[str, Any],
    architecture: str,
    sampling_strategy: str,
    problem_type: str,
) -> dict[str, Any]:
    dim = max(1, len(spec.get("independent_variables", [])))
    text = " ".join(
        [str(spec.get("description", "")), *(item.get("expression", "") for item in spec.get("equations", []))]
    ).lower()
    stiff_like = any(token in text for token in ["stiff", "shock", "boundary layer", "multiscale", "high-frequency"])
    return {
        "coordinate_scaling": True,
        "residual_normalization": True,
        "gradient_clipping": problem_type in {"inverse", "data_assisted"} or stiff_like,
        "fourier_features": architecture in {"fourier", "multiscale"} or stiff_like,
        "adaptive_activation": architecture in {"coordinate", "multiscale"} or dim >= 3,
        "domain_decomposition": architecture == "multiscale" or sampling_strategy == "residual_adaptive",
    }


def recommend_hyperparameters(
    spec: dict[str, Any],
    problem_type: str,
    observed_info: dict[str, Any] | None,
    architecture: str,
    hidden_layers: int | None = None,
    hidden_units: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    defaults = infer_default_counts(
        len(spec.get("independent_variables", [])),
        len(spec.get("dependent_variables", [])),
        problem_type,
        observed_info is not None,
    )
    model = defaults["model"]
    training = defaults["training"]
    sampling = defaults["sampling"]
    if architecture == "transformer_operator":
        model["hidden_layers"] = max(4, int(model["hidden_layers"]))
        model["hidden_units"] = max(128, int(model["hidden_units"]))
    if hidden_layers is not None:
        model["hidden_layers"] = int(hidden_layers)
    if hidden_units is not None:
        model["hidden_units"] = int(hidden_units)
    if epochs is not None:
        training["epochs"] = int(epochs)
        training["adam_epochs"] = max(1, int(0.8 * epochs))
    if learning_rate is not None:
        training["learning_rate"] = float(learning_rate)
    return {"model": model, "training": training, "sampling": sampling}


def _collect_questions(
    spec: dict[str, Any],
    observed_info: dict[str, Any] | None,
    observation_mapping: dict[str, Any],
    weighting: dict[str, Any],
    plots: list[str] | None,
) -> tuple[list[str], list[str]]:
    pending: list[str] = []
    blocking: list[str] = []
    if not spec.get("independent_variables"):
        blocking.append("The problem specification is missing independent variables.")
    if not spec.get("dependent_variables"):
        blocking.append("The problem specification is missing dependent variables.")
    if not spec.get("domains"):
        blocking.append("The problem specification is missing domain bounds.")
    if not spec.get("equations"):
        blocking.append("The problem specification is missing governing equations or residual expressions.")
    if "t" in set(spec.get("independent_variables", [])) and not spec.get("initial_conditions"):
        pending.append("No initial conditions were detected for a time-dependent problem. Confirm whether they are required.")
    if not spec.get("boundary_conditions"):
        pending.append("No boundary conditions were detected. Confirm whether the domain is periodic, unconstrained, or still underspecified.")
    pending.extend(observation_mapping.get("questions", []))
    if weighting.get("needs_confirmation"):
        pending.append(
            "Which loss-weighting strategy do you want? Supported: fixed, dynamic_balance, gradient_norm, uncertainty, adaptive_residual."
        )
    if not plots:
        pending.append(
            "Which visualizations do you want? Supported: solution_field, time_evolution, residual_heatmap, loss_curves, analytical_comparison, uncertainty_bands."
        )
    if observed_info is not None and not observation_mapping.get("output_columns"):
        blocking.append("Observed data is present but target columns could not be identified.")
    return pending, blocking


def inspect_problem_inputs(
    problem_path: str | None,
    observed_path: str | None = None,
    physics_text: str | None = None,
    callable_name: str | None = None,
    request_text: str | None = None,
    problem_type: str | None = None,
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    context_columns: list[str] | None = None,
    architecture: str | None = None,
    framework: str | None = None,
    sampling_strategy: str | None = None,
    loss_weighting: str | None = None,
    optimizer: str | None = None,
    plots: list[str] | None = None,
    hidden_layers: int | None = None,
    hidden_units: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    try:
        spec = load_problem_spec(problem_path, physics_text=physics_text, callable_name=callable_name)
    except ProblemSpecError as exc:
        raise AnalysisError(str(exc)) from exc
    observed_info = load_observed_data(observed_path) if observed_path else None
    observation_mapping = infer_observation_mapping(
        observed_info,
        spec.get("independent_variables", []),
        spec.get("dependent_variables", []),
        input_columns=input_columns,
        output_columns=output_columns,
        context_columns=context_columns,
    )
    detected_problem_type = detect_problem_type(
        spec,
        observed_info,
        request_text=request_text,
        explicit_problem_type=problem_type,
        observation_mapping=observation_mapping,
    )
    architecture_recommendation = recommend_architecture(
        spec,
        detected_problem_type["problem_type"],
        requested=architecture,
        has_data=observed_info is not None,
    )
    sampling_recommendation = recommend_sampling_strategy(spec, requested=sampling_strategy)
    weighting_recommendation = recommend_loss_weighting(
        spec,
        detected_problem_type["problem_type"],
        observed_info,
        requested=loss_weighting,
    )
    optimizer_recommendation = recommend_optimizer(detected_problem_type["problem_type"], requested=optimizer)
    stabilization_recommendation = recommend_stabilization(
        spec,
        architecture_recommendation["architecture"],
        sampling_recommendation["strategy"],
        detected_problem_type["problem_type"],
    )
    hyperparameters = recommend_hyperparameters(
        spec,
        detected_problem_type["problem_type"],
        observed_info,
        architecture_recommendation["architecture"],
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    environment = peek_environment()
    framework_recommendation = recommend_framework(environment, requested=framework)
    pending_questions, blocking_questions = _collect_questions(
        spec,
        observed_info,
        observation_mapping,
        weighting_recommendation,
        plots,
    )
    if plots:
        unknown_plots = [plot for plot in plots if plot not in SUPPORTED_PLOTS]
        if unknown_plots:
            raise AnalysisError(f"Unsupported plot names: {unknown_plots}")
    return {
        "problem_summary": summarize_problem(spec),
        "problem_spec": spec,
        "observed_analysis": observed_info,
        "observation_mapping": observation_mapping,
        "problem_type": detected_problem_type,
        "architecture_recommendation": architecture_recommendation,
        "framework_recommendation": framework_recommendation,
        "sampling_recommendation": sampling_recommendation,
        "loss_weighting_recommendation": weighting_recommendation,
        "optimizer_recommendation": optimizer_recommendation,
        "stabilization_recommendation": stabilization_recommendation,
        "default_hyperparameters": hyperparameters,
        "visualization_recommendation": {
            "enabled": bool(plots),
            "plots": list(plots or []),
        },
        "pending_questions": pending_questions,
        "blocking_questions": blocking_questions,
        "environment": environment,
    }
