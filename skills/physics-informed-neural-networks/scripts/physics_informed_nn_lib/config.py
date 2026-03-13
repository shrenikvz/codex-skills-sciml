"""Configuration helpers for the PINNs skill."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

CONFIG_VERSION = 1

DEFAULT_CONFIG: dict[str, Any] = {
    "config_version": CONFIG_VERSION,
    "objective": {
        "text": "",
        "problem_type": "auto",
        "observation_path": None,
        "input_columns": [],
        "output_columns": [],
        "context_columns": [],
        "unknown_parameters": [],
    },
    "problem": {
        "path": "",
        "format": "auto",
        "callable": None,
        "independent_variables": [],
        "dependent_variables": [],
        "domains": {},
        "equations": [],
        "boundary_conditions": [],
        "initial_conditions": [],
        "algebraic_constraints": [],
        "constitutive_relations": [],
        "parameters": {},
        "unknown_parameters": [],
        "analytical_solution": None,
        "metadata": {},
    },
    "model": {
        "framework": "auto",
        "architecture": "auto",
        "hidden_layers": None,
        "hidden_units": None,
        "activation": "tanh",
        "residual_blocks": 0,
        "fourier_features": {
            "enabled": False,
            "num_features": 64,
            "sigma": 2.0,
        },
        "multiscale": {
            "enabled": False,
            "scales": [1.0, 2.0, 4.0],
        },
        "transformer": {
            "enabled": False,
            "width": 128,
            "heads": 4,
            "layers": 4,
            "context_dim": 0,
        },
        "adaptive_activation": False,
    },
    "sampling": {
        "strategy": "auto",
        "interior_points": None,
        "boundary_points": None,
        "initial_points": None,
        "validation_points": None,
        "adaptive": {
            "enabled": False,
            "interval": 250,
            "candidate_pool": 4096,
            "top_k": 512,
        },
    },
    "loss": {
        "weighting_strategy": "auto",
        "weights": {
            "pde": 1.0,
            "bc": 1.0,
            "ic": 1.0,
            "data": 1.0,
            "algebraic": 1.0,
        },
    },
    "training": {
        "optimizer": "hybrid",
        "learning_rate": None,
        "epochs": None,
        "adam_epochs": None,
        "lbfgs_steps": None,
        "batch_size": None,
        "gradient_clip_norm": None,
        "early_stopping_patience": 500,
        "log_interval": 50,
        "seed": 7,
        "domain_decomposition": {
            "enabled": False,
            "num_subdomains": 1,
        },
    },
    "stabilization": {
        "coordinate_scaling": "auto",
        "residual_normalization": "auto",
        "gradient_clipping": "auto",
        "fourier_features": "auto",
        "adaptive_activation": "auto",
        "domain_decomposition": "auto",
    },
    "compute": {
        "device_preference": "auto",
        "distributed": False,
        "data_parallel": "auto",
        "mixed_precision": False,
    },
    "evaluation": {
        "prediction_points": 1024,
        "residual_points": 2048,
        "uncertainty_method": "none",
    },
    "visualization": {
        "enabled": False,
        "plots": [],
        "dpi": 140,
    },
    "output": {
        "results_dir": "results",
        "artifacts_dir": "artifacts",
    },
}


class ConfigError(RuntimeError):
    """Configuration failure."""


def clone_default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def migrate_config(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = deep_merge(DEFAULT_CONFIG, payload)
    migrated["config_version"] = CONFIG_VERSION
    return migrated


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a JSON object.")
    cfg = migrate_config(raw)
    validate_config(cfg)
    return cfg


def save_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def validate_config(cfg: dict[str, Any]) -> None:
    problem = cfg.get("problem")
    if not isinstance(problem, dict):
        raise ConfigError("config.problem must be an object.")
    if not isinstance(problem.get("independent_variables", []), list):
        raise ConfigError("config.problem.independent_variables must be a list.")
    if not isinstance(problem.get("dependent_variables", []), list):
        raise ConfigError("config.problem.dependent_variables must be a list.")
    if not problem.get("path"):
        raise ConfigError("config.problem.path must be a non-empty string.")
    if not problem.get("equations"):
        raise ConfigError("config.problem.equations must contain at least one equation.")
    domains = problem.get("domains", {})
    if not isinstance(domains, dict) or not domains:
        raise ConfigError("config.problem.domains must be a non-empty object.")
    for name in problem.get("independent_variables", []):
        if name not in domains:
            raise ConfigError(f"Missing domain bounds for independent variable {name!r}.")
    model = cfg.get("model", {})
    if model.get("architecture") not in {
        "auto",
        "mlp",
        "fourier",
        "resnet",
        "multiscale",
        "coordinate",
        "transformer_operator",
    }:
        raise ConfigError("Unsupported config.model.architecture.")
    sampling = cfg.get("sampling", {})
    for key in ["interior_points", "boundary_points", "initial_points", "validation_points"]:
        value = sampling.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ConfigError(f"config.sampling.{key} must be a positive integer or null.")
    training = cfg.get("training", {})
    for key in ["epochs", "adam_epochs", "lbfgs_steps", "early_stopping_patience", "log_interval", "seed"]:
        value = training.get(key)
        if value is not None and (not isinstance(value, int) or value < 0):
            raise ConfigError(f"config.training.{key} must be a non-negative integer or null.")
    learning_rate = training.get("learning_rate")
    if learning_rate is not None and (not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0):
        raise ConfigError("config.training.learning_rate must be positive or null.")
    visualization = cfg.get("visualization", {})
    if not isinstance(visualization.get("plots", []), list):
        raise ConfigError("config.visualization.plots must be a list.")


def infer_default_counts(problem_dim: int, output_dim: int, problem_type: str, has_data: bool) -> dict[str, Any]:
    dim = max(1, int(problem_dim or 1))
    outputs = max(1, int(output_dim or 1))
    scale = 2 if has_data or problem_type in {"inverse", "operator_learning"} else 1
    interior = 1024 * max(1, dim) * scale
    boundary = 256 * outputs
    initial = 256 * outputs if dim > 1 else 0
    validation = min(max(512 * dim, 512), 4096)
    epochs = 2500 if problem_type == "forward" else 4000
    adam_epochs = int(0.8 * epochs)
    lbfgs_steps = 300 if problem_type == "forward" else 500
    learning_rate = 1e-3 if dim <= 3 else 5e-4
    hidden_layers = 4 if dim == 1 else 6 if dim <= 3 else 8
    hidden_units = 96 if outputs == 1 else 128 if outputs <= 3 else 160
    return {
        "sampling": {
            "interior_points": int(interior),
            "boundary_points": int(boundary),
            "initial_points": int(initial) if initial else int(boundary),
            "validation_points": int(validation),
        },
        "training": {
            "epochs": int(epochs),
            "adam_epochs": int(adam_epochs),
            "lbfgs_steps": int(lbfgs_steps),
            "learning_rate": float(learning_rate),
        },
        "model": {
            "hidden_layers": int(hidden_layers),
            "hidden_units": int(hidden_units),
        },
    }


def resolve_runtime_recommendations(cfg: dict[str, Any]) -> dict[str, Any]:
    problem = cfg.get("problem", {})
    objective = cfg.get("objective", {})
    recommendations = infer_default_counts(
        len(problem.get("independent_variables", [])),
        len(problem.get("dependent_variables", [])),
        str(objective.get("problem_type", "forward")),
        bool(objective.get("observation_path")),
    )
    for section, values in recommendations.items():
        bucket = cfg.setdefault(section, {})
        for key, value in values.items():
            if bucket.get(key) in {None, "", 0}:
                bucket[key] = value
    return cfg

