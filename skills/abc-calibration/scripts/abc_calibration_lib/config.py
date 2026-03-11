"""Configuration helpers for the ABC calibration skill."""

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
        "observed_path": "",
        "observed_format": "auto",
        "observed_output_names": [],
        "observed_output_indices": [],
        "likelihood_hint": "auto",
    },
    "model": {
        "adapter": "auto",
        "path": "",
        "callable": "simulate",
        "call_style": "auto",
        "command_template": None,
        "working_directory": None,
        "runtime_hint": None,
        "parameter_names": [],
        "parameter_defaults": {},
        "output_names": [],
        "observed_output_names": [],
        "observed_output_indices": [],
        "supports_batch": False,
        "stochastic": None,
        "timeout_seconds": 300,
    },
    "priors": {},
    "summary_statistics": {
        "kind": "auto",
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
        "lag_count": 3,
        "path": None,
        "callable": None,
        "command_template": None,
    },
    "distance": {
        "metric": "auto",
        "custom_python_path": None,
        "custom_callable": None,
        "custom_command_template": None,
    },
    "scaling": {
        "enabled": None,
        "mode": "auto",
    },
    "algorithm": {
        "name": "abc_rejection",
        "random_seed": 7,
        "two_phase": {
            "pilot_size": None,
            "epsilon_quantile": 0.05,
            "main_budget": None,
            "accepted_samples": None,
            "batch_size": 32,
            "proceed_if_likelihood_available": False,
        },
    },
    "compute": {
        "backend": "local",
        "max_workers": "auto",
        "chunk_size": 32,
    },
    "posterior_predictive": {
        "enabled": True,
        "draws": 100,
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
    if not isinstance(cfg.get("model"), dict):
        raise ConfigError("config.model must be an object.")
    if not isinstance(cfg.get("priors"), dict):
        raise ConfigError("config.priors must be an object keyed by parameter name.")
    model = cfg["model"]
    parameter_names = model.get("parameter_names", [])
    if not isinstance(parameter_names, list) or not all(isinstance(item, str) for item in parameter_names):
        raise ConfigError("config.model.parameter_names must be a list of strings.")
    observed_path = cfg.get("objective", {}).get("observed_path")
    if not isinstance(observed_path, str):
        raise ConfigError("config.objective.observed_path must be a string.")
    two_phase = cfg.get("algorithm", {}).get("two_phase", {})
    quantile = two_phase.get("epsilon_quantile", 0.05)
    if not isinstance(quantile, (int, float)) or not (0 < float(quantile) < 1):
        raise ConfigError("config.algorithm.two_phase.epsilon_quantile must be in (0, 1).")
    timeout = model.get("timeout_seconds")
    if timeout is not None and (not isinstance(timeout, int) or timeout <= 0):
        raise ConfigError("config.model.timeout_seconds must be a positive integer or null.")



def infer_default_hyperparameters(parameter_count: int, observed_size: int) -> dict[str, int | float]:
    dim = max(1, int(parameter_count or 1))
    obs = max(1, int(observed_size or 1))
    pilot = max(400, 60 * dim, 10 * obs)
    accepted = max(250, 40 * dim)
    budget = max(accepted * 15, pilot * 2)
    batch_size = min(128, max(16, dim * 4))
    quantile = 0.05 if dim <= 6 else 0.03
    return {
        "pilot_size": int(pilot),
        "accepted_samples": int(accepted),
        "main_budget": int(budget),
        "batch_size": int(batch_size),
        "epsilon_quantile": float(quantile),
    }



def resolve_runtime_hyperparameters(cfg: dict[str, Any], observed_size: int | None = None) -> dict[str, Any]:
    parameter_names = cfg.get("model", {}).get("parameter_names", [])
    defaults = infer_default_hyperparameters(len(parameter_names), observed_size or 1)
    two_phase = cfg.setdefault("algorithm", {}).setdefault("two_phase", {})
    for key, value in defaults.items():
        if two_phase.get(key) in {None, "", 0}:
            two_phase[key] = value
    return two_phase
