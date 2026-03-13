"""Environment probing and framework selection."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from typing import Any


def _safe_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_module(module_name: str, probe_code: str) -> dict[str, Any]:
    if importlib.util.find_spec(module_name) is None:
        return {"available": False, "version": None, "gpu_count": 0, "devices": [], "error": "not_installed"}
    version = _package_version(module_name)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_code],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
            env=_safe_env(),
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "version": version, "gpu_count": 0, "devices": [], "error": str(exc)}
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "").strip() or f"{module_name} probe failed"
        return {"available": False, "version": version, "gpu_count": 0, "devices": [], "error": error}
    try:
        payload = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {}
    return {
        "available": True,
        "version": version,
        "gpu_count": int(payload.get("gpu_count", 0)),
        "devices": list(payload.get("devices", ["cpu"])),
        "error": None,
    }


def _probe_torch() -> dict[str, Any]:
    code = """
import json
import torch
payload = {
    "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"],
}
print(json.dumps(payload))
"""
    return _probe_module("torch", code)


def _probe_jax() -> dict[str, Any]:
    code = """
import json
import jax
devices = jax.devices()
payload = {
    "gpu_count": sum(1 for device in devices if device.platform == "gpu"),
    "devices": [f"{device.platform}:{device.id}" for device in devices],
}
print(json.dumps(payload))
"""
    return _probe_module("jax", code)


def _probe_tensorflow() -> dict[str, Any]:
    code = """
import json
import tensorflow as tf
gpus = list(tf.config.list_physical_devices("GPU"))
payload = {
    "gpu_count": len(gpus),
    "devices": [device.name for device in gpus] if gpus else ["cpu"],
}
print(json.dumps(payload))
"""
    return _probe_module("tensorflow", code)


def peek_environment() -> dict[str, Any]:
    report = {
        "torch": {
            "available": importlib.util.find_spec("torch") is not None,
            "version": _package_version("torch"),
            "gpu_count": 0,
            "devices": [],
            "error": None,
        },
        "jax": {
            "available": importlib.util.find_spec("jax") is not None,
            "version": _package_version("jax"),
            "gpu_count": 0,
            "devices": [],
            "error": None,
        },
        "tensorflow": {
            "available": importlib.util.find_spec("tensorflow") is not None,
            "version": _package_version("tensorflow"),
            "gpu_count": 0,
            "devices": [],
            "error": None,
        },
        "matplotlib": importlib.util.find_spec("matplotlib") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
    }
    report["default_framework"] = recommend_framework(report)["framework"]
    return report


@lru_cache(maxsize=1)
def probe_environment() -> dict[str, Any]:
    torch_report = _probe_torch()
    jax_report = _probe_jax()
    tensorflow_report = _probe_tensorflow()
    report = {
        "torch": torch_report,
        "jax": jax_report,
        "tensorflow": tensorflow_report,
        "matplotlib": importlib.util.find_spec("matplotlib") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
    }
    report["default_framework"] = recommend_framework(report)["framework"]
    return report


def recommend_framework(environment: dict[str, Any], requested: str | None = None) -> dict[str, Any]:
    requested_name = str(requested or "auto").lower()
    framework_reports = {
        "torch": environment.get("torch", {}),
        "jax": environment.get("jax", {}),
        "tensorflow": environment.get("tensorflow", {}),
    }
    if requested_name in framework_reports:
        selected = framework_reports[requested_name]
        if selected.get("available"):
            return {"framework": requested_name, "reason": "user_preference", "fallback": None}
    gpu_ready = [
        name
        for name in ["torch", "jax", "tensorflow"]
        if framework_reports[name].get("available") and int(framework_reports[name].get("gpu_count", 0)) > 0
    ]
    if gpu_ready:
        return {"framework": gpu_ready[0], "reason": "gpu_available", "fallback": requested_name if requested_name != "auto" else None}
    for name in ["torch", "jax", "tensorflow"]:
        if framework_reports[name].get("available"):
            return {"framework": name, "reason": "first_available", "fallback": requested_name if requested_name != "auto" else None}
    return {"framework": None, "reason": "no_supported_framework", "fallback": requested_name if requested_name != "auto" else None}
