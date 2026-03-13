"""Environment probing for JAX and BlackJAX."""

from __future__ import annotations

import platform
import sys
from typing import Any


def _failure_hint(message: str) -> str | None:
    lowered = message.lower()
    if "avx" in lowered and "macos" in lowered:
        return "The installed jaxlib appears to target the wrong CPU architecture. On Apple Silicon, use an ARM Python environment and reinstall jax/jaxlib."
    if "avx" in lowered:
        return "The installed jaxlib was compiled for unsupported CPU features. Install a build compatible with this machine."
    if "cuda" in lowered:
        return "JAX found a CUDA-related issue. Verify the installed CUDA-enabled jaxlib matches the local driver stack."
    return None


def probe_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": {"installed": False, "importable": False},
        "blackjax": {"installed": False, "importable": False},
        "devices": [],
    }
    try:
        import jax  # noqa: PLC0415

        result["jax"] = {
            "installed": True,
            "importable": True,
            "version": getattr(jax, "__version__", "unknown"),
        }
        try:
            result["devices"] = [
                {
                    "id": idx,
                    "kind": device.platform,
                    "device": str(device),
                    "process_index": getattr(device, "process_index", None),
                }
                for idx, device in enumerate(jax.devices())
            ]
        except Exception as exc:  # noqa: BLE001
            result["jax"]["device_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        result["jax"] = {
            "installed": True,
            "importable": False,
            "error": str(exc),
            "hint": _failure_hint(str(exc)),
        }
    try:
        import blackjax  # noqa: PLC0415

        result["blackjax"] = {
            "installed": True,
            "importable": True,
            "version": getattr(blackjax, "__version__", "unknown"),
        }
    except Exception as exc:  # noqa: BLE001
        result["blackjax"] = {
            "installed": True,
            "importable": False,
            "error": str(exc),
            "hint": _failure_hint(str(exc)),
        }
    result["recommended_backend"] = "blackjax" if result["jax"]["importable"] and result["blackjax"]["importable"] else None
    return result
