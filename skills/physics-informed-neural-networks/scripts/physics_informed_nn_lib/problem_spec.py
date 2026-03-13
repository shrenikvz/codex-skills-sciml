"""Problem specification loaders and normalizers."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any


class ProblemSpecError(RuntimeError):
    """Problem-spec loading or normalization failure."""


_DERIVATIVE_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)__([A-Za-z_][A-Za-z0-9_]*)\b")
_RANGE_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+in\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]")


def _load_python_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ProblemSpecError(f"Could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_location_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if text in {"min", "max"}:
            return text
        try:
            return float(text)
        except ValueError:
            return text
    return value


def _normalize_domains(raw: Any) -> dict[str, dict[str, float]]:
    if not raw:
        return {}
    domains: dict[str, dict[str, float]] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                items.append((str(item["name"]), item))
    else:
        items = []
    for name, value in items:
        if isinstance(value, dict):
            lower = value.get("min", value.get("lower", value.get("start")))
            upper = value.get("max", value.get("upper", value.get("end")))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            lower, upper = value[0], value[1]
        else:
            continue
        try:
            domains[str(name)] = {"min": float(lower), "max": float(upper)}
        except (TypeError, ValueError):
            continue
    return domains


def _normalize_expression(text: Any) -> str | None:
    if text is None:
        return None
    expression = str(text).strip().replace("^", "**")
    if "=" in expression and "==" not in expression:
        lhs, rhs = expression.split("=", 1)
        expression = f"({lhs.strip()}) - ({rhs.strip()})"
    return expression


def _normalize_equations(items: Any, key_name: str) -> list[dict[str, Any]]:
    if not items:
        return []
    if isinstance(items, dict):
        items = [items]
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if isinstance(item, str):
            expression = _normalize_expression(item)
            if expression:
                out.append({"name": f"{key_name}_{idx}", "expression": expression, "weight": 1.0})
            continue
        if not isinstance(item, dict):
            continue
        expression = _normalize_expression(item.get("expression", item.get("equation", item.get("residual"))))
        if not expression:
            continue
        out.append(
            {
                "name": str(item.get("name", f"{key_name}_{idx}")),
                "expression": expression,
                "weight": float(item.get("weight", 1.0)),
            }
        )
    return out


def _condition_expression(item: dict[str, Any], prefix: str, index: int) -> dict[str, Any]:
    payload = dict(item)
    payload["name"] = str(payload.get("name", f"{prefix}_{index}"))
    payload["type"] = str(payload.get("type", "dirichlet")).lower()
    payload["location"] = {
        str(key): _parse_location_value(value) for key, value in dict(payload.get("location", payload.get("where", {}))).items()
    }
    payload["field"] = payload.get("field")
    payload["value"] = payload.get("value")
    payload["expression"] = _normalize_expression(payload.get("expression", payload.get("residual")))
    if not payload["expression"] and payload.get("field") and payload.get("value") is not None:
        field = str(payload["field"])
        value = str(payload["value"])
        if payload["type"] == "dirichlet":
            payload["expression"] = f"{field} - ({value})"
        elif payload["type"] == "neumann":
            normal = payload.get("normal") or next(iter(payload["location"]), None)
            if normal:
                payload["expression"] = f"{field}__{normal} - ({value})"
    return payload


def _normalize_conditions(items: Any, prefix: str) -> list[dict[str, Any]]:
    if not items:
        return []
    if isinstance(items, dict):
        items = [items]
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            expression = _normalize_expression(item)
            if expression:
                out.append({"name": f"{prefix}_{index}", "type": "raw", "location": {}, "expression": expression})
            continue
        if isinstance(item, dict):
            out.append(_condition_expression(item, prefix, index))
    return out


def _normalize_parameters(raw: Any) -> dict[str, dict[str, Any]]:
    if not raw:
        return {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw, list):
        items = []
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                items.append((str(item["name"]), item))
    elif isinstance(raw, dict):
        items = list(raw.items())
    else:
        items = []
    for name, value in items:
        if isinstance(value, dict):
            bounds = value.get("bounds")
            if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                bounds = [float(bounds[0]), float(bounds[1])]
            else:
                bounds = None
            out[str(name)] = {
                "value": value.get("value"),
                "trainable": bool(value.get("trainable", False)),
                "bounds": bounds,
            }
        else:
            out[str(name)] = {"value": value, "trainable": False, "bounds": None}
    return out


def _extract_variables_from_text(expressions: list[str]) -> set[str]:
    variables: set[str] = set()
    for expression in expressions:
        for match in _DERIVATIVE_PATTERN.finditer(expression):
            variables.add(match.group(1))
    return variables


def _parse_text_spec(text: str) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "description": "",
        "independent_variables": [],
        "dependent_variables": [],
        "domains": {},
        "equations": [],
        "boundary_conditions": [],
        "initial_conditions": [],
        "parameters": {},
        "unknown_parameters": [],
        "metadata": {"raw_text": text},
    }
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    spec["description"] = lines[0] if lines else ""
    for match in _RANGE_PATTERN.finditer(text):
        spec.setdefault("domains", {})[match.group(1)] = {"min": float(match.group(2)), "max": float(match.group(3))}
    lower_text = text.lower()
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("independent variables:"):
            payload = line.split(":", 1)[1]
            spec["independent_variables"] = [item.strip() for item in payload.split(",") if item.strip()]
        elif lowered.startswith("dependent variables:"):
            payload = line.split(":", 1)[1]
            spec["dependent_variables"] = [item.strip() for item in payload.split(",") if item.strip()]
        elif lowered.startswith(("equation:", "ode:", "pde:")):
            payload = line.split(":", 1)[1].strip()
            spec["equations"].append({"expression": _normalize_expression(payload)})
        elif lowered.startswith(("boundary:", "bc:")):
            payload = line.split(":", 1)[1].strip()
            location_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_.+-]+)\s*->\s*(.+)", payload)
            if location_match:
                lhs = location_match.group(3).strip()
                field_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", lhs)
                item: dict[str, Any] = {
                    "location": {location_match.group(1): _parse_location_value(location_match.group(2))},
                }
                if field_match:
                    item["field"] = field_match.group(1).strip()
                    item["value"] = field_match.group(2).strip()
                else:
                    item["expression"] = lhs
                spec["boundary_conditions"].append(item)
        elif lowered.startswith(("initial:", "ic:")):
            payload = line.split(":", 1)[1].strip()
            location_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_.+-]+)\s*->\s*(.+)", payload)
            if location_match:
                lhs = location_match.group(3).strip()
                field_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)", lhs)
                item = {"location": {location_match.group(1): _parse_location_value(location_match.group(2))}}
                if field_match:
                    item["field"] = field_match.group(1).strip()
                    item["value"] = field_match.group(2).strip()
                else:
                    item["expression"] = lhs
                spec["initial_conditions"].append(item)
    if not spec["equations"] and ("__" in text or "d/d" in lower_text or "du/d" in lower_text):
        for clause in re.split(r"[;\n]+", text):
            if "__" in clause or "d/d" in clause.lower():
                spec["equations"].append({"expression": _normalize_expression(clause)})
    return spec


def load_problem_spec(
    path: str | None = None,
    physics_text: str | None = None,
    callable_name: str | None = None,
) -> dict[str, Any]:
    if physics_text and not path:
        return normalize_problem_spec(_parse_text_spec(physics_text), source="inline_text")
    if not path:
        raise ProblemSpecError("Either problem_path or physics_text is required.")
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise ProblemSpecError(f"Problem path does not exist: {target}")
    suffix = target.suffix.lower()
    if suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
        return normalize_problem_spec(payload, source="json", source_path=str(target))
    if suffix in {".md", ".txt"}:
        payload = _parse_text_spec(target.read_text(encoding="utf-8"))
        return normalize_problem_spec(payload, source="text", source_path=str(target))
    if suffix == ".py":
        module = _load_python_module(target)
        if callable_name and hasattr(module, callable_name):
            payload = getattr(module, callable_name)()
        elif hasattr(module, "build_problem") and callable(getattr(module, "build_problem")):
            payload = module.build_problem()
        elif hasattr(module, "problem_spec"):
            payload = getattr(module, "problem_spec")
        else:
            raise ProblemSpecError(f"Python problem file {target.name} must expose build_problem() or problem_spec.")
        return normalize_problem_spec(payload, source="python", source_path=str(target), callable_name=callable_name)
    raise ProblemSpecError(f"Unsupported problem format: {suffix}")


def normalize_problem_spec(
    payload: Any,
    source: str | None = None,
    source_path: str | None = None,
    callable_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProblemSpecError("Problem specification must be a JSON-like object.")
    independent = list(payload.get("independent_variables", payload.get("inputs", [])) or [])
    domains = _normalize_domains(payload.get("domains", payload.get("domain", {})))
    if not independent and domains:
        independent = list(domains.keys())
    equations = _normalize_equations(payload.get("equations", payload.get("residuals", payload.get("pde", []))), "equation")
    algebraic = _normalize_equations(payload.get("algebraic_constraints", []), "algebraic")
    constitutive = _normalize_equations(payload.get("constitutive_relations", []), "constitutive")
    dependent = list(payload.get("dependent_variables", payload.get("outputs", [])) or [])
    if not dependent:
        dependent = sorted(_extract_variables_from_text([item["expression"] for item in equations + algebraic + constitutive]))
    boundary = _normalize_conditions(payload.get("boundary_conditions", payload.get("boundary", [])), "bc")
    initial = _normalize_conditions(payload.get("initial_conditions", payload.get("initial", [])), "ic")
    parameters = _normalize_parameters(payload.get("parameters", {}))
    unknown_parameters = list(payload.get("unknown_parameters", []))
    if not unknown_parameters:
        unknown_parameters = [name for name, spec in parameters.items() if spec.get("trainable")]
    for name in unknown_parameters:
        if name not in parameters:
            parameters[name] = {"value": None, "trainable": True, "bounds": None}
        else:
            parameters[name]["trainable"] = True
    analytical = payload.get("analytical_solution", payload.get("exact_solution"))
    return {
        "description": str(payload.get("description", "")),
        "source": source,
        "source_path": source_path,
        "callable": callable_name,
        "independent_variables": independent,
        "dependent_variables": dependent,
        "domains": domains,
        "equations": equations,
        "boundary_conditions": boundary,
        "initial_conditions": initial,
        "algebraic_constraints": algebraic,
        "constitutive_relations": constitutive,
        "parameters": parameters,
        "unknown_parameters": unknown_parameters,
        "analytical_solution": analytical,
        "metadata": dict(payload.get("metadata", {})),
    }


def summarize_problem(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": spec.get("source"),
        "source_path": spec.get("source_path"),
        "independent_variables": list(spec.get("independent_variables", [])),
        "dependent_variables": list(spec.get("dependent_variables", [])),
        "domain_count": len(spec.get("domains", {})),
        "equation_count": len(spec.get("equations", [])),
        "boundary_condition_count": len(spec.get("boundary_conditions", [])),
        "initial_condition_count": len(spec.get("initial_conditions", [])),
        "unknown_parameters": list(spec.get("unknown_parameters", [])),
        "has_analytical_solution": spec.get("analytical_solution") is not None,
    }

