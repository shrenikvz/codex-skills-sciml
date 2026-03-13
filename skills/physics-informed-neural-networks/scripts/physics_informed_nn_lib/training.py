"""Training runtime for Physics-Informed Neural Networks."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency at import time
    import torch
    from torch import nn
except Exception:  # noqa: BLE001 pragma: no cover
    torch = None
    nn = None

from .architectures import build_torch_model, count_parameters, model_summary
from .io_utils import extract_table_columns, write_json, write_jsonl
from .problem_spec import load_problem_spec
from .sampling import build_prediction_points, iter_subdomain_bounds, sample_domain_points
from .visualization import generate_figures


class TrainingError(RuntimeError):
    """Training failure."""


_DERIVATIVE_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)__([A-Za-z_][A-Za-z0-9_]*(?:__[A-Za-z_][A-Za-z0-9_]*)*)\b")


def _require_torch():
    if torch is None or nn is None:  # pragma: no cover
        raise TrainingError("PyTorch is not available. Install torch to run PINN training.")


def _set_seed(seed: int) -> None:
    _require_torch()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _select_device(cfg: dict[str, Any], environment: dict[str, Any] | None = None):
    _require_torch()
    preference = str(cfg.get("compute", {}).get("device_preference", "auto")).lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "gpu":
        if not torch.cuda.is_available():
            raise TrainingError("GPU execution was requested but CUDA is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _domain_arrays(problem: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(problem.get("independent_variables", []))
    domains = problem.get("domains", {})
    low = np.asarray([float(domains[name]["min"]) for name in names], dtype=float)
    high = np.asarray([float(domains[name]["max"]) for name in names], dtype=float)
    return names, low, high


def _maybe_scale_coordinates(coords, problem: dict[str, Any], enabled: bool):
    if not enabled:
        return coords
    names, low, high = _domain_arrays(problem)
    low_t = torch.as_tensor(low, dtype=coords.dtype, device=coords.device)
    high_t = torch.as_tensor(high, dtype=coords.dtype, device=coords.device)
    span = torch.clamp(high_t - low_t, min=1e-8)
    return 2.0 * (coords - low_t) / span - 1.0


def _split_derivative_sequence(spec: str, coordinate_names: list[str]) -> list[str]:
    if "__" in spec:
        return [token for token in spec.split("__") if token]
    ordered = sorted(coordinate_names, key=len, reverse=True)
    remaining = spec
    sequence: list[str] = []
    while remaining:
        matched = False
        for name in ordered:
            if remaining.startswith(name):
                sequence.append(name)
                remaining = remaining[len(name) :]
                matched = True
                break
        if not matched:
            raise TrainingError(f"Could not interpret derivative token suffix {spec!r} against coordinates {coordinate_names}.")
    return sequence


def _extract_derivative_tokens(expressions: list[str]) -> set[str]:
    tokens: set[str] = set()
    for expression in expressions:
        if not expression:
            continue
        for match in _DERIVATIVE_PATTERN.finditer(expression):
            tokens.add(match.group(0))
    return tokens


def _parameter_transform(raw, spec: dict[str, Any]):
    bounds = spec.get("bounds")
    if bounds and len(bounds) == 2:
        low, high = float(bounds[0]), float(bounds[1])
        return low + (high - low) * torch.sigmoid(raw)
    return raw


def _parameter_inverse(value: float, spec: dict[str, Any]) -> float:
    bounds = spec.get("bounds")
    if bounds and len(bounds) == 2:
        low, high = float(bounds[0]), float(bounds[1])
        clipped = min(max((float(value) - low) / max(high - low, 1e-8), 1e-6), 1.0 - 1e-6)
        return math.log(clipped / (1.0 - clipped))
    return float(value)


class PhysicsParameterModule(nn.Module):
    def __init__(self, parameter_specs: dict[str, dict[str, Any]]):
        super().__init__()
        self.specs = parameter_specs
        self.raw_parameters = nn.ParameterDict()
        for name, spec in parameter_specs.items():
            if spec.get("trainable"):
                initial = spec.get("value")
                if initial is None:
                    bounds = spec.get("bounds")
                    if bounds and len(bounds) == 2:
                        initial = 0.5 * (float(bounds[0]) + float(bounds[1]))
                    else:
                        initial = 0.1
                self.raw_parameters[name] = nn.Parameter(
                    torch.tensor(_parameter_inverse(float(initial), spec), dtype=torch.float32)
                )

    def resolved(self, device):
        values: dict[str, Any] = {}
        for name, spec in self.specs.items():
            if spec.get("trainable"):
                values[name] = _parameter_transform(self.raw_parameters[name], spec)
            else:
                values[name] = torch.tensor(float(spec.get("value", 0.0) or 0.0), dtype=torch.float32, device=device)
        return values


class UncertaintyWeightModule(nn.Module):
    def __init__(self, term_names: list[str]):
        super().__init__()
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.tensor(0.0)) for name in term_names})


def _build_math_namespace():
    return {
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "exp": torch.exp,
        "sqrt": torch.sqrt,
        "log": torch.log,
        "abs": torch.abs,
        "tanh": torch.tanh,
        "sinh": torch.sinh,
        "cosh": torch.cosh,
        "minimum": torch.minimum,
        "maximum": torch.maximum,
        "pi": math.pi,
    }


def _compute_derivative(token: str, outputs, coords, field_names: list[str], coordinate_names: list[str], cache: dict[str, Any]):
    if token in cache:
        return cache[token]
    field_name, derivative_spec = token.split("__", 1)
    if field_name not in field_names:
        raise TrainingError(f"Unknown field in derivative token: {token}")
    field_index = field_names.index(field_name)
    sequence = _split_derivative_sequence(derivative_spec, coordinate_names)
    current = outputs[:, field_index : field_index + 1]
    for coordinate_name in sequence:
        coordinate_index = coordinate_names.index(coordinate_name)
        grad = torch.autograd.grad(
            current,
            coords,
            grad_outputs=torch.ones_like(current),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        current = grad[:, coordinate_index : coordinate_index + 1]
    cache[token] = current
    return current


def _evaluate_expression(
    expression: str,
    coords,
    outputs,
    field_names: list[str],
    coordinate_names: list[str],
    parameters: dict[str, Any],
    cache: dict[str, Any],
):
    namespace = _build_math_namespace()
    namespace.update({name: coords[:, index : index + 1] for index, name in enumerate(coordinate_names)})
    namespace.update({name: outputs[:, index : index + 1] for index, name in enumerate(field_names)})
    namespace.update(parameters)
    for token in _extract_derivative_tokens([expression]):
        namespace[token] = _compute_derivative(token, outputs, coords, field_names, coordinate_names, cache)
    try:
        value = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        raise TrainingError(f"Could not evaluate expression {expression!r}: {exc}") from exc
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=coords.dtype, device=coords.device)
    if value.ndim == 0:
        value = value.reshape(1, 1).expand(coords.shape[0], 1)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    return value


def _active_loss_terms(problem: dict[str, Any], observations: dict[str, Any] | None) -> list[str]:
    terms = ["pde"]
    if problem.get("boundary_conditions"):
        terms.append("bc")
    if problem.get("initial_conditions"):
        terms.append("ic")
    if problem.get("algebraic_constraints") or problem.get("constitutive_relations"):
        terms.append("algebraic")
    if observations is not None:
        terms.append("data")
    return terms


def _prepare_observations(cfg: dict[str, Any], observed_info: dict[str, Any] | None, device):
    if observed_info is None:
        return None
    objective = cfg.get("objective", {})
    input_columns = list(objective.get("input_columns", []))
    output_columns = list(objective.get("output_columns", []))
    context_columns = list(objective.get("context_columns", []))
    coords = extract_table_columns(observed_info, input_columns)
    targets = extract_table_columns(observed_info, output_columns)
    context = extract_table_columns(observed_info, context_columns) if context_columns else np.zeros((coords.shape[0], 0), dtype=float)
    return {
        "coords": torch.as_tensor(coords, dtype=torch.float32, device=device),
        "targets": torch.as_tensor(targets, dtype=torch.float32, device=device),
        "context": torch.as_tensor(context, dtype=torch.float32, device=device),
        "input_columns": input_columns,
        "output_columns": output_columns,
        "context_columns": context_columns,
    }


def _sample_context(observations: dict[str, Any] | None, n: int, seed: int, device):
    if observations is None or observations["context"].shape[1] == 0:
        return torch.zeros((int(n), 0), dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, observations["context"].shape[0], size=int(n))
    return observations["context"][choices]


def _build_model_input(coords, context):
    if context is None or context.shape[1] == 0:
        return coords
    return torch.cat([coords, context], dim=1)


def _grad_norm(modules: list[nn.Module]) -> float:
    total = 0.0
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            total += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _normalize_loss(value, enabled: bool):
    if not enabled:
        return value
    scale = value.detach().abs().mean().clamp_min(1e-8)
    return value / scale


def _combine_loss_terms(
    cfg: dict[str, Any],
    loss_terms: dict[str, Any],
    model,
    modules: list[nn.Module],
    uncertainty_module: UncertaintyWeightModule | None,
):
    active = {name: value for name, value in loss_terms.items() if value is not None}
    strategy = str(cfg.get("loss", {}).get("weighting_strategy", "fixed")).lower()
    configured = cfg.get("loss", {}).get("weights", {})
    weights: dict[str, float] = {}
    if strategy == "fixed":
        for name in active:
            weights[name] = float(configured.get(name, 1.0))
        total = sum(float(weights[name]) * active[name] for name in active)
        return total, weights
    if strategy == "dynamic_balance":
        base = float(sum(value.detach().item() for value in active.values()) / max(1, len(active)))
        for name, value in active.items():
            weights[name] = float(base / max(abs(float(value.detach().item())), 1e-8))
        total = sum(float(weights[name]) * active[name] for name in active)
        return total, weights
    if strategy == "gradient_norm":
        anchor = next((parameter for parameter in model.parameters() if parameter.requires_grad), None)
        if anchor is None:
            raise TrainingError("Could not find trainable model parameters for gradient-norm balancing.")
        norms: dict[str, float] = {}
        for name, value in active.items():
            grad = torch.autograd.grad(value, anchor, retain_graph=True, allow_unused=True)[0]
            norms[name] = float(torch.norm(grad).detach().item()) if grad is not None else 1.0
        baseline = sum(norms.values()) / max(1, len(norms))
        for name, norm in norms.items():
            weights[name] = float(baseline / max(norm, 1e-8))
        total = sum(float(weights[name]) * active[name] for name in active)
        return total, weights
    if strategy == "uncertainty":
        if uncertainty_module is None:
            raise TrainingError("Uncertainty weighting was requested but the uncertainty module is unavailable.")
        total = None
        for name, value in active.items():
            log_var = uncertainty_module.log_vars[name]
            term = torch.exp(-log_var) * value + log_var
            total = term if total is None else total + term
            weights[name] = float(torch.exp(-log_var.detach()).item())
        if total is None:
            raise TrainingError("No active loss terms were available.")
        return total, weights
    if strategy == "adaptive_residual":
        pde = active.get("pde")
        base = max(abs(float(pde.detach().item())) if pde is not None else 1.0, 1e-8)
        for name in active:
            weights[name] = float(configured.get(name, 1.0))
        if "pde" in active:
            weights["pde"] = float(1.0 / base)
        total = sum(float(weights[name]) * active[name] for name in active)
        return total, weights
    raise TrainingError(f"Unsupported loss-weighting strategy: {strategy}")


def _residual_bundle(model, parameter_module, problem: dict[str, Any], coords_np: np.ndarray, context_np: np.ndarray | None, scale_inputs: bool, device):
    coordinate_names = list(problem.get("independent_variables", []))
    field_names = list(problem.get("dependent_variables", []))
    coords = torch.as_tensor(coords_np, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
    context = torch.as_tensor(context_np, dtype=torch.float32, device=device) if context_np is not None else torch.zeros((coords.shape[0], 0), dtype=torch.float32, device=device)
    inputs = _build_model_input(_maybe_scale_coordinates(coords, problem, scale_inputs), context)
    outputs = model(inputs)
    parameters = parameter_module.resolved(device)
    cache: dict[str, Any] = {}
    equations = list(problem.get("equations", []))
    algebraic = list(problem.get("algebraic_constraints", [])) + list(problem.get("constitutive_relations", []))
    pde_terms = [
        _evaluate_expression(item["expression"], coords, outputs, field_names, coordinate_names, parameters, cache)
        for item in equations
    ]
    algebraic_terms = [
        _evaluate_expression(item["expression"], coords, outputs, field_names, coordinate_names, parameters, cache)
        for item in algebraic
    ]
    return {"coords": coords, "outputs": outputs, "parameters": parameters, "pde_terms": pde_terms, "algebraic_terms": algebraic_terms}


def _compute_loss_terms(
    cfg: dict[str, Any],
    problem: dict[str, Any],
    model,
    parameter_module: PhysicsParameterModule,
    observations: dict[str, Any] | None,
    seed: int,
    adaptive_points: np.ndarray | None,
    device,
):
    sampling_cfg = cfg.get("sampling", {})
    training_cfg = cfg.get("training", {})
    stabilization_cfg = cfg.get("stabilization", {})
    scale_inputs = bool(stabilization_cfg.get("coordinate_scaling", True) not in {False, "false", "none"})
    normalize_residuals = bool(stabilization_cfg.get("residual_normalization", True) not in {False, "false", "none"})
    strategy = sampling_cfg.get("strategy", "uniform")
    subdomain_count = int(training_cfg.get("domain_decomposition", {}).get("num_subdomains", 1))
    subdomains = iter_subdomain_bounds(problem, subdomain_count)
    coordinate_names = list(problem.get("independent_variables", []))
    field_names = list(problem.get("dependent_variables", []))
    parameter_values = parameter_module.resolved(device)
    pde_losses: list[Any] = []
    algebraic_losses: list[Any] = []
    for sub_index, bounds in enumerate(subdomains):
        sub_seed = int(seed + sub_index * 97)
        points = sample_domain_points(problem, int(sampling_cfg["interior_points"]) // max(1, subdomain_count), strategy, sub_seed, bounds_override=bounds)
        if adaptive_points is not None and adaptive_points.size:
            points = np.vstack([points, adaptive_points])
        context_np = None
        if observations is not None and observations["context"].shape[1] > 0:
            context_np = _sample_context(observations, points.shape[0], sub_seed + 13, device).detach().cpu().numpy()
        bundle = _residual_bundle(model, parameter_module, problem, points, context_np, scale_inputs, device)
        for residual in bundle["pde_terms"]:
            pde_losses.append(_normalize_loss(torch.mean(residual.pow(2)), normalize_residuals))
        for residual in bundle["algebraic_terms"]:
            algebraic_losses.append(_normalize_loss(torch.mean(residual.pow(2)), normalize_residuals))
    loss_terms: dict[str, Any] = {
        "pde": torch.stack(pde_losses).mean() if pde_losses else None,
        "algebraic": torch.stack(algebraic_losses).mean() if algebraic_losses else None,
        "bc": None,
        "ic": None,
        "data": None,
    }

    def condition_loss(items: list[dict[str, Any]], count: int, seed_offset: int):
        values = []
        for index, item in enumerate(items):
            location = dict(item.get("location", {}))
            points = sample_domain_points(problem, count, strategy, seed + seed_offset + index * 37, fixed_location=location)
            context_np = None
            if observations is not None and observations["context"].shape[1] > 0:
                context_np = _sample_context(observations, points.shape[0], seed + seed_offset + index * 41, device).detach().cpu().numpy()
            bundle = _residual_bundle(model, parameter_module, problem, points, context_np, scale_inputs, device)
            expression = item.get("expression")
            if not expression:
                continue
            residual = _evaluate_expression(
                expression,
                bundle["coords"],
                bundle["outputs"],
                field_names,
                coordinate_names,
                bundle["parameters"],
                {},
            )
            values.append(_normalize_loss(torch.mean(residual.pow(2)), normalize_residuals))
        return torch.stack(values).mean() if values else None

    loss_terms["bc"] = condition_loss(list(problem.get("boundary_conditions", [])), int(sampling_cfg["boundary_points"]), 1000)
    loss_terms["ic"] = condition_loss(list(problem.get("initial_conditions", [])), int(sampling_cfg["initial_points"]), 2000)

    if observations is not None:
        coords = observations["coords"]
        context = observations["context"]
        scaled = _maybe_scale_coordinates(coords, problem, scale_inputs)
        inputs = _build_model_input(scaled, context)
        predictions = model(inputs)
        if predictions.shape[1] < observations["targets"].shape[1]:
            raise TrainingError("Model output dimension is smaller than the observation target dimension.")
        data_residual = predictions[:, : observations["targets"].shape[1]] - observations["targets"]
        loss_terms["data"] = torch.mean(data_residual.pow(2))

    return loss_terms


def _adaptive_refinement(
    cfg: dict[str, Any],
    problem: dict[str, Any],
    model,
    parameter_module: PhysicsParameterModule,
    observations: dict[str, Any] | None,
    epoch: int,
    device,
) -> np.ndarray | None:
    adaptive_cfg = cfg.get("sampling", {}).get("adaptive", {})
    if not adaptive_cfg.get("enabled"):
        return None
    if epoch % int(adaptive_cfg.get("interval", 250)) != 0:
        return None
    candidate_pool = int(adaptive_cfg.get("candidate_pool", 4096))
    top_k = int(adaptive_cfg.get("top_k", 512))
    strategy = cfg.get("sampling", {}).get("strategy", "uniform")
    points = sample_domain_points(problem, candidate_pool, strategy, epoch + 901)
    context_np = None
    if observations is not None and observations["context"].shape[1] > 0:
        context_np = _sample_context(observations, points.shape[0], epoch + 917, device).detach().cpu().numpy()
    bundle = _residual_bundle(model, parameter_module, problem, points, context_np, True, device)
    residuals = []
    for residual in bundle["pde_terms"]:
        residuals.append(residual.detach().abs().reshape(-1))
    if not residuals:
        return None
    score = torch.stack(residuals).mean(dim=0)
    top_indices = torch.topk(score, k=min(top_k, score.shape[0])).indices.detach().cpu().numpy()
    return points[top_indices]


def _prepare_analytical(problem: dict[str, Any], coords, parameters: dict[str, Any]):
    analytical = problem.get("analytical_solution")
    if analytical is None:
        return None
    expressions: dict[str, str]
    if isinstance(analytical, str):
        field_names = list(problem.get("dependent_variables", []))
        if len(field_names) != 1:
            return None
        expressions = {field_names[0]: analytical}
    elif isinstance(analytical, dict):
        expressions = {str(key): str(value) for key, value in analytical.items()}
    else:
        return None
    names = list(problem.get("independent_variables", []))
    namespace = _build_math_namespace()
    namespace.update({name: coords[:, index : index + 1] for index, name in enumerate(names)})
    namespace.update(parameters)
    outputs: dict[str, Any] = {}
    for field_name, expression in expressions.items():
        value = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=coords.dtype, device=coords.device)
        outputs[field_name] = value.reshape(-1).detach().cpu().numpy()
    return outputs


def run_torch_training(cfg: dict[str, Any], workdir: Path, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    _require_torch()
    problem_path = Path(cfg["problem"]["path"])
    if not problem_path.is_absolute():
        problem_path = (workdir / problem_path).resolve()
    problem = load_problem_spec(str(problem_path), callable_name=cfg["problem"].get("callable"))
    problem.update({key: value for key, value in cfg.get("problem", {}).items() if key in problem})
    observed_path = cfg.get("objective", {}).get("observation_path")
    observed_info = None
    if observed_path:
        from .io_utils import load_observed_data

        observed_info = load_observed_data(str((workdir / observed_path).resolve() if not Path(observed_path).is_absolute() else observed_path))
    _set_seed(int(cfg.get("training", {}).get("seed", 7)))
    device = _select_device(cfg, environment=environment)
    observations = _prepare_observations(cfg, observed_info, device)
    input_dim = len(problem.get("independent_variables", [])) + (observations["context"].shape[1] if observations is not None else 0)
    output_dim = len(problem.get("dependent_variables", []))
    model = build_torch_model(cfg.get("model", {}), input_dim=input_dim, output_dim=output_dim).to(device)
    parameter_module = PhysicsParameterModule(problem.get("parameters", {})).to(device)
    active_terms = _active_loss_terms(problem, observations)
    uncertainty_module = UncertaintyWeightModule(active_terms).to(device) if cfg.get("loss", {}).get("weighting_strategy") == "uncertainty" else None
    modules: list[nn.Module] = [model, parameter_module]
    if uncertainty_module is not None:
        modules.append(uncertainty_module)
    trainable_parameters = list(model.parameters()) + list(parameter_module.parameters())
    if uncertainty_module is not None:
        trainable_parameters.extend(list(uncertainty_module.parameters()))
    if not trainable_parameters:
        raise TrainingError("No trainable parameters were found for the PINN run.")
    optimizer_name = str(cfg.get("training", {}).get("optimizer", "hybrid")).lower()
    learning_rate = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    adam = torch.optim.Adam(trainable_parameters, lr=learning_rate)
    grad_clip = cfg.get("training", {}).get("gradient_clip_norm")
    if grad_clip is None and cfg.get("stabilization", {}).get("gradient_clipping", True) not in {False, "false", "none"}:
        grad_clip = 1.0
    epochs = int(cfg.get("training", {}).get("epochs", 2500))
    adam_epochs = int(cfg.get("training", {}).get("adam_epochs", epochs if optimizer_name == "adam" else max(1, int(0.8 * epochs))))
    lbfgs_steps = int(cfg.get("training", {}).get("lbfgs_steps", 0 if optimizer_name == "adam" else 300))
    patience = int(cfg.get("training", {}).get("early_stopping_patience", 500))
    log_interval = max(1, int(cfg.get("training", {}).get("log_interval", 50)))
    history: list[dict[str, Any]] = []
    adaptive_points: np.ndarray | None = None
    best_loss = float("inf")
    stagnant_epochs = 0
    instability_events: list[str] = []

    def record(epoch: int, total_loss_value: float, loss_terms: dict[str, Any], weights: dict[str, float], grad_norm_value: float, stage: str):
        history.append(
            {
                "epoch": int(epoch),
                "stage": stage,
                "total_loss": float(total_loss_value),
                "pde_loss": float(loss_terms["pde"].detach().item()) if loss_terms.get("pde") is not None else None,
                "bc_loss": float(loss_terms["bc"].detach().item()) if loss_terms.get("bc") is not None else None,
                "ic_loss": float(loss_terms["ic"].detach().item()) if loss_terms.get("ic") is not None else None,
                "data_loss": float(loss_terms["data"].detach().item()) if loss_terms.get("data") is not None else None,
                "algebraic_loss": float(loss_terms["algebraic"].detach().item()) if loss_terms.get("algebraic") is not None else None,
                "gradient_norm": float(grad_norm_value),
                "weights": dict(weights),
            }
        )

    for epoch in range(1, adam_epochs + 1):
        adam.zero_grad(set_to_none=True)
        loss_terms = _compute_loss_terms(cfg, problem, model, parameter_module, observations, seed=epoch + 17, adaptive_points=adaptive_points, device=device)
        total_loss, weights = _combine_loss_terms(cfg, loss_terms, model, modules, uncertainty_module)
        if not torch.isfinite(total_loss):
            raise TrainingError(f"Encountered a non-finite loss at epoch {epoch}.")
        total_loss.backward()
        grad_norm_value = _grad_norm(modules)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(grad_clip))
        adam.step()
        total_loss_value = float(total_loss.detach().item())
        record(epoch, total_loss_value, loss_terms, weights, grad_norm_value, stage="adam")
        if not math.isfinite(grad_norm_value) or grad_norm_value > 1e4:
            instability_events.append(f"Large gradient norm detected at epoch {epoch}: {grad_norm_value:.3e}")
        if total_loss_value + 1e-9 < best_loss:
            best_loss = total_loss_value
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
        if stagnant_epochs >= patience:
            instability_events.append(f"Early stopping triggered after {epoch} epochs without improvement.")
            break
        new_adaptive = _adaptive_refinement(cfg, problem, model, parameter_module, observations, epoch, device)
        if new_adaptive is not None:
            adaptive_points = new_adaptive

    if optimizer_name in {"hybrid", "lbfgs"} and lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(trainable_parameters, max_iter=1, history_size=50, line_search_fn="strong_wolfe")
        for step in range(1, lbfgs_steps + 1):
            seed = 200000 + step

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                terms = _compute_loss_terms(cfg, problem, model, parameter_module, observations, seed=seed, adaptive_points=adaptive_points, device=device)
                total, _ = _combine_loss_terms(cfg, terms, model, modules, uncertainty_module)
                if not torch.isfinite(total):
                    raise TrainingError(f"Encountered a non-finite loss during L-BFGS step {step}.")
                total.backward()
                return total

            total_loss = lbfgs.step(closure)
            loss_terms = _compute_loss_terms(cfg, problem, model, parameter_module, observations, seed=seed + 1, adaptive_points=adaptive_points, device=device)
            _, weights = _combine_loss_terms(cfg, loss_terms, model, modules, uncertainty_module)
            grad_norm_value = _grad_norm(modules)
            record(adam_epochs + step, float(total_loss.detach().item()), loss_terms, weights, grad_norm_value, stage="lbfgs")
            if step % log_interval == 0:
                adaptive_points = _adaptive_refinement(cfg, problem, model, parameter_module, observations, adam_epochs + step, device) or adaptive_points

    results_dir = workdir / cfg.get("output", {}).get("results_dir", "results")
    artifacts_dir = workdir / cfg.get("output", {}).get("artifacts_dir", "artifacts")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    prediction_points, prediction_shape = build_prediction_points(problem, int(cfg.get("evaluation", {}).get("prediction_points", 1024)))
    context_np = None
    if observations is not None and observations["context"].shape[1] > 0:
        context_np = observations["context"].mean(dim=0, keepdim=True).repeat(prediction_points.shape[0], 1).detach().cpu().numpy()
    bundle = _residual_bundle(model, parameter_module, problem, prediction_points, context_np, True, device)
    coordinates = {
        name: prediction_points[:, index]
        for index, name in enumerate(problem.get("independent_variables", []))
    }
    outputs = {
        name: bundle["outputs"][:, index].detach().cpu().numpy()
        for index, name in enumerate(problem.get("dependent_variables", []))
    }
    analytical = None
    analytical_outputs = _prepare_analytical(problem, bundle["coords"], bundle["parameters"])
    if analytical_outputs:
        field_name = sorted(analytical_outputs.keys())[0]
        analytical = {"field": field_name, "values": analytical_outputs[field_name]}
    predictions = {
        "coordinates": coordinates,
        "outputs": outputs,
        "shape": prediction_shape,
    }

    residual_points, residual_shape = build_prediction_points(problem, int(cfg.get("evaluation", {}).get("residual_points", 2048)))
    residual_context = None
    if observations is not None and observations["context"].shape[1] > 0:
        residual_context = observations["context"].mean(dim=0, keepdim=True).repeat(residual_points.shape[0], 1).detach().cpu().numpy()
    residual_bundle = _residual_bundle(model, parameter_module, problem, residual_points, residual_context, True, device)
    residual_magnitudes = []
    equation_names = []
    for item, residual in zip(problem.get("equations", []), residual_bundle["pde_terms"], strict=False):
        equation_names.append(item["name"])
        residual_magnitudes.append(residual.detach().abs().reshape(-1))
    if residual_magnitudes:
        stacked_residuals = torch.stack(residual_magnitudes)
        residual_mean = stacked_residuals.mean(dim=0)
        residual_diagnostics = {
            "equations": {
                name: {
                    "mean_abs_residual": float(residual.detach().abs().mean().item()),
                    "max_abs_residual": float(residual.detach().abs().max().item()),
                }
                for name, residual in zip(equation_names, residual_bundle["pde_terms"], strict=False)
            },
            "grid": {
                "coordinates": {name: residual_points[:, index] for index, name in enumerate(problem.get("independent_variables", []))},
                "values": residual_mean.detach().cpu().numpy(),
                "shape": residual_shape,
            },
        }
    else:
        residual_diagnostics = {"equations": {}, "grid": None}

    inferred_parameters = {
        name: float(value.detach().item()) if torch.is_tensor(value) else float(value)
        for name, value in parameter_module.resolved(device).items()
    }

    evaluation_summary: dict[str, Any] = {
        "prediction_shape": prediction_shape,
        "parameter_count": count_parameters(model) + sum(parameter.numel() for parameter in parameter_module.parameters()),
        "framework": "torch",
        "device": str(device),
    }
    if observations is not None:
        scaled = _maybe_scale_coordinates(observations["coords"], problem, True)
        data_predictions = model(_build_model_input(scaled, observations["context"]))
        targets = observations["targets"]
        residual = data_predictions[:, : targets.shape[1]] - targets
        evaluation_summary["data_metrics"] = {
            "rmse": float(torch.sqrt(torch.mean(residual.pow(2))).detach().item()),
            "mae": float(torch.mean(residual.abs()).detach().item()),
        }
    if analytical_outputs:
        field_name = sorted(analytical_outputs.keys())[0]
        analytical_tensor = torch.as_tensor(analytical_outputs[field_name], dtype=torch.float32, device=device)
        prediction_tensor = bundle["outputs"][:, problem["dependent_variables"].index(field_name)]
        diff = prediction_tensor.reshape(-1) - analytical_tensor.reshape(-1)
        evaluation_summary["analytical_comparison"] = {
            "field": field_name,
            "rmse": float(torch.sqrt(torch.mean(diff.pow(2))).detach().item()),
            "max_abs_error": float(torch.max(torch.abs(diff)).detach().item()),
        }

    if history:
        recent = history[-min(50, len(history)) :]
        losses = [record["total_loss"] for record in recent]
        span = max(losses) - min(losses) if losses else 0.0
    else:
        span = 0.0
    diagnostics = {
        "final_loss": history[-1]["total_loss"] if history else None,
        "history_points": len(history),
        "loss_span_last_window": float(span),
        "instability_events": instability_events,
        "warnings": [],
    }
    if span < 1e-6 and history:
        diagnostics["warnings"].append("Training appears to have stagnated. Consider changing scaling, collocation density, or architecture.")
    if residual_diagnostics.get("equations"):
        worst = max(
            (report["mean_abs_residual"] for report in residual_diagnostics["equations"].values()),
            default=0.0,
        )
        if worst > 1.0:
            diagnostics["warnings"].append("Residual errors remain large. Increase collocation density or adjust loss balancing.")

    plot_report = generate_figures(
        requested_plots=list(cfg.get("visualization", {}).get("plots", [])),
        output_dir=figures_dir,
        history=history,
        predictions=predictions,
        residual_grid=residual_diagnostics.get("grid"),
        analytical=analytical,
        dpi=int(cfg.get("visualization", {}).get("dpi", 140)),
    ) if cfg.get("visualization", {}).get("enabled") else {"generated": [], "skipped": []}

    model_state_path = artifacts_dir / "model_state.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "parameter_state_dict": parameter_module.state_dict(),
            "config": cfg,
            "problem": problem,
        },
        model_state_path,
    )

    write_jsonl(results_dir / "training_history.jsonl", history)
    write_json(results_dir / "diagnostics.json", diagnostics)
    write_json(results_dir / "residual_diagnostics.json", residual_diagnostics)
    write_json(results_dir / "evaluation_summary.json", evaluation_summary)
    write_json(results_dir / "inferred_parameters.json", inferred_parameters)
    write_json(results_dir / "solution_predictions.json", predictions)
    artifact_index = {
        "results": [
            str(results_dir / "training_history.jsonl"),
            str(results_dir / "diagnostics.json"),
            str(results_dir / "residual_diagnostics.json"),
            str(results_dir / "evaluation_summary.json"),
            str(results_dir / "inferred_parameters.json"),
            str(results_dir / "solution_predictions.json"),
        ],
        "artifacts": [str(model_state_path)],
        "figures": plot_report["generated"],
    }
    write_json(results_dir / "artifact_index.json", artifact_index)
    run_summary = {
        "framework": "torch",
        "device": str(device),
        "model": model_summary(cfg.get("model", {}), input_dim, output_dim),
        "trainable_parameters": count_parameters(model),
        "physics_parameters": inferred_parameters,
        "diagnostics": diagnostics,
        "figures": plot_report,
    }
    write_json(results_dir / "run_summary.json", run_summary)
    return {
        "results_dir": str(results_dir),
        "artifacts_dir": str(artifacts_dir),
        "framework": "torch",
        "device": str(device),
        "trainable_parameters": count_parameters(model),
        "physics_parameters": inferred_parameters,
        "figures": plot_report,
        "artifact_index": artifact_index,
    }
