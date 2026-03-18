"""Backend execution for NUTS calibration."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import simulate_model
from .io_utils import payload_to_array
from .likelihoods import loglikelihood_numpy
from .transforms import default_unconstrained_position, vector_to_parameter_dict_numpy


class BackendError(RuntimeError):
    """Backend execution failure."""


def _import_jax_stack(enable_x64: bool) -> tuple[Any, Any, Any, Any, Any]:
    try:
        jax = importlib.import_module("jax")
        if enable_x64:
            jax.config.update("jax_enable_x64", True)
        jnp = importlib.import_module("jax.numpy")
        jsp_special = importlib.import_module("jax.scipy.special")
        jnn = importlib.import_module("jax.nn")
        blackjax = importlib.import_module("blackjax")
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"Could not import JAX/BlackJAX: {exc}") from exc
    return jax, jnp, jsp_special, jnn, blackjax


def _fit_scaling(observed: np.ndarray, scaling_cfg: dict[str, Any]) -> dict[str, Any]:
    mode = str(scaling_cfg.get("mode", "none")).lower()
    enabled = bool(scaling_cfg.get("enabled")) and mode != "none"
    observed = np.asarray(observed, dtype=float).reshape(-1)
    if not enabled or observed.size == 0:
        return {"enabled": False, "mode": "none", "center": np.zeros_like(observed), "scale": np.ones_like(observed)}
    if mode == "zscore":
        center = np.mean(observed)
        scale = np.std(observed)
    elif mode == "minmax":
        lower = np.min(observed)
        upper = np.max(observed)
        center = lower
        scale = upper - lower
    elif mode == "variance":
        center = 0.0
        scale = np.std(observed)
    else:
        return {"enabled": False, "mode": "none", "center": np.zeros_like(observed), "scale": np.ones_like(observed)}
    scale = np.where(np.asarray(scale) == 0, 1.0, scale)
    return {
        "enabled": True,
        "mode": mode,
        "center": np.full(observed.shape, float(center), dtype=float) if np.asarray(center).ndim == 0 else np.asarray(center, dtype=float),
        "scale": np.full(observed.shape, float(scale), dtype=float) if np.asarray(scale).ndim == 0 else np.asarray(scale, dtype=float),
    }


def _apply_scaling_numpy(values: np.ndarray, scaling_state: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not scaling_state.get("enabled"):
        return array
    return (array - np.asarray(scaling_state["center"], dtype=float)) / np.asarray(scaling_state["scale"], dtype=float)


def _prior_logpdf_numpy(value: float, spec: dict[str, Any]) -> float:
    dist = spec["dist"]
    params = spec["params"]
    x = float(value)
    if dist == "uniform":
        if not params["lower"] < x < params["upper"]:
            return -math.inf
        return -math.log(params["upper"] - params["lower"])
    if dist == "normal":
        std = params["std"]
        return -0.5 * ((x - params["mean"]) / std) ** 2 - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    if dist == "lognormal":
        sigma = params["sigma"]
        if x <= 0:
            return -math.inf
        return -0.5 * ((math.log(x) - params["mean"]) / sigma) ** 2 - math.log(x * sigma) - 0.5 * math.log(2.0 * math.pi)
    if dist == "gamma":
        if x <= 0:
            return -math.inf
        shape = params["shape"]
        scale = params["scale"]
        return (shape - 1.0) * math.log(x) - x / scale - shape * math.log(scale) - math.lgamma(shape)
    if dist == "beta":
        lower = params["lower"]
        upper = params["upper"]
        if not lower < x < upper:
            return -math.inf
        unit = (x - lower) / (upper - lower)
        return (
            (params["alpha"] - 1.0) * math.log(unit)
            + (params["beta"] - 1.0) * math.log1p(-unit)
            - (math.lgamma(params["alpha"]) + math.lgamma(params["beta"]) - math.lgamma(params["alpha"] + params["beta"]))
            - math.log(upper - lower)
        )
    if dist == "halfnormal":
        lower = params.get("lower", 0.0)
        if x < lower:
            return -math.inf
        y = x - lower
        scale = params["scale"]
        return math.log(math.sqrt(2.0 / math.pi) / scale) - 0.5 * (y / scale) ** 2
    if dist == "student_t":
        df = params["df"]
        scale = params["scale"]
        standardized = (x - params["loc"]) / scale
        return (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            - 0.5 * math.log(df * math.pi)
            - math.log(scale)
            - 0.5 * (df + 1.0) * math.log1p((standardized**2) / df)
        )
    raise BackendError(f"Unsupported prior distribution: {dist}")


def _build_numpy_logdensity(runtime: dict[str, Any]) -> Any:
    parameter_names = runtime["parameter_names"]
    transform_specs = runtime["transform_specs"]
    priors = runtime["priors"]
    observed = runtime["observed_scaled"]
    scaling_state = runtime["scaling_state"]
    likelihood = runtime["likelihood"]
    model_cfg = runtime["model_cfg"]
    workdir = runtime["workdir"]
    selected_names = runtime["selected_output_names"]
    selected_indices = runtime["selected_output_indices"]

    def numpy_logdensity(unconstrained: np.ndarray) -> float:
        try:
            params, log_det = vector_to_parameter_dict_numpy(np.asarray(unconstrained, dtype=float), parameter_names, transform_specs)
            prior_value = sum(_prior_logpdf_numpy(params[name], priors[name]) for name in parameter_names)
            if not math.isfinite(prior_value):
                return -math.inf
            payload = simulate_model(model_cfg, params, workdir=workdir)
            simulated, _ = payload_to_array(payload, output_names=selected_names or None, output_indices=selected_indices or None)
            simulated_scaled = _apply_scaling_numpy(simulated, scaling_state)
            loglik = loglikelihood_numpy(observed, simulated_scaled, likelihood, params=params)
            value = prior_value + loglik + float(log_det)
            return float(value) if math.isfinite(value) else -math.inf
        except Exception:
            return -math.inf

    return numpy_logdensity


def _select_output_payload(payload: Any, jnp: Any, output_names: list[str], output_indices: list[int]) -> Any:
    if isinstance(payload, dict):
        keys = output_names or list(payload.keys())
        arrays = [jnp.ravel(jnp.asarray(payload[key], dtype=float)) for key in keys if key in payload]
        if not arrays:
            raise BackendError("Selected outputs are missing from the model payload.")
        return jnp.concatenate(arrays)
    array = jnp.ravel(jnp.asarray(payload, dtype=float))
    if output_indices:
        return array[jnp.asarray(output_indices, dtype=int)]
    return array


def _constrained_params_jax(unconstrained: Any, parameter_names: list[str], transform_specs: dict[str, Any], jnp: Any, jnn: Any) -> tuple[dict[str, Any], Any]:
    params: dict[str, Any] = {}
    log_det = 0.0
    for idx, name in enumerate(parameter_names):
        spec = transform_specs[name]
        z = unconstrained[idx]
        kind = spec["kind"]
        lower = spec.get("lower")
        upper = spec.get("upper")
        if kind == "identity":
            x = z
            log_j = 0.0
        elif kind == "log":
            shift = float(lower or 0.0)
            x = shift + jnp.exp(z)
            log_j = z
        elif kind == "softplus":
            shift = float(lower or 0.0)
            x = shift + jnn.softplus(z)
            log_j = -jnp.logaddexp(0.0, -z)
        elif kind == "logit":
            if lower is None or upper is None:
                raise BackendError("Logit transform requires lower and upper bounds.")
            unit = jnn.sigmoid(z)
            x = float(lower) + (float(upper) - float(lower)) * unit
            log_j = math.log(float(upper) - float(lower)) + jnp.log(unit) + jnp.log1p(-unit)
        else:
            raise BackendError(f"Unsupported transform kind: {kind}")
        params[name] = x
        log_det = log_det + log_j
    return params, log_det


def _prior_logpdf_jax(value: Any, spec: dict[str, Any], jnp: Any, jsp_special: Any) -> Any:
    dist = spec["dist"]
    params = spec["params"]
    if dist == "uniform":
        inside = (value > params["lower"]) & (value < params["upper"])
        return jnp.where(inside, -jnp.log(params["upper"] - params["lower"]), -jnp.inf)
    if dist == "normal":
        std = params["std"]
        return -0.5 * ((value - params["mean"]) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
    if dist == "lognormal":
        sigma = params["sigma"]
        safe = jnp.where(value > 0, value, 1.0)
        logpdf = -0.5 * ((jnp.log(safe) - params["mean"]) / sigma) ** 2 - jnp.log(safe * sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
        return jnp.where(value > 0, logpdf, -jnp.inf)
    if dist == "gamma":
        shape = params["shape"]
        scale = params["scale"]
        safe = jnp.where(value > 0, value, 1.0)
        logpdf = (shape - 1.0) * jnp.log(safe) - safe / scale - shape * jnp.log(scale) - jsp_special.gammaln(shape)
        return jnp.where(value > 0, logpdf, -jnp.inf)
    if dist == "beta":
        lower = params["lower"]
        upper = params["upper"]
        safe = jnp.clip((value - lower) / (upper - lower), 1e-8, 1.0 - 1e-8)
        logpdf = (
            (params["alpha"] - 1.0) * jnp.log(safe)
            + (params["beta"] - 1.0) * jnp.log1p(-safe)
            - jsp_special.betaln(params["alpha"], params["beta"])
            - jnp.log(upper - lower)
        )
        inside = (value > lower) & (value < upper)
        return jnp.where(inside, logpdf, -jnp.inf)
    if dist == "halfnormal":
        lower = params.get("lower", 0.0)
        scale = params["scale"]
        shifted = value - lower
        logpdf = jnp.log(jnp.sqrt(2.0 / jnp.pi) / scale) - 0.5 * (shifted / scale) ** 2
        return jnp.where(value >= lower, logpdf, -jnp.inf)
    if dist == "student_t":
        df = params["df"]
        scale = params["scale"]
        centered = (value - params["loc"]) / scale
        return (
            jsp_special.gammaln((df + 1.0) / 2.0)
            - jsp_special.gammaln(df / 2.0)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(scale)
            - 0.5 * (df + 1.0) * jnp.log1p((centered**2) / df)
        )
    raise BackendError(f"Unsupported prior distribution: {dist}")


def _loglikelihood_jax(observed: Any, simulated: Any, likelihood: dict[str, Any], jnp: Any, jsp_special: Any) -> Any:
    name = likelihood["name"]
    params = likelihood.get("params", {})
    if name == "gaussian":
        sigma = params["sigma"]
        resid = (observed - simulated) / sigma
        return -0.5 * jnp.sum(resid**2 + jnp.log(2.0 * jnp.pi * sigma * sigma))
    if name == "student_t":
        sigma = params["sigma"]
        df = params["df"]
        resid = (observed - simulated) / sigma
        log_norm = (
            jsp_special.gammaln((df + 1.0) / 2.0)
            - jsp_special.gammaln(df / 2.0)
            - 0.5 * jnp.log(df * jnp.pi)
            - jnp.log(sigma)
        )
        return jnp.sum(log_norm - 0.5 * (df + 1.0) * jnp.log1p((resid**2) / df))
    if name == "poisson":
        rate = jnp.clip(simulated, 1e-8)
        return jnp.sum(observed * jnp.log(rate) - rate - jsp_special.gammaln(observed + 1.0))
    if name == "binomial":
        n_trials = params["n_trials"]
        prob = jnp.clip(simulated, 1e-8, 1.0 - 1e-8)
        coeff = jsp_special.gammaln(n_trials + 1.0) - jsp_special.gammaln(observed + 1.0) - jsp_special.gammaln(n_trials - observed + 1.0)
        return jnp.sum(coeff + observed * jnp.log(prob) + (n_trials - observed) * jnp.log1p(-prob))
    if name == "negative_binomial":
        dispersion = params["dispersion"]
        mean = jnp.clip(simulated, 1e-8)
        probs = dispersion / (dispersion + mean)
        return jnp.sum(
            jsp_special.gammaln(observed + dispersion)
            - jsp_special.gammaln(dispersion)
            - jsp_special.gammaln(observed + 1.0)
            + dispersion * jnp.log(probs)
            + observed * jnp.log1p(-probs)
        )
    raise BackendError(f"Likelihood {name!r} is not available in JAX mode.")


def _build_direct_logdensity(runtime: dict[str, Any], jax: Any, jnp: Any, jsp_special: Any, jnn: Any) -> Any:
    model_cfg = runtime["model_cfg"]
    workdir = runtime["workdir"]
    parameter_names = runtime["parameter_names"]
    transform_specs = runtime["transform_specs"]
    priors = runtime["priors"]
    likelihood = runtime["likelihood"]
    observed = jnp.asarray(runtime["observed_scaled"], dtype=float)
    scaling_state = runtime["scaling_state"]
    selected_output_names = runtime["selected_output_names"]
    selected_output_indices = runtime["selected_output_indices"]

    from .adapters import _load_python_callable  # local import to avoid importing at module load

    model_path = Path(model_cfg["path"])
    if not model_path.is_absolute():
        model_path = (workdir / model_path).resolve()
    fn = _load_python_callable(model_path, model_cfg.get("callable") or "simulate")
    call_style = model_cfg.get("call_style") or "kwargs"
    enabled = bool(scaling_state.get("enabled"))
    center = jnp.asarray(scaling_state["center"], dtype=float)
    scale = jnp.asarray(scaling_state["scale"], dtype=float)

    def apply_scaling(values):
        array = jnp.ravel(jnp.asarray(values, dtype=float))
        if not enabled:
            return array
        return (array - center) / scale

    def model_output(params_dict):
        if call_style == "mapping":
            payload = fn(params_dict)
        elif call_style == "positional":
            payload = fn(*[params_dict[name] for name in parameter_names])
        else:
            payload = fn(**params_dict)
        return _select_output_payload(payload, jnp, selected_output_names, selected_output_indices)

    def logdensity(unconstrained):
        params, log_det = _constrained_params_jax(unconstrained, parameter_names, transform_specs, jnp, jnn)
        prior_value = jnp.asarray(0.0, dtype=float)
        for name in parameter_names:
            prior_value = prior_value + _prior_logpdf_jax(params[name], priors[name], jnp, jsp_special)
        simulated = apply_scaling(model_output(params))
        likelihood_value = _loglikelihood_jax(observed, simulated, likelihood, jnp, jsp_special)
        return prior_value + likelihood_value + log_det

    return logdensity


def _build_callback_logdensity(runtime: dict[str, Any], jax: Any, jnp: Any) -> Any:
    numpy_logdensity = _build_numpy_logdensity(runtime)
    dtype = jnp.float64 if runtime["enable_x64"] else jnp.float32
    shape_scalar = jax.ShapeDtypeStruct((), dtype)
    shape_grad = jax.ShapeDtypeStruct((len(runtime["parameter_names"]),), dtype)
    step = 1e-4

    def finite_difference_grad(theta_np: np.ndarray) -> np.ndarray:
        theta_np = np.asarray(theta_np, dtype=float)
        grad = np.zeros_like(theta_np, dtype=float)
        for idx in range(theta_np.size):
            delta = step * max(1.0, abs(theta_np[idx]))
            forward = theta_np.copy()
            backward = theta_np.copy()
            forward[idx] += delta
            backward[idx] -= delta
            grad[idx] = (numpy_logdensity(forward) - numpy_logdensity(backward)) / (2.0 * delta)
        return grad

    @jax.custom_vjp
    def logdensity(theta):
        return jax.pure_callback(lambda x: np.asarray(numpy_logdensity(np.asarray(x, dtype=float)), dtype=float), shape_scalar, theta)

    def fwd(theta):
        value = jax.pure_callback(lambda x: np.asarray(numpy_logdensity(np.asarray(x, dtype=float)), dtype=float), shape_scalar, theta)
        return value, theta

    def bwd(theta, g):
        grad = jax.pure_callback(lambda x: np.asarray(finite_difference_grad(np.asarray(x, dtype=float)), dtype=float), shape_grad, theta)
        return (g * grad,)

    logdensity.defvjp(fwd, bwd)
    return logdensity


def _initial_positions(runtime: dict[str, Any], num_chains: int, seed: int) -> np.ndarray:
    default_points = runtime["default_points"]
    base = default_unconstrained_position(runtime["parameter_names"], default_points, runtime["transform_specs"])
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.1, size=(num_chains, base.size))
    return np.asarray(base, dtype=float)[None, :] + jitter


def _select_devices(jax: Any, preference: str | None) -> list[Any]:
    devices = list(jax.devices())
    token = str(preference or "auto").lower()
    if token == "auto":
        return devices
    selected = [device for device in devices if device.platform == token]
    return selected or devices


def _choose_chain_method(requested: str | None, execution_mode: str, num_chains: int, devices: list[Any]) -> str:
    choice = str(requested or "auto").lower()
    if choice in {"sequential", "vmap", "pmap"}:
        return choice
    if execution_mode != "direct_jax" or num_chains <= 1:
        return "sequential"
    if len(devices) > 1 and num_chains % len(devices) == 0:
        return "pmap"
    return "vmap"


def _tree_stack(jax: Any, states: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: np.stack(items, axis=0), *states)


def _tree_reshape(jax: Any, tree: Any, shape: tuple[int, ...]) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value).reshape(shape + np.asarray(value).shape[1:]), tree)


def _sample_chains_sequential(
    jax: Any,
    blackjax: Any,
    logdensity_fn: Any,
    states: list[Any],
    tuned: list[dict[str, Any]],
    num_samples: int,
    thin: int,
    max_tree_depth: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total_steps = num_samples * thin
    rng_master = jax.random.PRNGKey(seed)
    chain_samples = []
    info_traces: dict[str, list[np.ndarray]] = {
        "acceptance_rate": [],
        "energy": [],
        "is_divergent": [],
        "tree_depth": [],
        "num_integration_steps": [],
    }
    for chain_idx, (state, tuned_params) in enumerate(zip(states, tuned)):
        algorithm = blackjax.nuts(
            logdensity_fn,
            tuned_params["step_size"],
            tuned_params["inverse_mass_matrix"],
            max_num_doublings=max_tree_depth,
        )
        chain_key = jax.random.fold_in(rng_master, chain_idx)
        keys = jax.random.split(chain_key, total_steps)

        def one_step(carry, key):
            return algorithm.step(key, carry)

        _, history = jax.lax.scan(one_step, state, keys)
        positions = np.asarray(jax.device_get(history[0].position))[thin - 1 :: thin]
        info = history[1]
        chain_samples.append(positions)
        info_traces["acceptance_rate"].append(np.asarray(jax.device_get(info.acceptance_rate))[thin - 1 :: thin])
        info_traces["energy"].append(np.asarray(jax.device_get(info.energy))[thin - 1 :: thin])
        info_traces["is_divergent"].append(np.asarray(jax.device_get(info.is_divergent))[thin - 1 :: thin])
        info_traces["tree_depth"].append(np.asarray(jax.device_get(info.num_trajectory_expansions))[thin - 1 :: thin])
        info_traces["num_integration_steps"].append(np.asarray(jax.device_get(info.num_integration_steps))[thin - 1 :: thin])
    return np.asarray(chain_samples), {key: np.asarray(value) for key, value in info_traces.items()}


def _sample_chains_vmap(
    jax: Any,
    blackjax: Any,
    logdensity_fn: Any,
    states: list[Any],
    tuned: list[dict[str, Any]],
    num_samples: int,
    thin: int,
    max_tree_depth: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    total_steps = num_samples * thin
    kernel = blackjax.nuts.build_kernel()
    state_stack = _tree_stack(jax, states)
    step_sizes = np.asarray([item["step_size"] for item in tuned], dtype=float)
    inverse_mass = np.stack([np.asarray(item["inverse_mass_matrix"], dtype=float) for item in tuned], axis=0)
    keys = np.asarray(jax.device_get(jax.random.split(jax.random.PRNGKey(seed), total_steps * len(states)))).reshape(total_steps, len(states), -1)
    keys = jax.device_put(keys)

    def single_step(key, state, step_size, mass):
        return kernel(key, state, logdensity_fn, step_size, mass, max_tree_depth)

    def one_step(carry, rng_keys):
        new_states, infos = jax.vmap(single_step)(rng_keys, carry, step_sizes, inverse_mass)
        return new_states, (new_states, infos)

    _, history = jax.lax.scan(one_step, state_stack, keys)
    state_history, info_history = history
    positions = np.asarray(jax.device_get(state_history.position))[thin - 1 :: thin]
    samples = np.transpose(positions, (1, 0, 2))
    return samples, {
        "acceptance_rate": np.transpose(np.asarray(jax.device_get(info_history.acceptance_rate))[thin - 1 :: thin], (1, 0)),
        "energy": np.transpose(np.asarray(jax.device_get(info_history.energy))[thin - 1 :: thin], (1, 0)),
        "is_divergent": np.transpose(np.asarray(jax.device_get(info_history.is_divergent))[thin - 1 :: thin], (1, 0)),
        "tree_depth": np.transpose(np.asarray(jax.device_get(info_history.num_trajectory_expansions))[thin - 1 :: thin], (1, 0)),
        "num_integration_steps": np.transpose(np.asarray(jax.device_get(info_history.num_integration_steps))[thin - 1 :: thin], (1, 0)),
    }


def _sample_chains_pmap(
    jax: Any,
    blackjax: Any,
    logdensity_fn: Any,
    states: list[Any],
    tuned: list[dict[str, Any]],
    num_samples: int,
    thin: int,
    max_tree_depth: int,
    seed: int,
    devices: list[Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    num_chains = len(states)
    if len(devices) <= 1 or num_chains % len(devices) != 0:
        return _sample_chains_vmap(jax, blackjax, logdensity_fn, states, tuned, num_samples, thin, max_tree_depth, seed)
    total_steps = num_samples * thin
    local_chains = num_chains // len(devices)
    kernel = blackjax.nuts.build_kernel()
    state_stack = _tree_stack(jax, states)
    state_stack = _tree_reshape(jax, state_stack, (len(devices), local_chains))
    step_sizes = np.asarray([item["step_size"] for item in tuned], dtype=float).reshape(len(devices), local_chains)
    inverse_mass = np.stack([np.asarray(item["inverse_mass_matrix"], dtype=float) for item in tuned], axis=0).reshape((len(devices), local_chains) + np.asarray(tuned[0]["inverse_mass_matrix"]).shape)
    keys = np.asarray(jax.device_get(jax.random.split(jax.random.PRNGKey(seed), total_steps * num_chains))).reshape(total_steps, len(devices), local_chains, -1)
    keys = jax.device_put_sharded([keys[:, idx, :, :] for idx in range(len(devices))], devices)

    def single_step(key, state, step_size, mass):
        return kernel(key, state, logdensity_fn, step_size, mass, max_tree_depth)

    def mapped_step(rng_keys, carry, step_size, mass):
        return jax.vmap(single_step)(rng_keys, carry, step_size, mass)

    parallel_step = jax.pmap(mapped_step, in_axes=(0, 0, 0, 0))

    def one_step(carry, rng_keys):
        new_states, infos = parallel_step(rng_keys, carry, step_sizes, inverse_mass)
        return new_states, (new_states, infos)

    _, history = jax.lax.scan(one_step, state_stack, keys)
    state_history, info_history = history
    positions = np.asarray(jax.device_get(state_history.position))[thin - 1 :: thin].reshape(num_samples, num_chains, -1)
    samples = np.transpose(positions, (1, 0, 2))
    return samples, {
        "acceptance_rate": np.asarray(jax.device_get(info_history.acceptance_rate))[thin - 1 :: thin].reshape(num_samples, num_chains).T,
        "energy": np.asarray(jax.device_get(info_history.energy))[thin - 1 :: thin].reshape(num_samples, num_chains).T,
        "is_divergent": np.asarray(jax.device_get(info_history.is_divergent))[thin - 1 :: thin].reshape(num_samples, num_chains).T,
        "tree_depth": np.asarray(jax.device_get(info_history.num_trajectory_expansions))[thin - 1 :: thin].reshape(num_samples, num_chains).T,
        "num_integration_steps": np.asarray(jax.device_get(info_history.num_integration_steps))[thin - 1 :: thin].reshape(num_samples, num_chains).T,
    }


def run_blackjax_nuts(runtime: dict[str, Any]) -> dict[str, Any]:
    jax, jnp, jsp_special, jnn, blackjax = _import_jax_stack(runtime["enable_x64"])
    selected_devices = _select_devices(jax, runtime["device_preference"])
    execution_mode = "direct_jax" if runtime["gradient_strategy"] == "jax_autodiff" and runtime["likelihood"]["name"] not in {"custom_python", "custom_command"} else "callback_fd"
    if execution_mode == "direct_jax":
        logdensity_fn = _build_direct_logdensity(runtime, jax, jnp, jsp_special, jnn)
    else:
        logdensity_fn = _build_callback_logdensity(runtime, jax, jnp)

    num_chains = runtime["num_chains"]
    warmup_steps = runtime["warmup_steps"]
    initial_step_size = runtime["initial_step_size"]
    target_acceptance = runtime["target_acceptance"]
    mass_matrix_kind = runtime["mass_matrix"]
    num_samples = runtime["num_samples"]
    thin = runtime["thin"]
    max_tree_depth = runtime["max_tree_depth"]

    initial_positions = _initial_positions(runtime, num_chains, runtime["seed"])
    if selected_devices:
        initial_positions = np.asarray([jax.device_get(jax.device_put(position, selected_devices[0])) for position in initial_positions])
    states = []
    tuned = []
    warmup_report = []
    for chain_idx in range(num_chains):
        key = jax.random.PRNGKey(runtime["seed"] + chain_idx)
        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            is_mass_matrix_diagonal=mass_matrix_kind != "dense",
            initial_step_size=initial_step_size or 1.0,
            target_acceptance_rate=target_acceptance,
            progress_bar=bool(runtime["progress_bar"]),
            max_num_doublings=max_tree_depth,
        )
        result, _ = adaptation.run(key, jnp.asarray(initial_positions[chain_idx], dtype=float), num_steps=warmup_steps)
        states.append(result.state)
        tuned.append(
            {
                "step_size": float(jax.device_get(result.parameters["step_size"])),
                "inverse_mass_matrix": np.asarray(jax.device_get(result.parameters["inverse_mass_matrix"]), dtype=float),
            }
        )
        warmup_report.append(
            {
                "chain": chain_idx,
                "step_size": tuned[-1]["step_size"],
                "mass_matrix_shape": list(np.asarray(tuned[-1]["inverse_mass_matrix"]).shape),
                "mass_matrix_type": "diagonal" if np.asarray(tuned[-1]["inverse_mass_matrix"]).ndim == 1 else "dense",
            }
        )

    chain_method = _choose_chain_method(runtime["chain_method"], execution_mode, num_chains, selected_devices)
    if chain_method == "pmap":
        unconstrained_samples, info = _sample_chains_pmap(
            jax,
            blackjax,
            logdensity_fn,
            states,
            tuned,
            num_samples,
            thin,
            max_tree_depth,
            runtime["seed"] + 1000,
            selected_devices,
        )
    elif chain_method == "vmap":
        unconstrained_samples, info = _sample_chains_vmap(
            jax,
            blackjax,
            logdensity_fn,
            states,
            tuned,
            num_samples,
            thin,
            max_tree_depth,
            runtime["seed"] + 1000,
        )
    else:
        unconstrained_samples, info = _sample_chains_sequential(
            jax,
            blackjax,
            logdensity_fn,
            states,
            tuned,
            num_samples,
            thin,
            max_tree_depth,
            runtime["seed"] + 1000,
        )

    constrained = np.zeros_like(unconstrained_samples, dtype=float)
    for chain in range(unconstrained_samples.shape[0]):
        for draw in range(unconstrained_samples.shape[1]):
            params, _ = vector_to_parameter_dict_numpy(
                unconstrained_samples[chain, draw, :],
                runtime["parameter_names"],
                runtime["transform_specs"],
            )
            constrained[chain, draw, :] = np.asarray([params[name] for name in runtime["parameter_names"]], dtype=float)

    return {
        "backend": "blackjax",
        "execution_mode": execution_mode,
        "chain_method": chain_method,
        "devices": [f"{device.platform}:{device.id}" for device in selected_devices] if selected_devices else [],
        "warmup": warmup_report,
        "tuned_parameters": tuned,
        "samples_unconstrained": unconstrained_samples,
        "samples_constrained": constrained,
        "info": info,
    }

