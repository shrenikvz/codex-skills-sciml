"""Backend registry and BlackJAX sampler implementation."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .io_utils import payload_to_array
from .priors import normalize_prior_spec
from .transforms import constrain_array, jacobian_correction


class BackendError(RuntimeError):
    """Backend execution failure."""


def _import_blackjax_stack():
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415
        import blackjax  # noqa: PLC0415
        from blackjax.adaptation.base import get_filter_adapt_info_fn  # noqa: PLC0415

        return jax, jnp, blackjax, get_filter_adapt_info_fn
    except Exception as exc:  # noqa: BLE001
        raise BackendError(f"BlackJAX backend is unavailable: {exc}") from exc


def _load_direct_model_callable(model_cfg: dict[str, Any]) -> Callable:
    target = Path(model_cfg["path"]).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(target.stem, target)
    if spec is None or spec.loader is None:
        raise BackendError(f"Could not load Python module: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, model_cfg.get("callable") or "simulate", None)
    if not callable(fn):
        raise BackendError(f"Callable {model_cfg.get('callable')!r} not found in {target}")
    return fn


def _select_device(jax, preference: str):
    devices = jax.devices()
    if not devices:
        raise BackendError("JAX did not report any devices.")
    pref = (preference or "auto").lower()
    if pref == "auto":
        for platform in ["gpu", "tpu", "cpu"]:
            for device in devices:
                if device.platform == platform:
                    return device
        return devices[0]
    for device in devices:
        if device.platform == pref:
            return device
    raise BackendError(f"Requested JAX device platform {preference!r} is not available.")


def _log_prior_density(value, spec: dict[str, Any], xp, gammaln):
    spec = normalize_prior_spec(spec)
    dist = spec["dist"]
    params = spec["params"]
    if dist == "uniform":
        lower = params["lower"]
        upper = params["upper"]
        inside = xp.logical_and(value >= lower, value <= upper)
        return xp.where(inside, -xp.log(upper - lower), -xp.inf)
    if dist == "normal":
        mean = params["mean"]
        std = params["std"]
        z = (value - mean) / std
        return -0.5 * (z**2 + xp.log(2.0 * xp.pi * std**2))
    if dist == "lognormal":
        sigma = params["sigma"]
        mean = params["mean"]
        positive = value > 0
        z = (xp.log(xp.maximum(value, 1e-12)) - mean) / sigma
        logp = -0.5 * (z**2 + xp.log(2.0 * xp.pi * sigma**2)) - xp.log(xp.maximum(value, 1e-12))
        return xp.where(positive, logp, -xp.inf)
    if dist == "gamma":
        shape = params["shape"]
        scale = params["scale"]
        positive = value > 0
        logp = (shape - 1.0) * xp.log(xp.maximum(value, 1e-12)) - value / scale - shape * xp.log(scale) - gammaln(shape)
        return xp.where(positive, logp, -xp.inf)
    if dist == "beta":
        alpha = params["alpha"]
        beta = params["beta"]
        lower = params["lower"]
        upper = params["upper"]
        scaled = (value - lower) / (upper - lower)
        inside = xp.logical_and(scaled > 0, scaled < 1)
        log_norm = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta) + xp.log(upper - lower)
        logp = (alpha - 1.0) * xp.log(scaled) + (beta - 1.0) * xp.log1p(-scaled) - log_norm
        return xp.where(inside, logp, -xp.inf)
    if dist == "halfnormal":
        scale = params["scale"]
        positive = value >= 0
        logp = 0.5 * xp.log(2.0 / xp.pi) - xp.log(scale) - 0.5 * (value / scale) ** 2
        return xp.where(positive, logp, -xp.inf)
    if dist == "student_t":
        df = params["df"]
        loc = params["loc"]
        scale = params["scale"]
        z = (value - loc) / scale
        return (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * (xp.log(df) + xp.log(xp.pi))
            - xp.log(scale)
            - 0.5 * (df + 1.0) * xp.log1p((z**2) / df)
        )
    raise BackendError(f"Unsupported prior distribution: {dist}")


def run_backend(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    backend = (cfg.get("sampler", {}).get("backend") or "blackjax").lower()
    if backend == "blackjax":
        return run_blackjax_backend(cfg, context)
    if backend in {"numpyro", "pymc", "stan", "tensorflow_probability", "tfp"}:
        raise BackendError(f"Backend {backend!r} is reserved by the skill architecture but not implemented in this revision.")
    raise BackendError(f"Unsupported backend: {backend}")


def run_blackjax_backend(cfg: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    jax, jnp, blackjax, get_filter_adapt_info_fn = _import_blackjax_stack()
    if cfg.get("compute", {}).get("enable_x64"):
        jax.config.update("jax_enable_x64", True)
    selected_device = _select_device(jax, cfg.get("compute", {}).get("device_preference", "auto"))
    parameter_names = context["parameter_names"]
    model_parameter_names = context["model_parameter_names"]
    transform_specs = context["transform_specs"]
    priors = context["priors"]
    observed_eval = context["observed_eval"]
    gradient_strategy = context["gradient_strategy"]
    gradient_step = float(context["gradient_step"])
    workdir = context["workdir"]
    likelihood_cfg = context["likelihood_cfg"]
    scaler_apply = context["scaler_apply"]
    model_cfg = context["model_cfg"]
    max_tree_depth = int(cfg["sampler"]["max_tree_depth"])
    direct_callable = None
    if gradient_strategy == "jax":
        if model_cfg.get("adapter") != "python_callable":
            raise BackendError("Direct JAX gradients require a Python callable model.")
        direct_callable = _load_direct_model_callable(model_cfg)

    def gammaln_host(x):
        return math.lgamma(float(x))

    def host_logdensity(unconstrained: np.ndarray) -> float:
        params = constrain_array(np.asarray(unconstrained, dtype=float), parameter_names, transform_specs, xp=np)
        model_params = {name: float(params[name]) for name in model_parameter_names}
        simulated_payload = context["simulate_host"](model_params)
        simulated_array, _ = payload_to_array(
            simulated_payload,
            output_names=model_cfg.get("observed_output_names") or None,
            output_indices=model_cfg.get("observed_output_indices") or None,
        )
        simulated_eval = scaler_apply(np.asarray(simulated_array, dtype=float), xp=np)
        logprior = 0.0
        for idx, name in enumerate(parameter_names):
            logprior += float(_log_prior_density(params[name], priors[name], np, gammaln_host))
        logprior += float(jacobian_correction(np.asarray(unconstrained, dtype=float), parameter_names, transform_specs, xp=np))
        loglik = float(context["likelihood_fn"](likelihood_cfg, params, simulated_eval, observed_eval, context["metadata"], xp=np, workdir=workdir))
        total = logprior + loglik
        return total if np.isfinite(total) else -np.inf

    def host_gradient(unconstrained: np.ndarray) -> np.ndarray:
        base = np.asarray(unconstrained, dtype=float)
        grad = np.zeros_like(base)
        for idx in range(base.size):
            step = gradient_step * max(1.0, abs(base[idx]))
            plus = base.copy()
            minus = base.copy()
            plus[idx] += step
            minus[idx] -= step
            grad[idx] = (host_logdensity(plus) - host_logdensity(minus)) / (2.0 * step)
        return grad

    def flatten_payload_jax(payload):
        if isinstance(payload, dict):
            keys = model_cfg.get("observed_output_names") or list(payload.keys())
            arrays = [jnp.asarray(payload[key], dtype=jnp.float32).reshape(-1) for key in keys]
            return jnp.concatenate(arrays)
        array = jnp.asarray(payload, dtype=jnp.float32)
        indices = model_cfg.get("observed_output_indices") or []
        if indices:
            axis = 0 if array.ndim == 1 else 1
            array = jnp.take(array, jnp.asarray(indices), axis=axis)
        return array.reshape(-1)

    def gammaln_jax(x):
        import jax.scipy.special as jsp  # noqa: PLC0415

        return jsp.gammaln(x)

    def direct_logdensity(unconstrained):
        params = constrain_array(unconstrained, parameter_names, transform_specs, xp=jnp)
        model_params = {name: params[name] for name in model_parameter_names}
        payload = direct_callable(model_params) if model_cfg.get("call_style") == "mapping" else direct_callable(**model_params)
        simulated_eval = scaler_apply(flatten_payload_jax(payload), xp=jnp)
        logprior = jnp.sum(
            jnp.asarray([
                _log_prior_density(params[name], priors[name], jnp, gammaln_jax)
                for name in parameter_names
            ])
        )
        logprior = logprior + jacobian_correction(unconstrained, parameter_names, transform_specs, xp=jnp)
        loglik = context["likelihood_fn"](likelihood_cfg, params, simulated_eval, observed_eval, context["metadata"], xp=jnp, workdir=workdir)
        total = logprior + loglik
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    def callback_logdensity_factory():
        scalar_shape = jax.ShapeDtypeStruct((), jnp.float32)
        grad_shape = jax.ShapeDtypeStruct((len(parameter_names),), jnp.float32)

        @jax.custom_vjp
        def callback_logdensity(unconstrained):
            return jax.pure_callback(
                lambda x: np.asarray(host_logdensity(np.asarray(x, dtype=float)), dtype=np.float32),
                scalar_shape,
                unconstrained,
                vmap_method="sequential",
            )

        def fwd(unconstrained):
            value = callback_logdensity(unconstrained)
            return value, unconstrained

        def bwd(residual, cotangent):
            grad = jax.pure_callback(
                lambda x: np.asarray(host_gradient(np.asarray(x, dtype=float)), dtype=np.float32),
                grad_shape,
                residual,
                vmap_method="sequential",
            )
            return (cotangent * grad,)

        callback_logdensity.defvjp(fwd, bwd)
        return callback_logdensity

    logdensity_fn = direct_logdensity if gradient_strategy == "jax" else callback_logdensity_factory()

    rng = np.random.default_rng(int(cfg["sampler"]["random_seed"]))
    base_unconstrained = np.asarray(context["unconstrain_params"]({name: context["initial_point"][name] for name in parameter_names}), dtype=np.float32)
    init_unconstrained = np.stack(
        [
            base_unconstrained + rng.normal(0.0, 0.05, size=base_unconstrained.shape).astype(np.float32)
            for _ in range(int(cfg["sampler"]["num_chains"]))
        ]
    )

    adaptation_info_fn = get_filter_adapt_info_fn(info_keys={"acceptance_rate", "is_divergent"})
    is_diagonal = str(cfg["sampler"].get("mass_matrix", "diagonal")).lower() != "dense"
    initial_step_size = float(cfg["sampler"].get("step_size") or 1.0)
    adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=is_diagonal,
        initial_step_size=initial_step_size,
        target_acceptance_rate=float(cfg["sampler"]["target_acceptance_rate"]),
        progress_bar=False,
        adaptation_info_fn=adaptation_info_fn,
        max_num_doublings=max_tree_depth,
    )

    def run_one_chain(chain_key, init_position):
        warmup_key, sample_key = jax.random.split(chain_key)
        adaptation_result, warmup_info = adaptation.run(
            warmup_key,
            init_position,
            num_steps=int(cfg["sampler"]["num_warmup"]),
        )
        nuts = blackjax.nuts(
            logdensity_fn,
            adaptation_result.parameters["step_size"],
            adaptation_result.parameters["inverse_mass_matrix"],
            max_num_doublings=max_tree_depth,
        )
        sample_keys = jax.random.split(sample_key, int(cfg["sampler"]["num_samples"]))

        def one_step(state, key):
            new_state, info = nuts.step(key, state)
            return new_state, (new_state.position, new_state.logdensity, info)

        _, (positions, logdensity, info) = jax.lax.scan(one_step, adaptation_result.state, sample_keys)
        return positions, logdensity, info, adaptation_result.parameters, warmup_info

    num_chains = int(cfg["sampler"]["num_chains"])
    with jax.default_device(selected_device):
        master_key = jax.random.PRNGKey(int(cfg["sampler"]["random_seed"]))
        chain_keys = jax.random.split(master_key, num_chains)
        positions = jnp.asarray(init_unconstrained)
        strategy_used = "vmap"
        if (
            cfg.get("compute", {}).get("parallel_mode", "auto") in {"auto", "pmap"}
            and len({device.platform for device in jax.devices()}) == 1
            and len(jax.devices()) > 1
            and num_chains % len(jax.devices()) == 0
        ):
            chains_per_device = num_chains // len(jax.devices())
            vmapped = jax.vmap(run_one_chain)
            pmapped = jax.pmap(vmapped)
            result = pmapped(
                chain_keys.reshape(len(jax.devices()), chains_per_device, -1),
                positions.reshape(len(jax.devices()), chains_per_device, positions.shape[-1]),
            )
            strategy_used = "pmap"
            positions_u, logdensity_u, info_u, tuned_u, warm_u = result
            unconstrained_samples = np.asarray(positions_u).reshape(num_chains, int(cfg["sampler"]["num_samples"]), -1)
            logdensity = np.asarray(logdensity_u).reshape(num_chains, int(cfg["sampler"]["num_samples"]))
            info = {field: np.asarray(getattr(info_u, field)).reshape(num_chains, int(cfg["sampler"]["num_samples"])) for field in info_u._fields}
            tuned = {key: np.asarray(value).reshape(num_chains, *np.asarray(value).shape[2:]) for key, value in tuned_u.items()}
            warmup = warm_u
        else:
            vmapped = jax.vmap(run_one_chain)
            positions_u, logdensity_u, info_u, tuned_u, warm_u = vmapped(chain_keys, positions)
            unconstrained_samples = np.asarray(positions_u)
            logdensity = np.asarray(logdensity_u)
            info = {field: np.asarray(getattr(info_u, field)) for field in info_u._fields}
            tuned = {key: np.asarray(value) for key, value in tuned_u.items()}
            warmup = warm_u

    constrained_samples = np.zeros_like(unconstrained_samples, dtype=float)
    for chain in range(num_chains):
        for draw in range(int(cfg["sampler"]["num_samples"])):
            params = constrain_array(unconstrained_samples[chain, draw], parameter_names, transform_specs, xp=np)
            constrained_samples[chain, draw] = np.asarray([float(params[name]) for name in parameter_names], dtype=float)

    tuned_summary = {
        "step_size": tuned.get("step_size", []).tolist() if "step_size" in tuned else [],
        "inverse_mass_matrix_shape": list(np.asarray(tuned.get("inverse_mass_matrix")).shape) if "inverse_mass_matrix" in tuned else [],
        "mass_matrix": cfg["sampler"]["mass_matrix"],
        "parallel_strategy": strategy_used,
        "device": str(selected_device),
    }
    return {
        "samples": constrained_samples,
        "unconstrained_samples": unconstrained_samples,
        "logdensity": logdensity,
        "info": info,
        "tuned": tuned_summary,
        "warmup_info_present": warmup is not None,
    }
