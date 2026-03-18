---
name: nuts-calibration
description: General-purpose Bayesian parameter estimation and model calibration for differentiable models, simulators, mathematical expressions, and probabilistic model components using No-U-Turn Sampler (NUTS). Use when Codex needs to inspect a user-provided model and observed data, infer parameters and observed outputs, construct priors and likelihoods, apply parameter transforms and scaling, run BlackJAX-based NUTS with JAX gradients or finite-difference fallback, diagnose convergence, and return posterior summaries, predictive checks, and requested visualizations.
---

# NUTS Calibration

Use this skill when users want likelihood-based Bayesian calibration with NUTS.

Natural-language trigger examples:

- "Estimate these parameters with NUTS."
- "Calibrate this differentiable simulator against observed data with BlackJAX."
- "Build a reusable Bayesian inference project for this model and data."
- "Infer posterior distributions of model parameters with Hamiltonian Monte Carlo."
- "Use JAX and BlackJAX to fit this model, then give me diagnostics and posterior predictive checks."

This skill follows a strict reasoning sequence:

1. Inspect the model and observed data before sampling.
2. Infer parameter names, defaults, outputs, and which outputs map to observations.
3. Ask for clarification when the observed-output mapping is materially ambiguous.
4. Determine whether the model is JAX-differentiable.
5. Use JAX autodiff by default when possible.
6. If the model is not directly differentiable, fall back to a differentiable wrapper strategy:
   - finite-difference gradients through a callback wrapper by default
   - surrogate or alternate backend hooks later if the user requests them
7. Build priors, likelihood, parameter transforms, and scaling choices explicitly.
8. Ask whether normalization should be applied when the user has not already decided.
9. Ask which visualizations are wanted before enabling them.
10. Run adaptive warmup, then NUTS sampling, then diagnostics and posterior predictive checks.

## First-run workflow

Check the runtime first:

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py doctor
```

Inspect the model and data:

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Estimate the unknown parameters with NUTS"
```

Create a runnable project:

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py create-project \
  --project-dir ./nuts-run \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --backend blackjax \
  --plot posterior_marginals \
  --plot trace \
  --plot posterior_predictive
```

Run the generated project:

```bash
cd nuts-run
python3 run.py
```

Or run directly:

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py run \
  --config ./nuts-run/config.json
```

## Supported model forms

- Python callables returning scalar, vector, array, or dict outputs
- command-line simulators in other languages via generated wrappers
- mathematical expressions wrapped into generated JAX-ready Python models
- partially observed models through explicit output-name or output-index selection
- user-defined likelihood functions through Python or command hooks

Model handling rules:

- infer parameter names from Python signatures or explicit `--parameter` values
- infer observed outputs by name overlap when possible
- stop and ask when multiple outputs remain ambiguous
- detect whether the model appears JAX-compatible
- default to finite-difference gradients when JAX autodiff is not feasible

## Priors, likelihoods, and transforms

Supported priors:

- `uniform`
- `normal`
- `lognormal`
- `gamma`
- `beta`
- `halfnormal`
- `student_t`

Likelihood support:

- `gaussian`
- `student_t`
- `poisson`
- `binomial`
- `negative_binomial`
- `custom_python`
- `custom_command`

Transform support:

- `log` for strictly positive parameters
- `logit` for bounded parameters
- `softplus` for non-negative parameters
- `identity` for unconstrained parameters

Always record the prior, likelihood, and transform used for each parameter in the generated project and run outputs.

## Scaling and preprocessing

- Ask whether scaling should be applied.
- If the user does not specify, recommend `zscore`, `minmax`, `variance`, or `none`.
- Apply the same scaling to observed and simulated outputs during likelihood evaluation.
- Record the scaling decision in `config.json` and the run summary.

## NUTS behavior

Default backend:

- `blackjax` with `jax`

Warmup behavior:

- adaptive warmup tunes step size and mass matrix
- follows BlackJAX window adaptation, which mirrors the standard three-phase schedule
- records tuned step size and mass matrix per chain

Parallel execution behavior:

- prefer `vmap` on single-device JAX runs when the model is directly JAX-compatible
- use `pmap` when multiple devices are available and the chain count matches device partitioning
- fall back to sequential chains when the model uses callback-based finite-difference gradients

## Diagnostics and outputs

Run outputs include:

- `results/posterior_samples.csv`
- `results/posterior_samples.jsonl`
- `results/posterior_summary.json`
- `results/sampling_diagnostics.json`
- `results/tuned_hyperparameters.json`
- `results/posterior_predictive_summary.json`
- `results/run_summary.json`
- `results/artifact_index.json`
- `results/figures/*.png` for requested plots

Diagnostics include:

- split `R-hat`
- effective sample size
- energy diagnostics including E-BFMI
- divergent transitions
- tree depth saturation
- acceptance statistics

## Failure handling

Handle common failures explicitly:

- broken JAX or BlackJAX imports in the current Python environment
- ambiguous observed-output mapping
- non-finite model outputs
- invalid parameter transforms or impossible priors
- gradient failures in direct JAX mode
- divergences, low E-BFMI, low ESS, or large `R-hat`

If a run cannot proceed safely, stop with a concrete remediation message instead of returning misleading posterior summaries.

## Extensibility

Keep the runtime modular:

- backend selection is separate from inference orchestration
- priors, likelihoods, transforms, diagnostics, and plotting are separate modules
- `algorithm.name` is separate from `algorithm.backend`
- future extensions can add:
  - standard `hmc`
  - Riemannian HMC
  - variational inference
  - alternate probabilistic backends such as `pymc`, `numpyro`, `stan`, or `tensorflow_probability`

## Recommended command sequence

1. `doctor`
2. `inspect-model`
3. resolve pending clarification questions in `project_summary.json`
4. `create-project`
5. review `config.json`
6. `python3 run.py`
7. inspect posterior summaries, diagnostics, and predictive checks

## References

- Quickstart: `references/quickstart.md`
- Editable config template: `references/config.template.json`
- Model and likelihood adapter contracts: `references/model-adapters.md`

