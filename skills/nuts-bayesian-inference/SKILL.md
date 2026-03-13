---
name: nuts-bayesian-inference
description: General-purpose Bayesian inference for scientific and engineering models with tractable likelihoods using the No-U-Turn Sampler (NUTS). Use when Codex needs to estimate posterior distributions for model parameters from observed data, construct priors and likelihoods, run adaptive warmup and multi-chain posterior sampling with BlackJAX and JAX, diagnose convergence, and produce posterior predictive checks for arbitrary differentiable or wrapped simulator code.
---

# NUTS Bayesian Inference

Use this skill when users want likelihood-based Bayesian calibration or posterior sampling with NUTS.

Natural-language trigger examples:

- "Fit this scientific model with NUTS."
- "Estimate posterior distributions for these simulator parameters with BlackJAX."
- "Run Bayesian inference for my engineering model with a Gaussian likelihood."
- "Use JAX and NUTS for posterior sampling on this differentiable function."
- "Build a reusable posterior sampling project for this model and dataset."
- "Calibrate this model with Bayesian inference instead of ABC."

This skill assumes the likelihood is tractable or can be explicitly written down. If the likelihood is unavailable, implicit, or computationally intractable, use `abc-calibration` instead.

## First-run workflow

Check the runtime first:

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py doctor
```

Inspect the model and observed data:

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Infer posterior parameters with NUTS and a Gaussian likelihood"
```

Create a runnable project:

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py create-project \
  --project-dir ./nuts-run \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Run BlackJAX NUTS posterior inference" \
  --num-warmup 1000 \
  --num-samples 1000 \
  --num-chains 4 \
  --target-acceptance 0.85
```

Run the generated project:

```bash
cd nuts-run
python3 run.py
```

Or run directly from a config file:

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py run \
  --config ./nuts-run/config.json
```

## Required reasoning sequence

1. Inspect the model implementation and the observed data.
2. Infer parameter names, defaults, outputs, and which outputs are observed.
3. If output mapping is ambiguous, stop and ask which outputs correspond to the observations.
4. Determine whether the likelihood is tractable and what observation family is appropriate.
5. Use user-specified priors if present; otherwise infer priors heuristically and document the provenance.
6. Determine parameter constraints and choose transformations with Jacobian adjustments.
7. Ask whether normalization should be applied. If the user does not specify, choose automatically and document the choice.
8. Ask which visualizations are wanted before generating them.
9. Check JAX and BlackJAX availability with `doctor` when the environment is uncertain.
10. Run adaptive warmup, multi-chain NUTS sampling, diagnostics, and optional posterior predictive checks.

## Supported model forms

- Python callables that map parameters to outputs
- Mathematical expressions wrapped into a generated Python model
- Command-line models in other languages through a generated wrapper or explicit command template
- Python log-likelihood callables for user-defined likelihoods
- Partially observed models through explicit output-name or output-index selection

Model adapter behavior:

- Python model files use signature inspection to infer parameter names and defaults.
- Equation text is wrapped into a JAX-friendly Python callable.
- Non-Python or external models use command adapters with the contract `<params_json> <output_json>` unless an explicit command template is supplied.
- If the model cannot be differentiated directly by JAX, the runtime falls back to host callbacks with numerical gradients so BlackJAX can still be used.

## Priors

Supported priors:

- `uniform`
- `normal`
- `lognormal`
- `gamma`
- `beta`
- `halfnormal`
- `student_t`

Prior behavior:

- `--prior-file priors.json` for explicit priors
- `--prior theta=normal(mean=0,std=1)` for inline overrides
- `--parameter-bound theta=0:5` for bound-driven inference
- missing priors are inferred from parameter names, defaults, bounds, and common scientific heuristics
- likelihood nuisance parameters such as observation scale can be introduced automatically and are documented in `results/prior_report.json`

## Likelihoods

Supported likelihood families:

- `gaussian`
- `student_t`
- `poisson`
- `binomial`
- `negative_binomial`
- `custom`

Likelihood behavior:

- Continuous observations default to Gaussian unless the user requests Student-t or provides a custom likelihood.
- Non-negative integer observations default to Poisson or Negative Binomial depending on dispersion heuristics.
- Binary observations default to Binomial with `total_count=1`.
- If the noise model is ambiguous, the inspection report records a clarification question.
- Custom likelihoods can be supplied through Python or command hooks in `config.json`.

## Transformations and scaling

Supported parameter transforms:

- `identity`
- `log`
- `logit`
- `softplus`

Behavior:

- constraints are inferred from priors and bounds
- Jacobian adjustments are included in the log-posterior
- scaling modes include `none`, `zscore`, `minmax`, and `variance`
- the same scaling transformation is applied to observed and simulated outputs during likelihood evaluation
- count likelihoods default to no scaling unless the user explicitly overrides it

## Sampler defaults

The default backend is `blackjax`.

Warmup and sampler defaults are inferred from dimension and output size, then written into `config.json`:

- `num_warmup`
- `num_samples`
- `num_chains`
- `target_acceptance_rate`
- `max_tree_depth`
- `mass_matrix`
- optional `step_size`

BlackJAX window adaptation is used for the three-phase warmup schedule:

1. initial fast adaptation
2. mass matrix adaptation windows
3. final step-size stabilization

## Parallel execution and devices

- Multi-chain execution uses `jax.vmap` by default.
- If multiple matching accelerator devices are available, the runtime can switch to `jax.pmap`.
- Use `compute.device_preference` to request `cpu`, `gpu`, `tpu`, or `auto`.
- Use `doctor` before sampling on a new machine to verify JAX device availability.

## Diagnostics and outputs

Ask which plots the user wants before enabling them.

Supported plot names:

- `posterior_marginals`
- `pairwise`
- `trace`
- `autocorrelation`
- `energy`
- `posterior_predictive`

Run outputs include:

- `results/posterior_samples.csv`
- `results/posterior_samples.jsonl`
- `results/posterior_summary.json`
- `results/credible_intervals.json`
- `results/diagnostics.json`
- `results/acceptance_statistics.json`
- `results/tuned_hyperparameters.json`
- `results/prior_report.json`
- `results/likelihood_report.json`
- `results/posterior_predictive_summary.json`
- `results/run_summary.json`
- `results/artifact_index.json`
- `results/figures/*.png` for requested plots

Generated project scaffolding includes:

- `config.json`
- `run.py`
- `project_summary.json`
- `AGENT_RUNBOOK.md`
- `skill_runtime/nuts_inference_lib/`

## Failure handling

Handle common failure modes explicitly:

- ambiguous observed-output mapping
- non-finite log-posterior or model outputs
- JAX import or device failures
- gradient failures in direct autodiff mode
- fallback to numerical-gradient host callbacks
- divergent transitions
- tree-depth saturation
- poor chain mixing or low effective sample size
- unstable scaling or invalid count likelihood settings

If the environment blocks JAX or BlackJAX, the skill should stop with a concrete remediation message instead of emitting a partial run.

## Extensibility

Keep the runtime modular:

- backend registry is separate from project scaffolding
- `blackjax` is the default backend
- backend hooks are reserved for `numpyro`, `pymc`, `stan`, and `tensorflow_probability`
- algorithm names stay separate from the backend so future extensions can add `hmc`, `rmhmc`, or variational inference without breaking the project format

## Recommended command sequence

1. `doctor`
2. `inspect-model`
3. resolve any pending questions in `project_summary.json`
4. `create-project`
5. review `config.json`
6. `python3 run.py`
7. inspect summaries, diagnostics, and posterior predictive artifacts

## References

- Quickstart: `references/quickstart.md`
- Editable config template: `references/config.template.json`
- Model adapter contracts: `references/model-adapters.md`
