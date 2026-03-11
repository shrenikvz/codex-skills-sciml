---
name: abc-calibration
description: General-purpose parameter estimation and model calibration for scientific simulators, black-box models, deterministic/stochastic codes, engineering digital twins, and mathematical models using Approximate Bayesian Computation with rejection sampling. Use when Codex needs to infer unknown parameters from observed data and the likelihood is unavailable, implicit, or computationally intractable; when a user asks for ABC, likelihood-free inference, simulator calibration, or posterior predictive validation for arbitrary model code.
---

# ABC Calibration

Use this skill when users want to calibrate model parameters from observed data with likelihood-free inference.

Natural-language trigger examples:

- "Calibrate this simulator with ABC rejection."
- "Estimate the unknown parameters of my stochastic model from experimental data."
- "This model has no tractable likelihood; use approximate Bayesian computation."
- "Infer the posterior of my simulator parameters and validate it with posterior predictive checks."
- "Fit this engineering digital twin to sensor data without writing a closed-form likelihood."
- "I have a mathematical model and observed outputs; recover the parameters."

This skill enforces a strict gating rule:

1. Inspect the model and observed data first.
2. Decide whether a tractable likelihood probably exists.
3. If likelihood-based inference is feasible, recommend MCMC or variational inference before running ABC.
4. Use ABC-Rejection only when the likelihood is unavailable, implicit, or intractable, or when the user explicitly insists on ABC.

## First-run workflow

Inspect the model first:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Calibrate this stochastic simulator with ABC rejection"
```

Then create a runnable project:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py create-project \
  --project-dir ./abc-run \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Estimate parameters with likelihood-free calibration" \
  --pilot-size 1000 \
  --main-budget 20000 \
  --accepted-samples 500 \
  --epsilon-quantile 0.05
```

Then run it:

```bash
cd abc-run
python3 run.py
```

Or run through the CLI:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py run \
  --config ./abc-run/config.json
```

## Supported model forms

- Python functions returning scalars, vectors, arrays, or dicts of outputs
- Command-line simulators in other languages via a generated command wrapper
- Mathematical expressions wrapped into Python with `--equation-text`
- Deterministic or stochastic simulators
- Partially observed models through explicit output-name or output-index selection

Model adapter behavior:

- Python model files use callable inspection to infer parameter names and defaults.
- Non-Python model files create `model/run_model.sh` and assume `<params_json> <output_json>` unless the user provides an explicit `--command-template`.
- When output mapping is ambiguous, stop and ask which model outputs correspond to the observed data.

## Decision workflow

1. Inspect model inputs, outputs, defaults, and runtime form.
2. Load observed data and infer its shape, names, and dimensionality.
3. Ask for clarification only when high-impact ambiguity remains:
   - which outputs are observed
   - missing parameter names for black-box command adapters
   - nonstandard command wrapper contracts
4. Build priors:
   - use user priors directly when provided
   - otherwise infer priors from parameter names, defaults, bounds, and common domain heuristics
5. Assess likelihood availability.
6. Choose scaling, summary statistics, and distance metric.
7. Run two-phase ABC-Rejection:
   - pilot phase estimates `epsilon` from the `zeta` quantile of prior-sampled distances
   - main phase samples until `N_accept` accepted draws or the budget is exhausted
8. Return posterior summaries, diagnostics, and optional posterior predictive checks.

## Priors and hyperparameters

Supported priors:

- `uniform`
- `normal`
- `lognormal`
- `gamma`
- `beta`

Prior behavior:

- `--prior-file priors.json` for explicit priors
- `--prior theta=uniform(0,5)` for direct overrides
- `--parameter-bound theta=0:5` for bound-driven inference
- if priors are missing, infer them heuristically and record provenance in `results/prior_report.json`

Hyperparameters are configurable in `config.json` or during `create-project`:

- `pilot_size`
- `main_budget`
- `accepted_samples`
- `epsilon_quantile`
- `distance.metric`
- `summary_statistics.kind`
- `compute.max_workers`

If not provided, the skill derives defaults from parameter dimension and observed output size.

## Distance, scaling, and summary statistics

Supported distance metrics:

- `rmse`
- `nrmse`
- `ks`
- `euclidean`
- `mahalanobis`
- `wasserstein`
- `custom`

Scaling behavior:

- Ask whether scaling should be applied.
- If the user does not specify, choose automatically from `zscore`, `minmax`, `variance`, or `none` based on magnitude and spread.
- Apply the same transformation to observed and simulated summaries before distance evaluation.

Summary-statistic behavior:

- Use `identity` for low-dimensional outputs.
- Use `timeseries` or `moments` automatically for high-dimensional outputs.
- Support custom Python or command-based summary functions through `config.json`.

## Parallel execution

- Parallelize simulator evaluations with `compute.max_workers`.
- Default `auto` uses available CPU cores conservatively.
- Keep batch sizes moderate for expensive simulators; tune `algorithm.two_phase.batch_size` when memory pressure is high.

## Visualizations

Ask which plots the user actually wants before enabling them.

Supported plot names:

- `posterior_marginals`
- `pairwise`
- `posterior_predictive`
- `trace`
- `calibration_diagnostics`

Plots are generated only when `visualization.enabled=true` and the plot name is present in `visualization.plots`.

## Outputs

Run outputs include:

- `results/posterior_samples.csv`
- `results/posterior_samples.jsonl`
- `results/posterior_summary.json`
- `results/pilot_phase.json`
- `results/sampling_diagnostics.json`
- `results/prior_report.json`
- `results/likelihood_assessment.json`
- `results/posterior_predictive_summary.json`
- `results/run_summary.json`
- `results/artifact_index.json`
- `results/figures/*.png` for requested plots

Generated project scaffolding includes:

- `config.json`
- `run.py`
- `project_summary.json`
- `AGENT_RUNBOOK.md`
- `skill_runtime/abc_calibration_lib/` for portable execution

## Failure handling

Handle common simulator issues without crashing the full run:

- simulator exceptions or command failures are recorded as failed proposals
- non-finite outputs are rejected
- invalid parameter draws remain in the trace with error status
- if no pilot samples succeed, stop and fix the simulator contract before continuing
- if the main budget is exhausted early, return a partial posterior with explicit diagnostics

## Extensibility

The runtime config keeps `algorithm.name` separate from `algorithm.two_phase` so the same scaffold can be extended later to:

- `abc_smc`
- `abc_mcmc`

Keep new algorithms compatible with the same model adapters, priors, summaries, scaling, diagnostics, and posterior predictive interfaces.

## Recommended command sequence

1. `inspect-model`
2. `create-project`
3. review `project_summary.json` and `config.json`
4. `python3 run.py`
5. inspect posterior summaries and predictive checks

## References

- Quickstart: `references/quickstart.md`
- Editable config template: `references/config.template.json`
- Model adapter contracts: `references/model-adapters.md`
