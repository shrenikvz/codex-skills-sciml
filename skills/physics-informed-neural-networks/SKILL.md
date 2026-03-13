---
name: physics-informed-neural-networks
description: General-purpose training and evaluation of Physics-Informed Neural Networks (PINNs) for forward problems, inverse parameter inference, data-assisted physics learning, and operator-learning-style setups governed by ODEs, PDEs, algebraic constraints, conservation laws, and constitutive relations. Use when Codex needs to inspect a physics problem definition, infer domains and conditions, choose a PINN architecture and autodiff backend, scaffold a reusable training project, run physics-constrained optimization, diagnose instability, and return predictions, inferred parameters, residual diagnostics, and requested visualizations.
---

# Physics-Informed Neural Networks

Use this skill when users want a general PINNs workflow for differential-equation-constrained learning.

Natural-language trigger examples:

- "Train a PINN for this PDE."
- "Solve this ODE with physics-informed neural networks."
- "Infer unknown coefficients in this diffusion equation from sparse observations."
- "Combine observational data and conservation laws in a PINN."
- "Build a reusable PINN project for this coupled PDE system."
- "Use a Fourier-feature PINN for this stiff multiscale problem."
- "Set up operator-learning-style physics-constrained training for this field mapping."

This skill follows a strict reasoning sequence:

1. Inspect the problem specification, not just the equations.
2. Infer independent variables, dependent variables, domains, boundary conditions, initial conditions, and observational data.
3. Classify the task as `forward`, `inverse`, `data_assisted`, or `operator_learning`.
4. Ask for clarification when the physics specification is materially ambiguous.
5. Ask which loss-weighting strategy the user wants before finalizing training.
6. Ask which visualizations are wanted before enabling them.
7. Probe the runtime environment before training if the framework situation is uncertain.
8. Build a portable project, then train, evaluate, and diagnose.

## First-run workflow

Check the runtime first:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py doctor
```

Inspect the physics problem:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py inspect-problem \
  --problem-path ./burgers_problem.json \
  --observed-path ./observations.csv \
  --request-text "Train a PINN for Burgers' equation and infer viscosity"
```

Create a runnable project:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py create-project \
  --project-dir ./pinn-run \
  --problem-path ./burgers_problem.json \
  --observed-path ./observations.csv \
  --architecture fourier \
  --sampling-strategy sobol \
  --loss-weighting gradient_norm \
  --optimizer hybrid \
  --plot solution_field \
  --plot loss_curves
```

Run the generated project:

```bash
cd pinn-run
python3 run.py
```

Or run directly from a config file:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py run \
  --config ./pinn-run/config.json
```

## Supported problem inputs

- structured JSON problem specifications
- Python problem modules exposing `problem_spec` or `build_problem()`
- lightweight text or Markdown physics specs for inspection and initial scaffolding
- symbolic residual expressions using derivative tokens such as `u__x`, `u__xx`, `u__t`
- observational datasets in `csv`, `tsv`, `json`, `npy`, or `npz`

If the problem is underspecified, stop and ask for clarification instead of guessing high-impact details such as:

- missing domains
- missing dependent variables
- ambiguous boundary or initial conditions
- unclear observational column mapping
- unknown parameter roles in inverse problems

## Problem classification

The workflow classifies the task automatically:

- `forward`: solve the governing equations given conditions
- `inverse`: infer unknown physical parameters
- `data_assisted`: blend observational data with physics residuals
- `operator_learning`: use conditioning features or context fields to learn families of solutions

If the user explicitly specifies the task type, use that directly.

## Architecture selection

Supported architectures:

- `mlp`
- `fourier`
- `resnet`
- `multiscale`
- `coordinate`
- `transformer_operator`

Selection behavior:

- use the user-specified architecture when given
- otherwise choose from domain dimension, expected smoothness, multiscale behavior, data availability, and task type
- reserve `transformer_operator` for operator-learning-style problems or rich conditioning inputs

## Autodiff and frameworks

The runtime probes and reports support for:

- `torch`
- `jax`
- `tensorflow`

Framework selection is automatic unless the user overrides it. The generated project records the chosen framework and device preference in `config.json`.

## Sampling, losses, and optimization

Supported sampling strategies:

- `uniform`
- `latin_hypercube`
- `sobol`
- `adaptive`
- `residual_adaptive`

Supported loss-weighting strategies:

- `fixed`
- `dynamic_balance`
- `gradient_norm`
- `uncertainty`
- `adaptive_residual`

Supported optimizers and schedules:

- `adam`
- `lbfgs`
- `hybrid`
- `curriculum`
- `domain_decomposition`

If the user does not provide hyperparameters, the skill writes defensible defaults into `config.json` and explains the reasoning in `project_summary.json`.

## Stabilization behavior

The skill can enable or recommend:

- coordinate scaling
- residual normalization
- gradient clipping
- adaptive activations
- Fourier features
- domain decomposition

When instability is detected, the runtime records the event and recommends concrete remediation in diagnostics and the run summary.

## Visualizations

Ask which plots the user wants before enabling them.

Supported plot names:

- `solution_field`
- `time_evolution`
- `residual_heatmap`
- `loss_curves`
- `analytical_comparison`
- `uncertainty_bands`

Plots are generated only when `visualization.enabled=true` and plot names are listed in `visualization.plots`.

## Outputs

Run outputs include:

- `results/training_history.jsonl`
- `results/diagnostics.json`
- `results/residual_diagnostics.json`
- `results/evaluation_summary.json`
- `results/inferred_parameters.json`
- `results/solution_predictions.json`
- `results/run_summary.json`
- `results/artifact_index.json`
- `results/figures/*.png` for requested plots

Generated project scaffolding includes:

- `config.json`
- `run.py`
- `project_summary.json`
- `AGENT_RUNBOOK.md`
- `skill_runtime/physics_informed_nn_lib/`

## Failure handling

Handle common failure modes explicitly:

- incompatible or incomplete physics definitions
- unresolved clarification questions
- framework import failures
- non-finite residuals or losses
- gradient explosions
- poor collocation coverage
- optimizer stagnation
- unstable parameter inference

If a run cannot proceed safely, stop with a concrete remediation message instead of emitting misleading outputs.

## Extensibility

Keep the runtime modular:

- framework probing is separate from training orchestration
- architecture selection is separate from residual construction
- sampling, weighting, diagnostics, and visualization modules are separate
- configuration leaves room for future `deeponet`, `neural_operator`, `probabilistic_pinn`, and `bayesian_pinn` extensions

## Recommended command sequence

1. `doctor`
2. `inspect-problem`
3. resolve pending clarification questions in `project_summary.json`
4. `create-project`
5. review `config.json`
6. `python3 run.py`
7. inspect diagnostics, inferred parameters, predictions, and requested figures

## References

- Quickstart: `references/quickstart.md`
- Editable config template: `references/config.template.json`
- Problem specification contracts: `references/problem-adapters.md`
