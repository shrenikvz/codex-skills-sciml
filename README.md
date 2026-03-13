# Codex Skills for Scientific Machine Learning

This repository contains Codex skills focused on scientific machine learning workflows. At the moment, the repository includes three production skills:

- `abc-calibration`: likelihood-free parameter estimation and simulator calibration with Approximate Bayesian Computation (ABC) rejection sampling
- `nuts-bayesian-inference`: likelihood-based posterior sampling for tractable scientific models with BlackJAX NUTS
- `physics-informed-neural-networks`: forward and inverse differential-equation-constrained learning with Physics-Informed Neural Networks (PINNs)

## Overview

The current skill is designed for cases where a model or simulator must be calibrated from observed data, but the likelihood is unavailable, implicit, or too expensive to evaluate directly.

Use `abc-calibration` for tasks such as:

- calibrating scientific simulators or engineering digital twins
- estimating unknown parameters from experimental or observational data
- running likelihood-free inference for deterministic or stochastic models
- producing posterior summaries and posterior predictive checks

The skill supports Python callables, mathematical expressions wrapped into Python, and command-line simulators in other languages through generated adapters.

Use `nuts-bayesian-inference` for tasks such as:

- estimating posterior distributions when a tractable likelihood can be written down
- calibrating differentiable or wrapped simulator models with adaptive NUTS sampling
- running multi-chain BlackJAX workflows with diagnostics and posterior predictive checks
- building reusable Bayesian inference projects with priors, transformations, diagnostics, and plots

Use `physics-informed-neural-networks` for tasks such as:

- solving ODEs and PDEs with PINNs under boundary and initial conditions
- inferring unknown physical parameters from sparse observations
- combining data losses with physics residuals in a reusable training project
- generating collocation samplers, diagnostics, predictions, and physics-aware visualizations

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── scripts/
│   └── install-all-skills.sh
└── skills/
    ├── abc-calibration/
    │   ├── SKILL.md
    │   ├── agents/
    │   ├── references/
    │   └── scripts/
    ├── physics-informed-neural-networks/
    │   ├── SKILL.md
    │   ├── agents/
    │   ├── references/
    │   └── scripts/
    └── nuts-bayesian-inference/
        ├── SKILL.md
        ├── agents/
        ├── references/
        └── scripts/
```

## Available Skills

### `abc-calibration`

General-purpose parameter estimation and model calibration for scientific simulators, black-box models, deterministic or stochastic codes, engineering digital twins, and mathematical models using Approximate Bayesian Computation with rejection sampling.

Key behavior:

- inspects the model and observed data before running inference
- checks whether likelihood-based inference is more appropriate before defaulting to ABC
- scaffolds runnable calibration projects with configuration, runtime code, and diagnostics
- returns posterior samples, summaries, diagnostics, and optional posterior predictive plots

Primary references:

- [skills/abc-calibration/SKILL.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/abc-calibration/SKILL.md)
- [skills/abc-calibration/references/quickstart.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/abc-calibration/references/quickstart.md)
- [skills/abc-calibration/references/model-adapters.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/abc-calibration/references/model-adapters.md)

### `nuts-bayesian-inference`

General-purpose Bayesian inference for scientific and engineering models with tractable likelihoods using the No-U-Turn Sampler (NUTS), BlackJAX, and JAX.

Key behavior:

- inspects model inputs, outputs, observed-output mapping, and differentiability
- constructs priors, likelihoods, parameter transformations, and optional scaling
- uses BlackJAX window adaptation plus multi-chain NUTS sampling by default
- reports R-hat, effective sample size, acceptance rates, energy diagnostics, divergences, and tree-depth warnings
- supports posterior predictive checks and optional diagnostic plots

Primary references:

- [skills/nuts-bayesian-inference/SKILL.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-bayesian-inference/SKILL.md)
- [skills/nuts-bayesian-inference/references/quickstart.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-bayesian-inference/references/quickstart.md)
- [skills/nuts-bayesian-inference/references/model-adapters.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-bayesian-inference/references/model-adapters.md)

### `physics-informed-neural-networks`

General-purpose training and evaluation of Physics-Informed Neural Networks (PINNs) for forward problems, inverse parameter inference, data-assisted physics learning, and operator-learning-style setups governed by ODEs, PDEs, algebraic constraints, conservation laws, and constitutive relations.

Key behavior:

- inspects a physics problem specification before training
- infers variables, domains, conditions, and observation mappings
- recommends architectures, sampling strategies, loss weighting, and stabilization
- scaffolds portable PINN projects with diagnostics, predictions, and optional figures

Primary references:

- [skills/physics-informed-neural-networks/SKILL.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/physics-informed-neural-networks/SKILL.md)
- [skills/physics-informed-neural-networks/references/quickstart.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/physics-informed-neural-networks/references/quickstart.md)
- [skills/physics-informed-neural-networks/references/problem-adapters.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/physics-informed-neural-networks/references/problem-adapters.md)

## Installation

### Install this repository's skills

Clone the repository and run the installer:

```bash
git clone git@github.com:shrenikvz/codex-skills-sciml.git
cd codex-skills-sciml
./scripts/install-all-skills.sh --force
```

This installs every skill under `skills/` into `${CODEX_HOME:-$HOME/.codex}/skills`. Right now, that means `abc-calibration`, `nuts-bayesian-inference`, and `physics-informed-neural-networks`.

### Install a single skill from GitHub

If you prefer installing only `abc-calibration`, use the Codex skill installer directly:

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo shrenikvz/codex-skills-sciml \
  --path skills/abc-calibration
```

Restart Codex after installation so the new skill is loaded.

If you prefer installing only `nuts-bayesian-inference`, use:

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo shrenikvz/codex-skills-sciml \
  --path skills/nuts-bayesian-inference
```

If you prefer installing only `physics-informed-neural-networks`, use:

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo shrenikvz/codex-skills-sciml \
  --path skills/physics-informed-neural-networks
```

## Quick Start

Inspect the model first:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Calibrate this stochastic simulator with ABC rejection"
```

Create a runnable project:

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

Run the generated project:

```bash
cd abc-run
python3 run.py
```

For `nuts-bayesian-inference`, start with:

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py doctor
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json
```

For `physics-informed-neural-networks`, start with:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py doctor
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py inspect-problem \
  --problem-path ./problem.json \
  --observed-path ./observations.csv
```

## Testing

Unit tests for `abc-calibration` live in [skills/abc-calibration/scripts](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/abc-calibration/scripts).

Run them with:

```bash
cd skills/abc-calibration/scripts
PYTHONPATH=. python3 -m unittest -q \
  test_abc_calibration_unit.py \
  test_abc_calibration_cli_unit.py
```

Unit tests for `nuts-bayesian-inference` live in [skills/nuts-bayesian-inference/scripts](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-bayesian-inference/scripts).

Run them with:

```bash
cd skills/nuts-bayesian-inference/scripts
PYTHONPATH=. python3 -m unittest -q \
  test_nuts_inference_unit.py \
  test_nuts_inference_cli_unit.py
```

Unit tests for `physics-informed-neural-networks` live in [skills/physics-informed-neural-networks/scripts](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/physics-informed-neural-networks/scripts).

Run them with:

```bash
cd skills/physics-informed-neural-networks/scripts
PYTHONPATH=. python3 -m unittest -q \
  test_physics_informed_neural_networks_unit.py \
  test_physics_informed_neural_networks_cli_unit.py
```

## Notes

- The README now reflects the skills currently present in this repository.
- If additional skills are added under `skills/`, this document should be updated to keep the inventory and install guidance accurate.
