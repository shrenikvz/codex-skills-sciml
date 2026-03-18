# Codex Skills for Scientific Machine Learning

This repository contains Codex skills focused on scientific machine learning workflows. At the moment, the repository includes three production skills:

- `abc-calibration`: likelihood-free parameter estimation and simulator calibration with Approximate Bayesian Computation (ABC) rejection sampling
- `physics-informed-neural-networks`: forward and inverse differential-equation-constrained learning with Physics-Informed Neural Networks (PINNs)
- `nuts-calibration`: Bayesian parameter estimation and calibration with No-U-Turn Sampler (NUTS) using BlackJAX and JAX

## Overview

The current skill is designed for cases where a model or simulator must be calibrated from observed data, but the likelihood is unavailable, implicit, or too expensive to evaluate directly.

Use `abc-calibration` for tasks such as:

- calibrating scientific simulators or engineering digital twins
- estimating unknown parameters from experimental or observational data
- running likelihood-free inference for deterministic or stochastic models
- producing posterior summaries and posterior predictive checks

The skill supports Python callables, mathematical expressions wrapped into Python, and command-line simulators in other languages through generated adapters.

Use `physics-informed-neural-networks` for tasks such as:

- solving ODEs and PDEs with PINNs under boundary and initial conditions
- inferring unknown physical parameters from sparse observations
- combining data losses with physics residuals in a reusable training project
- generating collocation samplers, diagnostics, predictions, and physics-aware visualizations

Use `nuts-calibration` for tasks such as:

- fitting differentiable models or simulators to observed data with Bayesian inference
- building priors, likelihoods, parameter transforms, and scaling choices for calibration
- running adaptive warmup and multi-chain NUTS posterior sampling
- reporting convergence diagnostics, credible intervals, and posterior predictive checks

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
    ├── nuts-calibration/
    │   ├── SKILL.md
    │   ├── agents/
    │   ├── references/
    │   └── scripts/
    └── physics-informed-neural-networks/
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

### `nuts-calibration`

General-purpose Bayesian parameter estimation and model calibration for differentiable models, simulators, mathematical expressions, and probabilistic model components using NUTS with BlackJAX and JAX by default.

Key behavior:

- inspects the model and observed data before sampling
- infers parameters, observed-output mappings, priors, likelihoods, transforms, and scaling recommendations
- probes JAX and BlackJAX availability before attempting sampling
- scaffolds runnable calibration projects with diagnostics, posterior summaries, and posterior predictive checks

Primary references:

- [skills/nuts-calibration/SKILL.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-calibration/SKILL.md)
- [skills/nuts-calibration/references/quickstart.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-calibration/references/quickstart.md)
- [skills/nuts-calibration/references/model-adapters.md](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-calibration/references/model-adapters.md)

## Installation

### Install this repository's skills

Clone the repository and run the installer:

```bash
git clone git@github.com:shrenikvz/codex-skills-sciml.git
cd codex-skills-sciml
./scripts/install-all-skills.sh --force
```

This installs every skill under `skills/` into `${CODEX_HOME:-$HOME/.codex}/skills`. Right now, that means `abc-calibration`, `nuts-calibration`, and `physics-informed-neural-networks`.

### Install a single skill from GitHub

If you prefer installing only `abc-calibration`, use the Codex skill installer directly:

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo shrenikvz/codex-skills-sciml \
  --path skills/abc-calibration
```

Restart Codex after installation so the new skill is loaded.

If you prefer installing only `physics-informed-neural-networks`, use:

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo shrenikvz/codex-skills-sciml \
  --path skills/physics-informed-neural-networks
```

If you prefer installing only `nuts-calibration`, use:

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo shrenikvz/codex-skills-sciml \
  --path skills/nuts-calibration
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

For `physics-informed-neural-networks`, start with:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py doctor
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py inspect-problem \
  --problem-path ./problem.json \
  --observed-path ./observations.csv
```

For `nuts-calibration`, start with:

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py doctor
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Estimate parameters with NUTS"
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

Unit tests for `physics-informed-neural-networks` live in [skills/physics-informed-neural-networks/scripts](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/physics-informed-neural-networks/scripts).

Run them with:

```bash
cd skills/physics-informed-neural-networks/scripts
PYTHONPATH=. python3 -m unittest -q \
  test_physics_informed_neural_networks_unit.py \
  test_physics_informed_neural_networks_cli_unit.py
```

Unit tests for `nuts-calibration` live in [skills/nuts-calibration/scripts](/Users/shrenikzinage/Documents/Purdue%20Research/Codes/github_repositaries/codex-skills-sciml/skills/nuts-calibration/scripts).

Run them with:

```bash
cd skills/nuts-calibration/scripts
PYTHONPATH=. python3 -m unittest -q \
  test_nuts_calibration_unit.py \
  test_nuts_calibration_cli_unit.py
```

## Notes

- The README now reflects the skills currently present in this repository.
- If additional skills are added under `skills/`, this document should be updated to keep the inventory and install guidance accurate.
