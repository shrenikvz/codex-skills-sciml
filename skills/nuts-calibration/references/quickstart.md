# Quickstart

## 1. Verify the environment

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py doctor
```

This reports whether `jax` and `blackjax` can actually be imported in the current Python runtime, which is more important than package presence alone.

## 2. Inspect the model and observed data

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Calibrate this model with NUTS"
```

Inspection returns:

- inferred parameter names and defaults
- observed-output mapping
- prior recommendations
- likelihood recommendation
- transform recommendations
- scaling recommendation
- backend readiness
- pending clarification questions

## 3. Create a portable project

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

Inspect:

- `nuts-run/project_summary.json`
- `nuts-run/config.json`
- `nuts-run/AGENT_RUNBOOK.md`

## 4. Run sampling

```bash
cd nuts-run
python3 run.py
```

Main outputs:

- `results/posterior_summary.json`
- `results/sampling_diagnostics.json`
- `results/posterior_predictive_summary.json`
- `results/figures/`

## 5. Mathematical model example

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py create-project \
  --project-dir ./nuts-equation \
  --equation-text "a * jnp.exp(-b * jnp.arange(25))" \
  --parameter a \
  --parameter b \
  --parameter-bound a=0:10 \
  --parameter-bound b=0:2 \
  --observed-path ./decay_curve.json
```

## 6. Non-Python simulator example

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py create-project \
  --project-dir ./nuts-julia \
  --model-path ./simulate.jl \
  --observed-path ./observed.csv \
  --parameter theta \
  --parameter sigma
```

The generated project creates `model/run_model.sh` and assumes:

```text
simulate.jl <params_json> <output_json>
```

Adjust `model/run_model.sh` if the simulator uses a different interface.

## 7. Tips

- Use `inspect-model` before choosing hyperparameters manually.
- Increase warmup or target acceptance when divergences appear.
- Prefer tighter priors and better transforms before increasing tree depth aggressively.
- Callback-based finite-difference gradients are slower; use a JAX-compatible model when possible.

