# Quickstart

## 1. Verify the environment

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py doctor
```

If `jax` or `blackjax` cannot be imported, fix the environment before sampling.

## 2. Inspect the model

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Estimate posterior parameters with NUTS"
```

This reports:

- inferred parameter names
- observed-output mapping
- prior recommendations
- likelihood recommendation
- differentiability strategy
- scaling recommendation
- default hyperparameters

## 3. Create a portable project

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py create-project \
  --project-dir ./nuts-run \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --plot posterior_marginals \
  --plot trace \
  --plot posterior_predictive
```

Inspect:

- `nuts-run/project_summary.json`
- `nuts-run/config.json`
- `nuts-run/AGENT_RUNBOOK.md`

## 4. Run inference

```bash
cd nuts-run
python3 run.py
```

Main outputs:

- `results/posterior_summary.json`
- `results/diagnostics.json`
- `results/tuned_hyperparameters.json`
- `results/figures/`

## 5. Custom likelihoods

Set these fields in `config.json` for a custom Python likelihood:

```json
{
  "likelihood": {
    "family": "custom",
    "custom_python_path": "model/custom_likelihood.py",
    "custom_callable": "log_likelihood"
  }
}
```

The callable should accept:

```python
def log_likelihood(params, simulated, observed, metadata):
    ...
```

## 6. External simulators

For non-Python models, provide either:

- `--command-template 'julia {model_path} {params_json} {output_json}'`
- or a model path with a recognized suffix so the generated wrapper can call it

The model must write JSON to `output_json` or print JSON to stdout.
