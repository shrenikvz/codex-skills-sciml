# Model Adapters

## Python callable adapter

Expected forms:

```python
def simulate(theta, sigma):
    ...
```

```python
def simulate(params):
    ...
```

```python
def log_likelihood(params, simulated, observed, metadata):
    ...
```

The model output may be:

- scalar
- list
- NumPy array
- dict of named outputs

If the output is a dict, observed-output selection prefers matching keys.

## Equation wrapper

You can provide `--equation-text` instead of a model file. The runtime generates a Python wrapper with a JAX-friendly `simulate(...)` function.

Example:

```bash
python3 ~/.codex/skills/nuts-bayesian-inference/scripts/nuts_bayesian_inference.py create-project \
  --project-dir ./run \
  --equation-text "np.array([a * x0 + b, a * x1 + b])" \
  --parameter a \
  --parameter b \
  --observed-path ./observed.json
```

## Command adapter

For non-Python models, the runtime expects a command template with these placeholders:

- `{model_path}`
- `{params_json}`
- `{output_json}`

Example:

```bash
--command-template 'julia {model_path} {params_json} {output_json}'
```

The external program must:

1. read parameter values from the JSON file
2. write model outputs to the output JSON file, or print JSON to stdout

## User-defined likelihoods

Custom likelihoods are separate from the model adapter. The simplest pattern is a Python callable:

```python
def log_likelihood(params, simulated, observed, metadata):
    return ...
```

The runtime passes:

- `params`: dict of current parameter values, including likelihood nuisance parameters
- `simulated`: raw model outputs selected for calibration
- `observed`: raw observed data aligned to the selected outputs
- `metadata`: config and scaling context

## Differentiability fallback

When a model is not directly JAX-compatible:

1. the runtime tries direct JAX evaluation if the callable appears JAX-native
2. otherwise it wraps the host model with `jax.pure_callback`
3. gradients are supplied through numerical differentiation

This preserves BlackJAX compatibility while keeping external models usable.
