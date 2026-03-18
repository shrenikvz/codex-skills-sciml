# Model and Likelihood Adapter Contracts

## Python callable adapter

Use when the model lives in a Python module.

Required config:

- `model.adapter = "python_callable"`
- `model.path = "model/model.py"`
- `model.callable = "simulate"`
- `model.call_style = "kwargs"`, `"mapping"`, or `"positional"`

Supported callable signatures:

```python
def simulate(theta, sigma, ...):
    ...
    return output
```

```python
def simulate(params: dict[str, float]):
    ...
    return output
```

Return types:

- scalar
- list or tuple
- NumPy or JAX array
- dict of arrays keyed by output name

## Command adapter

Use when the model is written in another language or exposed as a binary.

Required config:

- `model.adapter = "command"`
- `model.command_template = "julia {model_path} {params_json} {output_json}"`

Available template fields:

- `{model_path}`
- `{params_json}`
- `{output_json}`
- `{parameter_name}` for each sampled parameter

Expected command behavior:

1. Read parameters from `params_json` or the expanded template placeholders.
2. Run the simulator.
3. Write JSON outputs to `output_json`, or print JSON to stdout.

Accepted JSON output shapes:

- numeric scalar
- list of numeric values
- list of lists
- object or dict mapping output names to numeric arrays

## Mathematical expression wrapper

When the user provides a mathematical expression instead of source code, `create-project` can generate a JAX-ready Python wrapper automatically with `--equation-text`.

Example:

```bash
python3 ~/.codex/skills/nuts-calibration/scripts/nuts_calibration.py create-project \
  --project-dir ./nuts-run \
  --equation-text "a * jnp.exp(-b * jnp.arange(50))" \
  --parameter a \
  --parameter b \
  --observed-path ./curve.json
```

## Custom likelihood hooks

`likelihood.name = "custom_python"` expects:

- `likelihood.custom_python_path`
- `likelihood.custom_callable`

Callable contract:

```python
def loglikelihood(observed, simulated, likelihood_spec, params) -> float:
    ...
```

`likelihood.name = "custom_command"` expects:

- `likelihood.custom_command_template`

Available template fields:

- `{observed_json}`
- `{simulated_json}`
- `{likelihood_json}`

The command should print a single numeric log-likelihood value to stdout.

