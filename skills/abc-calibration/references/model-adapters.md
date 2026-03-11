# model adapter contracts

## Python callable adapter

Use when the simulator lives in a Python module.

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
- list / tuple / NumPy array
- dict of numeric arrays keyed by output name

## Command adapter

Use when the simulator is written in another language or is exposed as a binary.

Required config:

- `model.adapter = "command"`
- `model.command_template = "julia {model_path} {params_json} {output_json}"`

Available template fields:

- `{model_path}`
- `{params_json}`
- `{output_json}`
- `{parameter_name}` for each sampled parameter

Expected command behavior:

1. Read parameters from `params_json` or command-line placeholders.
2. Run the simulator.
3. Write JSON outputs to `output_json` or print JSON to stdout.

Accepted JSON output shapes:

- numeric scalar
- list of numeric values
- list of lists
- object/dict mapping output names to numeric arrays

## Mathematical expression wrapper

When the user provides a mathematical model instead of source code, `create-project` can generate a Python wrapper automatically with `--equation-text`.

Example:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py create-project \
  --project-dir ./abc-run \
  --equation-text "a * np.exp(-b * np.arange(50))" \
  --parameter a \
  --parameter b \
  --observed-path ./curve.json
```

## Custom summary and custom distance hooks

`summary_statistics.kind = "python_callable"` expects:

- `summary_statistics.path`
- `summary_statistics.callable`

Callable contract:

```python
def summarize(output_array) -> list[float] | np.ndarray:
    ...
```

`distance.metric = "custom"` can use either:

- Python callable with `distance.custom_python_path` + `distance.custom_callable`
- command adapter with `distance.custom_command_template`

Callable contract:

```python
def distance(observed_summary, simulated_summary) -> float:
    ...
```
