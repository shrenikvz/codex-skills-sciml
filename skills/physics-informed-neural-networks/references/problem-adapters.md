# Problem Adapters

## Structured JSON problem specification

Preferred contract:

```json
{
  "description": "1D Burgers equation with unknown viscosity",
  "independent_variables": ["x", "t"],
  "dependent_variables": ["u"],
  "domains": {
    "x": {"min": -1.0, "max": 1.0},
    "t": {"min": 0.0, "max": 1.0}
  },
  "equations": [
    {
      "name": "burgers",
      "expression": "u__t + u * u__x - nu * u__xx"
    }
  ],
  "parameters": {
    "nu": {"value": 0.01, "trainable": false, "bounds": [0.0001, 1.0]}
  },
  "unknown_parameters": [],
  "boundary_conditions": [
    {"location": {"x": "min"}, "type": "dirichlet", "field": "u", "value": "0.0"},
    {"location": {"x": "max"}, "type": "dirichlet", "field": "u", "value": "0.0"}
  ],
  "initial_conditions": [
    {"location": {"t": "min"}, "type": "dirichlet", "field": "u", "value": "-sin(pi * x)"}
  ],
  "algebraic_constraints": [],
  "constitutive_relations": [],
  "analytical_solution": null
}
```

Use derivative tokens in residual expressions:

- `u__x`
- `u__xx`
- `u__t`
- `u__xt`

This avoids needing symbolic PDE parsers while still supporting autodiff-based residual construction.

## Python problem module

Supported forms:

```python
problem_spec = {
    "independent_variables": ["x"],
    "dependent_variables": ["u"],
    "domains": {"x": {"min": 0.0, "max": 1.0}},
    "equations": [{"expression": "u__xx + omega**2 * u"}],
    "boundary_conditions": [
        {"location": {"x": "min"}, "type": "dirichlet", "field": "u", "value": "0.0"},
        {"location": {"x": "max"}, "type": "dirichlet", "field": "u", "value": "1.0"}
    ],
    "parameters": {"omega": {"value": 2.0}}
}
```

Or:

```python
def build_problem():
    return {...}
```

The returned object must be JSON-like. Python modules are the best route when the user wants to compute parts of the spec programmatically before training.

## Observational data

Supported formats:

- `csv`
- `tsv`
- `json`
- `npy`
- `npz`

For tabular data, columns should include independent-variable names and dependent-variable names when possible.

Example CSV:

```text
x,t,u
0.0,0.0,0.0
0.1,0.0,0.309
```

If the columns are ambiguous, the inspection report returns a clarification question.

## Text and Markdown specs

Text specs are useful for early inspection and scaffolding, but they are intentionally conservative.

Supported cues:

- `independent variables: x, t`
- `dependent variables: u`
- `x in [0,1]`
- `t in [0,1]`
- `equation: u__t - alpha * u__xx`
- `boundary: x=min -> u=0`
- `initial: t=min -> u=sin(pi*x)`

If the text omits key structural details, do not proceed to training until the missing pieces are resolved.
