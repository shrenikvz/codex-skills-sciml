# abc-calibration quickstart

## Assumptions

- Python 3.10+ with `numpy` is available.
- `matplotlib` is optional and only required for requested plots.
- Observed data is numeric and stored in CSV, TSV, JSON, NPY, or NPZ.
- Models can be exposed as:
  - a Python function,
  - a command-line simulator that reads parameters and writes JSON outputs, or
  - a mathematical expression wrapped into a generated Python model.

## Install skill from this repo

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo gg2uah/codex-skills \
  --path skills/abc-calibration
```

Restart Codex after install.

## Inspect a model before creating a project

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py inspect-model \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Calibrate this stochastic simulator with ABC rejection"
```

Inspection returns:

- inferred parameter names and defaults
- prior recommendations
- likelihood assessment
- scaling, summary statistic, and distance recommendations
- pending clarification questions when outputs are ambiguous

## Create a runnable project

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py create-project \
  --project-dir ./abc-run \
  --model-path ./model.py \
  --observed-path ./observed.json \
  --request-text "Estimate the unknown simulator parameters using likelihood-free calibration" \
  --pilot-size 1000 \
  --main-budget 20000 \
  --accepted-samples 500 \
  --epsilon-quantile 0.05 \
  --plot posterior_marginals \
  --plot posterior_predictive
```

Generated project files include:

- `config.json`
- `run.py`
- `project_summary.json`
- `AGENT_RUNBOOK.md`
- `skill_runtime/abc_calibration_lib/` for portable execution

## Run the calibration

```bash
cd abc-run
python3 run.py
```

Or run directly through the CLI:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py run \
  --config ./abc-run/config.json
```

If the tool detects that a tractable likelihood probably exists, it stops by default and recommends likelihood-based inference. Override only when you explicitly want ABC:

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py run \
  --config ./abc-run/config.json \
  --force-abc
```

## Mathematical model example

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py create-project \
  --project-dir ./abc-equation \
  --equation-text "a * np.exp(-b * np.arange(25))" \
  --parameter a \
  --parameter b \
  --parameter-bound a=0:10 \
  --parameter-bound b=0:2 \
  --observed-path ./decay_curve.json
```

## Non-Python simulator example

```bash
python3 ~/.codex/skills/abc-calibration/scripts/abc_calibration.py create-project \
  --project-dir ./abc-julia \
  --model-path ./simulate.jl \
  --observed-path ./observed.csv
```

The generated project creates `model/run_model.sh` and assumes the simulator accepts:

```text
simulate.jl <params_json> <output_json>
```

Adjust `model/run_model.sh` if the simulator uses a different interface.

## Output artifacts

- `results/posterior_samples.csv`
- `results/posterior_samples.jsonl`
- `results/posterior_summary.json`
- `results/pilot_phase.json`
- `results/sampling_diagnostics.json`
- `results/posterior_predictive_summary.json`
- `results/run_summary.json`
- `results/artifact_index.json`
- `results/figures/*.png` for requested plots

## Tips

- Use `rmse` or `nrmse` for vector or time-series outputs.
- Use `wasserstein` or `ks` for distributional outputs.
- Use `mahalanobis` when correlated summary statistics matter.
- When outputs are high-dimensional, prefer `timeseries` or `moments` summaries over raw identity summaries.
- Increase `pilot_size` for higher-dimensional parameter spaces so `epsilon` is estimated from a stable pilot distribution.
