# Quickstart

## 1. Verify the environment

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py doctor
```

This reports framework availability, accelerator visibility, and the default framework recommendation.

## 2. Inspect the problem

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py inspect-problem \
  --problem-path ./problem.json \
  --observed-path ./observations.csv \
  --request-text "Train a PINN for this inverse diffusion problem"
```

The inspection result includes:

- inferred variables and domains
- detected problem type
- pending clarification questions
- recommended architecture
- recommended sampling strategy
- recommended stabilization
- default training hyperparameters

## 3. Create a portable project

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py create-project \
  --project-dir ./pinn-run \
  --problem-path ./problem.json \
  --observed-path ./observations.csv \
  --architecture fourier \
  --sampling-strategy sobol \
  --loss-weighting gradient_norm \
  --optimizer hybrid \
  --plot solution_field \
  --plot loss_curves
```

Inspect:

- `pinn-run/project_summary.json`
- `pinn-run/config.json`
- `pinn-run/AGENT_RUNBOOK.md`

## 4. Run training

```bash
cd pinn-run
python3 run.py
```

Main outputs:

- `results/diagnostics.json`
- `results/evaluation_summary.json`
- `results/solution_predictions.json`
- `results/inferred_parameters.json`
- `results/figures/`

## 5. Text-first problem specs

If you do not yet have a structured JSON or Python problem file, start with:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py inspect-problem \
  --physics-text "u_t - alpha u_xx = 0 on x in [0,1], t in [0,1], u(x,0)=sin(pi x), u(0,t)=0, u(1,t)=0" \
  --request-text "Forward solve with a PINN"
```

If the specification is still ambiguous, the inspection report returns `pending_questions` instead of pretending the problem is fully defined.

## 6. Operator-learning-style setups

Provide conditioning or operator context columns in the observation file and choose `transformer_operator` explicitly when needed:

```bash
python3 ~/.codex/skills/physics-informed-neural-networks/scripts/physics_informed_neural_networks.py create-project \
  --project-dir ./operator-run \
  --problem-path ./operator_problem.json \
  --observed-path ./operator_data.csv \
  --architecture transformer_operator
```
