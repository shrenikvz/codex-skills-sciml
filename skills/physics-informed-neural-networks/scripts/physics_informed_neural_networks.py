#!/usr/bin/env python3
"""Physics-informed neural networks skill CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physics_informed_nn_lib.analysis import (
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_FRAMEWORKS,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_PLOTS,
    SUPPORTED_SAMPLING,
    SUPPORTED_WEIGHTING,
    inspect_problem_inputs,
)
from physics_informed_nn_lib.config import clone_default_config, load_config
from physics_informed_nn_lib.environment import probe_environment
from physics_informed_nn_lib.inference import run_training
from physics_informed_nn_lib.project import create_project


def cmd_doctor(_args: argparse.Namespace) -> dict[str, object]:
    return probe_environment()


def cmd_list_capabilities(_args: argparse.Namespace) -> dict[str, object]:
    return {
        "frameworks": SUPPORTED_FRAMEWORKS,
        "architectures": SUPPORTED_ARCHITECTURES,
        "sampling_strategies": SUPPORTED_SAMPLING,
        "loss_weighting": SUPPORTED_WEIGHTING,
        "optimizers": SUPPORTED_OPTIMIZERS,
        "plots": SUPPORTED_PLOTS,
    }


def cmd_show_template(_args: argparse.Namespace) -> dict[str, object]:
    return clone_default_config()


def _inspect_args(args: argparse.Namespace) -> dict[str, object]:
    return inspect_problem_inputs(
        problem_path=args.problem_path,
        observed_path=args.observed_path,
        physics_text=args.physics_text,
        callable_name=args.callable,
        request_text=args.request_text,
        problem_type=args.problem_type,
        input_columns=args.input_column,
        output_columns=args.output_column,
        context_columns=args.context_column,
        architecture=args.architecture,
        framework=args.framework,
        sampling_strategy=args.sampling_strategy,
        loss_weighting=args.loss_weighting,
        optimizer=args.optimizer,
        plots=args.plot,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )


def cmd_create_project(args: argparse.Namespace) -> dict[str, object]:
    return create_project(
        project_dir=args.project_dir,
        problem_path=args.problem_path,
        observed_path=args.observed_path,
        physics_text=args.physics_text,
        callable_name=args.callable,
        request_text=args.request_text,
        problem_type=args.problem_type,
        input_columns=args.input_column,
        output_columns=args.output_column,
        context_columns=args.context_column,
        architecture=args.architecture,
        framework=args.framework,
        sampling_strategy=args.sampling_strategy,
        loss_weighting=args.loss_weighting,
        optimizer=args.optimizer,
        plots=args.plot,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        overwrite=args.overwrite,
    )


def cmd_run(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    if args.framework:
        cfg["model"]["framework"] = args.framework
    return run_training(cfg, workdir=config_path.parent)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Physics-informed neural networks workflow")
    sub = p.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("doctor", help="Check framework availability and devices")
    sub.add_parser("list-capabilities", help="List frameworks, architectures, samplers, and plots")
    sub.add_parser("show-template", help="Print default config template")

    for name, help_text in [("inspect-problem", "Inspect a physics problem and training recommendations"), ("inspect-model", "Alias for inspect-problem")]:
        inspect_problem = sub.add_parser(name, help=help_text)
        inspect_problem.add_argument("--problem-path", default=None)
        inspect_problem.add_argument("--observed-path", default=None)
        inspect_problem.add_argument("--physics-text", default=None)
        inspect_problem.add_argument("--callable", default=None)
        inspect_problem.add_argument("--request-text", default="")
        inspect_problem.add_argument("--problem-type", default="auto")
        inspect_problem.add_argument("--input-column", action="append", default=[])
        inspect_problem.add_argument("--output-column", action="append", default=[])
        inspect_problem.add_argument("--context-column", action="append", default=[])
        inspect_problem.add_argument("--architecture", default="auto")
        inspect_problem.add_argument("--framework", default="auto")
        inspect_problem.add_argument("--sampling-strategy", default="auto")
        inspect_problem.add_argument("--loss-weighting", default="auto")
        inspect_problem.add_argument("--optimizer", default="auto")
        inspect_problem.add_argument("--plot", action="append", default=[])
        inspect_problem.add_argument("--hidden-layers", type=int, default=None)
        inspect_problem.add_argument("--hidden-units", type=int, default=None)
        inspect_problem.add_argument("--epochs", type=int, default=None)
        inspect_problem.add_argument("--learning-rate", type=float, default=None)

    create = sub.add_parser("create-project", help="Create a runnable PINN project")
    create.add_argument("--project-dir", required=True)
    create.add_argument("--problem-path", default=None)
    create.add_argument("--observed-path", default=None)
    create.add_argument("--physics-text", default=None)
    create.add_argument("--callable", default=None)
    create.add_argument("--request-text", default="")
    create.add_argument("--problem-type", default="auto")
    create.add_argument("--input-column", action="append", default=[])
    create.add_argument("--output-column", action="append", default=[])
    create.add_argument("--context-column", action="append", default=[])
    create.add_argument("--architecture", default="auto")
    create.add_argument("--framework", default="auto")
    create.add_argument("--sampling-strategy", default="auto")
    create.add_argument("--loss-weighting", default="auto")
    create.add_argument("--optimizer", default="auto")
    create.add_argument("--plot", action="append", default=[])
    create.add_argument("--hidden-layers", type=int, default=None)
    create.add_argument("--hidden-units", type=int, default=None)
    create.add_argument("--epochs", type=int, default=None)
    create.add_argument("--learning-rate", type=float, default=None)
    create.add_argument("--overwrite", action="store_true")

    run = sub.add_parser("run", help="Run PINN training from a config.json")
    run.add_argument("--config", required=True)
    run.add_argument("--framework", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.subcommand == "doctor":
            result = cmd_doctor(args)
        elif args.subcommand == "list-capabilities":
            result = cmd_list_capabilities(args)
        elif args.subcommand == "show-template":
            result = cmd_show_template(args)
        elif args.subcommand in {"inspect-problem", "inspect-model"}:
            result = _inspect_args(args)
        elif args.subcommand == "create-project":
            result = cmd_create_project(args)
        elif args.subcommand == "run":
            result = cmd_run(args)
        else:
            raise RuntimeError(f"Unsupported subcommand: {args.subcommand}")
        print(json.dumps({"ok": True, "result": result}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
