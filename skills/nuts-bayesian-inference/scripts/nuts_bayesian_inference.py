#!/usr/bin/env python3
"""NUTS Bayesian inference skill CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nuts_inference_lib.config import clone_default_config, load_config
from nuts_inference_lib.environment import probe_environment
from nuts_inference_lib.inference import run_inference
from nuts_inference_lib.project import create_project, inspect_model_inputs


SUPPORTED_PLOTS = [
    "posterior_marginals",
    "pairwise",
    "trace",
    "autocorrelation",
    "energy",
    "posterior_predictive",
]


def cmd_doctor(_args: argparse.Namespace) -> dict[str, object]:
    return probe_environment()


def cmd_create_project(args: argparse.Namespace) -> dict[str, object]:
    return create_project(
        project_dir=args.project_dir,
        model_path=args.model_path,
        observed_path=args.observed_path,
        command_template=args.command_template,
        equation_text=args.equation_text,
        callable_name=args.callable,
        request_text=args.request_text,
        observed_output_names=args.observed_output,
        parameter_names=args.parameter,
        prior_file=args.prior_file,
        prior_overrides=args.prior,
        parameter_bounds=args.parameter_bound,
        likelihood_family=args.likelihood_family,
        scaling_mode=args.scaling,
        gradient_strategy=args.gradient_strategy,
        plots=args.plot,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_acceptance=args.target_acceptance,
        max_tree_depth=args.max_tree_depth,
        mass_matrix=args.mass_matrix,
        overwrite=args.overwrite,
    )


def cmd_inspect_model(args: argparse.Namespace) -> dict[str, object]:
    return inspect_model_inputs(
        model_path=args.model_path,
        observed_path=args.observed_path,
        command_template=args.command_template,
        equation_text=args.equation_text,
        callable_name=args.callable,
        request_text=args.request_text,
        observed_output_names=args.observed_output,
        user_parameter_names=args.parameter,
        prior_file=args.prior_file,
        prior_overrides=args.prior,
        parameter_bounds=args.parameter_bound,
        likelihood_family=args.likelihood_family,
        scaling_mode=args.scaling,
        gradient_strategy=args.gradient_strategy,
        plots=args.plot,
    )


def cmd_run(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    if args.backend:
        cfg["sampler"]["backend"] = args.backend
    if args.device:
        cfg["compute"]["device_preference"] = args.device
    return run_inference(cfg, workdir=config_path.parent)


def cmd_show_template(_args: argparse.Namespace) -> dict[str, object]:
    return clone_default_config()


def cmd_list_plots(_args: argparse.Namespace) -> dict[str, object]:
    return {"plots": SUPPORTED_PLOTS}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NUTS Bayesian inference workflow")
    sub = p.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("doctor", help="Check JAX/BlackJAX availability and devices")
    sub.add_parser("list-plots", help="List supported visualization names")
    sub.add_parser("show-template", help="Print default config template")

    inspect_model = sub.add_parser("inspect-model", help="Inspect model, data, priors, and likelihood")
    inspect_model.add_argument("--model-path", default=None)
    inspect_model.add_argument("--observed-path", required=True)
    inspect_model.add_argument("--command-template", default=None)
    inspect_model.add_argument("--equation-text", default=None)
    inspect_model.add_argument("--callable", default=None)
    inspect_model.add_argument("--request-text", default="")
    inspect_model.add_argument("--observed-output", action="append", default=[])
    inspect_model.add_argument("--parameter", action="append", default=[])
    inspect_model.add_argument("--prior-file", default=None)
    inspect_model.add_argument("--prior", action="append", default=[])
    inspect_model.add_argument("--parameter-bound", action="append", default=[])
    inspect_model.add_argument("--likelihood-family", default="auto")
    inspect_model.add_argument("--scaling", default="auto")
    inspect_model.add_argument("--gradient-strategy", default="auto")
    inspect_model.add_argument("--plot", action="append", default=[])

    create = sub.add_parser("create-project", help="Create a runnable NUTS project")
    create.add_argument("--project-dir", required=True)
    create.add_argument("--model-path", default=None)
    create.add_argument("--observed-path", required=True)
    create.add_argument("--command-template", default=None)
    create.add_argument("--equation-text", default=None)
    create.add_argument("--callable", default=None)
    create.add_argument("--request-text", default="")
    create.add_argument("--observed-output", action="append", default=[])
    create.add_argument("--parameter", action="append", default=[])
    create.add_argument("--prior-file", default=None)
    create.add_argument("--prior", action="append", default=[])
    create.add_argument("--parameter-bound", action="append", default=[])
    create.add_argument("--likelihood-family", default="auto")
    create.add_argument("--scaling", default="auto")
    create.add_argument("--gradient-strategy", default="auto")
    create.add_argument("--plot", action="append", default=[])
    create.add_argument("--num-warmup", type=int, default=None)
    create.add_argument("--num-samples", type=int, default=None)
    create.add_argument("--num-chains", type=int, default=None)
    create.add_argument("--target-acceptance", type=float, default=None)
    create.add_argument("--max-tree-depth", type=int, default=None)
    create.add_argument("--mass-matrix", default=None)
    create.add_argument("--overwrite", action="store_true")

    run = sub.add_parser("run", help="Run NUTS inference from a config.json")
    run.add_argument("--config", required=True)
    run.add_argument("--backend", default=None)
    run.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.subcommand == "doctor":
            result = cmd_doctor(args)
        elif args.subcommand == "inspect-model":
            result = cmd_inspect_model(args)
        elif args.subcommand == "create-project":
            result = cmd_create_project(args)
        elif args.subcommand == "run":
            result = cmd_run(args)
        elif args.subcommand == "show-template":
            result = cmd_show_template(args)
        elif args.subcommand == "list-plots":
            result = cmd_list_plots(args)
        else:
            raise RuntimeError(f"Unsupported subcommand: {args.subcommand}")
        print(json.dumps({"ok": True, "result": result}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
