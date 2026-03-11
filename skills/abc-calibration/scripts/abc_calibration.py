#!/usr/bin/env python3
"""ABC calibration skill CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abc_calibration_lib.config import clone_default_config, load_config
from abc_calibration_lib.inference import run_calibration
from abc_calibration_lib.project import create_project, inspect_model_inputs


class CliError(RuntimeError):
    """Command failure."""


SUPPORTED_DISTANCE_METRICS = [
    "rmse",
    "nrmse",
    "ks",
    "euclidean",
    "mahalanobis",
    "wasserstein",
    "custom",
]

SUPPORTED_PLOTS = [
    "posterior_marginals",
    "pairwise",
    "posterior_predictive",
    "trace",
    "calibration_diagnostics",
]



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
        distance_metric=args.distance_metric,
        scaling_mode=args.scaling,
        summary_kind=args.summary_statistics,
        likelihood_hint=args.likelihood_hint,
        plots=args.plot,
        pilot_size=args.pilot_size,
        main_budget=args.main_budget,
        accepted_samples=args.accepted_samples,
        epsilon_quantile=args.epsilon_quantile,
        max_workers=args.max_workers,
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
        distance_metric=args.distance_metric,
        scaling_mode=args.scaling,
        summary_kind=args.summary_statistics,
        likelihood_hint=args.likelihood_hint,
        plots=args.plot,
    )



def cmd_run(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    if args.force_abc:
        cfg["algorithm"]["two_phase"]["proceed_if_likelihood_available"] = True
    return run_calibration(cfg, workdir=config_path.parent)



def cmd_list_distance_metrics(_args: argparse.Namespace) -> dict[str, object]:
    return {
        "distance_metrics": SUPPORTED_DISTANCE_METRICS,
        "plots": SUPPORTED_PLOTS,
    }



def cmd_show_template(_args: argparse.Namespace) -> dict[str, object]:
    return clone_default_config()



def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ABC calibration workflow")
    sub = p.add_subparsers(dest="subcommand", required=True)

    create = sub.add_parser("create-project", help="Create a runnable ABC calibration project")
    create.add_argument("--project-dir", required=True)
    create.add_argument("--model-path", default=None, help="Path to the model or simulator source file")
    create.add_argument("--observed-path", required=True, help="Path to observed calibration data")
    create.add_argument("--command-template", default=None, help="Explicit command adapter template")
    create.add_argument("--equation-text", default=None, help="Mathematical expression to wrap as a Python model")
    create.add_argument("--callable", default=None, help="Python callable name when model_path is a Python file")
    create.add_argument("--request-text", default="", help="Natural-language problem description")
    create.add_argument("--observed-output", action="append", default=[], help="Observed output name to calibrate against")
    create.add_argument("--parameter", action="append", default=[], help="Parameter name when inference is ambiguous")
    create.add_argument("--prior-file", default=None, help="JSON file with explicit prior specs")
    create.add_argument("--prior", action="append", default=[], help="Prior override like theta=uniform(0,1)")
    create.add_argument("--parameter-bound", action="append", default=[], help="Parameter bound like theta=0:1")
    create.add_argument("--distance-metric", default="auto", help="Distance metric: auto, rmse, nrmse, ks, euclidean, mahalanobis, wasserstein, custom")
    create.add_argument("--scaling", default="auto", help="Scaling: auto, none, zscore, minmax, variance")
    create.add_argument("--summary-statistics", default="auto", help="Summary statistics: auto, identity, moments, quantiles, timeseries")
    create.add_argument("--likelihood-hint", default="auto", help="Likelihood hint: auto, available, intractable, unavailable")
    create.add_argument("--plot", action="append", default=[], help="Requested plot type")
    create.add_argument("--pilot-size", type=int, default=None)
    create.add_argument("--main-budget", type=int, default=None)
    create.add_argument("--accepted-samples", type=int, default=None)
    create.add_argument("--epsilon-quantile", type=float, default=None)
    create.add_argument("--max-workers", default=None)
    create.add_argument("--overwrite", action="store_true")

    inspect_model = sub.add_parser("inspect-model", help="Inspect model, data, and ABC recommendations")
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
    inspect_model.add_argument("--distance-metric", default="auto")
    inspect_model.add_argument("--scaling", default="auto")
    inspect_model.add_argument("--summary-statistics", default="auto")
    inspect_model.add_argument("--likelihood-hint", default="auto")
    inspect_model.add_argument("--plot", action="append", default=[])

    run = sub.add_parser("run", help="Run ABC calibration from a config.json")
    run.add_argument("--config", required=True)
    run.add_argument("--force-abc", action="store_true", help="Proceed even if the likelihood appears tractable")

    sub.add_parser("list-distance-metrics", help="List supported metrics and plots")
    sub.add_parser("show-template", help="Print default config template")
    return p



def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.subcommand == "create-project":
            result = cmd_create_project(args)
        elif args.subcommand == "inspect-model":
            result = cmd_inspect_model(args)
        elif args.subcommand == "run":
            result = cmd_run(args)
        elif args.subcommand == "list-distance-metrics":
            result = cmd_list_distance_metrics(args)
        elif args.subcommand == "show-template":
            result = cmd_show_template(args)
        else:
            raise CliError(f"Unsupported subcommand: {args.subcommand}")
        print(json.dumps({"ok": True, "result": result}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
