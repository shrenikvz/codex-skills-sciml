import json
import pathlib
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from abc_calibration_lib.config import clone_default_config
from abc_calibration_lib.inference import InferenceError, run_calibration
from abc_calibration_lib.project import ProjectError, create_project, inspect_model_inputs


MODEL_SOURCE = '''from __future__ import annotations
import numpy as np

def simulate(theta):
    return np.asarray([theta, theta + 1.0, theta + 2.0], dtype=float)
'''


class AbcCalibrationUnitTests(unittest.TestCase):
    def test_inspect_model_requests_bounds_before_proceeding(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps([2.0, 3.0, 4.0]), encoding="utf-8")
            info = inspect_model_inputs(str(model_path), str(observed_path))
        self.assertEqual(info["model_analysis"]["adapter"], "python_callable")
        self.assertEqual(info["model_analysis"]["parameter_names"], ["theta"])
        self.assertEqual(info["prior_report"]["priors"], {})
        self.assertEqual(info["prior_report"]["missing_bounds"], ["theta"])
        self.assertFalse(info["prior_report"]["ready"])
        self.assertTrue(any("theta" in question for question in info["pending_questions"]))

    def test_create_project_requires_explicit_bounds(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            project_dir = root / "project"
            observed_path = root / "observed.json"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path.write_text(json.dumps([2.0, 3.0, 4.0]), encoding="utf-8")

            with self.assertRaises(ProjectError) as exc:
                create_project(
                    project_dir=str(project_dir),
                    model_path=str(model_path),
                    observed_path=str(observed_path),
                )

        self.assertIn("Explicit prior bounds are required", str(exc.exception))

    def test_two_phase_abc_recovers_scalar_parameter(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps([2.0, 3.0, 4.0]), encoding="utf-8")

            cfg = clone_default_config()
            cfg["objective"]["observed_path"] = str(observed_path)
            cfg["model"].update(
                {
                    "adapter": "python_callable",
                    "path": str(model_path),
                    "callable": "simulate",
                    "call_style": "kwargs",
                    "parameter_names": ["theta"],
                }
            )
            cfg["priors"] = {"theta": {"dist": "uniform", "params": {"lower": 0.0, "upper": 4.0}}}
            cfg["summary_statistics"]["kind"] = "identity"
            cfg["distance"]["metric"] = "rmse"
            cfg["scaling"] = {"enabled": False, "mode": "none"}
            cfg["algorithm"]["two_phase"].update(
                {
                    "pilot_size": 80,
                    "accepted_samples": 30,
                    "main_budget": 400,
                    "epsilon_quantile": 0.1,
                    "batch_size": 16,
                }
            )
            cfg["compute"]["max_workers"] = 1
            cfg["posterior_predictive"]["enabled"] = True

            result = run_calibration(cfg, workdir=root)
            self.assertTrue(pathlib.Path(result["results_dir"]).exists())

        self.assertGreaterEqual(result["posterior_samples"], 20)
        theta_mean = result["posterior_summary"]["parameters"]["theta"]["mean"]
        self.assertGreater(theta_mean, 1.5)
        self.assertLess(theta_mean, 2.5)
        self.assertGreater(result["acceptance_rate"], 0.0)

    def test_likelihood_guard_blocks_when_available(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps([2.0, 3.0, 4.0]), encoding="utf-8")

            cfg = clone_default_config()
            cfg["objective"].update(
                {
                    "observed_path": str(observed_path),
                    "text": "deterministic model with Gaussian noise likelihood",
                }
            )
            cfg["model"].update(
                {
                    "adapter": "python_callable",
                    "path": str(model_path),
                    "callable": "simulate",
                    "call_style": "kwargs",
                    "parameter_names": ["theta"],
                }
            )
            cfg["priors"] = {"theta": {"dist": "uniform", "params": {"lower": 0.0, "upper": 4.0}}}
            cfg["algorithm"]["two_phase"].update(
                {
                    "pilot_size": 20,
                    "accepted_samples": 5,
                    "main_budget": 40,
                    "epsilon_quantile": 0.2,
                }
            )
            with self.assertRaises(InferenceError):
                run_calibration(cfg, workdir=root)

    def test_run_requires_bounded_priors(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps([2.0, 3.0, 4.0]), encoding="utf-8")

            cfg = clone_default_config()
            cfg["objective"]["observed_path"] = str(observed_path)
            cfg["model"].update(
                {
                    "adapter": "python_callable",
                    "path": str(model_path),
                    "callable": "simulate",
                    "call_style": "kwargs",
                    "parameter_names": ["theta"],
                }
            )
            cfg["priors"] = {"theta": {"dist": "normal", "params": {"mean": 2.0, "std": 0.5}}}
            cfg["algorithm"]["two_phase"].update(
                {
                    "pilot_size": 10,
                    "accepted_samples": 3,
                    "main_budget": 20,
                    "epsilon_quantile": 0.2,
                }
            )

            with self.assertRaises(InferenceError) as exc:
                run_calibration(cfg, workdir=root)

        self.assertIn("Explicit prior bounds are required", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
