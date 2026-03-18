import json
import pathlib
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from nuts_calibration_lib.analysis import inspect_inputs
from nuts_calibration_lib.environment import peek_environment
from nuts_calibration_lib.priors import recommend_prior
from nuts_calibration_lib.project import create_project
from nuts_calibration_lib.transforms import build_transform_specs


MODEL_SOURCE = '''from __future__ import annotations
import jax.numpy as jnp

def simulate(theta, sigma=0.1):
    xs = jnp.arange(3.0)
    return theta + sigma * xs
'''


class NutsCalibrationUnitTests(unittest.TestCase):
    def test_recommend_prior_probability_name(self):
        spec = recommend_prior("mixing_probability")
        self.assertEqual(spec["dist"], "beta")

    def test_transform_recommendation_for_positive_scale(self):
        specs = build_transform_specs(
            ["sigma"],
            {"sigma": {"dist": "halfnormal", "params": {"scale": 1.0, "lower": 0.0}}},
        )
        self.assertEqual(specs["sigma"]["kind"], "softplus")

    def test_inspection_detects_jax_compatible_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            observed_path = root / "observed.json"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path.write_text(json.dumps([1.0, 1.1, 1.2]), encoding="utf-8")
            result = inspect_inputs(str(model_path), str(observed_path))
        self.assertEqual(result["model_analysis"]["parameter_names"], ["theta", "sigma"])
        self.assertEqual(result["differentiability_assessment"]["gradient_strategy"], "jax_autodiff")
        self.assertIn("theta", result["prior_report"]["priors"])

    def test_create_project_writes_config_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            observed_path = root / "observed.json"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path.write_text(json.dumps([1.0, 1.1, 1.2]), encoding="utf-8")
            project_dir = root / "nuts-run"
            summary = create_project(
                project_dir=str(project_dir),
                model_path=str(model_path),
                observed_path=str(observed_path),
                plots=["trace"],
            )
            self.assertTrue((project_dir / "config.json").exists())
            self.assertTrue((project_dir / "project_summary.json").exists())
            self.assertEqual(summary["backend"], "blackjax")

    def test_peek_environment_reports_expected_keys(self):
        report = peek_environment()
        self.assertIn("jax", report)
        self.assertIn("blackjax", report)
        self.assertIn("python_executable", report)


if __name__ == "__main__":
    unittest.main()
