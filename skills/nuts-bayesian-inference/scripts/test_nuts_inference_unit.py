import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from nuts_inference_lib.analysis import inspect_model_inputs
from nuts_inference_lib.environment import probe_environment
from nuts_inference_lib.likelihoods import infer_likelihood_family, maybe_add_likelihood_priors
from nuts_inference_lib.priors import build_prior_report, recommend_prior
from nuts_inference_lib.project import create_project
from nuts_inference_lib.transforms import build_transform_specs, forward_transform, inverse_transform


MODEL_SOURCE = '''from __future__ import annotations
import numpy as np

def simulate(theta, offset=1.0):
    return {"signal": np.asarray([theta + offset, theta + offset + 1.0], dtype=float)}
'''


JAX_MODEL_SOURCE = '''from __future__ import annotations
import jax.numpy as jnp

def simulate(theta):
    return jnp.asarray([theta, theta + 1.0], dtype=jnp.float32)
'''


class NutsInferenceUnitTests(unittest.TestCase):
    def test_recommend_prior_probability_name(self):
        spec = recommend_prior("mixing_probability")
        self.assertEqual(spec["dist"], "beta")

    def test_build_prior_report_adds_names(self):
        report = build_prior_report(["theta", "sigma"], {"theta": 2.0}, {})
        self.assertIn("theta", report["priors"])
        self.assertIn("sigma", report["priors"])

    def test_transform_round_trip_positive_parameter(self):
        specs = build_transform_specs({"sigma": {"dist": "halfnormal", "params": {"scale": 1.0}}})
        unconstrained = inverse_transform(0.7, specs["sigma"])
        recon = forward_transform(unconstrained, specs["sigma"])
        self.assertAlmostEqual(float(recon), 0.7, places=6)

    def test_inspect_model_infers_python_signature_and_output_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps({"signal": [2.1, 3.2]}), encoding="utf-8")
            info = inspect_model_inputs(str(model_path), str(observed_path))
        self.assertEqual(info["model_analysis"]["adapter"], "python_callable")
        self.assertEqual(info["model_analysis"]["parameter_names"], ["theta", "offset"])
        self.assertEqual(info["output_mapping"]["selected"], ["signal"])
        self.assertIn("obs_sigma", info["prior_report"]["priors"])

    def test_likelihood_inference_for_counts(self):
        observed = np.asarray([0, 4, 6, 2], dtype=float)
        report = infer_likelihood_family(observed)
        self.assertIn(report["family"], {"poisson", "negative_binomial"})

    def test_maybe_add_likelihood_priors_for_gaussian(self):
        priors, likelihood = maybe_add_likelihood_priors({}, {"family": "gaussian", "params": {}}, np.asarray([1.0, 2.0, 3.0]))
        self.assertIn("obs_sigma", priors)
        self.assertEqual(likelihood["params"]["sigma_parameter"], "obs_sigma")

    def test_create_project_writes_config_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps({"signal": [2.0, 3.0]}), encoding="utf-8")
            project_dir = root / "nuts-run"
            summary = create_project(
                project_dir=str(project_dir),
                model_path=str(model_path),
                observed_path=str(observed_path),
                plots=["trace"],
            )
            self.assertTrue((project_dir / "config.json").exists())
            self.assertTrue((project_dir / "project_summary.json").exists())
            self.assertEqual(summary["model_analysis"]["callable"], "simulate")

    def test_doctor_reports_status(self):
        report = probe_environment()
        self.assertIn("jax", report)
        self.assertIn("blackjax", report)

    def test_direct_jax_model_is_recognized(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(JAX_MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps([1.0, 2.0]), encoding="utf-8")
            info = inspect_model_inputs(str(model_path), str(observed_path))
        self.assertEqual(info["gradient_recommendation"]["strategy"], "jax")


if __name__ == "__main__":
    unittest.main()
