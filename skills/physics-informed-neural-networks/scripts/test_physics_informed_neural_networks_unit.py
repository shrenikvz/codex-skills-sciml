import json
import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from physics_informed_nn_lib.analysis import inspect_problem_inputs
from physics_informed_nn_lib.environment import peek_environment
from physics_informed_nn_lib.problem_spec import load_problem_spec
from physics_informed_nn_lib.project import create_project


PROBLEM_JSON = {
    "description": "Inverse ODE test problem",
    "independent_variables": ["x"],
    "dependent_variables": ["u"],
    "domains": {"x": {"min": 0.0, "max": 1.0}},
    "equations": [{"name": "ode", "expression": "u__x + k * u"}],
    "boundary_conditions": [{"location": {"x": "min"}, "type": "dirichlet", "field": "u", "value": "1.0"}],
    "parameters": {"k": {"value": 0.5, "trainable": True, "bounds": [0.0, 2.0]}},
}


class PhysicsInformedNeuralNetworksUnitTests(unittest.TestCase):
    def test_load_problem_spec_from_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            problem_path = root / "problem.json"
            problem_path.write_text(json.dumps(PROBLEM_JSON), encoding="utf-8")
            spec = load_problem_spec(str(problem_path))
        self.assertEqual(spec["independent_variables"], ["x"])
        self.assertIn("k", spec["unknown_parameters"])

    def test_inspect_problem_detects_inverse_problem_and_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            problem_path = root / "problem.json"
            problem_path.write_text(json.dumps(PROBLEM_JSON), encoding="utf-8")
            observed_path = root / "observed.csv"
            observed_path.write_text("x,u\n0.0,1.0\n0.5,0.8\n1.0,0.6\n", encoding="utf-8")
            info = inspect_problem_inputs(str(problem_path), str(observed_path))
        self.assertEqual(info["problem_type"]["problem_type"], "inverse")
        self.assertEqual(info["observation_mapping"]["input_columns"], ["x"])
        self.assertEqual(info["observation_mapping"]["output_columns"], ["u"])

    def test_create_project_writes_config_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            problem_path = root / "problem.json"
            problem_path.write_text(json.dumps(PROBLEM_JSON), encoding="utf-8")
            observed_path = root / "observed.csv"
            observed_path.write_text("x,u\n0.0,1.0\n0.5,0.8\n1.0,0.6\n", encoding="utf-8")
            project_dir = root / "pinn-run"
            summary = create_project(
                project_dir=str(project_dir),
                problem_path=str(problem_path),
                observed_path=str(observed_path),
                plots=["loss_curves"],
                loss_weighting="fixed",
            )
            self.assertTrue((project_dir / "config.json").exists())
            self.assertTrue((project_dir / "project_summary.json").exists())
            self.assertEqual(summary["problem_type"]["problem_type"], "inverse")

    def test_doctor_reports_framework_keys(self):
        report = peek_environment()
        self.assertIn("torch", report)
        self.assertIn("jax", report)
        self.assertIn("tensorflow", report)


if __name__ == "__main__":
    unittest.main()
