import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CLI_PATH = SCRIPT_DIR / "physics_informed_neural_networks.py"


PROBLEM_JSON = {
    "description": "Forward ODE test problem",
    "independent_variables": ["x"],
    "dependent_variables": ["u"],
    "domains": {"x": {"min": 0.0, "max": 1.0}},
    "equations": [{"name": "ode", "expression": "u__x + u"}],
    "boundary_conditions": [{"location": {"x": "min"}, "type": "dirichlet", "field": "u", "value": "1.0"}],
}


class PhysicsInformedNeuralNetworksCliUnitTests(unittest.TestCase):
    def _run(self, *args):
        completed = subprocess.run(
            [sys.executable, str(CLI_PATH), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"], msg=completed.stdout)
        return payload["result"]

    def test_show_template(self):
        result = self._run("show-template")
        self.assertIn("problem", result)
        self.assertIn("model", result)

    def test_list_capabilities(self):
        result = self._run("list-capabilities")
        self.assertIn("fourier", result["architectures"])
        self.assertIn("loss_curves", result["plots"])

    def test_inspect_problem(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            problem_path = root / "problem.json"
            problem_path.write_text(json.dumps(PROBLEM_JSON), encoding="utf-8")
            result = self._run("inspect-problem", "--problem-path", str(problem_path))
        self.assertEqual(result["problem_type"]["problem_type"], "forward")
        self.assertIn("architecture_recommendation", result)


if __name__ == "__main__":
    unittest.main()
