import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CLI_PATH = SCRIPT_DIR / "nuts_calibration.py"

MODEL_SOURCE = '''from __future__ import annotations
import jax.numpy as jnp

def simulate(theta):
    return jnp.asarray([theta, theta + 1.0, theta + 2.0], dtype=float)
'''


class NutsCalibrationCliUnitTests(unittest.TestCase):
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
        self.assertIn("model", result)
        self.assertIn("likelihood", result)

    def test_list_capabilities(self):
        result = self._run("list-capabilities")
        self.assertIn("blackjax", result["backends"])
        self.assertIn("posterior_predictive", result["plots"])

    def test_doctor(self):
        result = self._run("doctor")
        self.assertIn("jax", result)
        self.assertIn("blackjax", result)

    def test_inspect_and_create_project(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            observed_path = root / "observed.json"
            project_dir = root / "nuts-project"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path.write_text(json.dumps([1.0, 2.0, 3.0]), encoding="utf-8")
            inspect_result = self._run(
                "inspect-model",
                "--model-path",
                str(model_path),
                "--observed-path",
                str(observed_path),
            )
            self.assertEqual(inspect_result["model_analysis"]["parameter_names"], ["theta"])
            create_result = self._run(
                "create-project",
                "--project-dir",
                str(project_dir),
                "--model-path",
                str(model_path),
                "--observed-path",
                str(observed_path),
                "--plot",
                "trace",
            )
            self.assertTrue((project_dir / "config.json").exists())
            self.assertTrue((project_dir / "skill_runtime" / "nuts_calibration_lib").exists())
            self.assertEqual(create_result["backend"], "blackjax")


if __name__ == "__main__":
    unittest.main()

