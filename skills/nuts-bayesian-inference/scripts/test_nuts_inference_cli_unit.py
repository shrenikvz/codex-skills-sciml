import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CLI_PATH = SCRIPT_DIR / "nuts_bayesian_inference.py"


MODEL_SOURCE = '''from __future__ import annotations
import numpy as np

def simulate(theta):
    return np.asarray([theta, theta + 1.0], dtype=float)
'''


class NutsInferenceCliUnitTests(unittest.TestCase):
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
        self.assertEqual(result["sampler"]["backend"], "blackjax")

    def test_list_plots(self):
        result = self._run("list-plots")
        self.assertIn("trace", result["plots"])

    def test_inspect_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path = root / "observed.json"
            observed_path.write_text(json.dumps([1.0, 2.0]), encoding="utf-8")
            result = self._run(
                "inspect-model",
                "--model-path",
                str(model_path),
                "--observed-path",
                str(observed_path),
            )
        self.assertEqual(result["model_analysis"]["callable"], "simulate")
        self.assertIn("prior_report", result)


if __name__ == "__main__":
    unittest.main()
