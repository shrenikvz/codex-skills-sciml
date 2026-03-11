import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

MODEL_SOURCE = '''from __future__ import annotations
import numpy as np

def simulate(theta):
    return np.asarray([theta, theta + 1.0, theta + 2.0], dtype=float)
'''


class AbcCalibrationCliUnitTests(unittest.TestCase):
    def test_cli_create_project_and_run(self):
        script = SCRIPT_DIR / "abc_calibration.py"
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model_path = root / "model.py"
            observed_path = root / "observed.json"
            project_dir = root / "abc-project"
            model_path.write_text(MODEL_SOURCE, encoding="utf-8")
            observed_path.write_text(json.dumps([2.0, 3.0, 4.0]), encoding="utf-8")

            inspect_proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "inspect-model",
                    "--model-path",
                    str(model_path),
                    "--observed-path",
                    str(observed_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(inspect_proc.returncode, 0, msg=inspect_proc.stdout + "\n" + inspect_proc.stderr)
            inspect_payload = json.loads(inspect_proc.stdout)
            self.assertTrue(inspect_payload["ok"])
            self.assertEqual(inspect_payload["result"]["model_analysis"]["parameter_names"], ["theta"])

            create_proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "create-project",
                    "--project-dir",
                    str(project_dir),
                    "--model-path",
                    str(model_path),
                    "--observed-path",
                    str(observed_path),
                    "--pilot-size",
                    "60",
                    "--main-budget",
                    "240",
                    "--accepted-samples",
                    "20",
                    "--epsilon-quantile",
                    "0.1",
                    "--plot",
                    "posterior_marginals",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(create_proc.returncode, 0, msg=create_proc.stdout + "\n" + create_proc.stderr)
            create_payload = json.loads(create_proc.stdout)
            self.assertTrue(create_payload["ok"])
            self.assertTrue((project_dir / "config.json").exists())
            self.assertTrue((project_dir / "skill_runtime" / "abc_calibration_lib").exists())

            run_proc = subprocess.run(
                [sys.executable, str(project_dir / "run.py")],
                cwd=str(project_dir),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run_proc.returncode, 0, msg=run_proc.stdout + "\n" + run_proc.stderr)
            run_payload = json.loads(run_proc.stdout)
            self.assertTrue(run_payload["ok"])
            self.assertGreater(run_payload["result"]["posterior_samples"], 0)
            self.assertTrue((project_dir / "results" / "posterior_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
