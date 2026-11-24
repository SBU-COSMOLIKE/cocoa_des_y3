"""E5: hybrid matter-emulator comparison and bounded Fisher accuracy plot.

Run from activated Cocoa with python -m pytest and this file's path.
The study compares fifty prior cosmologies with CAMB, with fixed neutrino
mass and matched nonlinear prescriptions as documented by the comparison
CLI. Every invocation retains its vectors, diagnostics and figure under
chains/hybrid_emulator_tests/. It uses current example priors and does not
change frozen regression data. Missing emulator installations
skip this test; installed numerical-pipeline failures fail it. Accuracy
exceedances in the broad prior sample remain advisory.
"""

import os
import sys
import unittest


class TestHybridEmulator(unittest.TestCase):
    """E5: current hybrid matter emulator, including the Fisher/error figure.

    This uses current example priors, not frozen E1--E4 reference vectors.
    It runs when the candidate code and networks are installed. Broad-prior
    accuracy exceedances and individual nonfinite candidate predictions are
    advisory. Missing CAMB references, no valid comparisons, or incomplete
    Fisher integration or plotting fail the test. No separate opt-in is required.
    """

    def test_e5_hybrid_emulator_fisher(self):
        """Compare fifty cosmologies and save the bounded Fisher overlay."""
        from datetime import datetime, timezone
        from pathlib import Path
        import subprocess

        project = Path(__file__).resolve().parents[2]
        root = project.parents[1]
        code = Path(os.environ.get(
            "EMULATOR_CODE_DIR", root / "external_modules/code/emulators/emultrf",
        ))
        data = Path(os.environ.get(
            "EMULATOR_DATA_DIR", root / "external_modules/data/emultrf",
        ))
        # The CLI uses the installed linear network and the selected hybrid
        # nonlinear network. Missing installations are a skip, while an
        # installed but broken numerical pipeline is a failed test.
        linear = root / "external_modules/data/emultrf/emulmps/npce_emul"
        nonlinear = data / "emulmps/w0wa_mead2020"
        required = [
            code / "emulmps/emulmps.py",
            code / "emulmps/emulmps_emul/emulmps_w0wa.py",
            linear / "emulator_npce.keras",
            linear / "metadata.joblib",
            nonlinear / "emulator_mead2020.keras",
            nonlinear / "metadata.joblib",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            self.skipTest("Hybrid emulator inputs are not installed: " + ", ".join(missing))

        # Each invocation gets its own output folder. Earlier scientific
        # results are retained, including failed draws and their provenance.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = project / "chains/hybrid_emulator_tests" / (stamp + "_" + str(os.getpid()))
        environment = os.environ.copy()
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        print("E5 hybrid emulator comparison and Fisher figure:", output, flush=True)
        result = subprocess.run(
            [sys.executable, str(project / "tests/compare_emulator_camb.py"),
             "--models", "50", "--advisory", "--output", str(output)],
            cwd=root,
            env=environment,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "Hybrid emulator numerical/Fisher pipeline failed; inspect " + str(output),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
