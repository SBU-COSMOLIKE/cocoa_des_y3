"""E5: hybrid matter-emulator comparison and bounded Fisher accuracy plot.

Run from activated Cocoa with python -m pytest and this file's path.
The study compares fifty prior cosmologies and twenty-five focused Fisher
points with CAMB, using EXAMPLE_EVALUATE1 and fixed neutrino
mass and matched nonlinear prescriptions as documented by the comparison
CLI. Every invocation retains its vectors, diagnostics and figure under
chains/hybrid_emulator_tests/. It uses saved benchmark priors and does not
change frozen regression data. Missing emulator installations
skip this test; installed numerical-pipeline failures fail it. Accuracy
exceedances in the broad prior sample remain advisory.
"""

import os
import sys
import unittest


class TestHybridEmulator(unittest.TestCase):
    """E5: current hybrid matter emulator, with an optional saved Fisher overlay.

    This uses saved benchmark priors, not frozen E1--E4 reference vectors.
    It runs when the candidate code and networks are installed. Broad-prior
    accuracy exceedances and individual nonfinite candidate predictions are
    advisory. Missing CAMB predictions or a comparison with no valid vectors
    fail the test. Saved Fisher contours do not affect the numerical verdict.
    """

    def test_e5_hybrid_emulator_accuracy(self):
        """Compare prior and focused points and save their bounded Fisher overlay."""
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
        print("E5 hybrid emulator vector accuracy:", output, flush=True)
        frozen = project / "tests/fisher_reference"
        prior = frozen / "prior_50.txt"
        focused = frozen / "focused_25.txt"
        # A missing row changes the denominator even when CAMB itself succeeds.
        for path, count in ((prior, 50), (focused, 25)):
            rows = [s for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")]
            self.assertEqual(len(rows), count, str(path))
        result = subprocess.run(
            [sys.executable, str(project / "tests/compare_emulator_camb.py"),
             "--points", str(prior), "--local-points", str(focused),
             "--advisory", "--output", str(output)],
            cwd=root,
            env=environment,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "Emulator vector comparison failed; inspect " + str(output),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
