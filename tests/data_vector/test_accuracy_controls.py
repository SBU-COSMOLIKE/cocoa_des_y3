"""Check supported accuracy controls without running a cosmological model.

The signed control preserves historical integration levels: ordinary
quadratures use its magnitude, and some power-table callers use minus one.
The shared core caps its range at 126, where the largest pair rule saturates.
"""
from pathlib import Path
import os
import subprocess
import sys

import cocoa_test_utils as settings


def test_high_accuracy_uses_supported_ladder():
    assert 0 <= settings.HIGH_ACCURACY_LIKELIHOOD["integration_accuracy"] <= 126
    for name, overrides, unused in settings.ACCURACY_KNOBS:
        if "integration_accuracy" in overrides:
            assert 0 <= overrides["integration_accuracy"] <= 126, name


def test_native_accuracy_guard():
    project = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    for level in (-1, 10, 62, -127, 127):
        command = (
            "import sys; sys.path.insert(0, " + repr(str(project / "interface")) + "); "
            "import cosmolike_des_y3_interface as ci; "
            "ci.initial_setup(); "
            "ci.init_accuracy_boost(accuracy_boost=1., integration_accuracy="
            + str(level) + ")"
        )
        result = subprocess.run([sys.executable, "-c", command],
                                env=environment, capture_output=True, text=True)
        if abs(level) <= 126:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0
            assert "must be in [-126, 126]" in result.stdout + result.stderr
