"""Unit test 30: the OpenMP race check with EuclidEmulator2 on.

Cocoa pins a modified EuclidEmulator2 (the EE2_GIT_COMMIT of
set_installation_options.sh): OpenMP threading, a 1,010-redshift
capacity, the get_boost2 API with a pre-built emulator, memory-leak
fixes, and a bilinear interpolation with a border fix (the
repository's README, external_modules/code/euclidemu2/README.md,
documents them).

The standard race checks (the test_example*.py files) run with the
nonlinear P(k) from CAMB's Takahashi halofit (non_linear_emul: 2, the
setting of the frozen configurations), so EE2's compute never executes
inside them, and a race in it would go unseen. A race is a bug where
OpenMP threads write shared memory in an order that changes from run to
run. This test repeats
the race check on the DES-Y3 cosmic-shear configuration with the
nonlinear P(k) from EE2 (non_linear_emul: 1): the fiducial
evaluated fresh and again as the 10th of 10 cosmologies on one
model instance. EE2's compute is OpenMP-threaded, so leaked state
or a thread race inside it shifts the second fiducial value; the
two must agree within RACE_TOLERANCE (1e-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_ee2.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya or cosmolike import in the process.
# 4 is REQUIRED_OMP_THREADS of cocoa_testing.py: a race between
# OpenMP threads can only show up when several threads run.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The harness stays in the parent tests/ folder. Add it explicitly so
# direct execution and worker processes resolve this project's stored inputs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u


class TestEE2Race(unittest.TestCase):
    """Test 30, with one frozen-state verification.

    setUpClass runs once before the test: it moves to ROOTDIR (the
    Cocoa/ folder, exported by start_cocoa.sh) and verifies every
    frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: the race check compares one
    model against itself, so the frozen state only supplies the
    configuration and the data files.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen files before the test."""
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x30_ee2_race_ten_in_a_row(self):
        """The fiducial with EE2 as 10th of 10 matches a fresh run.

        EE2's OpenMP-threaded compute runs inside every evaluation
        of the row, so a thread race or leaked state in it moves
        the second fiducial value.
        """
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2("example1", tatt=False,
                                           ee2=True)
        u.report_race_test(
            30, "des_y3 example1 (cosmic shear, NLA+EE2) race "
            "check: 10 cosmologies in a row", fresh, tenth,
            u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh "
                f"{fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
