"""Advisory checks NL1-NL2: Halofit vs EuclidEmulator2.

The likelihoods can source the nonlinear matter power from CAMB's
Takahashi halofit (non_linear_emul: 2, the setting of the frozen
configurations) or from EuclidEmulator2 (EE2, non_linear_emul: 1),
which emulates the ratio of nonlinear to linear power and is applied
to CAMB's linear spectrum below z = 10. NL1 evaluates
the DES-Y3 cosmic-shear data vector (example1) and NL2 the DES-Y3
3x2pt data vector (example2) with both sources at ten fixed
cosmologies across the omegam/ns/As space
(NONLINEAR_COMPARISON_POINTS; every other parameter stays at the
frozen fiducial) and reports, at each cosmology, the chi2 of the
Halofit vector against the EE2 vector (delta^T C^-1 delta; the EE2
vector is that cosmology's fiducial, so the baseline is zero by
construction).

There is no pass/fail: the numbers say how much of the statistical
error budget the Halofit-vs-emulator difference consumes under the
chosen scale cuts, which answers whether Halofit is good enough for a
real-data analysis with that mask. The checks read the --mask option
of the comparison sweeps (conftest.py): --mask=frozen (the mask of the
frozen configuration, the default) or --mask=ones (every data point
kept).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_nonlinear.py

    python -m pytest ./projects/des_y3/tests/data_vector/test_nonlinear.py --mask=ones
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


class TestHalofitVsEE2(unittest.TestCase):
    """Advisory checks NL1-NL2, sharing the frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR (the
    Cocoa/ folder, exported by start_cocoa.sh) and verifies every
    frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: the checks compare the two
    nonlinear-P(k) sources against each other, so the frozen state
    only supplies the configuration and the data files.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen files before any test."""
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_nl1_halofit_vs_ee2_cosmic_shear(self):
        """Cosmic shear: Halofit scored against EE2 at ten cosmologies.

        Advisory: the printed report is the product. The only
        assertion is structural: every cosmology must have produced
        a number.
        """
        # conftest.py copies the --mask option into this environment
        # variable; .get returns "frozen" when the variable is absent,
        # as in a run outside pytest
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        dchi2s = u.halofit_vs_ee2_dchi2s("example1", mask=mask)
        u.report_nonlinear_comparison(
            f"NL1: example1 (cosmic shear, NLA, mask {mask}): HALOFIT "
            "vs EE2 at 10 fixed cosmologies", dchi2s)
        self.assertEqual(len(dchi2s), len(u.NONLINEAR_COMPARISON_POINTS))

    def test_nl2_halofit_vs_ee2_3x2pt(self):
        """3x2pt: Halofit scored against EE2 at ten cosmologies.

        Advisory: the printed report is the product. The only
        assertion is structural: every cosmology must have produced
        a number.
        """
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        dchi2s = u.halofit_vs_ee2_dchi2s("example2", mask=mask)
        u.report_nonlinear_comparison(
            f"NL2: example2 (3x2pt, NLA, mask {mask}): HALOFIT vs "
            "EE2 at 10 fixed cosmologies", dchi2s)
        self.assertEqual(len(dchi2s), len(u.NONLINEAR_COMPARISON_POINTS))


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
