"""Unit tests 27-29: cfastpt vs python FAST-PT, compared directly.

Cosmolike offers two implementations of the perturbation-theory
integrals that the TATT intrinsic-alignment model needs: cfastpt, a C
implementation built into the cosmolike interface (`IA_code: 0`, the
default), and the python FAST-PT package used through the fastpt
theory block (`IA_code: 1`).

27. example1 (cosmic shear): the same 30 hard-coded points
     across the intrinsic-alignment prior (FASTPT_COMPARISON_POINTS:
     20 drawn across the prior boxes plus a one-parameter-at-a-time
     family; cosmology fixed at the frozen fiducial) evaluated three
     times: with cfastpt, with FASTPT at the pass configuration
     (FASTPT_LOW_SETTINGS, written out in cocoa_test_utils), and with
     FASTPT at the doubled boosts (FASTPT_HIGH_SETTINGS). Every block
     prints its theory vector at every point (a block is one
     configuration), and the CFASTPT vector is the fiducial of that
     point: its own chi2 against it is zero by construction, so the
     pass rule is the chi2 of the FASTPT(low) vector against it
     (delta^T C^-1 delta, a pure second-order deviation; a chi2
     difference against the shipped data would instead follow the
     first-order slope of the chi2, since the point is far from the
     best fit). FASTPT(high)'s deviation is printed as the advisory
     FAST-PT grid response. Each configuration runs in its own
     subprocess, so no cache survives from one block to the next;
     inside a block the shared cosmology makes CAMB run once and the
     30 points cheap.
28. example2 (3x2pt): the same three-block sweep as test 27 on the
     3x2pt likelihood, so the TATT terms are also scored inside
     galaxy-galaxy lensing and under the 3x2pt covariance. The
     frozen configuration fixes the one-loop bias amplitudes
     (DES_B2_*) at zero, so the one-loop galaxy-bias tables both
     implementations compute multiply by zero here: this sweep
     compares the intrinsic-alignment tables only, on the wider
     data vector.
29. example2_2x2pt (2x2pt): the same sweep on the 2x2pt likelihood
     (galaxy clustering plus galaxy-galaxy lensing, cosmic shear
     dropped). Clustering carries no intrinsic alignment, so here
     the TATT tables are scored through galaxy-galaxy lensing
     alone, under the 2x2pt covariance; DES_B2_* stays zero as in
     test 28.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_fastpt.py

--high=1 repeats every block at the pushed camb/cosmolike settings
of the low-vs-high accuracy checks; the full comparison is one run
without the option and one with it:

    python -m pytest ./projects/des_y3/tests/data_vector/test_fastpt.py --high=1

The tests also read the --mask option (see conftest.py):
--mask=frozen (the default) keeps the baseline mask of the frozen
configuration, and --mask=ones keeps every data point (no scale cuts),
the strictest comparison; the 0.2 pass rule applies unchanged:

    python -m pytest ./projects/des_y3/tests/data_vector/test_fastpt.py --mask=ones

The design and the point values are shared with lsst_y1's tests
15-17; see that project's tests/README.md for the full discussion.
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


class TestCfastptVsFastptSweep(unittest.TestCase):
    """Tests 27-29, the direct CFASTPT-vs-FASTPT comparison.

    setUpClass runs once before the tests: it moves to ROOTDIR (the
    Cocoa/ folder, exported by start_cocoa.sh) and verifies every
    frozen file against the SHA-256 manifest. No
    frozen reference chi2 is loaded: these tests compare the two
    implementations against each other, so the frozen state only
    supplies the configuration and the data files.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen files before any test."""
        u.require_cocoa_environment()
        u.verify_frozen()

    def test_x27_cfastpt_vs_fastpt_sweep(self):
        """Cosmic shear: cfastpt and FASTPT agree at 30 IA points."""
        # the conftest copies the --high command line option into
        # this variable; .get with the "0" default keeps a run
        # outside pytest on the default settings unless the variable
        # is exported by hand
        high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
        setting = "high accuracy" if high else "default settings"
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        # five per-point lists come back: the chi2 of each block
        # against the data, then the delta^T C^-1 delta of the low and
        # high FASTPT vectors against the CFASTPT vector
        (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
         dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
            "example1", high=high, mask=mask)
        largest = u.report_fastpt_comparison(
            27,
            f"example1 (cosmic shear, TATT, camb/cosmolike {setting}, "
            f"mask {mask}): CFASTPT vs FASTPT at 30 hard-coded points",
            chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
            dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
        self.assertLess(
            largest, u.FASTPT_COMPARISON_TOLERANCE,
            msg="max chi2 of the FASTPT(low)-vs-CFASTPT data-vector "
                f"difference = {largest:.6f} over the comparison "
                f"points ({setting}); limit "
                f"{u.FASTPT_COMPARISON_TOLERANCE}")

    def test_x28_cfastpt_vs_fastpt_sweep_3x2pt(self):
        """3x2pt: cfastpt and FASTPT agree at the same 30 IA points.

        Test 27 on example2: the same three blocks, the same points,
        the same pass rule, with the TATT terms now entering
        galaxy-galaxy lensing as well and the difference weighted by
        the 3x2pt masked inverse covariance. The method name carries
        the x prefix only so unittest's alphabetical ordering runs it
        after test 27.
        """
        high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
        setting = "high accuracy" if high else "default settings"
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
         dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
            "example2", high=high, mask=mask)
        largest = u.report_fastpt_comparison(
            28,
            f"example2 (3x2pt, TATT, camb/cosmolike {setting}, "
            f"mask {mask}): CFASTPT vs FASTPT at 30 hard-coded points",
            chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
            dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
        self.assertLess(
            largest, u.FASTPT_COMPARISON_TOLERANCE,
            msg="max chi2 of the FASTPT(low)-vs-CFASTPT data-vector "
                f"difference = {largest:.6f} over the comparison "
                f"points ({setting}); limit "
                f"{u.FASTPT_COMPARISON_TOLERANCE}")

    def test_x29_cfastpt_vs_fastpt_sweep_2x2pt(self):
        """2x2pt: cfastpt and FASTPT agree at the same 30 IA points.

        Test 27 on example2_2x2pt: the same three blocks, the same
        points, the same pass rule. Clustering carries no intrinsic
        alignment, so the TATT tables enter through galaxy-galaxy
        lensing alone and the difference is weighted by the 2x2pt
        masked inverse covariance. The method name carries the x
        prefix only so unittest's alphabetical ordering runs it after
        test 28.
        """
        high = os.environ.get("COCOA_FASTPT_HIGH", "0") == "1"
        setting = "high accuracy" if high else "default settings"
        mask = os.environ.get("COCOA_FASTPT_MASK", "frozen")
        (chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
         dchi2_low, dchi2_high) = u.cfastpt_vs_fastpt_chi2s(
            "example2_2x2pt", high=high, mask=mask)
        largest = u.report_fastpt_comparison(
            29,
            f"example2_2x2pt (2x2pt, TATT, camb/cosmolike {setting}, "
            f"mask {mask}): CFASTPT vs FASTPT at 30 hard-coded points",
            chi2_cfastpt, chi2_fastpt_low, chi2_fastpt_high,
            dchi2_low, dchi2_high, u.FASTPT_COMPARISON_TOLERANCE)
        self.assertLess(
            largest, u.FASTPT_COMPARISON_TOLERANCE,
            msg="max chi2 of the FASTPT(low)-vs-CFASTPT data-vector "
                f"difference = {largest:.6f} over the comparison "
                f"points ({setting}); limit "
                f"{u.FASTPT_COMPARISON_TOLERANCE}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
