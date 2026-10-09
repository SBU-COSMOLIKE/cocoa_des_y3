"""Unit tests 11-14: the 2x2pt likelihood on the frozen test data.

2x2pt combines two of example2's three two-point correlations: galaxy
clustering and galaxy-galaxy lensing (cosmic shear is dropped). The
frozen configuration is example2's with the likelihood renamed to
des_y3.combo_2x2pt: same options, same data files, same evaluation
point; only the probe selection inside cosmolike changes. The four
tests mirror tests 5-8 (see cocoa_test_utils for what "frozen" means):

 11. chi2 at the frozen fiducial point, against the synthetic NLA
     data vector generated at that point (so the chi2 sits at its
     minimum), within CHI2_TOLERANCE (0.2) of the frozen reference
     value.
 12. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked
     between evaluations or OpenMP threads raced (wrote shared memory
     in an order that changes from run to run).
 13. the same comparison as test 11 with the TATT intrinsic-alignment
     model (IA_model: 1), against the TATT-generated data vector,
     with DES_A2_1 = 0.05, DES_BTA_1 = 0.05 and DES_A2_2 = -1.51541
     (all three are zero at the NLA point).
 14. the same race check as test 12 with the TATT model.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector
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

EXAMPLE = "example2_2x2pt"


class TestExample2TwoXTwo(unittest.TestCase):
    """Tests 11-14, sharing one frozen-state verification.

    setUpClass runs once before the tests: it moves to ROOTDIR (the
    Cocoa/ folder, exported by start_cocoa.sh), verifies every frozen
    file against the SHA-256 manifest (an edited frozen state must fail
    loudly before any physics runs), and loads the frozen reference chi2
    values.
    """

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen files; load the references."""
        u.require_cocoa_environment()
        u.verify_frozen()
        cls.reference = u.load_reference()

    def test_x11_chi2_matches_frozen_reference(self):
        """chi2 at the frozen NLA point stays within 0.2 of the reference.

        The x prefix on tests 11-14 only keeps unittest's alphabetical
        ordering aligned with the numbering (test_11 would sort before
        test_2).
        """
        chi2 = u.single_model_chi2(EXAMPLE, tatt=False)
        ref = self.reference[f"{EXAMPLE}_nla"]
        u.report_chi2_test(
            11, "des_y3 example2 (2x2pt, NLA) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        # assertLess(a, b) passes when a < b and fails with msg
        # otherwise; in that message, :.6f prints fixed six decimals
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_x12_no_race_condition_ten_in_a_row(self):
        """The fiducial as 10th of 10 cosmologies matches a fresh run."""
        u.assert_omp_threads()
        # the function returns a (fresh, tenth) pair; the assignment
        # unpacks it into the two names
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=False)
        u.report_race_test(
            12, "des_y3 example2 (2x2pt, NLA) race check: "
                "10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")

    def test_x13_chi2_matches_frozen_reference_tatt(self):
        """Test 11 repeated with the TATT IA model and nonzero A2/BTA."""
        chi2 = u.single_model_chi2(EXAMPLE, tatt=True)
        ref = self.reference[f"{EXAMPLE}_tatt"]
        u.report_chi2_test(
            13, "des_y3 example2 (2x2pt, TATT) chi2 vs frozen reference",
            chi2, ref, u.CHI2_TOLERANCE)
        self.assertLess(
            abs(chi2 - ref), u.CHI2_TOLERANCE,
            msg=f"TATT chi2 = {chi2:.6f} vs frozen reference {ref:.6f} "
                f"(|delta| >= {u.CHI2_TOLERANCE})")

    def test_x14_no_race_condition_ten_in_a_row_tatt(self):
        """Test 12 repeated with the TATT IA model."""
        u.assert_omp_threads()
        fresh, tenth = u.ten_in_a_row_chi2(EXAMPLE, tatt=True)
        u.report_race_test(
            14, "des_y3 example2 (2x2pt, TATT) race check: "
                "10 cosmologies in a row",
            fresh, tenth, u.RACE_TOLERANCE)
        self.assertLess(
            abs(tenth - fresh), u.RACE_TOLERANCE,
            msg=f"TATT 10th-in-a-row chi2 = {tenth:.8f} vs fresh {fresh:.8f}")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
