"""Accuracy advisory checks A1-A12: default vs high-accuracy settings.

Every reference in this suite is computed with the examples' default
numerical settings. These checks answer: how much numerical error do
those defaults carry? Each one re-evaluates a frozen configuration at
its frozen point with the numerical knobs pushed far beyond the
defaults (cosmolike: accuracyboost 3, internal_accuracyboost 2,
integration_accuracy 10, lmax 200000, kmax_boltzmann 40; CAMB:
AccuracyBoost 2.0, k_per_logint 50, kmax 50; the exact values live in
cocoa_test_utils.HIGH_ACCURACY_*) and reports

    delta chi2 = chi2(high accuracy) - chi2(default, frozen)

There is no pass/fail: how much numerical error an analysis tolerates
is a judgment call. The twelve checks cover the three probes with
both IA models, on the DES-Y3 data (A1-A6) and on the DES-Y1 data
(A7-A12):

  A1. des_y3 cosmic shear (example1), NLA        A2. cosmic shear, TATT
  A3. des_y3 2x2pt (example2_2x2pt), NLA         A4. 2x2pt, TATT
  A5. des_y3 3x2pt (example2), NLA               A6. 3x2pt, TATT
  A7. des_y1 cosmic shear (example3), NLA        A8. cosmic shear, TATT
  A9. des_y1 2x2pt (example4_2x2pt), NLA        A10. 2x2pt, TATT
 A11. des_y1 3x2pt (example4), NLA              A12. 3x2pt, TATT

Every check evaluates against a data vector generated at the frozen
point with its own intrinsic-alignment model (the synthetic NLA or the
TATT vector, written at freeze time), so the chi2 sits at a minimum
and the delta is a stable, quadratic response instead of a linear one.
Check A0 runs first: the same comparison one knob (one numerical
setting of cocoa_test_utils.ACCURACY_KNOBS) at a time.

A high-accuracy evaluation takes minutes, not seconds: the whole file
is far slower than the rest of the suite. To run only this file (from
the Cocoa/ folder, cocoa environment active, start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_accuracy.py

and to run the rest of the suite without it:

    python -m pytest ./projects/des_y3/tests --ignore \\
        ./projects/des_y3/tests/data_vector/test_accuracy.py
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


class TestAccuracyAdvisory(unittest.TestCase):
    """Advisory checks A1-A12, sharing one frozen-state verification.

    setUpClass runs once: it moves to ROOTDIR (the Cocoa/ folder,
    exported by start_cocoa.sh), verifies every frozen file against
    the SHA-256 manifest (an edited frozen state must fail loudly
    before any physics runs), and loads the frozen reference chi2
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

    def _accuracy_check(self, name, example, tatt, label):
        """Evaluate one configuration at high accuracy and report.

        Arguments:
          name    = the advisory label (A1-A12) for the report.
          example = a key of cocoa_test_utils.EXAMPLES (exact-physics
                    configurations only).
          tatt    = True evaluates the TATT variant against the
                    TATT-generated data vector, False the NLA one.
          label   = one line naming the probe and IA model.

        Returns:
          nothing; the printed report line is the result (advisory, no
          assertion).
        """
        chi2_high = u.single_model_chi2(example, tatt, high_accuracy=True)
        # ternary: the reference key ends in "tatt" or "nla", the
        # naming the frozen reference file uses
        suffix = "tatt" if tatt else "nla"
        default_ref = self.reference[f"{example}_{suffix}"]
        u.report_accuracy(f"{name}: {label}", chi2_high, default_ref)

    def test_a0_one_knob_at_a_time(self):
        """K scan: each accuracy knob alone on example2, NLA (check A0).

        Advisory: each knob's chi2 and its difference to the frozen
        default reference print as the scan runs. A knob whose delta
        rivals the all-knobs delta is the driver; a knob whose delta
        explodes (orders of magnitude beyond the others) signals an
        interface breakdown, not a numerics improvement. The scan
        stays on the DES-Y3 example2 configuration: the knobs act on
        the theory pipeline both data sets share, so one data set is
        enough to attribute an all-knobs delta.
        """
        default_ref = self.reference["example2_nla"]
        print("", flush=True)
        # each knob entry is (label, likelihood overrides, camb
        # overrides); the two _ discard the override tables here,
        # single_model_chi2 looks them up again by label
        for label, _, _ in u.ACCURACY_KNOBS:
            chi2 = u.single_model_chi2("example2", False, knob=label)
            u.report_knob(label, chi2, default_ref)

    def test_a1_cosmic_shear_nla(self):
        """A1: des_y3 cosmic shear, NLA, default vs high accuracy."""
        self._accuracy_check("A1", "example1", False,
                             "des_y3 example1 (cosmic shear, NLA)")

    def test_a2_cosmic_shear_tatt(self):
        """A2: des_y3 cosmic shear, TATT, default vs high accuracy."""
        self._accuracy_check("A2", "example1", True,
                             "des_y3 example1 (cosmic shear, TATT)")

    def test_a3_2x2pt_nla(self):
        """A3: des_y3 2x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A3", "example2_2x2pt", False,
                             "des_y3 example2_2x2pt (2x2pt, NLA)")

    def test_a4_2x2pt_tatt(self):
        """A4: des_y3 2x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A4", "example2_2x2pt", True,
                             "des_y3 example2_2x2pt (2x2pt, TATT)")

    def test_a5_3x2pt_nla(self):
        """A5: des_y3 3x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A5", "example2", False,
                             "des_y3 example2 (3x2pt, NLA)")

    def test_a6_3x2pt_tatt(self):
        """A6: des_y3 3x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A6", "example2", True,
                             "des_y3 example2 (3x2pt, TATT)")

    def test_a7_cosmic_shear_nla_y1(self):
        """A7: des_y1 cosmic shear, NLA, default vs high accuracy."""
        self._accuracy_check("A7", "example3", False,
                             "des_y1 example3 (cosmic shear, NLA)")

    def test_a8_cosmic_shear_tatt_y1(self):
        """A8: des_y1 cosmic shear, TATT, default vs high accuracy."""
        self._accuracy_check("A8", "example3", True,
                             "des_y1 example3 (cosmic shear, TATT)")

    def test_a9_2x2pt_nla_y1(self):
        """A9: des_y1 2x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A9", "example4_2x2pt", False,
                             "des_y1 example4_2x2pt (2x2pt, NLA)")

    def test_ax10_2x2pt_tatt_y1(self):
        """A10: des_y1 2x2pt, TATT, default vs high accuracy.

        The x prefix on A10-A12 only keeps unittest's alphabetical
        ordering aligned with the numbering (test_a10 would sort
        before test_a2).
        """
        self._accuracy_check("A10", "example4_2x2pt", True,
                             "des_y1 example4_2x2pt (2x2pt, TATT)")

    def test_ax11_3x2pt_nla_y1(self):
        """A11: des_y1 3x2pt, NLA, default vs high accuracy."""
        self._accuracy_check("A11", "example4", False,
                             "des_y1 example4 (3x2pt, NLA)")

    def test_ax12_3x2pt_tatt_y1(self):
        """A12: des_y1 3x2pt, TATT, default vs high accuracy."""
        self._accuracy_check("A12", "example4", True,
                             "des_y1 example4 (3x2pt, TATT)")


# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
