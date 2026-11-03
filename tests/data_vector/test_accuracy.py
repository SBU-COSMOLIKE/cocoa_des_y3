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
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the advisory class; assigning its return value to the usual
# module-level name is what makes pytest collect it here, with
# unchanged test ids. Each roster row is (method name, advisory
# label, example, TATT flag, report label); knob_scan binds the
# example the one-knob-at-a-time scan evaluates.
TestAccuracyAdvisory = u.accuracy_suite(
    u, name="TestAccuracyAdvisory",
    checks=(
        ("test_a1_cosmic_shear_nla", "A1", "example1",
         False, "des_y3 example1 (cosmic shear, NLA)"),
        ("test_a2_cosmic_shear_tatt", "A2", "example1",
         True, "des_y3 example1 (cosmic shear, TATT)"),
        ("test_a3_2x2pt_nla", "A3", "example2_2x2pt",
         False, "des_y3 example2_2x2pt (2x2pt, NLA)"),
        ("test_a4_2x2pt_tatt", "A4", "example2_2x2pt",
         True, "des_y3 example2_2x2pt (2x2pt, TATT)"),
        ("test_a5_3x2pt_nla", "A5", "example2",
         False, "des_y3 example2 (3x2pt, NLA)"),
        ("test_a6_3x2pt_tatt", "A6", "example2",
         True, "des_y3 example2 (3x2pt, TATT)"),
        ("test_a7_cosmic_shear_nla_y1", "A7", "example3",
         False, "des_y1 example3 (cosmic shear, NLA)"),
        ("test_a8_cosmic_shear_tatt_y1", "A8", "example3",
         True, "des_y1 example3 (cosmic shear, TATT)"),
        ("test_a9_2x2pt_nla_y1", "A9", "example4_2x2pt",
         False, "des_y1 example4_2x2pt (2x2pt, NLA)"),
        ("test_ax10_2x2pt_tatt_y1", "A10", "example4_2x2pt",
         True, "des_y1 example4_2x2pt (2x2pt, TATT)"),
        ("test_ax11_3x2pt_nla_y1", "A11", "example4",
         False, "des_y1 example4 (3x2pt, NLA)"),
        ("test_ax12_3x2pt_tatt_y1", "A12", "example4",
         True, "des_y1 example4 (3x2pt, TATT)"),
    ),
    knob_scan="example2")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
