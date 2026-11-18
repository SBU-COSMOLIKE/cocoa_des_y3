"""Unit test: the non-Limber early-exit tolerance, the FKEM hand-off
and the noise of the low-rank (SVD) engine.

The non-Limber real-space C_l switch a lens bin (gg) or a lens-source
pair (gs) to Limber once they agree with Limber within a tolerance; the
switch multipole moves with the parameters and puts steps into
finite-difference derivatives. init_nonlimber_tolerance sets the
tolerances (likelihood keys nonlimber_tolerance_gg = 0.002 and
nonlimber_tolerance_gs = 0.01 by default; 0 = no early exit). The
shared factory (cosmolike_core/cocoa_testing_suites.py,
nonlimber_tolerance_suite) documents the checks. On example2:

    This project's 3x2pt data vector is in real space: w(theta)
    (and gamma_t where adopt_limber_gs = 0) runs the non-Limber
    path with the early exit, so tests 3 and 4 measure what the
    tolerance does here.

The stencil noise is the whitened fourth difference of five evaluations
(cosmolike_notebook_utils.fisher.get_stencil_noise, step h = 0.025) of
the w0, wa and DES_B1_5 columns.

Measured on 2026-11-18 (OMP 4), then bounded:
  - test 3: delta^T C^-1 delta = 2.04e-4 (w(theta) only: gamma_t is
    Limber here); bound 5e-4;
  - test 4: sigma (w0, wa, bias) = 1.74e-3, 1.48e-4, 1.41e-3 at the
    default tolerances and 1.93e-4, 1.48e-4, 3e-12 at 0; bound 2e-3;
  - test 5: FKEM caps 150/200/300 give w0 1.93e-4/2.08e-4/2.38e-4 and
    wa 1.48e-4/1.49e-4/1.46e-4; margin 0.1;
  - test 6: SVD/FKEM ratio at most 1.14 (w0, cap 300); margin 0.3.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_nonlimber_tolerance.py -s
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

# The bounds below come from this project's measurement (the values
# are in the comment of the module docstring above).
TestNonLimberTolerance = u.nonlimber_tolerance_suite(
    u, name="TestNonLimberTolerance", example="example2",
    bias_parameter="DES_B1_5",
    change_bound=0.0005,
    noise_bound=0.002,
    handoff_margin=0.1,
    svd_rule='not_larger',
    svd_margin=0.3)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
