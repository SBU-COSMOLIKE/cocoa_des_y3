"""Unit test: the UV-fraction scale cut of w(theta) (cosmo2D_scuts).

The scale cut of arXiv:2603.13031 splits each galaxy clustering
spectrum C_l into the part sourced by modes below a cut k_max/D(z)
(the IR part) and the rest (the UV part), projects both with the
bin-averaged Legendre kernel of w(theta), and keeps an angular bin
while the UV fraction |w_UV|/(|w_UV| + |w_IR|) stays below alpha =
0.05. k_max is quoted at z = 0 in h/Mpc, and D is the linear growth
factor with D(0) = 1. Scanning from the largest angle down, the kept
bins of a lens bin are a suffix in theta; its smallest angle is
theta_min(k_max).

The checks (the shared factory w_gg_scale_cut_suite of
cosmolike_core/cocoa_testing_suites.py), on the frozen 3x2pt
fiducial, run on each clustering projection the cut can follow: Limber
at every multipole, the FKEM non-Limber split and the low-rank (SVD)
non-Limber engine:
  1. window off, the scale-cut driver against w_gg_tomo: bit for bit
     on the Limber path, within a stated tolerance on the non-Limber
     engines (the driver has no early exit there);
  2. w_IR + w_UV = w;
  3. on the Limber path, w_IR agrees with a reference summed from the
     exact Limber C_l at every integer multipole;
  4. at k_max = 20 h/Mpc the UV fraction is below alpha in every bin;
  5. theta_min does not increase with k_max, and the kept bins form
     nested suffixes in theta;
and the next ordinary evaluation after the cut gives the data vector
of before bit for bit.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_w_gg_scale_cuts.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya/cosmolike import in the process.
os.environ["OMP_NUM_THREADS"] = os.environ.get("COCOA_TEST_OMP_THREADS", "4")

import sys
import unittest

# The tests folder is not a package; put it on the import path so the
# shared harness resolves no matter where pytest was launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import cocoa_test_utils as u

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here.
TestWggScaleCut = u.w_gg_scale_cut_suite(
    u, name="TestWggScaleCut", example="example2")

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
