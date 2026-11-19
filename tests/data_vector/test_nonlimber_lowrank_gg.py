"""Unit test: FKEM vs low-rank SVD non-Limber clustering.

The likelihood yaml key nonlimber_lowrank_gg picks the engine behind
the non-Limber galaxy clustering C_l^gg (active when
adopt_limber_gg: 0):

  nonlimber_lowrank_gg: 0 - FKEM (Fang, Krause, Eifler & MacCrann,
      arXiv:1911.11947): an FFTLog integral of the separable linear
      spectrum plus, in Limber, what linear theory misses. The
      default of every project.
  nonlimber_lowrank_gg: 1 - the low-rank SVD engine
      (C_cl_tomo_lowrank in cosmo2D.c): the full nonlinear sqrt(P_g)
      decomposed into a short sum of separable terms, each projected
      by FFTLog, so the density-density term needs no Limber hand-off
      at any multipole. Built for thin spectroscopic lens bins; on
      this project's wide bins it is a consistency check.

Both are exact low-l projections, so the engine difference is small -
the SVD's rank truncation against FKEM's Limber hand-off of the
nonlinear density term - and the runtimes are comparable (wide bins
bring few rank rows). This test freezes both statements.

It evaluates the frozen 3x2pt fiducial (NLA) three times in one
process (FKEM, SVD, FKEM again) and computes

    delta chi2 = delta^T C^-1 delta,
    delta = dv(FKEM) - dv(SVD),

with C^-1 the masked inverse covariance. It also reports each
engine's steady-state evaluation time (informational).

Assertions (the shared factory's five checks):
  1. delta chi2 is above a dead-flag floor: the flag reaches the C
     code and the clustering cache notices the change;
  2. only clustering entries change: cosmic shear, galaxy-galaxy
     lensing, and every other block are bitwise equal;
  3. switching back to FKEM reproduces the first data vector bitwise;
  4. delta chi2 matches the value measured for this project
     (DCHI2_MEASURED below) to 10%: a change in either non-Limber
     engine, the kernels, or the covariance shows up here;
  5. the FKEM default reproduces the frozen reference chi2 (checked
     last, so an out-of-date snapshot cannot hide checks 1-4).

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_nonlimber_lowrank_gg.py
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

# The engine delta chi2 of this project's frozen fiducial, measured
# at OMP 6 with rank 5 and the path's l cap of 150 (the likelihood
# defaults): see assertion 4 of the module docstring.
DCHI2_MEASURED = 0.0001551837849

# The shared factory (cosmolike_core/cocoa_testing_suites.py) builds
# the class; assigning its return value to the usual module-level name
# is what makes pytest collect it here, with unchanged test ids.
TestNonLimberLowrankGG = u.lowrank_gg_suite(
    u, name="TestNonLimberLowrankGG", measured=DCHI2_MEASURED)

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
