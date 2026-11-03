"""Unit tests 19-22: the 3x2pt likelihood on the frozen DES-Y1 data.

3x2pt combines three two-point correlations: cosmic shear, galaxy
clustering, and galaxy-galaxy lensing; here it is the
des_y3.combo_3x2pt likelihood evaluated on the frozen copy of
example4's configuration: the same likelihood class as tests 5-8,
pointed at the DES-Y1 data (des_y1_real.dataset, with its own data
vector, covariance, masks, and n(z) tables) instead of the DES-Y3
data (see cocoa_test_utils for what "frozen" means and for the Y1/Y3
split). The four tests:

 19. chi2 at the frozen fiducial point, against the synthetic NLA
     data vector generated at that point (so the chi2 sits at its
     minimum), within CHI2_TOLERANCE (0.2) of the frozen reference
     value.
 20. race check: on one model, the fiducial evaluated fresh and again
     as the 10th of 10 cosmologies in a row must agree to
     RACE_TOLERANCE (1e-4). A disagreement means state leaked between
     evaluations or OpenMP threads raced (wrote shared memory in an
     order that changes from run to run).
 21. the same comparison as test 19 with the TATT intrinsic-alignment
     model (IA_model: 1), against the TATT-generated data vector,
     with DES_A2_1 = 0.05, DES_BTA_1 = 0.05 and DES_A2_2 = -1.51541
     (all three are zero at the NLA point).
 22. the same race check as test 20 with the TATT model.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector
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
# the chi2 + race test class for one frozen example; assigning its
# return value to the usual module-level name is what makes pytest
# collect it here, with unchanged test ids. The numbers name the
# methods and the printed reports; the label opens every report line.
TestExample4ThreeXTwo = u.example_suite(
    u, name="TestExample4ThreeXTwo", example="example4",
    label="des_y1 (3x2pt",
    numbers=("x19", "x20"), tatt_numbers=("x21", "x22"))

# __name__ is "__main__" only when this file runs directly as a
# script; pytest imports the module instead, so this block stays
# idle under pytest
if __name__ == "__main__":
    unittest.main(verbosity=2)
