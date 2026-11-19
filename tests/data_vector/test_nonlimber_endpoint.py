"""Check the non-Limber radial endpoint and its cached projections.

The endpoint is a maximum redshift, independent of the radial node count.
The core defaults to 4.0 and rejects non-finite values or values below 4.
This project's YAML explicitly selects 4.0. The shared tests check that
preset, resets, sampling-boost independence, invalid values, and endpoint
changes against fresh processes. The numerical tests retain fixed frozen
inputs and explicitly override the endpoint in each sequence.

Run from the Cocoa directory in the started environment:

    python -m pytest projects/des_y3/tests/data_vector/test_nonlimber_endpoint.py
"""

import os
import sys
import unittest

# Set the thread limit before loading any compiled scientific library.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u
from cocoa_testing_suites import nonlimber_endpoint_suite

TestNonLimberEndpoint = nonlimber_endpoint_suite(
    u=u,
    name="TestNonLimberEndpoint",
    example="example2",
    project_endpoint=4.0,
    tatt=True,
    cluster=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
