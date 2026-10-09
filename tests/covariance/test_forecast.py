"""Check this project's galaxy/shear covariance interface and catalog inputs.

The forecast covariance of covariance/des_y3_covariance.py is a layout
comparison: the likelihoods keep reading the published DES Y3 covariance.
This test calls the shared check_project_forecast (cosmolike_core,
cocoa_covariance_testing.py), which checks the catalog inputs and the
measurement layout (900 real-space and 525 Fourier-space entries), that
accuracy refinement leaves the scientific bins fixed, that the real and
Fourier matrices repeat with one and eight OpenMP threads, and that a
saved archive reads back intact. It uses small numerical settings and a
subset of the measured rows, so it does not certify the convergence of a
survey covariance.

Run it separately from tests/data_vector (pytest.ini collects only
data_vector by default), from the Cocoa/ folder after
`source start_cocoa.sh`, with covariance generation compiled in:

    python -m pytest projects/des_y3/tests/covariance
"""

from pathlib import Path
import sys

# parents[2] is the des_y3 project folder; project.parents[1] is Cocoa/.
# The shared core, the compiled interface and the covariance adapter go
# first on sys.path, the list of folders Python searches on import.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project.parents[1]/"external_modules/code/cosmolike_core"))
sys.path.insert(0, str(project/"interface"))
sys.path.insert(0, str(project/"covariance"))

from cocoa_covariance_testing import check_project_forecast
import cosmolike_des_y3_interface as ci
import des_y3_covariance as survey


def test_forecast_adapter(tmp_path):
    """Real and Fourier components repeat at one/eight threads and save intact.

    Arguments:
        tmp_path = a temporary folder pytest creates for this test (a
            built-in fixture); the check writes its archives there.
    """
    check_project_forecast(
        interface=ci, survey=survey, expected_sizes=(900, 525), directory=tmp_path,
    )
