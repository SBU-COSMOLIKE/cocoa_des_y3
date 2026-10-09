"""Locate the project interface for the optional covariance tests.

pytest reads this conftest.py (its per-folder configuration file) before
it collects the tests of covariance/. The file puts the folder of the
compiled interface on the import path and defines one fixture that skips
every test here when the interface was built without covariance support.
"""

from pathlib import Path
import sys

# parents[2] climbs from tests/covariance/ to the des_y3 project folder;
# sys.path is the list of folders Python searches on import.
project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project/"interface"))

import pytest
import cosmolike_des_y3_interface as ci


# A fixture is a function pytest runs around tests. autouse=True applies
# it to every test of this folder without the test asking for it, and
# scope="session" runs it once for the whole pytest session.
@pytest.fixture(scope="session", autouse=True)
def covariance_build():
    """Skip these tests when the interface was built without covariance support.

    The default Cocoa build omits covariance generation (the variable
    IGNORE_COSMOLIKE_DES_Y3_COVARIANCE is set), and such a build has no
    has_covariance attribute set to True. The skip names the two steps that
    enable it.

    Returns:
        nothing; pytest.skip marks every test of this folder as skipped.
    """
    if not getattr(ci, "has_covariance", False):
        pytest.skip(
            "Covariance generation is disabled. Unset "
            "IGNORE_COSMOLIKE_DES_Y3_COVARIANCE after start_cocoa.sh, "
            "then recompile this project."
        )
