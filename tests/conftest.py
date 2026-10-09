"""Command-line options of these tests (bound from cosmolike_core).

conftest.py is pytest's per-folder configuration file: pytest finds it by
walking up from the collected test files and calls the hook functions it
defines by their fixed names (pytest_addoption, pytest_configure). It
therefore has to live in this project's tests folder. Its content is the
shared implementation in cosmolike_core/cocoa_testing.py, bound here the
same way cocoa_test_utils.py binds the test harness. The options are
--high (repeat the CFASTPT-vs-FASTPT sweeps at the high-accuracy settings)
and --mask (the scale-cut mask of the comparison sweeps); the --mask
choices come from this project's harness (its fastpt_masks tuple,
"frozen" and "ones").
"""

import os
import sys

# The tests folder is not a package; put it first on sys.path (the list
# of folders Python searches on import) so cocoa_test_utils.py, the thin
# project module that binds the shared machinery to this project, resolves
# no matter where pytest was launched from. That module itself puts
# cosmolike_core on the path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u


def pytest_addoption(parser):
    """Register --high and --mask on pytest's parser.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_addoption.

    Arguments:
      parser = pytest's option parser (supplied by pytest).

    Returns:
      nothing; the options become readable through config.getoption.
    """
    u._cct.conftest_addoption(parser, u._H.fastpt_masks)


def pytest_configure(config):
    """Copy the option values where the test classes read them.

    The shared implementation and its documentation live in
    cocoa_testing.conftest_configure.

    Arguments:
      config = pytest's configuration object (supplied by pytest).

    Returns:
      nothing; the environment of this process gains the variables.
    """
    u._cct.conftest_configure(config)
