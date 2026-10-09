"""Shared harness for the des_y3 unit tests: the project's data bound
to the shared Cocoa test machinery.

The machinery itself (frozen-state verification, the chi2 pipeline,
the worker subprocesses, the race and baryon checks, the
CFASTPT-vs-FASTPT comparison, and the terminal reports) lives in
external_modules/code/cosmolike_core/cocoa_testing.py. This file holds
what belongs to des_y3 alone: the examples table, which covers both DES
data generations (examples 1 and 2, plus the 2x2pt reduction of example
2, evaluate the Y3 data set; examples 3 and 4, plus the 2x2pt reduction
of example 4, evaluate its Y1 twin, each data set with its own synthetic
NLA and TATT data vectors), the TATT point, the accuracy settings, and
the settings of the CFASTPT-vs-FASTPT comparison. It binds them to one
cocoa_testing.CocoaTestHarness instance and re-exports the harness
methods and the core functions as module-level names, the names the test
modules and generate_frozen_reference.py import from this module.

What "frozen" means: everything a test evaluates (the data files, the
fully resolved configurations, the reference chi2 values) lives under
tests/frozen/, pinned byte for byte by tests/manifest_sha256.json and
verified before any model is built. Later edits to the live data/
folder, the example yamls or the likelihood defaults therefore cannot
change what the tests evaluate. Refreshing the frozen state is a
deliberate maintainer action (generate_frozen_reference.py --overwrite).

Each test quantity is computed in a worker subprocess, a separate Python
process: CosmoLike keeps the data-vector dimensions in C global variables
and aborts a process in which a configuration of different dimensions
initializes after the first (cocoa_testing.py, WORKER ISOLATION).
"""

import os
import sys

# ---- tests/ paths -----------------------------------------------------------

# Everything the tests read or write lives relative to this folder, so
# the suite works no matter which directory pytest is launched from
# (__file__ is this module's own path; dirname strips the file name).
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_DIR = os.path.join(TESTS_DIR, "frozen")
MANIFEST_FILE = os.path.join(TESTS_DIR, "manifest_sha256.json")
REFERENCE_FILE = os.path.join(FROZEN_DIR, "reference_chi2.json")

# ---- the shared machinery ---------------------------------------------------

# The import is path-based (tests/ is three levels below Cocoa/, which
# holds external_modules/code/cosmolike_core) so it works before
# start_cocoa.sh's python-path setup runs; sys.path is the list of
# folders Python searches on import.
_CORE_DIR = os.path.abspath(os.path.join(
    TESTS_DIR, "..", "..", "..", "external_modules", "code",
    "cosmolike_core"))
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
import cocoa_testing as _cct

# ---- the project data ------------------------------------------------------

# The TATT (Tidal Alignment and Tidal Torquing, an intrinsic-alignment
# model with tidal second-order terms) tests replace these values in
# the frozen point. In the NLA reference point A2 and BTA are zero, so
# the nonzero values here make the TATT reference genuinely exercise
# the second-order terms.
TATT_POINT = {
    "DES_A2_1": 0.05,
    "DES_BTA_1": 0.05,
    "DES_A2_2": -1.51541,
}

# The TATT variants evaluate against a data vector generated with TATT
# at the fiducial point. Against an NLA-based vector the TATT chi2 would
# sit away from its minimum, where it responds linearly (not
# quadratically) to tiny numerical changes, and harmless rounding-level
# shifts would use up much of the 0.2 chi2 tolerance of the reference
# tests. The Y3 and Y1 configurations use different data sets, so each
# data set has its own generated vector, from its 3x2pt example; within
# a data set the full-length vector serves every probe (the masks of the
# other probes select their blocks). This table, {".dataset" file:
# generating example}, is read by no code: generate_frozen_reference.py
# works from SYNTHETIC_VECTORS below, which lists the same TATT vectors.
TATT_GENERATORS = {
    "tatt_des_y3.dataset": "example2",
    "tatt_des_y1.dataset": "example4",
}

# This project's shipped data_files are real data (the DES-Y3 and
# DES-Y1 measurements), and the example cosmology is not a best fit
# of either: the chi2 sits far from the minimum (hundreds to
# thousands), where it responds linearly to tiny theory changes. A
# chi2 comparison evaluated there reports alarming shifts that say
# nothing about the numerics near a fit. The NLA variants
# therefore evaluate against synthetic data vectors, generated with
# the default (NLA) model at the fiducial point during the freeze,
# one per data set, exactly like the TATT vectors: at its own minimum
# the chi2 response is quadratic and stable.
# Every generated vector: {".dataset" filename: (source example, TATT?)},
# where TATT? is True for a vector generated with the TATT model.
SYNTHETIC_VECTORS = {
    "synthetic_des_y3.dataset": ("example2", False),
    "tatt_des_y3.dataset": ("example2", True),
    "synthetic_des_y1.dataset": ("example4", False),
    "tatt_des_y1.dataset": ("example4", True),
}

# High-accuracy settings for the accuracy advisory checks
# (test_accuracy.py): the same physics evaluated with the numerical
# knobs pushed far beyond the defaults.
HIGH_ACCURACY_LIKELIHOOD = {
    # boost 3 is the highest value that stays healthy in every project
    # scanned (desy1xplanck breaks down above it), so the all-knobs
    # check compares the default against 3; the one-at-a-time scan
    # keeps 5 as a deliberate stress knob
    "accuracyboost": 3.0,       # default 1.0
    "internal_accuracyboost": 2.0, # default 1.0 (denser convolution grid)
    "integration_accuracy": 10,  # default 0
    "lmax": 200000,             # default 50000-75000
    "kmax_boltzmann": 40.0,     # default 5.0-7.5
}

# The one-at-a-time scan of test_accuracy.py: each entry is (label,
# likelihood overrides, camb extra_args overrides), evaluated alone on
# the example2 NLA configuration before the all-knobs checks, so a
# large all-knobs delta can be attributed to the knob causing it. A
# "knob" is one numerical accuracy setting. The accuracyboost = 5 entry
# is a stress test: it exceeds what measuring the default numerics
# needs, and it probes the interface for fixed-size tables (at this
# boost the desy1xplanck interface breaks down, a suspected fixed-size
# table).
# Investigation order when several knobs move the chi2: raise the
# cosmolike accuracyboost first (cheap), then camb k_per_logint, and
# only then camb AccuracyBoost (expensive at run time): an apparent
# CAMB sensitivity can masquerade as unresolved cosmolike-side
# resolution, so the cheap knobs must be settled before the expensive
# one is blamed. kmax_boltzmann and camb kmax are one physical cutoff
# seen from the two sides, so the scan moves them together.
ACCURACY_KNOBS = [
    ("accuracyboost -> 3", {"accuracyboost": 3.0}, {}),
    ("accuracyboost -> 5 (stress)", {"accuracyboost": 5.0}, {}),
    ("internal_accuracyboost 1->2", {"internal_accuracyboost": 2.0}, {}),
    ("integration_accuracy -> 10", {"integration_accuracy": 10}, {}),
    ("lmax -> 200000", {"lmax": 200000}, {}),
    ("kmax_boltzmann -> 40 + camb kmax -> 50",
     {"kmax_boltzmann": 40.0}, {"kmax": 50.0}),
    ("camb AccuracyBoost -> 2", {}, {"AccuracyBoost": 2.0}),
    ("camb k_per_logint -> 50", {}, {"k_per_logint": 50}),
]

# example1/2 (and example2's 2x2pt reduction) evaluate the DES-Y3
# data; example3/4 (and example4's 2x2pt reduction) evaluate the
# DES-Y1 data through the same des_y3.* likelihood classes. Keys of
# each entry:
#   frozen_module     = the frozen configuration module in tests/frozen/
#   provenance        = the example yaml it was built from; its frozen
#                       copy is a human-readable snapshot, never loaded
#   likelihood        = the cobaya component name, needed to reach that
#                       block inside the loaded info dictionary
#   source_likelihood = for a 2x2pt reduction, the likelihood name in
#                       the provenance yaml, renamed to "likelihood"
#   tatt_dataset      = the dataset the TATT variant evaluates against
#   nla_dataset       = the dataset the NLA variant evaluates against
EXAMPLES = {
    "example1": {
        "frozen_module": "frozen_config_example1.py",
        "provenance": "EXAMPLE_EVALUATE1.yaml",
        "likelihood": "des_y3.cosmic_shear",
        "tatt_dataset": "tatt_des_y3.dataset",
        "nla_dataset": "synthetic_des_y3.dataset",
    },
    "example2": {
        "frozen_module": "frozen_config_example2.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "likelihood": "des_y3.combo_3x2pt",
        "tatt_dataset": "tatt_des_y3.dataset",
        "nla_dataset": "synthetic_des_y3.dataset",
    },
    "example2_2x2pt": {
        "frozen_module": "frozen_config_example2_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE2.yaml",
        "source_likelihood": "des_y3.combo_3x2pt",
        "likelihood": "des_y3.combo_2x2pt",
        "tatt_dataset": "tatt_des_y3.dataset",
        "nla_dataset": "synthetic_des_y3.dataset",
    },
    "example3": {
        "frozen_module": "frozen_config_example3.py",
        "provenance": "EXAMPLE_EVALUATE3.yaml",
        "likelihood": "des_y3.cosmic_shear",
        "tatt_dataset": "tatt_des_y1.dataset",
        "nla_dataset": "synthetic_des_y1.dataset",
    },
    "example4": {
        "frozen_module": "frozen_config_example4.py",
        "provenance": "EXAMPLE_EVALUATE4.yaml",
        "likelihood": "des_y3.combo_3x2pt",
        "tatt_dataset": "tatt_des_y1.dataset",
        "nla_dataset": "synthetic_des_y1.dataset",
    },
    "example4_2x2pt": {
        "frozen_module": "frozen_config_example4_2x2pt.py",
        "provenance": "EXAMPLE_EVALUATE4.yaml",
        "source_likelihood": "des_y3.combo_3x2pt",
        "likelihood": "des_y3.combo_2x2pt",
        "tatt_dataset": "tatt_des_y1.dataset",
        "nla_dataset": "synthetic_des_y1.dataset",
    },
}

# Pass limit on the covariance-weighted difference of the two FAST-PT
# implementations (cfastpt, the C one inside CosmoLike, and the python
# FAST-PT package). At each point both blocks print their theory data
# vector, and the tested number is delta^T C^-1 delta, with
# delta = dv(FASTPT low) - dv(CFASTPT) and C^-1 the masked inverse
# covariance: the chi2 of the implementation difference, zero when the
# vectors agree. The raw chi2 values are printed only as information:
# across the IA prior they are large, so their difference follows the
# local slope of the chi2 and measures the distance from the data, not
# the numerics. The limit 0.2 equals CHI2_TOLERANCE, the tolerance of
# the reference tests. FASTPT_LOW_SETTINGS, the converged two-grid
# configuration, stays far below it: this project's sweep measured a
# largest delta chi2 of 0.000948 with it, and 4.3 across the prior with
# a single grid.
FASTPT_COMPARISON_TOLERANCE = 0.2

# The python FAST-PT side has numerical settings of its own, read by
# the fastpt theory block from its extra_args block
# (external_modules/code/PyFAST-PT/fastpt.py, symlinked into cobaya
# as theories/fastpt). The block computes on two grids: accuracyboost
# multiplies the density of the output table cosmolike reads with
# linear interpolation (the accuracy driver), and
# internal_accuracyboost the density of the internal grid the FFTLog
# convolutions run on; a cubic spline in log k upsamples the terms
# from one grid onto the other. Both boosts default to 1.0, the
# converged configuration, so low equals the default; the values are
# written out here so the test keeps evaluating this configuration if
# a default changes. High doubles both boosts, so the advisory column
# shows the residual grid response of low. kmax_boltzmann [1/Mpc] and
# extrap_kmax [1/Mpc] are the same in both, so only the FAST-PT grids
# differ.
FASTPT_LOW_SETTINGS = {
    "accuracyboost": 1.0,
    "internal_accuracyboost": 1.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

FASTPT_HIGH_SETTINGS = {
    "accuracyboost": 2.0,
    "internal_accuracyboost": 2.0,
    "kmax_boltzmann": 7.5,
    "extrap_kmax": 250.0,
}

# ---- project-independent constants ------------------------------------------

# These are identical in every project and live in the core module.
REQUIRED_OMP_THREADS = _cct.REQUIRED_OMP_THREADS
CHI2_TOLERANCE = _cct.CHI2_TOLERANCE
RACE_TOLERANCE = _cct.RACE_TOLERANCE
RACE_PERTURBATIONS = _cct.RACE_PERTURBATIONS
HIGH_ACCURACY_CAMB_EXTRA_ARGS = _cct.HIGH_ACCURACY_CAMB_EXTRA_ARGS
BARYON_METHODS = _cct.BARYON_METHODS
BARYON_POINT_OVERRIDES = _cct.BARYON_POINT_OVERRIDES

# The 30 CFASTPT-vs-FASTPT comparison points under this project's
# sampled-parameter prefix; the values are identical in every project.
FASTPT_COMPARISON_POINTS = _cct.fastpt_comparison_points("DES")

# The ten Halofit-vs-EE2 comparison cosmologies (checks NL1-NL2); the
# parameters are cosmology-level (omegam/ns/As, no project prefix), so
# the list is shared verbatim by every project.
NONLINEAR_COMPARISON_POINTS = _cct.NONLINEAR_COMPARISON_POINTS

# ---- the harness -----------------------------------------------------------

# One instance binds the shared machinery to this project's data. The
# assignments below give its methods, and the core functions above,
# module-level names: the names the test modules and
# generate_frozen_reference.py call through "u.". Binding a name to an
# existing function is an alias; it replaces no behavior.
_H = _cct.CocoaTestHarness(
    worker_file=__file__,
    interface_module="cosmolike_des_y3_interface",
    examples=EXAMPLES,
    tatt_point=TATT_POINT,
    accuracy_knobs=ACCURACY_KNOBS,
    high_accuracy_likelihood=HIGH_ACCURACY_LIKELIHOOD,
    fastpt_low_settings=FASTPT_LOW_SETTINGS,
    fastpt_high_settings=FASTPT_HIGH_SETTINGS,
    fastpt_points=FASTPT_COMPARISON_POINTS,
    fastpt_masks=("frozen", "ones"),
)

# ---- module functions re-exported from the core (no project state) ----------
require_cocoa_environment = _cct.require_cocoa_environment
assert_omp_threads = _cct.assert_omp_threads
sha256_of = _cct.sha256_of
make_model = _cct.make_model
evaluate_chi2 = _cct.evaluate_chi2
_evaluate_cached = _cct._evaluate_cached
_load_datavector = _cct._load_datavector
_baryon_method = _cct._baryon_method
_baryon_dataset = _cct._baryon_dataset
report_chi2_test = _cct.report_chi2_test
report_race_test = _cct.report_race_test
report_accuracy = _cct.report_accuracy
report_knob = _cct.report_knob
report_fastpt_comparison = _cct.report_fastpt_comparison
report_nonlinear_comparison = _cct.report_nonlinear_comparison

# ---- bound methods of the harness (the machinery, project-bound) ------------
compute_manifest = _H.compute_manifest
verify_frozen = _H.verify_frozen
load_reference = _H.load_reference
_frozen_module = _H._frozen_module
load_frozen_info = _H.load_frozen_info
load_frozen_point = _H.load_frozen_point
build_point = _H.build_point
_single_model_chi2_impl = _H._single_model_chi2_impl
_ten_in_a_row_impl = _H._ten_in_a_row_impl
_baryon_accuracy_delta_impl = _H._baryon_accuracy_delta_impl
_baryon_drift_chi2_impl = _H._baryon_drift_chi2_impl
single_model_chi2 = _H.single_model_chi2
ten_in_a_row_chi2 = _H.ten_in_a_row_chi2
baryon_accuracy_delta = _H.baryon_accuracy_delta
baryon_drift_chi2 = _H.baryon_drift_chi2
_worker = _H._worker
_run_isolated = _H._run_isolated
_fastpt_comparison_info = _H._fastpt_comparison_info
_fastpt_comparison_block = _H._fastpt_comparison_block
_run_fastpt_comparison_worker = _H._run_fastpt_comparison_worker
cfastpt_vs_fastpt_chi2s = _H.cfastpt_vs_fastpt_chi2s
# the worker subprocess of the nonlinear comparison calls
# u._nonlinear_comparison_block by name, so this re-export is required
_nonlinear_comparison_block = _H._nonlinear_comparison_block
halofit_vs_ee2_dchi2s = _H.halofit_vs_ee2_dchi2s
