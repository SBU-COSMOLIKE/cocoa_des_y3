"""Compare the candidate matter emulator with CAMB at random project cosmologies.

Run from activated Cocoa:
    python projects/des_y3/tests/compare_emulator_camb.py

The shared Cocoa test uses the numbered example's cosmological priors,
restricted to 0.2 < Omega_m < 0.5 and fixed mnu=0.06 eV. Nuisance parameters
stay at its declared point. Both calculations use plain mead2020 and CAMB
background distances. This does not change the production nonlinear prescription
or its sampled neutrino-mass prior.

Use EMULATOR_CODE_DIR and EMULATOR_DATA_DIR to select candidate checkouts.
COCOA_TEST_INTERFACE_DIR optionally selects an independently built native
interface. Results, including failed points, go under chains/.
"""

import argparse
import os
from pathlib import Path
import sys

# Set thread ownership before NumPy, TensorFlow or the native library loads.
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

project = Path(__file__).resolve().parents[1]
root = project.parents[1]
core = root / "external_modules/code/cosmolike_core"
interface = Path(os.environ.get("COCOA_TEST_INTERFACE_DIR", project / "interface"))
sys.path.insert(0, str(core))
sys.path.insert(0, str(interface))
sys.path.insert(0, str(root / "cobaya"))

import cosmolike_des_y3_interface as ci
from cocoa_testing import run_matter_emulator_prior_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="EXAMPLE_EVALUATE2.yaml")
    parser.add_argument("--models", type=int, default=50)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--limit", type=float, default=1.0)
    args = parser.parse_args()
    if args.models < 1 or args.limit <= 0:
        parser.error("models and limit must be positive")
    os.chdir(root)
    failed = run_matter_emulator_prior_check(
        project=project,
        configuration=args.configuration,
        get_precision=ci.get_inv_cov_masked,
        n_models=args.models,
        seed=args.seed,
        limit=args.limit,
    )
    sys.exit(1 if failed else 0)
