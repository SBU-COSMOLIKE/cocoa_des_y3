"""Compare the candidate matter emulator with CAMB at random des_y3 cosmologies.

Run from activated Cocoa:
    python projects/des_y3/tests/compare_emulator_camb.py

The shared Cocoa test uses the numbered example's cosmological priors,
restricted to 0.2 < Omega_m < 0.5 and fixed mnu=0.06 eV. Nuisance parameters
stay at its declared point. Both calculations use plain mead2020 and CAMB
background distances. This does not change the production nonlinear
prescription or its sampled neutrino-mass prior.

Use EMULATOR_CODE_DIR and EMULATOR_DATA_DIR to select candidate checkouts.
COCOA_TEST_INTERFACE_DIR optionally selects an independently built native
interface. Results, including failed points, go under chains/.

The optional --points argument reads explicit named-column parameter points.
The default Fisher calculation varies the sampled cosmological and nuisance
parameters for the selected example's retained observable. It also selects
and evaluates 25 focused points from the bounded Fisher distribution.
--local-points replaces that automatic selection with a supplied table.
The E5 official test uses --advisory, retaining all accuracy outcomes.
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
from cocoa_testing import (
    run_matter_emulator_prior_check,
    plot_matter_emulator_fisher,
    sample_matter_emulator_fisher,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="EXAMPLE_EVALUATE1.yaml")
    parser.add_argument("--models", type=int, default=50)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--points", type=Path, help="Named-column parameter table instead of random draws")
    parser.add_argument("--no-fisher", action="store_true",
                        help="Skip the derivative and contour stage")
    parser.add_argument("--output", type=Path,
                        help="New result directory below project chains/")
    parser.add_argument("--plot-only", type=Path,
                        help="Redraw a completed comparison directory")
    parser.add_argument("--fisher-sample-power", type=int, default=18,
                        help="log2 bounded-Fisher proposals per repetition")
    parser.add_argument("--resample-fisher", action="store_true",
                        help="With --plot-only, repeat bounded sampling")
    parser.add_argument("--advisory", action="store_true",
                        help="Report error sizes; fail only incomplete calculations")
    parser.add_argument("--local-points", type=Path,
                        default=None,
                        help="Override the automatically selected focused Fisher points")
    args = parser.parse_args()
    if args.models < 1 or args.limit <= 0:
        parser.error("models and limit must be positive")
    points_file = args.points.resolve() if args.points is not None else None
    if args.plot_only is not None:
        output = args.plot_only.resolve()
        if args.resample_fisher:
            sample_matter_emulator_fisher(
                output=output / "fisher",
                sample_power=args.fisher_sample_power,
            )
        local = output / "local"
        plot_matter_emulator_fisher(
            output=output / "fisher",
            points_file=output / "points.txt",
            distances_file=output / "distances.txt",
            project_name=project.name,
            local_points_file=local / "points.txt" if local.is_dir() else None,
            local_distances_file=local / "distances.txt" if local.is_dir() else None,
        )
        sys.exit(0)
    output_dir = args.output.resolve() if args.output is not None else None
    local_points = args.local_points.resolve() if args.local_points is not None else None
    os.chdir(root)
    failed = run_matter_emulator_prior_check(
        project=project,
        configuration=args.configuration,
        get_precision=ci.get_inv_cov_masked,
        n_models=args.models,
        seed=args.seed,
        limit=args.limit,
        fisher=not args.no_fisher,
        output_dir=output_dir,
        fisher_sample_power=args.fisher_sample_power,
        points_file=points_file,
        local_points_file=local_points,
        advisory=args.advisory,
    )
    sys.exit(1 if failed else 0)
