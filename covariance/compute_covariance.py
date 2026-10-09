"""Compute the des_y3 forecast covariance from the command line and save it.

The likelihoods never read the matrix this program writes. They use the
published DES Y3 covariance named by the cov_file key of
data/des_y3_real.dataset (des_y3_cov_unblinded_final.txt). This program
computes a forecast covariance in the same measurement layout (900
real-space entries: 20 angular bins from 2.5 to 250 arcmin, 5 lens and 4
source bins), so the two matrices can be compared entry by entry. Its
physical choices (massless neutrinos, linear galaxy bias, a spherical-cap
footprint) differ from the published analysis, so an equal layout does
not mean an equal matrix; covariance/README.md lists those choices.

The matrix has three parts, saved separately and as their sum: the
Gaussian part G (the covariance a Gaussian density field would have, shape
and shot noise included), the super-sample covariance SSC (from density
modes larger than the survey) and the connected non-Gaussian part cNG (the
connected four-point function inside the survey). The program reads one
evaluate YAML with Cobaya's YAML reader (no sampler runs), runs CAMB once
and calls the C kernels compiled into the project interface; the survey
numbers come from des_y3_covariance.py, the same adapter the notebook
EXAMPLE_EVALUATE_COVARIANCE.ipynb uses.

From the Cocoa/ folder, after `source start_cocoa.sh`, with covariance
generation compiled in (unset IGNORE_COSMOLIKE_DES_Y3_COVARIANCE, then
recompile with scripts/compile_des_y3.sh):

    python projects/des_y3/covariance/compute_covariance.py \\
        projects/des_y3/EXAMPLE_EVALUATE_COVARIANCE.yaml

The result is the .npz archive (a numpy file holding several named arrays)
named by the YAML's output key; --output overrides the name, and an
existing archive is replaced only with --overwrite. --help lists the
options; covariance/README.md explains the space and accuracy settings.
"""

import os
from pathlib import Path
import sys

# OpenBLAS, MKL and Apple's vecLib are BLAS (linear algebra) libraries; they
# read these variables when they load, so the limit of one thread must be
# set before their first import. CosmoLike's own OpenMP threads are set by
# OMP_NUM_THREADS in the shell, not here.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# This runner evaluates one matrix in one process. Cobaya supplies the YAML
# reader; it does not launch MPI workers or a sampler for this calculation.
os.environ["COBAYA_NOMPI"] = "1"

# project is the des_y3 folder, one level above this file's covariance/
# folder; project.parents[1], two levels above it, is the Cocoa/ folder.
# sys.path is the list of folders Python searches on import: the shared
# core and the folder of the compiled interface go first. The adapter
# des_y3_covariance is found because Python also puts the folder of the
# running script (covariance/) on sys.path.
project = Path(__file__).resolve().parents[1]
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))
sys.path.insert(0, str(project/"interface"))

import cosmolike_des_y3_interface as ci
import des_y3_covariance as survey
from cosmolike_notebook_utils.covariance.command_line import run_covariance


# __name__ is "__main__" only when this file runs as a script. The real
# angular-bin measurement is the native DES Y3 space (YAML covariance.space
# can select "fourier"); joint=False selects the galaxy/shear adapter rather
# than the cluster 6x2pt+N one.
if __name__ == "__main__":
    run_covariance(
        interface=ci, survey=survey, default_space="real", joint=False,
    )
