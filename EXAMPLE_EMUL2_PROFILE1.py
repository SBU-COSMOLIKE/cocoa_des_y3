"""Parameter profile of the des_y3 hybrid example 1 (cosmic shear, TATT).

In a hybrid example (use_emulator: 2 in the likelihood block), trained
emulators replace the Boltzmann code for the background expansion and the
matter power spectra, while CosmoLike still computes the survey
projections, galaxy bias and intrinsic alignments. Example 1 reads
EXAMPLE_EMUL2_EVALUATE1.yaml: the des_y3.cosmic_shear likelihood with the
TATT intrinsic-alignment model on des_y3_real.dataset.

The profile mode fixes one sampled parameter (--profile, a name or a
zero-based index) at each value of a grid and repeats the annealed
minimization over the other parameters; the minimized score as a function
of the fixed value is the profile. The grid is centered on a saved
minimization, so outside --check the run needs --minfile, the .json record
written by EXAMPLE_EMUL2_MINIMIZE1.py.

This file only chooses the mode and the example number. The work is done
by run() in cosmolike_core/cocoa_hybrid_sampling.py, whose module
docstring (printed by --help) explains the profile mode and every
command-line option.

Run from the Cocoa/ folder after `source start_cocoa.sh`:

    python ./projects/des_y3/EXAMPLE_EMUL2_PROFILE1.py --check
    mpirun -n 2 --bind-to none python \\
        ./projects/des_y3/EXAMPLE_EMUL2_PROFILE1.py \\
        --minfile ./projects/des_y3/chains/EXAMPLE_EMUL2_MINIMIZE1.json

--check evaluates the fiducial point, prints the sampled-parameter order
and stops without sampling. Results go to projects/des_y3/chains/, named
by --outroot (default EXAMPLE_EMUL2_PROFILE1); an existing result is never
overwritten. The project README (Running Hybrid Cosmolike-ML emulators)
lists the MPI and multi-node commands.
"""

from pathlib import Path
import sys

# The folder of this file is the project folder (projects/des_y3);
# project.parents[1], two levels up, is the Cocoa/ folder.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
# sys.path is the list of folders Python searches on import: putting
# cosmolike_core first lets the import below find the shared driver.
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# __name__ is "__main__" only when this file runs as a script, so an
# import of this file does not start a run.
if __name__ == "__main__":
    run(mode="profile", project=project, example=1)
