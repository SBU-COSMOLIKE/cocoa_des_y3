# Covariance tests

These tests require the optional covariance build. Follow the
[project build instructions](../../README.md#computing_covariances): unset
`IGNORE_COSMOLIKE_DES_Y3_COVARIANCE` after activating Cocoa and rebuild.
With the default data-vector-only build this sector reports skips; the
separate `tests/data_vector` suite remains available.

These checks exercise this project's galaxy/shear forecast adapter and
compiled covariance bindings. They use the project's redshift files and
catalog inputs, with small numerical grids and three measured rows.

We assume Cocoa is installed, users have run `conda activate cocoa`, the
shell is Bash and the current folder is `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile this project's interface.

    unset IGNORE_COSMOLIKE_DES_Y3_CODE
    unset IGNORE_COSMOLIKE_DES_Y3_COVARIANCE
    source projects/des_y3/scripts/compile_des_y3.sh

**Step :three:**: run the covariance checks.

    python -m pytest projects/des_y3/tests/covariance

| Check | Purpose |
| --- | --- |
| Project layout | Verify the full angular and Fourier vector lengths from the measured row map. |
| Accuracy refinement | Keep measurement bins fixed while refining the tables (`accuracy_boost: 2`, with the 96-node quadrature unchanged) and the quadrature (`integration_accuracy: 1`, 128 nodes). |
| Real-space components | Check finite, symmetric G, SSC, cNG and total matrices for a measured subset. |
| Fourier components | Check the same properties for bandpowers. |
| Component sum | Check that the total equals G + SSC + cNG to a relative 2e-15. |
| Production backend | Check that the production (CLI) backend reproduces the notebook-wrapper matrices bitwise. |
| Thread repeatability | Compare every component bitwise with one and eight OpenMP threads. |
| Positive total | Check variance positivity for every direction of the tested subset. |
| Output archive | Reload the saved total and row map without pickle and check the resolved survey metadata. |

The [shared component tests](https://github.com/CosmoLike/cocoa_lsst_y1/blob/main/tests/covariance/README.md)
contain independent algebra and projection references. This project check
covers its binding and inputs; it does not duplicate those references.
Use the [covariance notebook](../../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
for full matrices and numerical refinement. Subset positivity alone does
not establish full-matrix positivity or parameter-error convergence.

The [data-vector tests](../data_vector/README.md) check likelihood signals
and stored reference values separately. No stored covariance or reference
value is replaced by this test.
