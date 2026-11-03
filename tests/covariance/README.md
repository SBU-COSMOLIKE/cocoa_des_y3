# Covariance tests

These tests require the optional covariance build. Follow the
[project build instructions](../../README.md#computing_covariances): unset
`IGNORE_COSMOLIKE_DES_Y3_COVARIANCE` after activating Cocoa and rebuild.
With the default data-vector-only build this sector reports skips; the
separate `tests/data_vector` suite remains available.

The sector runs the shared
[project forecast check](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#cov_forecast_check)
against this project's galaxy/shear forecast adapter and compiled
covariance bindings, with this project's redshift files and catalog
inputs.

This project's run of the shared check pins these specifics:

- Accuracy refinement: `accuracy_boost: 2` with the 96-node quadrature
  unchanged, and `integration_accuracy: 1` selecting 128 nodes.
- Component sum: the total equals G + SSC + cNG to a relative 2e-15.
- Production backend: the production (CLI) backend reproduces the
  notebook-wrapper matrices bitwise.
- Output archive: the saved total and row map reload without pickle,
  with the resolved survey metadata checked.

The independent component references live in
[LSST Y1's covariance suite](https://github.com/CosmoLike/cocoa_lsst_y1/blob/main/tests/covariance/README.md);
use the [covariance notebook](../../EXAMPLE_EVALUATE_COVARIANCE.ipynb)
for full matrices and numerical refinement. The
[data-vector tests](../data_vector/README.md) check likelihood signals
and stored reference values separately.

We assume Cocoa is installed, users have run `conda activate cocoa`, the
shell is Bash and the current folder is `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile this project's interface with covariance enabled.

    unset IGNORE_COSMOLIKE_DES_Y3_CODE
    unset IGNORE_COSMOLIKE_DES_Y3_COVARIANCE
    source projects/des_y3/scripts/compile_des_y3.sh

**Step :three:**: run the covariance checks.

    python -m pytest projects/des_y3/tests/covariance
