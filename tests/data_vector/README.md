# Unit tests for the likelihoods

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads).

The project ships two generations of DES data, and the tests cover
both: tests 1-8, 11-14 and 27-30 run on the DES-Y3 data set
(`des_y3_real.dataset`), tests 15-26 on the DES-Y1 data set
(`des_y1_real.dataset`), through the same `des_y3.*` likelihood
classes. Each uses its data set's covariance, mask and n(z) with a
synthetic data vector ([Appendix](#synthetic_vectors)). Numbers 9 and
10 are unused.

Most checks build each model in its own worker subprocess: the Y1 and
Y3 data sets have different masked dimensions and n(z) table lengths,
and the cosmolike C layer aborts the whole process when a second
configuration with different dimensions initializes after the first.
The photo-z, non-Limber and cache checks build their models in the
pytest process itself. The isolation is internal; the commands below
stay the same.

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The CFASTPT vs FASTPT comparison](#cfastpt_fastpt)
    2. [The Halofit vs EE2 checks](#halofit_ee2)
    3. [The EE2 race test](#ee2_race)
    4. [Accuracy checks](#accuracy_checks)
    5. [Baryonic feedback accuracy checks](#baryon_accuracy_checks)
    6. [Baryonic feedback drift tests](#baryon_drift_tests)
    7. [The photo-z convention checks](#photoz_conventions)
    8. [The non-Limber galaxy-galaxy lensing check](#nonlimber_ggl)
    9. [The non-Limber galaxy clustering check](#nonlimber_gg)
    10. [The sector-ladder cache check](#cache_ladder)
3. [Appendix](#appendix)
    1. [FAQ: Do the tests keep their own data?](#frozen_copy)
    2. [FAQ: Why do the tests use their own data vectors?](#synthetic_vectors)
    3. [FAQ: Why does the non-Limber clustering subtract a separable-growth term?](#gg_growth_fix)
    4. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the tests of this project

    python -m pytest ./projects/des_y3/tests/data_vector

Without pytest:

    python -m unittest discover -s ./projects/des_y3/tests/data_vector -v

Under unittest the `--high` and `--mask` options below do not exist;
the FAST-PT and Halofit vs EE2 checks then run at their defaults
unless `COCOA_FASTPT_HIGH` and `COCOA_FASTPT_MASK` are exported.

The tests change no project files. Each test prints a progress line
per model build and per evaluation, then a report with the
computed $\chi^2$, the stored reference, the difference, and the pass
limit.

A full run performs about 600 likelihood evaluations; the
high-accuracy checks alone take minutes each. The test files force
`OMP_NUM_THREADS=4` internally.

## The tests <a name="the_tests"></a>

The standard configurations get four tests each: a $\chi^2$ drift check
and a race-condition check, both in the NLA and in the TATT intrinsic-alignment
model. The TATT variants set

    IA_model: 1
    DES_A2_1: 0.05
    DES_BTA_1: 0.05
    DES_A2_2: -1.51541

The two checks and their pass limits:

| check | pass limit                                        | a failure means                    |
|-------|---------------------------------------------------|------------------------------------|
| $\Delta\chi^2$ | the recomputed $\chi^2$ must stay within 0.2 of the value stored in `frozen/reference_chi2.json` | code or data changed the numbers |
| race condition | the fiducial evaluated on its own vs evaluated again after nine other cosmologies; the two must agree within $10^{-4}$ | leftover state or an OpenMP race |

Everything the tests compare against lives under `frozen/`: one
snapshot of configurations, data, and reference values, written by
`generate_frozen_reference.py` (the baryonic-feedback vectors by its
separate `--baryons` run). The [Appendix](#appendix) explains how the
snapshot is protected.

The checks split into asserted regressions, which fail when a number
moves, and advisory reports, which print numbers without pass/fail:

```mermaid
flowchart TD
  A["generate_frozen_reference.py: --overwrite, then --baryons"] --> B["frozen/: configurations, data, reference chi2"]
  B --> C["Asserted: tests 1-30, BD1-BD7, photo-z, non-Limber, cache"]
  B --> D["Advisory: A0-A12, BF0-BF7, NL1-NL2"]
```

The test files and the configurations they cover:

| test | file | data | configuration | what it checks |
|---|---|---|---|---|
| 1 | `test_example1.py` | DES-Y3 | cosmic shear; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 2 | `test_example1.py` | DES-Y3 | cosmic shear; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 3 | `test_example1.py` | DES-Y3 | cosmic shear; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 4 | `test_example1.py` | DES-Y3 | cosmic shear; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 5 | `test_example2.py` | DES-Y3 | 3x2pt; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 6 | `test_example2.py` | DES-Y3 | 3x2pt; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 7 | `test_example2.py` | DES-Y3 | 3x2pt; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 8 | `test_example2.py` | DES-Y3 | 3x2pt; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 11 | `test_example2_2x2pt.py` | DES-Y3 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 12 | `test_example2_2x2pt.py` | DES-Y3 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 13 | `test_example2_2x2pt.py` | DES-Y3 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 14 | `test_example2_2x2pt.py` | DES-Y3 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 15 | `test_example3.py` | DES-Y1 | cosmic shear; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 16 | `test_example3.py` | DES-Y1 | cosmic shear; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 17 | `test_example3.py` | DES-Y1 | cosmic shear; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 18 | `test_example3.py` | DES-Y1 | cosmic shear; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 19 | `test_example4.py` | DES-Y1 | 3x2pt; IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 20 | `test_example4.py` | DES-Y1 | 3x2pt; IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 21 | `test_example4.py` | DES-Y1 | 3x2pt; IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 22 | `test_example4.py` | DES-Y1 | 3x2pt; IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 23 | `test_example4_2x2pt.py` | DES-Y1 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: NLA | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 24 | `test_example4_2x2pt.py` | DES-Y1 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: NLA | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 25 | `test_example4_2x2pt.py` | DES-Y1 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: TATT | $\Delta\chi^2$ against the stored reference at the fiducial point |
| 26 | `test_example4_2x2pt.py` | DES-Y1 | 2x2pt (`des_y3.combo_2x2pt`); IA modeling: TATT | race condition (OpenMP threading): fiducial alone vs after nine other cosmologies |
| 27 | `test_fastpt.py` | DES-Y3 | cosmic shear; IA modeling: TATT; the C cfastpt (`IA_code: 0`) vs the python FAST-PT package (`IA_code: 1`) at 30 fixed points (20 across the intrinsic-alignment prior plus a one-parameter-at-a-time family), cosmology at the fiducial | $\Delta\chi^2$ of the FAST-PT data vector against the cfastpt data vector at the same point; the cfastpt vector is that point's fiducial, so agreement means zero |
| 28 | `test_fastpt.py` | DES-Y3 | 3x2pt; IA modeling: TATT; the same comparison as test 27 on the 3x2pt likelihood (`des_y3.combo_3x2pt`) | the same pass rule as test 27, with the data-vector difference weighted by the 3x2pt masked inverse covariance |
| 29 | `test_fastpt.py` | DES-Y3 | 2x2pt; IA modeling: TATT; the same comparison as test 27 on the 2x2pt likelihood (`des_y3.combo_2x2pt`) | the same pass rule as test 27; clustering carries no intrinsic alignment, so the TATT tables enter through galaxy-galaxy lensing alone, weighted by the 2x2pt masked inverse covariance |
| 30 | `test_ee2.py` | DES-Y3 | cosmic shear; IA modeling: NLA; nonlinear $P(k)$ from EE2 (`non_linear_emul: 1`) | race condition (OpenMP threading, including EE2's own threaded compute): fiducial alone vs after nine other cosmologies |

### The CFASTPT vs FASTPT comparison (`test_fastpt.py`, tests 27-29) <a name="cfastpt_fastpt"></a>

Cosmolike computes the TATT perturbation-theory integrals with two
implementations: cfastpt, the C code built into the interface
(`IA_code: 0`), and the python FAST-PT package through the fastpt
theory block (`IA_code: 1`). Both are evaluated at 30 fixed points
across the intrinsic-alignment prior and checked for agreement:
test 27 on cosmic shear, test 28 on the 3x2pt likelihood, test 29
on the 2x2pt likelihood.

At every point the cfastpt data vector is the fiducial: the reported
quantity is the $\Delta\chi^2$ of the FAST-PT vector against it,
zero for identical vectors and quadratic in their difference. A
comparison against the shipped data vector would measure the slope
of the distance to the data instead of the numerics.

The fastpt block computes on two grids: `accuracyboost` scales the
density of the output table cosmolike reads with linear
interpolation (the accuracy driver), and `internal_accuracyboost`
sets the internal grid the FFTLog convolutions run on (1,100 points,
plus 200 per unit above 1), with a cubic spline in log k upsampling
the terms from one grid onto the other.

`accuracyboost` is rebased so 1.0 gives the converged output table.
The tests run FAST-PT at the defaults with the 0.2 band of the other
checks as the pass limit; a doubled configuration repeats the
measurement as an advisory.

In test 28 the TATT terms also enter galaxy-galaxy lensing, and the
difference is weighted by the 3x2pt masked inverse covariance. The
frozen configuration fixes the one-loop bias amplitudes
(`DES_B2_*`) at zero, so the one-loop galaxy-bias tables both
implementations compute multiply by zero: the sweep scores the
intrinsic-alignment tables alone, on the wider data vector.

Test 29 repeats the sweep on the 2x2pt likelihood. Clustering
carries no intrinsic alignment, so there the TATT tables are scored
through galaxy-galaxy lensing alone, under the 2x2pt masked
inverse covariance.

> [!NOTE]
> With one shared grid and no upsampling (the first row of the table
> below), the difference reaches $\Delta\chi^2 = 4.3$ across the prior.

The point values, the design, and the decision record live with the
lsst_y1 project (its tests/README.md carries the full discussion);
the table below is this project's own measurement:

| output table (points) | internal grid (points) | max $\Delta\chi^2$ | cost per cosmology |
|---|---|---|---|
| 1,100 | 1,100 (shared) | 4.32 | 1.0 s |
| 1,024,900 (`accuracyboost: 1`, the default) | 1,100 (the default) | 0.000948 | 1.4 s |
| 2,048,900 (`accuracyboost: 2`) | 1,300 (`internal_accuracyboost: 2`) | 0.000892 | 1.6 s |

![The 30 comparison points, colored by the per-point difference](../cfastpt_vs_fastpt_points.png)

Measured:

- Test 28 (3x2pt): max $\Delta\chi^2 = 0.0010$ at the defaults,
  $0.000013$ at the pushed camb/cosmolike settings.
- Test 29 (2x2pt): $0.00010$ and $0.000001$, the mildest of the
  three, with the TATT tables entering through galaxy-galaxy
  lensing alone.
- `--mask=ones` (no scale cuts, all 900 points weighted): every
  sweep still passes. Cosmic shear measures max
  $\Delta\chi^2 = 0.055$ at the default camb/cosmolike settings,
  3x2pt 0.062, and 2x2pt 0.011; at the pushed settings they fall
  to 0.00004, 0.00006, and 0.00003.

The default-settings values are insensitive to the FAST-PT boosts
(cosmic shear's 0.055 is fifty-seven times its baseline-mask
value): the unmasked small scales amplify cosmolike's default
integration error, not a FAST-PT grid deficiency.

> [!NOTE]
> The fastpt defaults hold this accuracy on their own; raising the
> boosts is a convergence test, not a need. cfastpt (`IA_code: 0`)
> remains the reference implementation.

#### Running the comparison <a name="run_cfastpt_fastpt"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the comparison at the default camb/cosmolike
settings

    python -m pytest ./projects/des_y3/tests/data_vector/test_fastpt.py

**Step :three:**: repeat it at the pushed camb/cosmolike settings

    python -m pytest ./projects/des_y3/tests/data_vector/test_fastpt.py --high=1

> [!NOTE]
> `--high=1`: applies the pushed camb/cosmolike settings of the
> accuracy checks to every block of tests 27-29 (the other tests do
> not read it). The full comparison is both invocations.

> [!NOTE]
> `--mask`: reruns tests 27-29, and the Halofit vs EE2 checks NL1-NL2,
> under a different scale-cut mask. `--mask=frozen` (the default) keeps
> the baseline mask of the frozen contract (533 of the 900 data
> points); `--mask=ones` keeps every point (no scale cuts), the
> strictest comparison. `ones` selects a frozen TATT dataset variant
> that differs from the base one only in its `mask_file` line, and the
> 0.2 pass rule applies unchanged.


### The Halofit vs EE2 checks (`test_nonlinear.py`, NL1-NL2) <a name="halofit_ee2"></a>

The likelihoods can source the nonlinear matter power from CAMB's
Takahashi halofit (`non_linear_emul: 2`, the frozen contract's
setting) or from EuclidEmulator2 (`non_linear_emul: 1`). NL1
evaluates the DES-Y3 cosmic-shear data vector and NL2 the DES-Y3
3x2pt data vector with both sources at ten fixed cosmologies across
the omegam/ns/As space, in the NLA configuration; every other
parameter keeps the frozen fiducial.

At every cosmology the EE2 data vector is the fiducial: the reported
quantity is the $\Delta\chi^2$ of the Halofit vector against it
($\delta^T C^{-1} \delta$), zero for identical vectors. Each
cosmology carries its own regenerated fiducial, so no stored data
vector enters the metric.

The checks are advisory: no pass/fail. The numbers say how much of
the statistical error budget the Halofit-vs-emulator difference
consumes under the chosen scale cuts. The `--mask` option of the
comparison sweeps applies: `--mask=frozen` (the default) weights the
difference with the contract mask, `--mask=ones` with every data
point kept (no scale cuts).

Measured (the figure below, frozen mask):

- NL1 (cosmic shear): max $\Delta\chi^2 = 11.5$, median $1.4$;
  max $21.5$, median $2.4$ under `--mask=ones`.
- NL2 (3x2pt): max $53.0$, median $6.0$; max $243.0$, median
  $42.6$ under `--mask=ones`.

![The ten cosmologies, colored by the Halofit-vs-EE2 difference](../halofit_vs_ee2_points.png)

#### Running the Halofit vs EE2 checks <a name="run_halofit_ee2"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the checks under the frozen contract mask

    python -m pytest ./projects/des_y3/tests/data_vector/test_nonlinear.py

**Step :three:**: repeat them with every data point kept

    python -m pytest ./projects/des_y3/tests/data_vector/test_nonlinear.py --mask=ones

### The EE2 race test (`test_ee2.py`, test 30) <a name="ee2_race"></a>

Cocoa pins a modified EuclidEmulator2: OpenMP threading, a
1,010-redshift capacity, the `get_boost2` API with a pre-built
emulator, memory-leak fixes, and a bilinear interpolation with a
border fix. The modifications and their measured speed-up are
documented in the repository's own README
(`external_modules/code/euclidemu2/README.md`).

Test 30 is the race check with EE2 on: the fiducial evaluated fresh
and again as the 10th of 10 cosmologies on one model instance, on
the DES-Y3 cosmic-shear configuration with the nonlinear $P(k)$
from EE2 (`non_linear_emul: 1`). Every other race check in the
table runs with halofit (`non_linear_emul: 2`, the frozen
contract's setting), so only this row executes EE2's
OpenMP-threaded compute; a thread race or leaked state inside it
shifts the second fiducial value, and the two must agree within
$10^{-4}$.

Measured:

- Test 30: the fresh and 10th-in-a-row fiducial agree to all eight
  printed decimals.
- The modified EE2 build against the unmodified EuclidEmulator2, at ten
  fixed cosmologies on this project's cosmic shear under the frozen
  mask: max $\Delta\chi^2 = 3.0\times10^{-8}$, with a max
  fractional data-vector difference of $7.4\times10^{-6}$. The
  modification gate itself runs as the lsst_y1 project's test 18.

### Accuracy checks (`test_accuracy.py`, A0-A12) <a name="accuracy_checks"></a>

Checks A1-A12 re-evaluate cosmic shear, 3x2pt, and 2x2pt, with NLA
and TATT, on the DES-Y3 data (A1-A6) and on the DES-Y1 data
(A7-A12), with every setting pushed far beyond the defaults at
once. Before them, check A0 changes one accuracy parameter at a time
on the DES-Y3 3x2pt NLA configuration, so a large $\Delta\chi^2$ can
be attributed to the parameter causing it. The settings:

| setting | raised to | what it controls |
|---------|-----------|------------------|
| `accuracyboost` (cosmolike) | 3 (A0 also tries 5) | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables |
| `integration_accuracy` (cosmolike) | 10 | extra refinement passes of cosmolike's numerical integrals |
| `internal_accuracyboost` (cosmolike) | 2 | density of the C-FAST-PT convolution grid relative to the output table the likelihood interpolates; 1 keeps the two grids equal |
| `lmax` (cosmolike) | 200000 | highest multipole of the internal harmonic-space $C_\ell$ tables that cosmolike transforms into the real-space correlation functions; arcminute scales need very high $\ell$ |
| `kmax_boltzmann` (cosmolike) | 40 | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | 2 | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | 50 | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | 50 | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

`accuracyboost` refines a nested z grid in the power-spectrum
tables: every coarser grid's nodes are a subset of every finer
grid's, so a higher boost tightens the same interpolation instead of
moving the nodes (the construction is commented in
`likelihood/_cosmolike_prototype_base.py`).

`internal_accuracyboost` scales only the C-FAST-PT convolution
grid; the output table the likelihood interpolates is unchanged.
This project's default, 1.0, keeps the two grids equal: the exact
single-grid path. The lsst_y1 scan measured
$\Delta^T C^{-1} \Delta \le 10^{-9}$ against that path for grid
fractions down to 0.27.

When several settings move the $\chi^2$, settle them in cost order:
raise cosmolike `accuracyboost` first (cheap), then CAMB
`k_per_logint`, and CAMB `AccuracyBoost` last (expensive at run
time, and able to masquerade for the cheap settings).
`kmax_boltzmann` and CAMB `kmax` are one physical cutoff seen from
two sides; move them together.

Each check reports the $\Delta\chi^2$ between the high-accuracy
evaluation and the frozen reference, the default-settings $\chi^2$:
the numerical error of the default settings. No pass/fail.

> [!NOTE]
> High-accuracy evaluations take minutes.

#### Running Accuracy checks <a name="run_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the accuracy checks on their own

    python -m pytest ./projects/des_y3/tests/data_vector/test_accuracy.py

To run every other test while skipping these:

    python -m pytest ./projects/des_y3/tests/data_vector --ignore ./projects/des_y3/tests/data_vector/test_accuracy.py

### Baryonic feedback accuracy checks (`test_accuracy_baryons.py`, BF1-BF7) <a name="baryon_accuracy_checks"></a>

The file `test_accuracy_baryons.py` repeats the default-versus-high
accuracy comparison with the `bfmt` theory block switched on: one
advisory check per feedback method (the three SP(k) fb relations,
BCEmu, Flamingo, BACCOemu, and BCemu2025), at a fixed parameter
point per method.

Each check creates its data vector on the fly: the default-settings
model writes its own theory vector during evaluation, that vector becomes
the data of a temporary dataset, and the pushed-settings model
evaluates at the same point against it. The fiducial $\chi^2$ is
therefore zero by construction, nothing is stored in the snapshot,
and the single reported number, $\Delta\chi^2$, is a pure numerics
response.

The check BF0 additionally runs the one-setting-at-a-time scan with
the Akino SP(k) feedback on, so a large delta names the setting
causing it.

Every checked configuration is measurable by construction. The
BACCOemu check evaluates with `omegab: 0.049`, inside that
emulator's baryon-density training box, whose floor sits exactly
above the fiducial `omegab: 0.04`; and the double-power-law point is
chosen to keep the baryon fraction inside SP(k)'s calibrated band
over the full redshift grid.

#### Running the baryonic feedback checks <a name="run_baryon_accuracy"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the baryonic feedback checks of this project

    python -m pytest ./projects/des_y3/tests/data_vector/test_accuracy_baryons.py

### Baryonic feedback drift tests (`test_baryons.py`, BD1-BD7) <a name="baryon_drift_tests"></a>

The file `test_baryons.py` pins the feedback pipeline against change
over time, one test per method. Each method's default-settings
theory prediction was stored at freeze time
(`generate_frozen_reference.py --baryons`), and the test evaluates
today's prediction against that stored vector: zero at freeze time
by construction, so a $\chi^2$ above the tolerance means cosmolike
or the `bfmt` theory block changed its prediction since the freeze.

These tests complement the accuracy checks above: the accuracy
checks regenerate their vector on the fly per run, so they measure
the numerical settings and can never see drift; the drift tests hold
the frozen vector still, so they measure drift and nothing else.

#### Running the baryonic feedback drift tests <a name="run_baryon_drift"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: run the drift tests of this project

    python -m pytest ./projects/des_y3/tests/data_vector/test_baryons.py


### The photo-z convention checks (`test_photoz_conventions.py`) <a name="photoz_conventions"></a>

The likelihood exposes two runtime knobs for how the n(z) table files
become the smooth distributions the Limber integrals consume, both
declared in the likelihood yamls with the default 0:
`photoz_interpolation_type` (0 = cubic spline,
1 = linear, 2+ = Steffen monotone, which cannot overshoot below zero
around a sharp feature in the table) and `photoz_zmid_convention`
(0 = the z column of the n(z) file holds Z_LOW left bin edges, so
the tabulated value belongs at the cell center z + dz/2; 1 = the
column holds Z_MID sample points). The two z-column readings differ
by a rigid dz/2 shift of every distribution. `scripts/README.md` lists
the DES FITS files the n(z) tables come from and how to write the
tables with either z column.

The test evaluates the frozen cosmic-shear fiducial under five
settings in one process - the default, each alternative, and the
default again - and measures every alternative against the default
with the same second-order construction the CFASTPT sweep uses:
$\Delta\chi^2 = \delta^T C^{-1} \delta$, with $\delta$ the
data-vector difference and $C^{-1}$ the masked inverse covariance.
The assertions are a dead-flag floor on each alternative (a stale
n(z) cache would give exactly zero), the frozen-reference check on
the default, and a bit-identical round trip back to the default (a
cache that fails to rebuild on the way back would fail loudly).

Measured:

- default: the test requires the frozen reference of the cosmic-shear
  NLA configuration, $\chi^2 = 0.000103$.
- linear: $\Delta\chi^2 = 3.1\times10^{-4}$; Steffen:
  $\Delta\chi^2 = 1.6\times10^{-5}$. Either interpolant change is far below
  the survey's statistical precision.
- Z_MID: $\Delta\chi^2 = 0.30$ - the half-bin z-column reading
  is the one photo-z convention that matters at this precision.

The figures below are regenerated by
`generate_photoz_convention_figure.py`:

![The Z_LOW vs Z_MID reading of the n(z) z column](../photoz_zmid_dxi.png)

![Linear and Steffen n(z) interpolation vs cubic spline](../photoz_interp_dxi.png)

### The non-Limber galaxy-galaxy lensing check (`test_nonlimber_ggl.py`) <a name="nonlimber_ggl"></a>

The likelihood yaml key `adopt_limber_gs` chooses how the
galaxy-galaxy lensing spectrum $C_\ell^{gs}$ is computed: `1` (the
default) uses the Limber approximation at every multipole; `0` takes
the multipoles below $\ell = 150$ from the exact projection, computed
by cosmolike's `C_gs_tomo` with the split of Fang, Krause, Eifler &
MacCrann (arXiv:1911.11947): an FFTLog integral of the linear power
spectrum plus, in Limber, what linear theory misses. Galaxy-galaxy
lensing defaults to Limber because its lensing kernel is broad; galaxy
clustering has its own key, `adopt_limber_gg` (next section).

The test evaluates the frozen 3x2pt fiducial with Limber,
non-Limber, and Limber again in one process and reports
$\Delta\chi^2 = \delta^T C^{-1} \delta$, with $\delta$ the data-vector
difference and $C^{-1}$ the masked inverse covariance: the $\chi^2$ the
Limber model would score against a data set generated with non-Limber
galaxy-galaxy lensing. It also prints the contribution of each
lens-source pair. The assertions are a dead-flag floor
on $\Delta\chi^2$, that only galaxy-galaxy lensing entries change, a
bit-identical round trip back to Limber, agreement with the measured
$\Delta\chi^2$ to 5%, and, last, the frozen-reference check on the
Limber evaluation.

Measured:

- $\Delta\chi^2 = 0.012$ for the Y3 3x2pt data vector (example2),
  far below the survey's statistical precision; the largest pairs,
  (3,2) and (4,2), contribute 0.007 and 0.006. The test pins
  $\Delta\chi^2 = 0.01214$ to 5%.

### The non-Limber galaxy clustering check (`test_nonlimber_gg.py`) <a name="nonlimber_gg"></a>

The likelihood yaml key `adopt_limber_gg` chooses how the galaxy
clustering spectrum $C_\ell^{gg}$ is computed: `0` takes the
multipoles below $\ell = 150$ from the exact projection (cosmolike's
`C_cl_tomo`, the same FFTLog split as the galaxy-galaxy lensing check
above), `1` uses the Limber approximation at every multipole; `0`
(non-Limber) is this project's default. The lens galaxy redshift
distributions are narrow, so the Limber approximation fails at low
$\ell$ for the clustering auto spectra.

The test evaluates the frozen 3x2pt fiducial with the default, the
other setting, and the default again in one process and reports
$\Delta\chi^2 = \delta^T C^{-1} \delta$, with $\delta$ the non-Limber
minus the Limber data vector, and the contribution of each lens bin.
The assertions are a dead-flag floor on $\Delta\chi^2$, that only
clustering entries change, a bit-identical round trip back to the
default, agreement with the measured $\Delta\chi^2$ to 5%, and, last,
the frozen-reference check on the default evaluation.

Measured:

- $\Delta\chi^2 = 9.49$ for the Y3 3x2pt data vector (example2), against
  0.012 for the same comparison in galaxy-galaxy lensing. The test pins
  $\Delta\chi^2 = 9.492$ to 5%.
- Lens bins 4, 3, 2, 1, 0 contribute 4.56, 2.44, 1.75, 0.61, 0.19
  (each bin's block alone).


### The sector-ladder cache check (`test_cache_consistency.py`) <a name="cache_ladder"></a>

cosmolike caches every expensive stage behind its own key, and a
partial-invalidation bug - one sector's update path leaving a stale
static another sector consumes - produces wrong data vectors only in
MIXED update sequences, which the per-point checks never exercise.
The test walks a deterministic ladder in one process (three steps in
each sector in turn: cosmology, intrinsic alignment, source photo-z,
lens photo-z and shear calibration), evaluating after every step and
requiring every step to change the data vector, then scrambles every
sector at once, galaxy bias included, and returns to the ladder's
final point: the pipeline must reproduce the recorded data vector and $\chi^2$ bit for
bit, and a second instance walking the mirrored sector order must
land on the same vector. The shear-calibration steps must equal the
analytic $(1+m_i)(1+m_j)$ block rescale to $10^{-12}$, a no-op update
must change nothing, and both intrinsic-alignment models run (the
TATT ladder exercises the FAST-PT rebuild machinery).

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Do the tests keep their own data? <a name="frozen_copy"></a>

The tests read no live data or example files: not `../data` and not
the `EXAMPLE_EVALUATE` yaml files. Cobaya still loads the likelihood
classes with their default yaml files, but the frozen configurations
override every option and parameter they list. Instead, `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_example*.py` | the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariances, n(z), and masks of both data sets, plus the synthetic and baryonic-feedback data vectors the tests evaluate against |
| `reference_chi2.json` | the twelve reference $\chi^2$ values, one per configuration and intrinsic-alignment model |
| `EXAMPLE_EVALUATE{1,2,3,4}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

In the configuration modules every option and every parameter is
written out, including the ones that normally come from
`params_source.yaml` and the other default files, so editing those
files cannot change what the tests evaluate.


`tests/manifest_sha256.json` stores a SHA-256 hash (a fingerprint that
changes when any byte changes) of every file under `frozen/`. Each test
class verifies the manifest before its tests run and refuses to run when
a file under `frozen/` was edited, added or removed, naming the file.
The result: users may change the live data and examples freely, and
nobody can quietly edit the snapshot either.

## :interrobang: FAQ: Why do the tests use their own data vectors? <a name="synthetic_vectors"></a>

This project's shipped data vectors are real data (the DES-Y3 and
DES-Y1 measurements), and the example cosmology is not a best fit of
either, so the $\chi^2$ there sits far from the minimum, where it
responds linearly to tiny numerical changes. Every variant therefore
evaluates against a data vector generated at the fiducial point
when the snapshot was created, one pair per data set:

| data | NLA vector | TATT vector | generated from |
|------|-----------|-------------|----------------|
| DES-Y3 | `frozen/data/synthetic_des_y3.dataset` | `frozen/data/tatt_des_y3.dataset` | the 3x2pt model |
| DES-Y1 | `frozen/data/synthetic_des_y1.dataset` | `frozen/data/tatt_des_y1.dataset` | the DES-Y1 3x2pt model |

Within a data set the full-length 3x2pt vector serves every probe
(the masks select their sections); at its own minimum the $\chi^2$
response is quadratic and the drift and accuracy numbers stay
meaningful.

## :interrobang: FAQ: Why does the non-Limber clustering subtract a separable-growth term? <a name="gg_growth_fix"></a>

Below $\ell = 150$ cosmolike's `C_cl_tomo` computes

$$C_\ell = C_\ell^{\rm FFTLog}(P_{\rm lin}) + C_\ell^{\rm Limber}(P_\delta) - C_\ell^{\rm Limber}(P_{\rm lin}),$$

an exact FFTLog projection of the linear power spectrum plus, in the
Limber approximation, what linear theory misses. Where the Limber
approximation holds (high $\ell$) the first and third terms must
converge to the same number and cancel, leaving the Limber
$C_\ell^{gg}$. The FFTLog term needs a separable linear spectrum,
$(D(a)/D(a_{\rm piv}))^2 P_{\rm lin}(k, a_{\rm piv})$, anchored at a
pivot scale factor $a_{\rm piv}$ of each lens bin, so the subtracted
Limber term uses the identical separable form.

CAMB's $P_{\rm lin}(k, a)$ would not cancel the FFTLog term: massive
neutrinos make its growth scale dependent, and an uncancelled pair
leaves an offset in every $C_\ell^{gg}$ below the switch to Limber.
The scale-dependent part of the growth stays in the Limber $P_\delta$
term, where the split puts it. The non-Limber galaxy-galaxy lensing
spectrum of `C_gs_tomo` (`adopt_limber_gs: 0`) uses the same
separable form.

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/des_y3/tests/generate_frozen_reference.py --overwrite

It deletes `frozen/`, rebuilds it from the current project, prints the
twelve new reference $\chi^2$ values, and rewrites the manifest. It does
not regenerate the baryonic-feedback vectors of the drift tests.

**Step :three:**: regenerate the baryonic-feedback vectors and re-pin
the manifest

    python ./projects/des_y3/tests/generate_frozen_reference.py --baryons

Review the printed $\chi^2$ values against the committed references
before committing: they define what every later test run compares
against.
