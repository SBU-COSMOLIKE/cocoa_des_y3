# Redshift distributions from the DES FITS files

The redshift distributions $n(z)$ the likelihoods read (the `.nz` files
in `data/`) are copies of the histograms DES published in its 2pt FITS
files. Both FITS files live in `data/`, and the script
`make_nz_from_fits.py` in this folder writes the `.nz` files from them.

The script writes the redshift column of each table either as the left
edges of the histogram cells (`--zcolumn zleft`) or as their centers
(`--zcolumn zcenter`). The likelihood yaml key `photoz_zmid_convention`
tells cosmolike which of the two it reads, so each choice pairs with one
value of the key.

```mermaid
flowchart TD
  A["data/*.fits: the DES-Y3 and DES-Y1 2pt files"] --> B["make_nz_from_fits.py"]
  B --> C["--zcolumn zleft: the shipped .nz tables"]
  B --> D["--zcolumn zcenter: *_zcenter.nz and a *_zcenter.dataset"]
  C --> E["photoz_zmid_convention: 0, in every likelihood yaml"]
  D --> F["photoz_zmid_convention: 1"]
  C --> G["tests/frozen/data: the tests' copy, written by generate_frozen_reference.py"]
```

# Table of contents

1. [Running the n(z) script](#run_nz)
    1. [Rewriting the tables the project ships](#run_zleft)
    2. [Writing the zcenter tables](#run_zcenter)
2. [The FITS files](#fits_files)
3. [The z column and `photoz_zmid_convention`](#zcolumn)
4. [Appendix](#appendix)
    1. [FAQ: Why does the DES-Y3 lens z column start at 0.005?](#lens_z_column)
    2. [FAQ: Why do the zcenter tables not reproduce the zleft tables?](#zcenter_mismatch)
    3. [FAQ: Why does the script drop some cells?](#dropped_cells)
    4. [FAQ: What if the script reports a git LFS pointer?](#lfs_pointer)

## Running the n(z) script <a name="run_nz"></a>

The script writes one lens and one source table per run, both with the
same redshift column: `photoz_zmid_convention` is a single key for the
whole likelihood, so the two tables of one data set must follow the
same convention.

### Rewriting the tables the project ships <a name="run_zleft"></a>

The project ships zleft tables, read with `photoz_zmid_convention: 0`
(the value in every likelihood yaml). Rewriting them from the FITS files
reproduces them byte for byte.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rewrite the DES-Y3 tables

    python ./projects/des_y3/scripts/make_nz_from_fits.py --data des_y3 --zcolumn zleft --overwrite

**Step :three:**: rewrite the DES-Y1 tables

    python ./projects/des_y3/scripts/make_nz_from_fits.py --data des_y1 --zcolumn zleft --overwrite

**Step :four:**: check that the four tables did not change

    git -C ./projects/des_y3 status --short data

> [!NOTE]
> - `--data des_y3`: read `data/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits` and write `des_y3_lens.nz` and `des_y3_source.nz`.
> - `--data des_y1`: read `data/2pt_NG_mcal_1110.fits` and write `des_y1_lens.nz` and `des_y1_source.nz`.
> - `--zcolumn zleft`: write the FITS column `Z_LOW` (left cell edges) as the redshift column.
> - `--overwrite`: replace tables that already exist; without it the script lists them and stops.

> [!TIP]
> If the script reports a git LFS pointer, see the appendix
> [FAQ: What if the script reports a git LFS pointer?](#lfs_pointer).

### Writing the zcenter tables <a name="run_zcenter"></a>

The zcenter tables carry the cell centers as their redshift column. The
script writes them next to the shipped tables with a `_zcenter` suffix,
plus a dataset file (the small `.dataset` text file naming the files the
likelihood reads) that points at them; the shipped files stay untouched.

> [!Warning]
> With the cosmolike reader of this Cocoa version the zcenter tables do
> not reproduce the zleft tables. The example yamls and the tests use
> the zleft tables. See the appendix
> [FAQ: Why do the zcenter tables not reproduce the zleft tables?](#zcenter_mismatch).

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: write the DES-Y3 zcenter tables and their dataset file

    python ./projects/des_y3/scripts/make_nz_from_fits.py --data des_y3 --zcolumn zcenter

**Step :three:**: set these two keys in the likelihood block of the yaml
(for example `EXAMPLE_EVALUATE2.yaml`)

    data_file: des_y3_real_zcenter.dataset
    photoz_zmid_convention: 1

> [!NOTE]
> - `--zcolumn zcenter`: write the FITS column `Z_MID` (cell centers) as the redshift column, into `des_y3_lens_zcenter.nz` and `des_y3_source_zcenter.nz`, and write `des_y3_real_zcenter.dataset`.
> - `--data des_y1`: the same for the DES-Y1 data; the dataset file is then `des_y1_real_zcenter.dataset`.

## The FITS files <a name="fits_files"></a>

`data/` holds the two DES 2pt FITS files the project's data come from:

| FITS file in `data/` | analysis | lens sample |
|---|---|---|
| `2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits` | DES-Y3 3x2pt | redMaGiC, 5 bins |
| `2pt_NG_mcal_1110.fits` | DES-Y1 3x2pt | redMaGiC, 5 bins |

Both are the copies distributed with the cosmosis-standard-library
(`likelihood/des-y3/` and `likelihood/des-y1/`). The DES-Y3 file has the
byte size of the DES release file
`2pt_NG_final_2ptunblind_02_24_21_wnz_redmagic_covupdate.fits`
(desdr-server.ncsa.illinois.edu/despublic/y3a2_files/datavectors/). The
DES-Y3 MagLim FITS file (6 lens bins) is a different analysis and is not
in this project.

Every data vector, covariance and n(z) file the two dataset files name
is a copy of one FITS extension:

| file in `data/` | FITS file | FITS extension |
|---|---|---|
| `des_y3_unblinded_final.txt` | DES-Y3 | `xip`, `xim`, `gammat`, `wtheta` (column `VALUE`, in that order) |
| `des_y3_cov_unblinded_final.txt` | DES-Y3 | `COVMAT` |
| `des_y3_lens.nz` | DES-Y3 | `nz_lens`, cells 1 to 149 |
| `des_y3_source.nz` | DES-Y3 | `nz_source`, cells 0 to 299 |
| `2pt_NG_mcal_1101.txt` | DES-Y1 | `xip`, `xim`, `gammat`, `wtheta` (column `VALUE`, in that order) |
| `des_y1_cov.txt` | DES-Y1 | `COVMAT` |
| `des_y1_lens.nz` | DES-Y1 | `nz_lens`, cells 0 to 399 |
| `des_y1_source.nz` | DES-Y1 | `nz_source`, cells 0 to 399 |

Every number in these files equals its FITS value bit for bit. The
DES-Y1 data vector file says `1101` in its name, the
FITS file `1110`; the numbers are the same. The covariance
`COV2_Y1_mcal_v04.17_3column_rr` differs from the DES-Y1 `COVMAT`, and no
dataset file reads it.

The FITS files are stored with git LFS (`.gitattributes`: `*.fits`). No
test reads them: the snapshot the tests keep under `tests/frozen/data`
copies `data/` without the `.fits` files.

## The z column and `photoz_zmid_convention` <a name="zcolumn"></a>

A FITS n(z) extension stores one row per redshift cell of width
$\Delta z$, with three redshift columns

    Z_LOW  : left edge of the cell
    Z_MID  : center of the cell, (Z_LOW + Z_HIGH)/2
    Z_HIGH : right edge of the cell

and one column `BIN1`, `BIN2`, ... per tomographic bin. A `.nz` file keeps
one of the three redshift columns as its first column, followed by the
bin columns. The DES files place their cells as follows:

| FITS table | first `Z_LOW` | first `Z_MID` | $\Delta z$ |
|---|---|---|---|
| DES-Y3 `nz_lens` (redMaGiC) | -0.005 | 0.000 | 0.01 |
| DES-Y3 `nz_source` | 0.000 | 0.005 | 0.01 |
| DES-Y1 `nz_lens` | 0.0001 | 0.0051 | 0.01 |
| DES-Y1 `nz_source` | 0.0001 | 0.0051 | 0.01 |

The key `photoz_zmid_convention` in the likelihood block tells cosmolike
where each tabulated value belongs. With $z_0$ the first value of the
redshift column, the value of row $k$ goes to

$$z_k = z_0 + \left(k + \tfrac{1}{2}\right)\Delta z \quad \text{for } 0, \qquad z_k = z_0 + k\,\Delta z \quad \text{for } 1.$$

Both place every value at the cell center when the table and the key
match:

| `--zcolumn` | redshift column | `photoz_zmid_convention` | tables |
|---|---|---|---|
| `zleft` | `Z_LOW` | `0` | `des_y3_lens.nz`, `des_y3_source.nz`, `des_y1_lens.nz`, `des_y1_source.nz` |
| `zcenter` | `Z_MID` | `1` | the same names with a `_zcenter` suffix |

A table read with the other value of the key moves every distribution
rigidly by $\Delta z/2 = 0.005$ in redshift. The test file
`tests/data_vector/test_photoz_conventions.py` measures that shift on cosmic
shear ([the photo-z convention checks](../tests/data_vector/README.md#photoz_conventions)).

# Appendix <a name="appendix"></a>

## :interrobang: FAQ: Why does the DES-Y3 lens z column start at 0.005? <a name="lens_z_column"></a>

The redMaGiC cells of the DES-Y3 FITS file are centered on multiples of
0.01: the first cell is $[-0.005, 0.005]$, the second $[0.005, 0.015]$.
The value 0.005 is therefore the left edge of the second cell, not a
center, and `des_y3_lens.nz` starts there because the empty first cell
is dropped ([FAQ: Why does the script drop some cells?](#dropped_cells)).
The source cells start at 0 and are centered on
0.005, 0.015, ..., so the source column starts at 0. Both tables hold
`Z_LOW`, and `photoz_zmid_convention: 0` reads both correctly.

Shifting the lens column by $-\Delta z/2$ (reading 0.005 as a center)
would move every lens distribution half a cell below its FITS position.
Measured at the 3x2pt point of the tests: $\delta^T C^{-1} \delta = 2.10$
($\delta$ = the data-vector difference, $C^{-1}$ = the masked inverse
covariance), 2.03 from galaxy-galaxy lensing and 0.41 from clustering.

## :interrobang: FAQ: Why do the zcenter tables not reproduce the zleft tables? <a name="zcenter_mismatch"></a>

A zcenter table read with `photoz_zmid_convention: 1` places its values
at the same redshifts as the zleft table read with
`photoz_zmid_convention: 0`, so the two should give identical data
vectors. They do not, because of two effects in cosmolike's n(z) reader
(`redshift_spline.c` and `generic_interface.cpp` of cosmolike_core):

1. The reader finds the table row of a redshift by rounding
   $(z - z_0)/\Delta z$ down. With `1` the evaluation points sit exactly
   on the table's redshifts, and floating-point rounding sends some of
   them to the previous row: 19 of 149 points of the DES-Y3 zcenter lens
   table and 7 of 300 of the source table (counted with the reader's own
   formulas, compiled with the cosmolike flags). With `0` the points sit
   in the middle of their cells and every row is read correctly.
2. Each bin's redshift range (used for its mean redshift, for the limits
   of the lens integrals, and for the non-Limber clustering kernel) is
   taken from the redshift column directly, as the first and last
   nonzero rows, without the convention. With zleft the range ends half
   a cell below the last evaluation point; with zcenter it ends on it.

Measured, zcenter with `1` against zleft with `0`:

- `EXAMPLE_EVALUATE2.yaml` (3x2pt, DES-Y3): $\delta^T C^{-1} \delta = 0.12$;
  $\chi^2$ against the data from 897.776 to 898.799.
- `EXAMPLE_EVALUATE4.yaml` (3x2pt, DES-Y1): $\delta^T C^{-1} \delta = 0.0018$,
  all from the second effect (no DES-Y1 row is misread).

The zleft tables with `photoz_zmid_convention: 0` read every row
correctly, which is why the project ships them.

## :interrobang: FAQ: Why does the script drop some cells? <a name="dropped_cells"></a>

The script drops two kinds of cells, and refuses to run when a dropped
cell holds a nonzero value:

1. Cells with `Z_LOW` below zero: the DES-Y3 lens table starts with the
   empty cell $[-0.005, 0.005]$. Cosmolike raises a first redshift below
   $10^{-5}$ to $10^{-5}$ but computes the cell width from the original
   range, so a redshift column that starts below zero is read on a
   compressed grid: every value lands 0.004 to 0.005 above its cell
   center.
2. DES-Y3 lens cells with `Z_HIGH` above 1.5: the FITS lens table runs
   to $z = 6$, every bin is zero above $z = 1.195$, and the shipped table
   ends at the cell $[1.485, 1.495]$. The cut reproduces the shipped
   table row for row.

## :interrobang: FAQ: What if the script reports a git LFS pointer? <a name="lfs_pointer"></a>

A clone made without git-lfs holds a short text file (a pointer) in
place of each FITS file. The cocoa environment provides git-lfs. If that
is the case, follow the steps below.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: download the FITS files the pointers name

    git -C ./projects/des_y3 lfs pull
