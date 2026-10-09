#!/usr/bin/env python
"""Write the n(z) tables of the des_y3 project from the DES 2pt FITS files.

What an n(z) table is: the redshift distribution of the galaxies in each
tomographic bin, tabulated on a uniform redshift grid. Cosmolike reads it
as a plain text file (the ".nz" files in data/): one row per redshift
cell, column 0 = a redshift, column k = n(z) of tomographic bin k. The
reader normalizes each bin itself, so the overall scale of a column does
not matter.

Where the numbers come from: DES publishes each 2pt analysis as one FITS
file. Its extensions nz_lens and nz_source hold the histograms, one row
per redshift cell of width dz, with three redshift columns

    Z_LOW  = left edge of the cell
    Z_MID  = center of the cell, (Z_LOW + Z_HIGH)/2
    Z_HIGH = right edge of the cell

and one column BIN1, BIN2, ... per tomographic bin. This script copies
the BIN columns unchanged and writes one of the redshift columns, chosen
with --zcolumn, as column 0 of the text table:

    --zcolumn zleft    column 0 = Z_LOW  -> set photoz_zmid_convention: 0
    --zcolumn zcenter  column 0 = Z_MID  -> set photoz_zmid_convention: 1

photoz_zmid_convention is the likelihood yaml key (likelihood/*.yaml)
that tells cosmolike how to read column 0. With 0 it treats the value as
a left edge and places the tabulated n(z) at the cell center z + dz/2;
with 1 it places the n(z) at z itself. Both choices put every value at
Z_MID, as long as the table and the key agree. A table read with the
other key value shifts every distribution rigidly by dz/2 in redshift.
The key is one value for the whole likelihood, so the lens and the
source table of one data set must be written with the same column: this
script always writes them as a pair.

The zleft tables are the ones the project ships (data/des_y3_lens.nz,
data/des_y3_source.nz, data/des_y1_lens.nz, data/des_y1_source.nz) and
the ones the example yamls read with photoz_zmid_convention: 0. The
zcenter tables get a _zcenter suffix and a matching dataset file (the
small ".dataset" text file listing, one `key = filename` line each,
which data files the likelihood reads), so they never replace the
shipped files. scripts/README.md explains why the zcenter pair does not
reproduce the zleft pair (two effects inside the cosmolike reader) and
why analyses use the zleft pair.

Usage (from the Cocoa/ folder, with the cocoa conda environment active):

    python ./projects/des_y3/scripts/make_nz_from_fits.py --data des_y3 \
      --zcolumn zleft --overwrite

    python ./projects/des_y3/scripts/make_nz_from_fits.py --data des_y1 \
      --zcolumn zcenter
"""

import argparse
import os
import sys

import numpy as np

# The project folder is the parent of scripts/; data/ holds both the FITS
# files and the tables written from them. Resolving the paths from this
# file's own location makes the script independent of the folder it is
# launched from.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# One entry per DES data set shipped by the project.
#   fits_file         = the DES 2pt FITS file in data/ the tables come from
#   dataset_file      = the project's dataset file that lists the tables
#   lens_file         = name of the lens table written with --zcolumn zleft
#   source_file       = name of the source table written with --zcolumn zleft
#   lens_zhigh_max    = keep only lens cells with Z_HIGH <= this value
#                       (None keeps every cell)
#   source_zhigh_max  = the same cut for the source table
#
# The 1.5 cut on the Y3 lens table: the FITS lens table runs to z = 6,
# but every redMaGiC bin is zero above z = 1.195. The shipped table stops
# at the cell [1.485, 1.495]; cutting at Z_HIGH <= 1.5 keeps exactly those
# rows, so a regenerated table matches the shipped one row for row. The
# script refuses to drop a cell that holds a nonzero value.
DATASETS = {
    "des_y3": {
        "fits_file": "2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits",
        "dataset_file": "des_y3_real.dataset",
        "lens_file": "des_y3_lens.nz",
        "source_file": "des_y3_source.nz",
        "lens_zhigh_max": 1.5,
        "source_zhigh_max": None,
    },
    "des_y1": {
        "fits_file": "2pt_NG_mcal_1110.fits",
        "dataset_file": "des_y1_real.dataset",
        "lens_file": "des_y1_lens.nz",
        "source_file": "des_y1_source.nz",
        "lens_zhigh_max": None,
        "source_zhigh_max": None,
    },
}

# The two choices of column 0 and the photoz_zmid_convention value each
# one needs. The suffix keeps the zcenter tables from overwriting the
# shipped zleft tables (des_y3_lens.nz -> des_y3_lens_zcenter.nz).
ZCOLUMNS = {
    "zleft": {
        "fits_column": "Z_LOW",
        "photoz_zmid_convention": 0,
        "suffix": "",
    },
    "zcenter": {
        "fits_column": "Z_MID",
        "photoz_zmid_convention": 1,
        "suffix": "_zcenter",
    },
}

# Cells must share one width: cosmolike rebuilds the grid from the first
# z, the last z, and the row count alone, so a nonuniform column would be
# silently misplaced. 1e-9 is far below the 0.01 cell width and far above
# the ~1e-16 rounding of the FITS values.
UNIFORM_TOLERANCE = 1.0e-9

# A git LFS pointer file starts with this line. The FITS files are stored
# with git LFS (see the project's .gitattributes); a clone made without
# git-lfs holds a ~130-byte pointer text file in their place.
LFS_POINTER_PREFIX = b"version https://git-lfs"


def read_dataset_keys(dataset_path):
    """Read the `key = value` lines of a dataset file into a dictionary.

    The dataset file (for example data/des_y3_real.dataset) names the files
    the likelihood reads and the number of tomographic bins. The script
    needs lens_ntomo and source_ntomo to check that the FITS tables have
    the expected number of bins before writing anything.

    Arguments:
      dataset_path = path of the dataset file.

    Returns:
      dict {key: value string}, both stripped of surrounding blanks.

    Raises:
      SystemExit when the file does not exist.
    """
    if not os.path.isfile(dataset_path):
        sys.exit(f"dataset file not found: {dataset_path}")
    keys = {}
    with open(dataset_path) as f:
        for line in f:
            # skip blank lines and lines commented out with '#'
            text = line.strip()
            if text == "" or text.startswith("#"):
                continue
            if "=" not in text:
                continue
            # split at the first '=' only: file names never contain one,
            # but the value keeps any later '=' intact this way
            key, value = text.split("=", 1)
            keys[key.strip()] = value.strip()
    return keys


def check_not_lfs_pointer(fits_path):
    """Refuse to continue when the FITS file is missing or a git LFS pointer.

    Without git-lfs, git checks out a short text file ("version
    https://git-lfs...") where the FITS file should be; astropy would then
    fail with an unrelated-looking parsing error. Checking the first bytes
    turns that into a message that says what to run.

    Arguments:
      fits_path = path of the FITS file in data/.

    Returns:
      nothing when the file exists and is not a pointer.

    Raises:
      SystemExit naming the file and the git lfs command to run.
    """
    if not os.path.isfile(fits_path):
        sys.exit(f"FITS file not found: {fits_path}")
    with open(fits_path, "rb") as f:
        head = f.read(len(LFS_POINTER_PREFIX))
    if head == LFS_POINTER_PREFIX:
        sys.exit(f"{fits_path} is a git LFS pointer, not the FITS file. "
                 "Activate the cocoa environment (it provides git-lfs) and "
                 f"run `git lfs pull` inside {PROJECT_DIR}.")


def load_fits_nz(fits_path, extension, ntomo):
    """Read one n(z) extension of a DES 2pt FITS file and check its layout.

    The checks run before anything is written: the extension must hold
    the three redshift columns and exactly ntomo bin columns, the cells
    must share one width, and Z_MID must be the center of each cell. The
    last check matters because the script lets the user pick Z_LOW or
    Z_MID as column 0, and the two choices are equivalent only when
    Z_MID = Z_LOW + dz/2.

    Arguments:
      fits_path = path of the FITS file.
      extension = "nz_lens" or "nz_source".
      ntomo     = number of tomographic bins the dataset file declares.

    Returns:
      dict with
        "Z_LOW", "Z_MID", "Z_HIGH" = 1D float arrays, one entry per cell
        "bins" = 2D float array, shape (cells, ntomo); column k is BIN(k+1)
        "dz"   = the common cell width

    Raises:
      SystemExit when a column is missing, the bin count differs from
      ntomo, the cells are not uniform, or Z_MID is not the cell center.
    """
    # astropy is imported here, not at the top, so `--help` works in an
    # environment without it
    from astropy.io import fits

    # `with` closes the FITS file when the block ends, even on an error
    with fits.open(fits_path) as hdul:
        # hdul[extension] looks the extension up by its EXTNAME
        names = [hdu.name.lower() for hdu in hdul]
        if extension not in names:
            sys.exit(f"{fits_path}: no extension {extension} "
                     f"(extensions: {names})")
        table = hdul[extension].data
        columns = list(table.columns.names)
        for required in ("Z_LOW", "Z_MID", "Z_HIGH"):
            if required not in columns:
                sys.exit(f"{fits_path}[{extension}]: column {required} "
                         f"missing (columns: {columns})")
        bin_columns = []
        for name in columns:
            if name.startswith("BIN"):
                bin_columns.append(name)
        if len(bin_columns) != ntomo:
            sys.exit(f"{fits_path}[{extension}]: {len(bin_columns)} bin "
                     f"columns {bin_columns}, the dataset file declares "
                     f"{ntomo}; check --data")
        # np.array(..., dtype=float) copies each FITS column (stored
        # big-endian on disk) into an ordinary float64 array; the values
        # are unchanged
        result = {}
        for name in ("Z_LOW", "Z_MID", "Z_HIGH"):
            result[name] = np.array(table[name], dtype=float)
        bins = np.zeros((len(table), ntomo))
        for k in range(ntomo):
            bins[:, k] = np.array(table[f"BIN{k + 1}"], dtype=float)
        result["bins"] = bins

    widths = result["Z_HIGH"] - result["Z_LOW"]
    dz = widths[0]
    if np.max(np.abs(widths - dz)) > UNIFORM_TOLERANCE:
        sys.exit(f"{fits_path}[{extension}]: cell widths vary between "
                 f"{widths.min()} and {widths.max()}; cosmolike needs one "
                 "width")
    steps = np.diff(result["Z_LOW"])
    if np.max(np.abs(steps - dz)) > UNIFORM_TOLERANCE:
        sys.exit(f"{fits_path}[{extension}]: Z_LOW does not advance by the "
                 f"cell width {dz}")
    centers = 0.5 * (result["Z_LOW"] + result["Z_HIGH"])
    if np.max(np.abs(result["Z_MID"] - centers)) > UNIFORM_TOLERANCE:
        sys.exit(f"{fits_path}[{extension}]: Z_MID is not the cell center "
                 "(Z_LOW + Z_HIGH)/2")
    result["dz"] = dz
    return result


def select_rows(nz, zhigh_max, label):
    """Choose the cells to write: drop empty cells below z = 0 and above a cut.

    Two kinds of cells are dropped, and only when every bin is zero there:

    1. cells with Z_LOW < 0. The DES Y3 redMaGiC lens table starts with
       the cell [-0.005, 0.005]. Cosmolike clips the lowest redshift of a
       table to 1e-5 but computes the cell width from the unclipped range,
       so a table whose column 0 starts below zero is read on a compressed
       grid: every value lands about +0.004 to +0.005 away from its cell
       center. Dropping the empty cell avoids that.
    2. cells with Z_HIGH > zhigh_max (see DATASETS for the one cut used).

    A nonzero value in a dropped cell would change the physics, so the
    function refuses instead of dropping it.

    Arguments:
      nz        = the dict returned by load_fits_nz.
      zhigh_max = upper cut on Z_HIGH, or None for no cut.
      label     = text naming the table in messages ("des_y3 lens").

    Returns:
      (first, last) = indices of the first and last cell kept; the kept
      cells are first, first + 1, ..., last.

    Raises:
      SystemExit when a cell to be dropped holds a nonzero value.
    """
    ncells = len(nz["Z_LOW"])
    keep = np.ones(ncells, dtype=bool)
    keep[nz["Z_LOW"] < 0.0] = False
    if zhigh_max is not None:
        keep[nz["Z_HIGH"] > zhigh_max] = False
    dropped = np.logical_not(keep)
    # any nonzero value in a dropped row is an error, not a trim
    if np.any(nz["bins"][dropped, :] != 0.0):
        rule = "Z_LOW >= 0"
        if zhigh_max is not None:
            rule = rule + f" and Z_HIGH <= {zhigh_max}"
        sys.exit(f"{label}: a cell outside {rule} holds a nonzero n(z); "
                 "refusing to drop it")
    kept = np.nonzero(keep)[0]
    first = int(kept[0])
    last = int(kept[-1])
    # the kept cells must be one contiguous block: a gap would break the
    # uniform grid cosmolike assumes
    if last - first + 1 != len(kept):
        sys.exit(f"{label}: the kept cells are not contiguous")
    return first, last


def format_table(nz, fits_column, first, last):
    """Format the kept cells as the lines of a cosmolike n(z) text table.

    Every number is written with Python's repr(), the shortest decimal
    text that converts back to exactly the same double. The text table
    therefore holds the FITS values bit for bit, and a regenerated table
    is byte-identical to the shipped one whenever the numbers agree
    (for example 0.024999999999999998, the FITS value of the Z_LOW 0.025
    edge, stays exactly that). Each line ends with a blank before the
    newline, as in the shipped files.

    Arguments:
      nz          = the dict returned by load_fits_nz.
      fits_column = "Z_LOW" or "Z_MID", the redshift column to write.
      first, last = the range of cells to write (inclusive).

    Returns:
      list of strings, one text line per cell, newline included.
    """
    lines = []
    for row in range(first, last + 1):
        fields = [repr(float(nz[fits_column][row]))]
        for k in range(nz["bins"].shape[1]):
            fields.append(repr(float(nz["bins"][row, k])))
        lines.append(" ".join(fields) + " \n")
    return lines


def dataset_text_for(dataset_path, lens_file, source_file):
    """Return the dataset file text with its two n(z) file names replaced.

    The zcenter tables need a dataset file that points at them. The new
    file is the project's dataset file with only the nz_lens_file and
    nz_source_file lines rewritten, so the data vector, covariance, mask,
    and binning stay identical.

    Arguments:
      dataset_path = the project's dataset file (for example
                     data/des_y3_real.dataset).
      lens_file    = the lens table name to write into nz_lens_file.
      source_file  = the source table name to write into nz_source_file.

    Returns:
      the new file content as one string.

    Raises:
      SystemExit when either key does not appear exactly once.
    """
    with open(dataset_path) as f:
        # keepends=True keeps each line's own newline, so joining the
        # lines back reproduces the file exactly
        lines = f.read().splitlines(keepends=True)
    replaced = {"nz_lens_file": 0, "nz_source_file": 0}
    for i in range(len(lines)):
        key = lines[i].split("=", 1)[0].strip()
        if key == "nz_lens_file":
            lines[i] = f"nz_lens_file = {lens_file}\n"
            replaced[key] += 1
        elif key == "nz_source_file":
            lines[i] = f"nz_source_file = {source_file}\n"
            replaced[key] += 1
    for key, count in replaced.items():
        if count != 1:
            sys.exit(f"{dataset_path}: {key} appears {count} times, "
                     "expected once")
    return "".join(lines)


def main():
    """Validate every input, then write the lens/source pair (and dataset).

    All reading and checking happens before the first file is written, so
    a failure never leaves a half-written pair behind.

    Side effects:
      writes data/<lens table> and data/<source table> (with a _zcenter
      suffix for --zcolumn zcenter) and, for zcenter, the dataset file
      data/<dataset name>_zcenter.dataset. Existing files are replaced
      only with --overwrite.

    Returns:
      0, the exit status of a successful run; every failure stops earlier
      through sys.exit with a message naming the problem.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="which DES data set to write")
    parser.add_argument(
        "--zcolumn",
        required=True,
        choices=sorted(ZCOLUMNS.keys()),
        help="zleft = FITS Z_LOW (photoz_zmid_convention: 0), "
             "zcenter = FITS Z_MID (photoz_zmid_convention: 1)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output files that already exist")
    args = parser.parse_args()

    config = DATASETS[args.data]
    choice = ZCOLUMNS[args.zcolumn]

    # ---- inputs: dataset file and FITS file --------------------------------
    dataset_path = os.path.join(DATA_DIR, config["dataset_file"])
    dataset = read_dataset_keys(dataset_path)
    ntomo = {
        "lens": int(dataset["lens_ntomo"]),
        "source": int(dataset["source_ntomo"]),
    }
    fits_path = os.path.join(DATA_DIR, config["fits_file"])
    check_not_lfs_pointer(fits_path)

    # ---- read and check both tables before writing anything ----------------
    outputs = {}
    for sample in ("lens", "source"):
        label = f"{args.data} {sample}"
        nz = load_fits_nz(
            fits_path=fits_path,
            extension=f"nz_{sample}",
            ntomo=ntomo[sample])
        first, last = select_rows(
            nz=nz,
            zhigh_max=config[f"{sample}_zhigh_max"],
            label=label)
        lines = format_table(
            nz=nz,
            fits_column=choice["fits_column"],
            first=first,
            last=last)
        # des_y3_lens.nz -> des_y3_lens_zcenter.nz for zcenter;
        # os.path.splitext splits "des_y3_lens.nz" into ("des_y3_lens", ".nz")
        stem, extension = os.path.splitext(config[f"{sample}_file"])
        name = stem + choice["suffix"] + extension
        outputs[sample] = {
            "name": name,
            "path": os.path.join(DATA_DIR, name),
            "lines": lines,
            "first": first,
            "last": last,
            "dz": nz["dz"],
            "z_first": float(nz[choice["fits_column"]][first]),
        }

    files_to_write = {}
    for sample in ("lens", "source"):
        files_to_write[outputs[sample]["path"]] = "".join(
            outputs[sample]["lines"])
    if args.zcolumn == "zcenter":
        stem, extension = os.path.splitext(config["dataset_file"])
        zcenter_dataset = stem + choice["suffix"] + extension
        files_to_write[os.path.join(DATA_DIR, zcenter_dataset)] = \
            dataset_text_for(
                dataset_path=dataset_path,
                lens_file=outputs["lens"]["name"],
                source_file=outputs["source"]["name"])

    existing = []
    for path in files_to_write:
        if os.path.exists(path):
            existing.append(path)
    if existing and not args.overwrite:
        sys.exit("these files exist; rerun with --overwrite to replace "
                 "them:\n  " + "\n  ".join(existing))

    # ---- write -------------------------------------------------------------
    for path, text in files_to_write.items():
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")

    for sample in ("lens", "source"):
        out = outputs[sample]
        ncells = out["last"] - out["first"] + 1
        print(f"  {sample}: {ncells} cells (FITS rows {out['first']} to "
              f"{out['last']}), dz = {out['dz']:.6g}, column 0 = "
              f"{choice['fits_column']} starting at {out['z_first']!r}")
    print("\nset in the likelihood block of the yaml:")
    print(f"    photoz_zmid_convention: {choice['photoz_zmid_convention']}")
    if args.zcolumn == "zcenter":
        print(f"    data_file: {zcenter_dataset}")
        print("\nNOTE: read scripts/README.md (FAQ: Why do the zcenter tables "
              "not reproduce the zleft tables?) before using these files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
