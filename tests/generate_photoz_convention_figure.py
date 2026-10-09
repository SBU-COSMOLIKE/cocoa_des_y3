"""Regenerate the photo-z convention figures the tests README shows.

Evaluates the frozen cosmic-shear fiducial under the four runtime
photo-z settings (cspline/Z_LOW default, linear, Steffen, Z_MID; see
test_photoz_conventions.py for what they mean) and plots the
fractional data-vector differences against the default,

    delta xi/xi = xi(setting)/xi(default) - 1,

per tomographic pair and per angular bin, for xi_plus (solid) and
xi_minus (dashed). Masked angular bins are left out. Two figures,
because the two knobs live on different scales:

    photoz_zmid_dxi.png   - the Z_LOW vs Z_MID reading of the n(z)
                            file z column (percent level),
    photoz_interp_dxi.png - linear and Steffen vs cubic spline
                            (1e-4 level).

Both PNG files are written next to this script, in tests/, replacing
the committed ones. To run (from the Cocoa/ folder, cocoa environment
active, start_cocoa.sh sourced):

    python ./projects/des_y3/tests/generate_photoz_convention_figure.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so it
# is set before any cobaya or cosmolike import; 4 is the thread count
# the frozen references were produced with.
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import shutil
import tempfile

import matplotlib
# Agg is matplotlib's file-only backend: no window opens, so the script
# also runs on a machine without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# tests/ is not a package; put it first on sys.path, the list of
# folders Python searches on import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cocoa_test_utils as u

# example1 is the DES-Y3 cosmic-shear configuration. Its data set has 4
# source bins and 20 logarithmic angular bins from 2.5 to 250 arcmin;
# these values repeat des_y3_real.dataset, and MASK_FILE is the frozen
# copy of its baseline mask (column 1: 1 = kept, 0 = cut).
EXAMPLE = "example1"
NTOMO = 4
NTHETA = 20
THETA_MIN, THETA_MAX = 2.5, 250.0
MASK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "frozen", "data",
                         "3x2pt_baseline.mask")

# (report tag, photoz_interpolation_type, photoz_zmid_convention); the
# first entry is the default every other setting is compared with
SETTINGS = (("cspline/Z_LOW (default)", 0, 0), ("linear", 1, 0),
            ("steffen", 2, 0), ("Z_MID", 0, 1))


def datavectors():
    """Return the theory vector of example1 under each photo-z setting.

    Each model prints its full-length theory vector (900 entries, the
    blocks the cosmic-shear likelihood does not compute set to zero) to
    a file in a temporary folder, which is read back and then deleted.

    Returns:
      dict {report tag: numpy array of the data vector}.
    """
    vectors_dir = tempfile.mkdtemp(prefix="photoz_conventions_fig_")
    out = {}
    try:
        for tag, interp, zmid in SETTINGS:
            print(f"building model ({EXAMPLE}, NLA, {tag}) ...", flush=True)
            info = u.load_frozen_info(EXAMPLE, tatt=False)
            block = info["likelihood"][u.EXAMPLES[EXAMPLE]["likelihood"]]
            block["photoz_interpolation_type"] = interp
            block["photoz_zmid_convention"] = zmid
            path = os.path.join(vectors_dir, f"dv_{interp}_{zmid}.modelvector")
            block["print_datavector"] = True
            block["print_datavector_file"] = path
            model = u.make_model(info)
            point = u.build_point(model, EXAMPLE, tatt=False)
            u.evaluate_chi2(model, point)
            out[tag] = u._load_datavector(path)
    finally:
        shutil.rmtree(vectors_dir, ignore_errors=True)
    return out


def plot(curves, fname, title, scale=100.0, unit="%", ylim=None):
    """Draw one 2 x 5 panel grid, one panel per source-bin pair, and save it.

    Arguments:
      curves = {label: (dxi_plus, dxi_minus)}, each a (npair, NTHETA)
               fractional-difference array with NaN at masked bins.
      fname  = name of the PNG file written next to this script.
      title  = the figure title.
      scale  = factor applied to the fractional differences (100 plots
               percent).
      unit   = the unit shown in the y label.
      ylim   = None, or a half-width: one fixed range [-ylim, ylim],
               in plotted units, for every panel.

    Returns:
      nothing; writes fname (120 dpi) and closes the figure.

    legend: npair = 10 source-bin pairs (i <= j) of the 4 source bins,
            NTHETA = 20 angular bins
    """
    # bin centers: the area-weighted mean angle of each bin, the mean of
    # theta over the annulus between two edges,
    # (2/3)(t_hi^3 - t_lo^3)/(t_hi^2 - t_lo^2)
    theta = np.geomspace(THETA_MIN, THETA_MAX, NTHETA + 1)
    theta = (2.0 / 3.0) * (theta[1:]**3 - theta[:-1]**3) \
                        / (theta[1:]**2 - theta[:-1]**2)
    pairs = [(i, j) for i in range(NTOMO) for j in range(i, NTOMO)]
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 7),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    cm = plt.get_cmap("gist_rainbow")
    for p, (i, j) in enumerate(pairs):
        ax = axes.ravel()[p]
        # each setting gets one color from the first 80% of the color
        # map; only the first panel labels its curves, for one legend
        for q, (label, (dp, dm)) in enumerate(curves.items()):
            color = cm(q / max(len(curves) - 1, 1) * 0.8)
            ax.semilogx(theta, scale * dp[p], color=color, lw=1.6,
                        label=rf"$\xi_+$: {label}" if p == 0 else None)
            ax.semilogx(theta, scale * dm[p], color=color, lw=1.6, ls="--",
                        label=rf"$\xi_-$: {label}" if p == 0 else None)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.text(0.08, 0.85, f"$({i+1},{j+1})$", transform=ax.transAxes,
                fontsize=15)
        if p >= 5:
            ax.set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
        if p % 5 == 0:
            ax.set_ylabel(rf"$\Delta\xi_\pm/\xi_\pm$ [{unit}]", fontsize=16)
    if ylim is not None:
        # one fixed range for every panel (sharey=True, so setting it on
        # the first panel sets them all) keeps the small offsets of most
        # pairs readable; a curve beyond it leaves its panel, as the
        # pairs with the lowest source bin do in the Z_MID figure
        axes.ravel()[0].set_ylim(-ylim, ylim)
    axes.ravel()[0].legend(fontsize=9, loc="lower left")
    fig.suptitle(title, fontsize=17)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             fname), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}")


def main():
    """Check the frozen state, evaluate the settings and draw both figures."""
    u.require_cocoa_environment()
    u.verify_frozen()
    dv = datavectors()

    # a two-column mask file holds (index, 0 or 1); keep the 0/1 column.
    # npair = 10 source-bin pairs; nxi = 200 entries per xi block
    mask = np.loadtxt(MASK_FILE)
    mask = mask[:, 1] if mask.ndim == 2 else mask
    npair = NTOMO * (NTOMO + 1) // 2
    nxi = npair * NTHETA

    def frac(tag):
        """Return [dxi_plus, dxi_minus] of one setting against the default.

        Each entry is an (npair, NTHETA) array of
        xi(setting)/xi(default) - 1, NaN where the mask removes the
        point. The first nxi entries of the data vector are xi_plus,
        the next nxi xi_minus.
        """
        ref, cur = dv[SETTINGS[0][0]], dv[tag]
        out = []
        for s in range(2):
            sl = slice(s * nxi, (s + 1) * nxi)
            # errstate silences numpy's division warnings: masked
            # entries are zero in both vectors, and np.where replaces
            # their 0/0 by NaN
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(mask[sl] > 0, cur[sl] / ref[sl] - 1.0, np.nan)
            out.append(d.reshape(npair, NTHETA))
        return out

    plot({"Z_MID": frac("Z_MID")},
         "photoz_zmid_dxi.png",
         "des_y3 cosmic shear: n(z) z-column read as Z_MID instead of "
         "Z_LOW (frozen fiducial)", scale=100.0, unit="%", ylim=4.0)
    plot({"linear": frac("linear"), "steffen": frac("steffen")},
         "photoz_interp_dxi.png",
         "des_y3 cosmic shear: linear and Steffen n(z) interpolation "
         "vs cubic spline (frozen fiducial)", scale=1.0e4,
         unit=r"$10^{-4}$", ylim=8.0)


if __name__ == "__main__":
    main()
