"""Survey adapter of the des_y3 forecast covariance.

An adapter is the small project-specific layer that the shared covariance
package (cosmolike_notebook_utils.covariance in cosmolike_core) calls: it
supplies this survey's numbers and initializes this project's compiled
interface, while the numerical algorithms stay in the shared package and
in the C kernels. The notebook EXAMPLE_EVALUATE_COVARIANCE.ipynb and the
command-line program compute_covariance.py both use its three functions:

    configuration()  resolve the survey, cosmology and accuracy settings
    initialize()     run CAMB once and install the forecast state
    compute()        compute the G, SSC, cNG and total matrices

The forecast has the measurement layout of data/des_y3_real.dataset: 5
lens and 4 source bins and 20 angular bins from 2.5 to 250 arcmin (900
real-space entries), plus 15 Fourier bands for the Fourier-space companion
(525 entries). The likelihoods do not use it: they read the published DES
Y3 covariance named by the dataset's cov_file key
(des_y3_cov_unblinded_final.txt), and the notebook compares the forecast
with that matrix after the same cuts. Its physics differs from the
published analysis: massless neutrinos, linear galaxy bias, zero
magnification and photo-z shifts, no intrinsic alignment unless the
optional Gaussian model adds it (Gaussian part only), and a spherical-cap
footprint.
"""

from pathlib import Path

import numpy as np

from cosmolike_notebook_utils.covariance.forecast import (
    initialize_forecast,
    gaussian_model,
    compute_forecast,
)
from cosmolike_notebook_utils import covariance as cov


def configuration(accuracy_boost=None, gaussian=None, **accuracy_overrides):
    """Return the resolved survey, cosmology and accuracy settings.

    The catalog numbers below come from the DES Y3 2pt release; the
    covariance README lists their sources. The redshift files supply only
    the shape of each n(z), because the reader normalizes every bin, so the
    number densities are given separately. The numerical controls come
    from default.yaml in this folder.

    Arguments:
        accuracy_boost = None keeps the default.yaml baseline; 1, 2, 4 or 8
            refines its tables and cutoffs. CAMB, the scientific bins and
            the catalog stay unchanged.
        gaussian = None, or a mapping of the Gaussian-covariance physics:
            nonlimber (bool), ia ("none", "NLA" or "TATT"), A1, A2, B_TA
            (a scalar or one value per source bin); see gaussian_model in
            cosmolike_notebook_utils/covariance/forecast.py. None keeps
            non-Limber on and intrinsic alignment off.
        accuracy_overrides = named controls of default.yaml to override,
            for example integration_accuracy=1.
    Returns:
        dict of fully resolved settings: the survey and cosmology entries
        below (units in their comments), the accuracy controls with both
        the base values and the boosted grids, and the resolved Gaussian
        model under "gaussian".
    Raises:
        TypeError for an unknown accuracy control; ValueError for an
        invalid gaussian mapping (gaussian_model lists the conditions).
    """
    numerical = cov.load_covariance_accuracy(
        filename=Path(__file__).with_name("default.yaml"),
        accuracy_boost=accuracy_boost, **accuracy_overrides,
    )

    # 16 integer edges, logarithmically spaced, give 15 Fourier bands
    # [band_first, band_last] covering multipoles 30 to 4000; both ends of
    # a band are included, so every integer multipole falls in one band.
    band_edges = np.rint(np.geomspace(30, 4001, 16)).astype(np.int32)

    # The fiducial is shared by G, SSC and cNG; CAMB runs only once. H0 is
    # in km/s/Mpc, As_1e9 = 10^9 A_s, w0pwa = w0 + wa (so wa = 0 here), mnu
    # in eV (massless neutrinos), and kmax in 1/Mpc; the remaining entries
    # are CAMB accuracy options and the nonlinear model (Takahashi
    # halofit).
    settings = {
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.05,
            "H0": 70.0,
            "ns": 0.965,
            "As_1e9": 2.1,
            "w": -1.0,
            "w0pwa": -1.0,
            "mnu": 0.0,
            "AccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "CAMBAccuracyBoost": 1.0,
            "kmax": 20.0,
            "k_per_logint": 20,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },

        # File columns describe radial shapes; these flags fix their z
        # convention, as the likelihood keys photoz_interpolation_type and
        # photoz_zmid_convention do: 0 = cubic spline, 0 = the z column
        # holds left cell edges (Z_LOW).
        "lens_file": "data/des_y3_lens.nz",
        "source_file": "data/des_y3_source.nz",
        "photoz_interpolation": 0,
        "photoz_zmid": 0,

        # Measured pair and band choices are fixed during accuracy
        # refinement. No (lens, source) pair is excluded from
        # galaxy-galaxy lensing; lnm_edges are the halo-mass integration
        # panels, in ln(M/[Msun/h]), from 1e-40 to 1e17 Msun/h.
        "excluded_gammat": [],
        "band_first": band_edges[:-1],
        "band_last": band_edges[1:]-1,
        "lnm_edges": cov.halo_mass_edges(),

        # area_deg2 is the survey area in square degrees, modeled as a
        # spherical cap. Densities are galaxies per square arcminute, one
        # per bin, from the NGAL headers of the DES Y3 2pt FITS file; shape
        # noise sigma_e is per ellipticity component (SIG_E headers). The
        # linear lens biases are the centers (loc) of the DES_B1_* ref
        # distributions in likelihood/params_lens.yaml, from which Cobaya
        # draws starting points.
        "area_deg2": 4143.0,
        "lens_density_arcmin2": [0.02206991865140178, 0.03811556894595951,
                                  0.05828770119609519, 0.02945872243588231,
                                  0.02514319207202297],
        "source_density_arcmin2": [1.475584985490327, 1.479383426887689,
                                    1.483671693529899, 1.461247850098986],
        "sigma_e_component": [0.2435002682964671, 0.2621945854120855,
                              0.2588927276307921, 0.3096827008528927],
        "bias": [1.7, 1.7, 1.7, 2.0, 2.0],

        # theta_edges_arcmin: 21 edges, 20 logarithmic angular bins.
        # a_edges: the redshift shells of the super-sample covariance, as
        # scale factors a = 1/(1+z), increasing from the distant boundary
        # (z = 3.1) to the observer (z = 1e-5).
        "theta_edges_arcmin": np.geomspace(start=2.5, stop=250.0, num=21),
        "a_edges": 1.0/(1.0+np.array([3.1, 2., 1.5, 1., .7, .4, .2, 1.e-5])),
    }
    settings.update(numerical)
    settings["gaussian"] = gaussian_model(
        gaussian=gaussian, nsource=len(settings["source_density_arcmin2"]),
    )
    return settings


def initialize(interface, settings):
    """Run CAMB once and install the complete forecast state without a covariance.

    The CAMB run uses settings["cosmology"]; the installed state is the
    cosmology, the redshift distributions, the densities and the galaxy
    bias that compute() reads.

    Arguments:
        interface = imported cosmolike_des_y3_interface module.
        settings = resolved mapping from configuration().
    Returns:
        CAMB input tables as a dict, suitable for saving beside results.
    Side effects:
        Replaces the interface's global cosmology and nuisance state. The
        likelihood covariance, data vector and mask are never loaded.
    """
    return initialize_forecast(
        interface=interface, settings=settings,
        project=Path(__file__).resolve().parents[1],
    )


def compute(interface, settings, space="real", rows=None, progress=None,
            backend=None):
    """Return the galaxy/shear forecast with G, SSC, connected and total matrices.

    Arguments:
        interface = the compiled project interface, already initialized by
            initialize().
        settings = the mapping returned by configuration().
        space = "real" (angular bins) or "fourier" (E-mode bandpowers).
        rows = None for every measured row, or an int32 array [n_rows, 3]
            of (probe type, bin A, bin B) selecting a subset; the crossed
            spectra the covariance needs are kept either way.
        progress = None, or a callable receiving (stage, elapsed_seconds).
        backend = None runs the notebook wrappers; interface.covariance
            (the command-line program's choice) calls the direct bindings
            to the same C calculations.
    Returns:
        dict from compute_forecast: the G, SSC, cNG and total matrices,
        mean signals, diagnostics, the bin coordinates (arcmin or
        multipole) and the resolved settings. The full layout has 900
        entries in real space and 525 in Fourier space (shear has only
        its E-mode spectrum there, no xi_minus).
    """
    return compute_forecast(
        interface=interface, settings=settings, space=space, rows=rows,
        progress=progress, backend=backend,
    )
