"""Notebook wrappers for the des_y3 Cosmolike interface.

The EXAMPLE_EVALUATE notebooks all drive the same compiled interface
(cosmolike_des_y3_interface) through the same steps: run CAMB once
(cnu.get_camb_cosmology), push the resulting power spectra and
distances into the interface with set_cosmology, set the nuisance
parameters, and read off a spectrum, a correlation function, or the
masked chi2. This module holds those steps once, so a notebook of
this project imports the same wrappers instead of redefining them:

    import cosmolike_des_y3_notebook_wrappers as nw
    nw.configure()                    # per-notebook yaml overrides
    nw.init_cosmolike(CLprobe="xi", with_data=True)
    nw.get_chi2(omegam=0.31)

Three layers of state matter here:

- The compiled interface is a C library with global state: every
  init_* and set_* call above replaces part of it, and the spectrum
  functions read whatever was set last. Each wrapper therefore
  resets everything it depends on (tables, accuracy, cosmology,
  nuisances) on every call, so no call depends on which wrapper ran
  before it.
- The project fiducial point lives in this module as plain
  constants (DES_B1_1, ...), shared by every notebook; a notebook
  overrides any of them per call (nw.C_ss_tomo_limber(ell=ell,
  omegam=x)) or imports the names for its own sweeps. The wrappers
  use them as default argument values, which Python evaluates once,
  when the module is imported: assigning nw.omegam = 0.31 later does
  not change the defaults; pass omegam=0.31 to the call instead.
- The few values that could differ between notebooks because each
  mirrors its own yaml (the lmax of the internal C_ell tables, the
  angular binning, the nonlinear emulator choice, the projection
  engines) live in _CONFIG and are set once per notebook with
  configure().

Every wrapper accepts the same accuracy arguments and combines them
the same way: CLAccuracyBoost is multiplied by AccuracyBoost, the
integration accuracy grows as |3 (CLAccuracyBoost - 1)|, and the
C_ell table reaches lmax + 20000 (CLAccuracyBoost - 1).

The wrappers, by family (s = source galaxies, whose shapes are sheared;
g = lens galaxies, whose positions trace the matter):
  harmonic space : C_ss_tomo_limber (shear-shear C_ell, E and B modes),
                   C_gs_tomo_limber (galaxy-shear), C_gg_tomo (clustering);
  real space     : xi (xi_+ and xi_-), gamma_t (tangential shear around
                   lens galaxies), w_theta (angular clustering);
  likelihood     : get_chi2 (chi2 of the masked data vector);
  responses      : dlnC_dlss_tomo_limber, dlnxi_dlnk_pm_tomo_limber,
                   rf_C_ss_tomo_limber, rf_xi_tomo_limber (sensitivity
                   of the shear statistics to the power at wavenumber k);
  Fisher forecast: get_dv, get_ddv, get_Fisher, get_ddv_dkit,
                   get_Fisher2, plot_Fisher (cosmic shear) and
                   get_dv_3x2pt, get_ddv_3x2pt, get_Fisher_3x2pt (3x2pt),
                   with w0 and wa free in both parameter vectors;
  baryons        : get_baryon_suppression, compute_probes.
"tomo" = tomographic: the galaxies are split into redshift bins and every
bin pair gets its own spectrum. "limber" = the Limber approximation, which
reduces the projection integrals over spherical Bessel functions to one
line-of-sight integral of P(k, z) at k = (ell + 1/2)/chi.
"""

import os
import sys

import numpy as np
from getdist import IniFile

# the shared notebook utilities live in cosmolike_core; the compiled
# interface is on the path already (each project's interface/
# directory is part of the Cocoa PYTHONPATH). ROOTDIR = the Cocoa
# folder, set by start_cocoa.sh; a KeyError here means Cocoa is not
# activated in this session.
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
# cnu = the shared notebook package (CAMB run, plots, Fisher helpers);
# ci = this project's compiled cosmolike library
import cosmolike_notebook_utils as cnu
from cosmolike_notebook_utils.fisher import nonlimber_derivative_mode
import cosmolike_des_y3_interface as ci


# ----------------------------------------------------------------------
# Project fiducial point (the evaluate point of the example notebooks)
# ----------------------------------------------------------------------
# Cosmology: As_1e9 = 10^9 A_s, ns, H0 [km/s/Mpc], omegab and omegam
# (Omega_b, Omega_m), mnu [eV] (sum of the neutrino masses), w = w0 and
# w0pwa = w0 + wa (equal values: wa = 0). Source side, one entry per
# source bin (4 bins): DES_DZ_S<i> = photo-z shift of the source n(z),
# DES_M<i> = shear calibration. Intrinsic alignment, TATT with a power
# law in (1+z)/1.62 (IA_redshift_evolution = 3): DES_A1_1, DES_A1_2 =
# the tidal-alignment amplitude A_1 and its redshift exponent eta_1;
# DES_A2_1, DES_A2_2 = the tidal-torquing amplitude A_2 and its exponent
# eta_2; DES_BTA_1 = b_TA, the source-density weighting of the
# alignment. Lens side, one entry per lens bin (5 bins): DES_DZ_L<i> =
# photo-z shift of the lens n(z), DES_B1_<i> = linear galaxy bias,
# DES_BMAG_<i> = magnification bias (fixed in the yaml, in cosmolike's
# convention), DES_PM_<i> = point mass of gamma_t (fixed to 0).
# These are the values of the sampler: evaluate: override blocks of
# EXAMPLE_EVALUATE1.yaml and EXAMPLE_EVALUATE2.yaml, except DES_A2_1:
# the yamls set 0.01, while the example notebooks use -1.7938938..., the
# value kept here so the wrappers and the notebooks share one point.
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
w = -0.9
w0pwa = -0.9
DES_DZ_S1 = 0.0414632
DES_DZ_S2 = 0.00147332
DES_DZ_S3 = 0.0237035
DES_DZ_S4 = -0.0773436
DES_M1 = 0.0191832
DES_M2 = -0.0431752
DES_M3 = -0.034961
DES_M4 = -0.0158096
DES_A1_1 = 0.606102            # TATT A_1 at 1+z = 1.62
DES_A1_2 = -1.51541            # its redshift exponent eta_1
DES_A2_1 = -1.7938938475734911 # TATT A_2 at 1+z = 1.62 (yamls: 0.01)
DES_A2_2 = -1.5448080290038528 # its redshift exponent eta_2
DES_BTA_1 = 0.8154011496506723 # b_TA
DES_DZ_L1 = 0.00457604
DES_DZ_L2 = 0.000309875
DES_DZ_L3 = 0.00855907
DES_DZ_L4 = -0.00316269
DES_DZ_L5 = -0.0146753
DES_B1_1 = 1.72716
DES_B1_2 = 1.65168
DES_B1_3 = 1.61423
DES_B1_4 = 1.92886
DES_B1_5 = 2.11633
DES_BMAG_1 = 0.63
DES_BMAG_2 = -3.04
DES_BMAG_3 = -1.33
DES_BMAG_4 = 2.50
DES_BMAG_5 = 1.93
DES_PM_1 = 0.0
DES_PM_2 = 0.0
DES_PM_3 = 0.0
DES_PM_4 = 0.0
DES_PM_5 = 0.0

# default nuisance vectors built from the constants above; wrappers
# take None and fall back to these, so a call overrides one vector
# without retyping the rest. Four entries for the source side, one per
# source bin, including the IA vectors, whose entries are the parameter
# slots of the IA model (IA_redshift_evolution = 3: slot 1 = amplitude,
# slot 2 = redshift exponent, the rest unused; set_nuisance_ia needs at
# least one entry per source bin). Five entries for the lens side, one
# per lens bin.
A1_FID = [DES_A1_1, DES_A1_2, 0, 0]
A2_FID = [DES_A2_1, DES_A2_2, 0, 0]
BTA_FID = [DES_BTA_1, 0, 0, 0]
SHEAR_PHOTOZ_FID = [DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4]
M_FID = [DES_M1, DES_M2, DES_M3, DES_M4]
LENS_PHOTOZ_FID = [DES_DZ_L1, DES_DZ_L2, DES_DZ_L3, DES_DZ_L4, DES_DZ_L5]
B1_FID = [DES_B1_1, DES_B1_2, DES_B1_3, DES_B1_4, DES_B1_5]
BMAG_FID = [DES_BMAG_1, DES_BMAG_2, DES_BMAG_3, DES_BMAG_4, DES_BMAG_5]
ZEROS5 = [0, 0, 0, 0, 0]
PM_FID = [DES_PM_1, DES_PM_2, DES_PM_3, DES_PM_4, DES_PM_5]

# ----------------------------------------------------------------------
# Per-notebook configuration
# ----------------------------------------------------------------------
# Every notebook mirrors its own yaml; the example yamls of this project
# agree on every entry here (lmax = 50000 for the shear and the 3x2pt
# notebooks alike), so configure() exists for per-notebook overrides the
# same way it does in the other projects. _CONFIG is module-global state:
# configure() changes it, and every later wrapper call reads it.
_CONFIG = {
    # Base early-exit tolerances; Fisher reads temporarily use zero.
    "nonlimber_tolerance_gg": 0.002,
    "nonlimber_tolerance_gs": 0.01,
    # FKEM multipole cap, separate from the low-rank engine cap.
    "lmax_nonlimber": 150,
    # Upper redshift, independent of sampling; matches this project YAML.
    "nonlimber_zmax": 4.0,
    "lmax": 50000,              # base of the internal C_ell tables
    "ntheta": 20,               # angular bins of the real-space vector
    "theta_min_arcmin": 2.5,
    "theta_max_arcmin": 250.0,
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    "path": "../../external_modules/data/des_y3",
    "data_file": "des_y3_real.dataset",
    "IA_model": 1,              # 0 = NLA, 1 = TATT
    "IA_redshift_evolution": 3,
    "IA_code": 0,               # 0 = C FASTPT (NLA always uses 0)
    # bias_model = redshift-evolution code of each bias term
    # [b1, b2, bs2, b3, bmag, bK] (0 = one amplitude per lens bin;
    # b3 = 1: computed from b1), as likelihood/combo_3x2pt.yaml sets it
    "bias_model": [0, 0, 0, 1, 0, 0],
    # n(z) photo-z conventions (mirror the likelihood yaml keys):
    # interpolation 0 = cspline, 1 = linear, 2+ = Steffen monotone;
    # z column 0 = Z_LOW (left bin edges), 1 = Z_MID (sample points)
    "photoz_histogram_akima": 0,
    "photoz_histogram_akima_factor": 3,  # 3, 5 or 7; independent of accuracy
    "photoz_interpolation_type": 0,
    "photoz_zmid_convention": 0,
    # C-FAST-PT internal (convolution) grid / output grid; 1.0 = equal
    "internal_accuracyboost": 1.0,
    # the projection of galaxy-galaxy lensing (gs) and clustering (gg),
    # mirroring the likelihood yaml keys and their defaults in
    # likelihood/_cosmolike_prototype_base.py: gs = 1, Limber at every
    # multipole; gg = 0, non-Limber (FKEM) below the multipole cap
    "adopt_limber_gs": 1,
    "adopt_limber_gg": 0,
    # the non-Limber FFTLog chi grid, refined on top of the accuracy boost
    "nonlimber_accuracyboost": 1.0,
    # 0 = perturbative galaxy bias, 1 = halo-model (HOD) galaxy power
    "include_HOD_GX": 0,
    # 0 = the init_IA model, 1 = halo-model intrinsic alignment
    "include_halo_IA": 0,
    # low-rank (SVD) non-Limber clustering: on/off, kept SVD ranks, and
    # its multipole cap
    "nonlimber_lowrank_gg": 0,
    "nonlimber_lowrank_rank": 5,
    "nonlimber_lowrank_lmax": 150,
}

# filled by init_cosmolike: the HDF5 file with every hydro simulation
# used by init_baryons_contamination
allsims = None


def configure(**overrides):
    """Sets this notebook's yaml-mirroring values, once per notebook.

    Every keyword must already exist in _CONFIG; an unknown name is
    almost always a typo, so it raises instead of being stored
    silently.

    The endpoint can also change after initialization. Updating the compiled
    setting here makes the next prediction rebuild the affected projections,
    including helpers that prepare CAMB directly instead of using _set_state.
    It must be finite and at least 4.0; invalid values leave _CONFIG intact.

    Arguments:
      overrides = keyword form of any _CONFIG entry, e.g.
                  configure(lmax=70000).

    Returns:
      nothing; later wrapper calls read the stored values.

    Raises:
      KeyError naming the unknown keyword and the valid names;
      ValueError when photoz_histogram_akima is not 0 or 1.
    """
    if "nonlimber_zmax" in overrides:
        endpoint = float(overrides["nonlimber_zmax"])
        if not np.isfinite(endpoint) or endpoint < 4.0:
            raise ValueError(
                f"nonlimber_zmax = {endpoint}; choose a finite value >= 4.0")

    if overrides.get("photoz_histogram_akima", 0) not in (0, 1):
        raise ValueError("photoz_histogram_akima must be 0 or 1")

    for name, value in overrides.items():
        if name not in _CONFIG:
            raise KeyError(
                f"configure() got unknown option '{name}'; valid options: "
                + ", ".join(sorted(_CONFIG)))
        _CONFIG[name] = value
    if "nonlimber_zmax" in overrides:
        ci.init_nonlimber_zmax(zmax=endpoint)


def init_cosmolike(CLprobe=None, with_data=False, lmax=None):
    """Set up the compiled interface once per notebook session.

    Reads the project's .dataset file (the small text file listing
    the n(z), covariance, mask, and data-vector files), then runs
    the init sequence of the likelihood (initialize in
    likelihood/_cosmolike_prototype_base.py), in its order: the probe
    set, the angular binning, the photo-z conventions, the C-FAST-PT
    grid, the projection choices (Limber or non-Limber for
    galaxy-galaxy lensing and clustering, the low-rank engine), the
    C_ell table length and accuracy, the n(z) tables, the data, the IA
    model and the galaxy-bias model. The chi2 machinery (covariance,
    mask, data vector) only loads when asked, because the
    plotting-only cells never need it.

    Arguments:
      CLprobe   = "xi", "3x2pt", ... to select the probe set and
                  (except for "xi") the galaxy-bias model, or None
                  to skip init_probes for a plotting-only session.
      with_data = True also loads covariance, mask, and data vector,
                  which get_chi2, get_dv and compute_probes need.
      lmax      = base lmax of the internal C_ell tables, or None
                  for the configure()d value.

    Returns:
      the parsed IniFile, for notebooks that read extra entries.

    Side effects:
      replaces the compiled interface's global state and sets this
      module's allsims to the hydro-simulation file of the dataset.
    """
    global allsims
    if lmax is None:
        lmax = _CONFIG["lmax"]
    ini = IniFile(os.path.normpath(
        os.path.join(_CONFIG["path"], _CONFIG["data_file"])))
    allsims = ini.relativeFileName('all_sims_hdf5_file')
    ci.initial_setup()
    ci.init_nonlimber_zmax(zmax=float(_CONFIG["nonlimber_zmax"]))
    ci.init_nonlimber_tolerance(
        gg_tol=float(_CONFIG["nonlimber_tolerance_gg"]),
        gs_tol=float(_CONFIG["nonlimber_tolerance_gs"]))
    ci.init_lmax_nonlimber(lmax=int(_CONFIG["lmax_nonlimber"]))
    if CLprobe is not None:
        ci.init_probes(possible_probes=CLprobe)
    ci.init_binning(int(ini.int("n_theta")),
                    ini.float("theta_min_arcmin"),
                    ini.float("theta_max_arcmin"))
    # the likelihood's yaml sets debug: false, which selects info logging
    ci.set_log_level_info()
    ci.init_photoz_histogram_akima(
        enabled=int(_CONFIG["photoz_histogram_akima"]),
        preparation_factor=int(_CONFIG["photoz_histogram_akima_factor"]))
    ci.init_photoz_conventions(
        interpolation_type=int(_CONFIG["photoz_interpolation_type"]),
        zmid_convention=int(_CONFIG["photoz_zmid_convention"]))
    # init_fpt_internal_boost comes before the first init_accuracy_boost
    # of the process, as in the likelihood: that call stores the
    # C-FAST-PT internal grid fraction (internal_accuracyboost) it finds
    # as the base every later call multiplies by the boost; called the
    # other way round, the base would be the C default 0.5 that
    # initial_setup restores
    ci.init_fpt_internal_boost(
        internal_boost=float(_CONFIG["internal_accuracyboost"]))
    # The projection options, set as the likelihood sets them: they are
    # cosmolike C globals, so without these calls the notebook would use
    # whatever an earlier init left behind.
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(_CONFIG["nonlimber_accuracyboost"]))
    ci.init_adopt_limber_gs(adopt_limber_gs=int(_CONFIG["adopt_limber_gs"]))
    ci.init_adopt_limber_gg(adopt_limber_gg=int(_CONFIG["adopt_limber_gg"]))
    ci.init_include_HOD_GX(include_HOD_GX=int(_CONFIG["include_HOD_GX"]))
    ci.init_include_halo_IA(include_halo_IA=int(_CONFIG["include_halo_IA"]))
    ci.init_lowrank_nonlimber_gg(
        flag=int(_CONFIG["nonlimber_lowrank_gg"]),
        nrank=int(_CONFIG["nonlimber_lowrank_rank"]),
        lmax=int(_CONFIG["nonlimber_lowrank_lmax"]))
    ci.init_ntable_lmax(lmax=int(lmax))
    # boost 1, integration level 0 for the start of the session; every
    # wrapper call sets both again (_set_state)
    ci.init_accuracy_boost(accuracy_boost=1.0, integration_accuracy=0)
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName('nz_lens_file'),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_multihisto_file=ini.relativeFileName('nz_source_file'),
        source_ntomo=int(ini.int("source_ntomo")))
    if with_data:
        ci.init_data_real(ini.relativeFileName('cov_file'),
                          ini.relativeFileName('mask_file'),
                          ini.relativeFileName('data_file'))
    ci.init_IA(ia_model=int(_CONFIG["IA_model"]),
               ia_redshift_evolution=int(_CONFIG["IA_redshift_evolution"]),
               ia_code=int(_CONFIG["IA_code"]))
    # galaxy bias exists only with lens galaxies: cosmic shear alone
    # ("xi") skips it, as the likelihood does
    if CLprobe is not None and CLprobe != "xi":
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    return ini


def _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=None, M=None, shear_photoz_bias=None,
               A1=None, A2=None, BTA=None,
               lens_photoz_bias=None, B1=None, B2=None,
               B_MAG=None, B3nl=None, BK=None, PM=None,
               baryon_sims=None, allsims_file=None):
    """Runs CAMB and pushes one complete state into the interface.

    This is the body every wrapper shares. The compiled interface
    keeps global state, so the sequence rebuilds everything a
    spectrum call reads: the accuracy settings and lookup tables, the
    binning when a real-space probe asked for it, the cosmology
    (power spectra, growth, distances from one CAMB run), and each
    nuisance group whose vectors were passed. A group passed as None
    is skipped, which leaves that part of the state at whatever the
    interface holds (a cosmic-shear wrapper never touches galaxy
    bias).

    Two differences from the likelihood: the neutrino mass is not an
    argument (the module constant mnu, 0.06 eV, is always used), and
    set_cosmology receives neither omegab nor the cold dark matter +
    baryon spectrum P_cb, so cosmolike holds Omega_b = 0 and no P_cb
    table. The spectra and correlation functions of this module read
    neither; a halo-model calculation would stop with an error, because
    cosmolike's sigma2 refuses to run without P_cb.

    Arguments:
      omegam ... non_linear_emul = the cosmology and accuracy
                 arguments, forwarded to cnu.get_camb_cosmology
                 (kmax in 1/Mpc; see its docstring for the grids).
      binning  = (ntheta, theta_min_arcmin, theta_max_arcmin) to
                 re-run init_binning (the real-space wrappers), or
                 None to keep the current binning.
      M, shear_photoz_bias, A1, A2, BTA = shear nuisance vectors
                 (M and the photo-z shifts gate the shear setters;
                 A1 gates the IA setter).
      lens_photoz_bias, B1, B2, B_MAG, B3nl, BK = clustering
                 nuisance vectors; passing them also selects the
                 configured galaxy-bias model.
      PM       = point-mass amplitudes, one per lens bin, or None.
      baryon_sims = a hydro simulation name to contaminate the
                 matter power with, or None to reset that state.
      allsims_file = HDF5 file for baryon_sims, or None for the one
                 init_cosmolike recorded.

    Returns:
      nothing; the interface state is the result.
    """
    ci.init_nonlimber_zmax(zmax=float(_CONFIG["nonlimber_zmax"]))
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_growth, z_interp_1D, chi,
     omegan2, lnPL_cb) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=non_linear_emul)
    # the accuracy combination every wrapper shares: the overall boost
    # multiplies the cosmolike boost, and the integration accuracy and
    # the C_ell table length grow with it
    CLAccuracyBoost = CLAccuracyBoost * AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_ntable_lmax(int(_CONFIG["lmax"] + 20000*(CLAccuracyBoost - 1)))
    ci.init_photoz_histogram_akima(
        enabled=int(_CONFIG["photoz_histogram_akima"]),
        preparation_factor=int(_CONFIG["photoz_histogram_akima_factor"]))
    ci.init_photoz_conventions(
        interpolation_type=int(_CONFIG["photoz_interpolation_type"]),
        zmid_convention=int(_CONFIG["photoz_zmid_convention"]))
    # init_fpt_internal_boost comes first, as in the likelihood:
    # init_accuracy_boost sets the C-FAST-PT internal grid fraction
    # (internal_accuracyboost) to base x CLAccuracyBoost, where base is
    # the fraction it found at its first call in the process; a fraction
    # set after init_accuracy_boost would discard the boost
    ci.init_fpt_internal_boost(
        internal_boost=float(_CONFIG["internal_accuracyboost"]))
    ci.init_accuracy_boost(CLAccuracyBoost, int(CLIntegrationAccuracy))
    if binning is not None:
        ci.init_binning(int(binning[0]), binning[1], binning[2])
    if B1 is not None:
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    # the growth table has its own z grid (z_growth, the dense 1D grid
    # cut at the last z_2D node), handed over as z_G, as the likelihood
    # does
    ci.set_cosmology(omegam=omegam,
                     H0=H0,
                     log10k_2D=log10k_interp_2D,
                     z_2D=z_interp_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_G=z_growth,
                     z_1D=z_interp_1D,
                     chi=chi,
                     omegan2=omegan2)
    if M is not None:
        ci.set_nuisance_shear_calib(M=M)
    if shear_photoz_bias is not None:
        ci.set_nuisance_shear_photoz(bias=shear_photoz_bias)
    if lens_photoz_bias is not None:
        ci.set_nuisance_clustering_photoz(bias=lens_photoz_bias)
    if B1 is not None:
        ci.set_nuisance_bias(B1=B1, B2=B2, B_MAG=B_MAG, B3nl=B3nl, BK=BK)
    if A1 is not None:
        ci.set_nuisance_ia(A1=A1, A2=A2, B_TA=BTA)
    if PM is not None:
        ci.set_point_mass(PMV=PM)
    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        if allsims_file is None:
            allsims_file = allsims
        ci.init_baryons_contamination(sim=baryon_sims, allsims=allsims_file)


def _shear_defaults(M, shear_photoz_bias, A1, A2, BTA):
    """Replaces None shear vectors with the fiducial ones.

    Arguments:
      M, shear_photoz_bias, A1, A2, BTA = shear nuisance vectors, each a
                 list of four values (one per source bin) or None.

    Returns:
      the same five vectors, in the argument order, with each None
      replaced by M_FID, SHEAR_PHOTOZ_FID, A1_FID, A2_FID or BTA_FID.
    """
    if M is None:
        M = M_FID
    if shear_photoz_bias is None:
        shear_photoz_bias = SHEAR_PHOTOZ_FID
    if A1 is None:
        A1 = A1_FID
    if A2 is None:
        A2 = A2_FID
    if BTA is None:
        BTA = BTA_FID
    return M, shear_photoz_bias, A1, A2, BTA


def _clustering_defaults(lens_photoz_bias, B1, B2, B_MAG, B3nl, BK):
    """Replaces None clustering vectors with the fiducial ones.

    Arguments:
      lens_photoz_bias, B1, B2, B_MAG, B3nl, BK = lens nuisance vectors
                 (photo-z shifts and bias terms), each a list of five
                 values (one per lens bin) or None.

    Returns:
      the same six vectors, in the argument order, with each None
      replaced by LENS_PHOTOZ_FID, B1_FID, BMAG_FID (the fixed
      magnification biases of the yaml) or, for the higher-order bias
      terms, ZEROS5.
    """
    if lens_photoz_bias is None:
        lens_photoz_bias = LENS_PHOTOZ_FID
    if B1 is None:
        B1 = B1_FID
    if B2 is None:
        B2 = ZEROS5
    if B_MAG is None:
        B_MAG = BMAG_FID
    if B3nl is None:
        B3nl = ZEROS5
    if BK is None:
        BK = ZEROS5
    return lens_photoz_bias, B1, B2, B_MAG, B3nl, BK


def C_ss_tomo_limber(ell, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
                     k_per_logint=10, CAMBAccuracyBoost=1.0,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Compute the cosmic-shear angular power spectra (EE, BB) at ell.

    Rebuilds the full interface state (see _set_state) and evaluates
    ci.C_ss_tomo_limber. The nuisance vectors default to the module
    fiducials when passed as None.

    Arguments:
      ell = 1D array of multipoles; the rest as in _set_state, with
      the shear group only (this wrapper never touches clustering).
      allsims = HDF5 file of the baryon_sims simulation, or None for
      the file init_cosmolike recorded.

    Returns:
      (EE, BB): two 3D arrays (n_ell, n_source, n_source), indexed
      [ell, bin i, bin j]; only entries with i <= j are filled, the
      others are 0. EE is the E-mode (lensing) spectrum; BB, the
      B-mode spectrum, is nonzero only through the TATT terms.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_ss_tomo_limber(l=ell)


def xi(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
       omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
       w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
       shear_photoz_bias=None, M=None, baryon_sims=None,
       AccuracyBoost=1.0, kmax=5.0, k_per_logint=10,
       CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
       CLIntegrationAccuracy=0, non_linear_emul=None, allsims=None):
    """Compute the shear correlation functions xi_+ and xi_- on a theta grid.

    Same state build as C_ss_tomo_limber plus a re-binning, so the
    binning can change between calls without restarting the kernel;
    the binning arguments default to the configure()d values.

    Arguments:
      ntheta, theta_min_arcmin, theta_max_arcmin = number of log-spaced
      angular bins and their range [arcmin]; the rest as in
      C_ss_tomo_limber.

    Returns:
      (theta, xi_plus, xi_minus): theta = the area-weighted bin centers
      in arcmin; xi_plus, xi_minus = 3D arrays (n_theta, n_source,
      n_source), both bin orderings filled.
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (xip, xim) = ci.xi_pm_tomo()
    return (ci.get_binning_real_space(), xip, xim)


def C_gs_tomo_limber(ell, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     lens_photoz_bias=None, galaxy_bias_b1=None,
                     galaxy_bias_b2=None, galaxy_bias_bmag=None,
                     galaxy_bias_b3nl=None, galaxy_bias_bk=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
                     k_per_logint=10, CAMBAccuracyBoost=1.0,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Compute the galaxy-galaxy lensing spectra C_gs at multipoles ell.

    Builds both the shear and the clustering state (both samples enter
    ggl) and evaluates ci.C_gs_tomo_limber.

    Arguments:
      ell = 1D array of multipoles; galaxy_bias_b1, galaxy_bias_b2,
      galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk = the bias
      vectors B1, B2, B_MAG, B3nl, BK of _set_state; the rest as in
      C_ss_tomo_limber.

    Returns:
      3D array (n_ell, n_lens, n_source); only the lens-source pairs of
      the data vector are filled, the other entries are 0.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_gs_tomo_limber(l=ell)


def C_gg_tomo(ell, limber, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
              As_1e9=As_1e9, w=w, w0pwa=w0pwa,
              lens_photoz_bias=None, galaxy_bias_b1=None,
              galaxy_bias_b2=None, galaxy_bias_bmag=None,
              galaxy_bias_b3nl=None, galaxy_bias_bk=None,
              baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
              k_per_logint=10, CAMBAccuracyBoost=1.0,
              CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
              non_linear_emul=None, allsims=None):
    """Compute the galaxy-clustering spectra C_gg at multipoles ell.

    Clustering state only. limber = 1 evaluates the Limber
    approximation, anything else the non-Limber computation, so the
    two can be compared on the same state.

    Arguments:
      ell = 1D array of multipoles; limber = 1 (Limber) or another
      value (non-Limber); the bias vectors as in C_gs_tomo_limber.

    Returns:
      3D array (n_ell, n_lens, n_lens); only the diagonal entries
      (auto-spectra of each lens bin) are filled, the others are 0.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    if limber == 1:
        return ci.C_gg_tomo_limber(l=ell)
    else:
        return ci.C_gg_tomo(l=ell)


def gamma_t(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
            omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
            w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
            shear_photoz_bias=None, M=None,
            lens_photoz_bias=None, galaxy_bias_b1=None,
            galaxy_bias_b2=None, galaxy_bias_bmag=None,
            galaxy_bias_b3nl=None, galaxy_bias_bk=None, PM=None,
            baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
            k_per_logint=10, CAMBAccuracyBoost=1.0,
            CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
            non_linear_emul=None, allsims=None):
    """Compute the tangential shear gamma_t on a theta grid.

    Full 3x2pt nuisance state (shear, clustering, point masses) plus
    a re-binning, then ci.w_gammat_tomo.

    Arguments:
      PM = point-mass amplitudes, one per lens bin, or None for PM_FID;
      the binning as in xi, the bias vectors as in C_gs_tomo_limber.

    Returns:
      (theta, gammat): theta in arcmin, gammat a 3D array
      (n_theta, n_lens, n_source); only the lens-source pairs of the
      data vector are filled, the other entries are 0.
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if PM is None:
        PM = PM_FID
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk, PM=PM,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.w_gammat_tomo())


def w_theta(ntheta=None, theta_min_arcmin=None, theta_max_arcmin=None,
            omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
            w=w, w0pwa=w0pwa,
            lens_photoz_bias=None, galaxy_bias_b1=None,
            galaxy_bias_b2=None, galaxy_bias_bmag=None,
            galaxy_bias_b3nl=None, galaxy_bias_bk=None,
            baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
            k_per_logint=10, CAMBAccuracyBoost=1.0,
            CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
            non_linear_emul=None, allsims=None):
    """Compute the angular clustering w(theta) on a theta grid.

    Clustering state plus a re-binning, then ci.w_gg_tomo. The state
    stays in the interface after the call, so the UV-fraction scale
    cut (ci.w_gg_scale_cuts_tomo, ci.w_gg_kwindow_tomo) can run right
    after it on the same model.

    Arguments:
      the binning as in xi, the bias vectors as in C_gs_tomo_limber.

    Returns:
      (theta, wtheta): theta in arcmin, wtheta a 3D array
      (n_theta, n_lens, n_lens); only the diagonal (each lens bin with
      itself) is filled, and the plots read it.
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return (ci.get_binning_real_space(), ci.w_gg_tomo())


def get_chi2(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
             w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
             shear_photoz_bias=None, M=None,
             lens_photoz_bias=None, galaxy_bias_b1=None,
             galaxy_bias_b2=None, galaxy_bias_bmag=None,
             galaxy_bias_b3nl=None, galaxy_bias_bk=None, PM=None,
             baryon_sims=None, AccuracyBoost=1.0, kmax=5.0,
             k_per_logint=10, CAMBAccuracyBoost=1.0,
             CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
             non_linear_emul=None, allsims=None):
    """Return the chi2 of the masked theory data vector against the data.

    Requires init_cosmolike(CLprobe=..., with_data=True) first: the
    probe selection fixes which blocks enter the masked vector, and
    with_data loads the covariance, mask, and data vector this chi2
    compares against. The full 3x2pt nuisance state is set every
    call; blocks outside the selected probes never read theirs (a
    "xi" run ignores the clustering state).

    Arguments:
      the cosmology, nuisance and accuracy arguments of gamma_t.

    Returns:
      float chi2 = (theory - data)^T C^-1 (theory - data) over the
      unmasked entries.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if PM is None:
        PM = PM_FID
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk, PM=PM,
               baryon_sims=baryon_sims, allsims_file=allsims)
    datavector = np.array(ci.compute_data_vector_masked())
    return ci.compute_chi2(datavector)


# ----------------------------------------------------------------------
# Responses of the shear statistics to the power at wavenumber k
# ----------------------------------------------------------------------
def dlnC_dlss_tomo_limber(k, ell, omegam=omegam, omegab=omegab, H0=H0,
                          ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                          A1=None, A2=None, BTA=None,
                          shear_photoz_bias=None, M=None,
                          baryon_sims=None, AccuracyBoost=1.0,
                          kmax=10.0, k_per_logint=10,
                          CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                          CLIntegrationAccuracy=0,
                          non_linear_emul=None, allsims=None):
    """Compute the response d ln C_ss / d ln k at wavenumbers k, multipoles ell.

    The response is the contribution of the matter power at wavenumber k
    to the shear spectrum per unit ln k. Shear state as in
    C_ss_tomo_limber, then the interface's response evaluation.

    Arguments:
      k = 1D array of wavenumbers [h/Mpc]; ell = 1D array of multipoles;
      the rest as in C_ss_tomo_limber (kmax defaults to 10/Mpc here).

    Returns:
      (EE, BB): two 4D arrays (n_k, n_ell, n_source, n_source), as
      ci.dlnC_ss_dlnk_tomo_limber returns them; only entries with
      source bin i <= j are filled.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.dlnC_ss_dlnk_tomo_limber(k=k, l=ell)


def dlnxi_dlnk_pm_tomo_limber(k, ntheta=None, theta_min_arcmin=None,
                              theta_max_arcmin=None, omegam=omegam,
                              omegab=omegab, H0=H0, ns=ns,
                              As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                              A1=None, A2=None, BTA=None,
                              shear_photoz_bias=None, M=None,
                              baryon_sims=None, AccuracyBoost=1.0,
                              kmax=10.0, k_per_logint=10,
                              CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                              CLIntegrationAccuracy=0,
                              non_linear_emul=None, allsims=None):
    """Compute the response d ln xi_+- / d ln k at wavenumbers k.

    Shear state plus a re-binning, as in xi.

    Arguments:
      k = 1D array of wavenumbers [h/Mpc]; the rest as in xi.

    Returns:
      (theta, dlnxip_dlnk, dlnxim_dlnk): theta in arcmin and two 4D
      arrays (n_k, n_theta, n_source, n_source), both bin orderings
      filled.
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (dlnxip_dlnk, dlnxim_dlnk) = ci.dlnxi_dlnk_pm_tomo_limber(k=k)
    return (ci.get_binning_real_space(), dlnxip_dlnk, dlnxim_dlnk)


def rf_C_ss_tomo_limber(k, ell, omegam=omegam, omegab=omegab, H0=H0,
                        ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                        A1=None, A2=None, BTA=None,
                        shear_photoz_bias=None, M=None,
                        baryon_sims=None, AccuracyBoost=1.0,
                        kmax=10.0, k_per_logint=10,
                        CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
                        CLIntegrationAccuracy=0,
                        non_linear_emul=None, allsims=None):
    """Compute the cumulative response RF(k_max) of C_ss.

    RF(k_max) = S(k_max)/S(k_hi), where S(K) integrates
    |d ln C_ss/d ln k| over ln k up to K on the tabulated k range,
    whose upper end is k_hi (arXiv:2011.06469, eq. 17): the fraction
    of the spectrum's response coming from wavenumbers below k_max,
    rising from 0 to 1. Shear state as in C_ss_tomo_limber, then
    ci.rf_C_ss_tomo_limber.

    Arguments:
      k = 1D array of cutoffs k_max [h/Mpc]; ell = 1D array of
      multipoles; the rest as in C_ss_tomo_limber.

    Returns:
      (EE, BB): two 4D arrays (n_k, n_ell, n_source, n_source), as
      ci.rf_C_ss_tomo_limber returns them; only entries with source
      bin i <= j are filled.
    """
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.rf_C_ss_tomo_limber(k=k, l=ell)


def rf_xi_tomo_limber(k, ntheta=None, theta_min_arcmin=None,
                      theta_max_arcmin=None, omegam=omegam,
                      omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
                      w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
                      shear_photoz_bias=None, M=None,
                      baryon_sims=None, AccuracyBoost=1.0, kmax=10.0,
                      k_per_logint=10, CAMBAccuracyBoost=1.0,
                      CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                      non_linear_emul=None, allsims=None):
    """Compute the cumulative response RF(k_max) of xi_+-.

    The real-space version of rf_C_ss_tomo_limber. Shear state plus a
    re-binning, then ci.rf_xi_tomo_limber.

    Arguments:
      k = 1D array of cutoffs k_max [h/Mpc]; the rest as in xi.

    Returns:
      (theta, rf_xip, rf_xim): theta in arcmin and two 4D arrays
      (n_k, n_theta, n_source, n_source), both bin orderings filled.
    """
    if ntheta is None:
        ntheta = _CONFIG["ntheta"]
    if theta_min_arcmin is None:
        theta_min_arcmin = _CONFIG["theta_min_arcmin"]
    if theta_max_arcmin is None:
        theta_max_arcmin = _CONFIG["theta_max_arcmin"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               binning=(ntheta, theta_min_arcmin, theta_max_arcmin),
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    (rf_xip, rf_xim) = ci.rf_xi_tomo_limber(k=k)
    return (ci.get_binning_real_space(), rf_xip, rf_xim)


# ----------------------------------------------------------------------
# Fisher forecasting
# ----------------------------------------------------------------------
# Two flat parameter vectors drive the Fisher machinery. A Fisher matrix
# F approximates the inverse parameter covariance from the derivatives
# of the data vector: F_ab = (d dv/d p_a)^T C^-1 (d dv/d p_b).
#
# Cosmic shear, 20 entries: the seven cosmological parameters (indices
# 0-6: 10^9 A_s, n_s, H_0, Omega_b, Omega_m, w0, wa), the five TATT
# numbers (7-11: A_1, eta_1, A_2, eta_2, b_TA), the four source photo-z
# shifts (12-15) and the four shear calibrations (16-19), in this order.
# w0 and wa are free, so the figure of merit of (w0, wa) exists; wa has
# the fiducial 0, so the derivatives step it by an absolute h (the
# |p0| <= 1e-12 rule of cnu.get_ddv). The neutrino mass stays at the
# module constant mnu.
#
# 3x2pt, 30 entries: the 20 above, then the five lens photo-z shifts
# (20-24) and the five linear galaxy biases (25-29). The magnification
# biases and the point masses are fixed, as in EXAMPLE_EVALUATE2.yaml.
#
# The names, LaTeX labels, and priors below index into the same vectors.
FISHER_PARAM_FID = np.array(
    [As_1e9, ns, H0, omegab, omegam, w, w0pwa - w,
     DES_A1_1, DES_A1_2, DES_A2_1, DES_A2_2, DES_BTA_1,
     DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4,
     DES_M1, DES_M2, DES_M3, DES_M4], dtype="float64")

FISHER_PARAM_NAMES = [
    "As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa",
    "DES_A1_1", "DES_A1_2", "DES_A2_1", "DES_A2_2", "DES_BTA_1",
    "DES_DZ_S1", "DES_DZ_S2", "DES_DZ_S3", "DES_DZ_S4",
    "DES_M1", "DES_M2", "DES_M3", "DES_M4",
]

FISHER_PARAM_LABELS = [
    r"10^{9} A_s", r"n_s", r"H_0", r"\Omega_b", r"\Omega_m",
    r"w_0", r"w_a",
    r"A_{1}", r"\eta_{1}", r"A_{2}", r"\eta_{2}", r"b_\mathrm{TA}",
    r"\Delta z_\mathrm{s}^1", r"\Delta z_\mathrm{s}^2",
    r"\Delta z_\mathrm{s}^3", r"\Delta z_\mathrm{s}^4",
    r"m^1", r"m^2", r"m^3", r"m^4",
]

FISHER_3X2PT_PARAM_FID = np.concatenate(
    [FISHER_PARAM_FID,
     np.array([DES_DZ_L1, DES_DZ_L2, DES_DZ_L3, DES_DZ_L4, DES_DZ_L5,
               DES_B1_1, DES_B1_2, DES_B1_3, DES_B1_4, DES_B1_5],
              dtype="float64")])

FISHER_3X2PT_PARAM_NAMES = FISHER_PARAM_NAMES + [
    "DES_DZ_L1", "DES_DZ_L2", "DES_DZ_L3", "DES_DZ_L4", "DES_DZ_L5",
    "DES_B1_1", "DES_B1_2", "DES_B1_3", "DES_B1_4", "DES_B1_5",
]

FISHER_3X2PT_PARAM_LABELS = FISHER_PARAM_LABELS + [
    r"\Delta z_\mathrm{l}^1", r"\Delta z_\mathrm{l}^2",
    r"\Delta z_\mathrm{l}^3", r"\Delta z_\mathrm{l}^4",
    r"\Delta z_\mathrm{l}^5",
    r"b_1^1", r"b_1^2", r"b_1^3", r"b_1^4", r"b_1^5",
]

# flat priors clip the getdist contours of the unbounded parameters;
# {index: (min, max)}, the prior ranges of the example yaml files for
# the cosmological parameters, the ones the notebooks draw (indices 0-6,
# the same in both layouts). wa has no entry: the yamls bound w0 + wa
# (-5 to -0.01), not wa itself. cnu.plot_Fisher reads the keys as
# positions in its select list, so the dictionary holds only entries
# that a select starting with 0, 1, ..., 6 reaches.
FISHER_FLAT_PRIORS = {
    0: (0.5, 5),
    1: (0.87, 1.07),
    2: (55.0, 91.0),
    3: (0.03, 0.07),
    4: (0.1, 0.9),
    5: (-3.0, -0.01),
}

# Gaussian priors on the calibration nuisances, (index: (mean, sigma)),
# as likelihood/params_source.yaml sets them: the source photo-z shifts
# and the shear calibrations of Table I of arXiv:2105.13549. Only sigma
# enters a Fisher matrix (cnu.add_gaussian_priors adds 1/sigma^2 to the
# diagonal); the means are recorded for reference.
FISHER_GAUSSIAN_PRIORS = {
    12: (0.0, 0.018),
    13: (0.0, 0.015),
    14: (0.0, 0.011),
    15: (0.0, 0.017),
    16: (-0.006, 0.009),
    17: (-0.02, 0.008),
    18: (-0.024, 0.008),
    19: (-0.037, 0.008),
}

# the 3x2pt vector adds the lens photo-z priors of params_lens.yaml
# (Table IV of arXiv:2105.13549) to the Gaussian ones; its flat priors
# for the contour plots are FISHER_FLAT_PRIORS, since the cosmological
# indices are the same in both layouts
FISHER_3X2PT_GAUSSIAN_PRIORS = dict(FISHER_GAUSSIAN_PRIORS)
FISHER_3X2PT_GAUSSIAN_PRIORS[20] = (0.006, 0.004)
FISHER_3X2PT_GAUSSIAN_PRIORS[21] = (0.001, 0.003)
FISHER_3X2PT_GAUSSIAN_PRIORS[22] = (0.004, 0.003)
FISHER_3X2PT_GAUSSIAN_PRIORS[23] = (-0.002, 0.005)
FISHER_3X2PT_GAUSSIAN_PRIORS[24] = (-0.007, 0.01)

# getdist samples for the Fisher contours come from this random
# generator, seeded with 0 and created once, at import: repeated plot
# calls draw successive samples from one reproducible stream, and
# re-importing the module restarts it
_FISHER_RNG = np.random.default_rng(0)


def fisher_fiducial_point():
    """Return a fresh copy of the cosmic-shear Fisher fiducial vector.

    Returns a copy so a notebook can shift entries for a forecast
    without editing the module's fiducial in place.

    Returns:
      1D float64 array of 20 entries in the FISHER_PARAM_NAMES order.
    """
    return FISHER_PARAM_FID.copy()


def fisher_3x2pt_fiducial_point():
    """Return a fresh copy of the 3x2pt Fisher fiducial vector.

    Returns:
      1D float64 array of 30 entries in the FISHER_3X2PT_PARAM_NAMES
      order.
    """
    return FISHER_3X2PT_PARAM_FID.copy()


def _check_fisher_vector(param, expected, caller):
    """Stops a Fisher data-vector call whose parameter vector has the wrong length.

    The two Fisher layouts differ only in length (20 and 30 entries), so
    a 3x2pt vector passed to the cosmic-shear get_dv, or the reverse,
    would read the wrong slots without this check.

    Arguments:
      param    = the parameter vector the caller received.
      expected = the length of the caller's layout.
      caller   = the caller's name, for the message.

    Raises:
      ValueError naming the caller, the received length and the layout.
    """
    if len(param) != expected:
        raise ValueError(
            f"{caller}: the parameter vector has {len(param)} entries, but "
            f"its layout has {expected}; pass fisher_fiducial_point() (20) "
            "to get_dv and fisher_3x2pt_fiducial_point() (30) to "
            "get_dv_3x2pt, or a shifted copy of one of them")


def get_dv(param=None, AccuracyBoost=1.0):
    """Return the masked cosmic-shear data vector at a flat parameter vector.

    The Fisher derivatives evaluate this at shifted copies of the
    fiducial vector; the layout is the one FISHER_PARAM_NAMES
    documents. w0 = param[5] and wa = param[6] reach CAMB as
    w = w0 and w0pwa = w0 + wa. The probes in the vector are the ones
    init_cosmolike selected (CLprobe = "xi" for cosmic shear); the
    clustering state is not set here.

    Arguments:
      param = 1D float array in the FISHER_PARAM_NAMES layout, or
              None for the fiducial vector.
      AccuracyBoost = the overall boost, passed to _set_state both as
              AccuracyBoost and as CLAccuracyBoost, so the cosmolike
              boost there is AccuracyBoost squared.

    Returns:
      1D float64 array: the masked data vector (masked entries 0).

    Raises:
      ValueError when param does not have the 20 entries of the layout.
    """
    if param is None:
        param = FISHER_PARAM_FID
    _check_fisher_vector(param, len(FISHER_PARAM_NAMES), "get_dv")
    A1 = [param[7], param[8], 0, 0]
    A2 = [param[9], param[10], 0, 0]
    BTA = [param[11], 0, 0, 0]
    shear_photoz_bias = [param[12], param[13], param[14], param[15]]
    M = [param[16], param[17], param[18], param[19]]
    # positional arguments of _set_state: omegam, omegab, H0, ns, As_1e9
    # (param[4] down to param[0]), w = w0 and w0pwa = w0 + wa,
    # AccuracyBoost, kmax = 5.0 [1/Mpc], k_per_logint = 10,
    # CAMBAccuracyBoost = 1, CLAccuracyBoost = AccuracyBoost,
    # CLIntegrationAccuracy = 0 and the configured non_linear_emul
    _set_state(param[4], param[3], param[2], param[1], param[0],
               param[5], param[5] + param[6],
               AccuracyBoost, 5.0, 10, 1.0,
               AccuracyBoost, 0, _CONFIG["non_linear_emul"],
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA)
    # A stencil must not change its stopping multipole between points.
    # Restore configured bases, not the accuracy-scaled getter values.
    with nonlimber_derivative_mode(
            interface=ci,
            production_gg=_CONFIG["nonlimber_tolerance_gg"],
            production_gs=_CONFIG["nonlimber_tolerance_gs"]):
        return np.array(ci.compute_data_vector_masked(), dtype=np.float64)


def get_dv_3x2pt(param=None, AccuracyBoost=1.0):
    """Return the masked 3x2pt data vector at a flat parameter vector.

    The 3x2pt counterpart of get_dv, in the FISHER_3X2PT_PARAM_NAMES
    layout: the cosmic-shear entries of get_dv, then the lens photo-z
    shifts and the linear galaxy biases. The magnification biases
    (BMAG_FID) and the point masses (PM_FID) stay fixed, and the
    higher-order bias terms are zero, as in EXAMPLE_EVALUATE2.yaml.
    Requires init_cosmolike(CLprobe="3x2pt", with_data=True).

    Arguments:
      param = 1D float array in the FISHER_3X2PT_PARAM_NAMES layout, or
              None for the fiducial vector.
      AccuracyBoost = as in get_dv (the cosmolike boost is its square).

    Returns:
      1D float64 array: the masked data vector (masked entries 0).

    Raises:
      ValueError when param does not have the 30 entries of the layout.
    """
    if param is None:
        param = FISHER_3X2PT_PARAM_FID
    _check_fisher_vector(param, len(FISHER_3X2PT_PARAM_NAMES),
                         "get_dv_3x2pt")
    A1 = [param[7], param[8], 0, 0]
    A2 = [param[9], param[10], 0, 0]
    BTA = [param[11], 0, 0, 0]
    shear_photoz_bias = [param[12], param[13], param[14], param[15]]
    M = [param[16], param[17], param[18], param[19]]
    lens_photoz_bias = [param[20], param[21], param[22], param[23],
                        param[24]]
    B1 = [param[25], param[26], param[27], param[28], param[29]]
    # the positional arguments as in get_dv
    _set_state(param[4], param[3], param[2], param[1], param[0],
               param[5], param[5] + param[6],
               AccuracyBoost, 5.0, 10, 1.0,
               AccuracyBoost, 0, _CONFIG["non_linear_emul"],
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=B1, B2=ZEROS5,
               B_MAG=BMAG_FID, B3nl=ZEROS5, BK=ZEROS5, PM=PM_FID)
    # A stencil must not change its stopping multipole between points.
    # Restore configured bases, not the accuracy-scaled getter values.
    with nonlimber_derivative_mode(
            interface=ci,
            production_gg=_CONFIG["nonlimber_tolerance_gg"],
            production_gs=_CONFIG["nonlimber_tolerance_gs"]):
        return np.array(ci.compute_data_vector_masked(), dtype=np.float64)


def get_ddv(index=0, h=0.02, CV=None, AccuracyBoost=1.0):
    """Return the derivative of get_dv along one parameter (cnu.get_ddv).

    Five-point finite-difference stencil: four get_dv evaluations at
    relative steps +-h and +-2h around CV.

    Arguments:
      index = position of the parameter in the FISHER_PARAM_NAMES order.
      h     = relative step (0.02 = 2%; absolute when the fiducial is 0,
              as for wa).
      CV    = fiducial vector, or None for FISHER_PARAM_FID.
      AccuracyBoost = forwarded to get_dv.

    Returns:
      1D array: d(data vector)/d(parameter), the data-vector length.
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    return cnu.get_ddv(get_dv, index=index, h=h, CV=CV,
                       AccuracyBoost=AccuracyBoost)


def get_ddv_3x2pt(index=0, h=0.02, CV=None, AccuracyBoost=1.0):
    """Return the derivative of get_dv_3x2pt along one parameter.

    Arguments:
      as get_ddv, in the FISHER_3X2PT_PARAM_NAMES layout; CV = None
      selects FISHER_3X2PT_PARAM_FID.

    Returns:
      1D array: d(data vector)/d(parameter), the data-vector length.
    """
    if CV is None:
        CV = FISHER_3X2PT_PARAM_FID
    return cnu.get_ddv(get_dv_3x2pt, index=index, h=h, CV=CV,
                       AccuracyBoost=AccuracyBoost)


def get_Fisher(CV=None, h=0.02, AccuracyBoost=3.1, ddv=None,
               priors=None, invcov=None):
    """Return the cosmic-shear Fisher matrix from five-point derivatives.

    Thin binding of cnu.get_Fisher to this project's data vector,
    priors, and masked inverse covariance (fetched from the
    interface when not passed, so init_cosmolike with data must
    have run). F = D^T C^-1 D, with one derivative column of D per
    parameter, plus the Gaussian priors on the diagonal.

    Arguments:
      CV    = fiducial vector, or None for FISHER_PARAM_FID.
      h     = relative step of the derivatives.
      AccuracyBoost = forwarded to every data-vector evaluation (see
              get_dv: the cosmolike boost becomes its square).
      ddv   = derivative function, or None for get_ddv.
      priors = {index: (mean, sigma)}, or None for
              FISHER_GAUSSIAN_PRIORS.
      invcov = inverse covariance (n_data, n_data), or None for the
              masked one loaded by init_cosmolike.

    Returns:
      2D array (len(CV), len(CV)): (20, 20), in the FISHER_PARAM_NAMES
      order, for the default CV.
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    if ddv is None:
        ddv = get_ddv
    if priors is None:
        priors = FISHER_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher(ddv, CV=CV, h=h, AccuracyBoost=AccuracyBoost,
                          priors=priors, invcov=invcov)


def get_Fisher_3x2pt(CV=None, h=0.02, AccuracyBoost=1.0, priors=None,
                     invcov=None):
    """Return the 3x2pt Fisher matrix from five-point derivatives.

    get_Fisher with get_ddv_3x2pt, the 3x2pt fiducial vector and the
    3x2pt priors; init_cosmolike(CLprobe="3x2pt", with_data=True) must
    have run.

    Arguments:
      CV, h, AccuracyBoost, invcov = as in get_Fisher (CV = None selects
              FISHER_3X2PT_PARAM_FID).
      priors = {index: (mean, sigma)}, or None for
              FISHER_3X2PT_GAUSSIAN_PRIORS.

    Returns:
      2D array (len(CV), len(CV)): (30, 30), in the
      FISHER_3X2PT_PARAM_NAMES order, for the default CV.
    """
    if CV is None:
        CV = FISHER_3X2PT_PARAM_FID
    if priors is None:
        priors = FISHER_3X2PT_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher(get_ddv_3x2pt, CV=CV, h=h,
                          AccuracyBoost=AccuracyBoost, priors=priors,
                          invcov=invcov)


def get_ddv_dkit(index=0, CV=None, AccuracyBoost=1.0, min_samples=7,
                 fallback_mode="poly_at_floor"):
    """Return the derivative of get_dv via derivkit (cnu.get_ddv_dkit).

    derivkit (an optional package) fits polynomials through adaptively
    chosen get_dv evaluations and differentiates the fit, which is less
    sensitive to numerical noise than a fixed-step stencil.

    Arguments:
      index = position of the parameter in the FISHER_PARAM_NAMES order.
      CV    = fiducial vector, or None for FISHER_PARAM_FID.
      AccuracyBoost = forwarded to get_dv.
      min_samples, fallback_mode = derivkit settings (minimum number of
              fitted points; the strategy when the fit is rejected).

    Returns:
      1D array: d(data vector)/d(parameter).
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    return cnu.get_ddv_dkit(get_dv, index=index, CV=CV,
                            AccuracyBoost=AccuracyBoost,
                            min_samples=min_samples,
                            fallback_mode=fallback_mode)


def get_Fisher2(CV=None, AccuracyBoost=3.0, priors=None, invcov=None,
                min_samples=7, fallback_mode="poly_at_floor"):
    """Return the cosmic-shear Fisher matrix from derivkit derivatives.

    Arguments:
      as get_Fisher, with min_samples and fallback_mode forwarded to
      get_ddv_dkit.

    Returns:
      2D array (len(CV), len(CV)): (20, 20), in the FISHER_PARAM_NAMES
      order, for the default CV.
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    if priors is None:
        priors = FISHER_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher2(get_dv, CV=CV, AccuracyBoost=AccuracyBoost,
                           priors=priors, invcov=invcov,
                           min_samples=min_samples,
                           fallback_mode=fallback_mode)


def plot_Fisher(F, mu, F2=None, root=None, select=None, labels=None,
                names=None, filled=True, flat_priors=None,
                chain_names=None):
    """Draw the Fisher contour triangle via getdist (cnu.plot_Fisher).

    Each Fisher matrix becomes a cloud of samples from the Gaussian with
    mean mu and covariance F^-1 (drawn with _FISHER_RNG), cut to the
    flat-prior boxes and plotted by getdist; an MCMC chain named by root
    can be overlaid.

    Arguments:
      F, F2 = Fisher matrix, or a list of them (F2 drawn after F).
      mu    = fiducial vector (the Gaussian mean), full length.
      root  = getdist chain root to overlay, or None.
      select = parameter indices to show, or None for all.
      labels, names = LaTeX labels and getdist names, or None for
              FISHER_PARAM_LABELS and FISHER_PARAM_NAMES (pass the
              FISHER_3X2PT_* lists for a 3x2pt matrix).
      filled = True fills the 2D contours.
      flat_priors = {position in select: (min, max)}, or None for
              FISHER_FLAT_PRIORS, whose keys match the full indices only
              when select keeps parameters 0 to 5 first (select = None
              keeps them too).
      chain_names = legend labels, one per drawn set, or None.

    Returns:
      the getdist subplot plotter holding the figure; nothing is saved.
    """
    if labels is None:
        labels = FISHER_PARAM_LABELS
    if names is None:
        names = FISHER_PARAM_NAMES
    if flat_priors is None:
        flat_priors = FISHER_FLAT_PRIORS
    return cnu.plot_Fisher(F, mu, F2=F2, root=root, select=select,
                           labels=labels, names=names, filled=filled,
                           flat_priors=flat_priors,
                           chain_names=chain_names, rng=_FISHER_RNG)


# ----------------------------------------------------------------------
# Baryonic feedback via the bfmt theory block
# ----------------------------------------------------------------------
def get_baryon_suppression(theory_options, point, z_grid, log10k_grid):
    """Return the suppression S(k, z) of the bfmt theory block on a grid.

    S = P(k) with baryonic feedback / P(k) without it. Builds a minimal
    Cobaya model (CAMB + bfmt + the likelihood "one", which returns
    ln L = 0 and only makes the model complete), requests the
    baryon_suppression product at the given grid (k in 1/Mpc, as the
    Cosmolike likelihoods send it; the block converts to h/Mpc
    internally), evaluates it at this module's fiducial cosmology, and
    returns {z: S array over k}.

    Arguments:
      theory_options = bfmt options dict, e.g. {"baryon_model": 2}.
      point   = {parameter name: value} for the method's feedback
                parameters, fixed in the model.
      z_grid  = redshifts of the evaluation grid.
      log10k_grid = log10 of the wavenumbers, read as k in 1/Mpc.

    Returns:
      {z: 1D S array over k}, one entry per z_grid value.
    """
    from cobaya.model import get_model
    # Cobaya model description: CAMB at this module's fiducial
    # cosmology (tau = 0.0543, a Planck 2018 value, only completes
    # CAMB's input; omch2 subtracts the massive-neutrino density),
    # bfmt with the caller's options and feedback parameters, and
    # debug = 50 (logging.CRITICAL: only critical messages print)
    info = {
        "likelihood": {"one": None},
        "theory": {
            # no "path" for camb: the session already imported it,
            # and cobaya accepts the loaded module as is
            "camb": {"extra_args": {"halofit_version": "takahashi",
                                    "dark_energy_model": "ppf"}},
            "bfmt": dict({"python_path": os.environ["ROOTDIR"]
                          + "/external_modules/code/baryon_suppression"},
                         **theory_options),
        },
        "params": dict({
            "As": {"value": As_1e9*1e-9},
            "ns": ns, "H0": H0, "mnu": mnu, "tau": 0.0543,
            "w": w,
            "ombh2": omegab*(H0/100)**2,
            "omch2": (omegam-omegab)*(H0/100)**2
                     - (mnu*(3.046/3)**0.75)/94.0708,
            "omegam": {"derived": True, "latex": r"\Omega_m"},
        }, **point),
        "debug": 50,
    }
    model = get_model(info)
    model.add_requirements({"baryon_suppression": {
        "z": z_grid, "k": np.power(10.0, log10k_grid)}})
    # every parameter is fixed, so the point to evaluate is the empty
    # dictionary; the call runs CAMB and bfmt once
    model.logposterior({})
    return model.provider.get_baryon_suppression()


def compute_probes(sup=None, ell=None):
    """Compute the fiducial 3x2pt statistics, data vector and chi2.

    Returns C_ss, C_gs, xi_+-, gamma_t, the masked data vector and its
    chi2 at this module's fiducial point, with optional baryonic
    suppression folded into the nonlinear power. Requires
    init_cosmolike(CLprobe="3x2pt", with_data=True). sup = None computes
    the dark-matter-only prediction; otherwise sup is the {z: S array}
    dictionary from get_baryon_suppression, applied the way the
    Cosmolike likelihoods apply it: lnPNL[i :: len(z_grid)] += ln S(z_i).

    Arguments:
      sup = suppression dictionary on the CAMB interpolation grid
            (one entry per z of z_grid, each an array over the k of
            log10k_grid), or None.
      ell = multipoles of the returned harmonic spectra, or None
            for np.arange(25, 3000, 15).

    Returns:
      dict with ell, C_ss (the EE spectra), C_gs, theta [arcmin], xip,
      xim, gammat, dv, chi2, and the interpolation grids z_grid and
      log10k_grid (log10 of k in 1/Mpc, the unit get_baryon_suppression
      takes; the notebooks feed both grids to that function).
    """
    if ell is None:
        ell = np.arange(25., 3000., 15.)
    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_growth, z_interp_1D, chi,
     omegan2, lnPL_cb) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, kmax=5.0, k_per_logint=10,
        CAMBAccuracyBoost=1.0,
        non_linear_emul=_CONFIG["non_linear_emul"])
    # a private copy of ln P_nonlinear, modified in place below
    lnPNL = np.array(lnPNL, copy=True)
    if sup is not None:
        for i, z_val in enumerate(z_interp_2D):
            # every k row of redshift z_i sits at stride len(z) in
            # the flattened table, the layout set_cosmology expects;
            # sup[z_val] looks the redshift up by exact float equality,
            # which holds when sup was computed on this same z_grid
            lnPNL[i :: len(z_interp_2D)] += np.log(sup[z_val])
    ci.init_ntable_lmax(int(_CONFIG["lmax"]))
    ci.init_photoz_histogram_akima(
        enabled=int(_CONFIG["photoz_histogram_akima"]),
        preparation_factor=int(_CONFIG["photoz_histogram_akima_factor"]))
    ci.init_photoz_conventions(
        interpolation_type=int(_CONFIG["photoz_interpolation_type"]),
        zmid_convention=int(_CONFIG["photoz_zmid_convention"]))
    # init_fpt_internal_boost comes first, as in the likelihood and in
    # _set_state, so the C-FAST-PT internal grid fraction is the
    # configured one even when this is the first init_accuracy_boost of
    # the process
    ci.init_fpt_internal_boost(
        internal_boost=float(_CONFIG["internal_accuracyboost"]))
    ci.init_accuracy_boost(1.0, 0)
    # the binning of the loaded data vector: the real-space wrappers may
    # have left another one
    ci.init_binning(int(_CONFIG["ntheta"]), _CONFIG["theta_min_arcmin"],
                    _CONFIG["theta_max_arcmin"])
    ci.init_bias(bias_model=_CONFIG["bias_model"])
    ci.set_cosmology(omegam=omegam, H0=H0,
                     log10k_2D=log10k_interp_2D, z_2D=z_interp_2D,
                     lnP_linear=lnPL, lnP_nonlinear=lnPNL,
                     G=G_growth, z_G=z_growth,
                     z_1D=z_interp_1D, chi=chi,
                     omegan2=omegan2)
    ci.set_nuisance_shear_calib(M=M_FID)
    ci.set_nuisance_shear_photoz(bias=SHEAR_PHOTOZ_FID)
    ci.set_nuisance_clustering_photoz(bias=LENS_PHOTOZ_FID)
    ci.set_nuisance_ia(A1=A1_FID, A2=A2_FID, B_TA=BTA_FID)
    ci.set_nuisance_bias(B1=B1_FID, B2=ZEROS5, B_MAG=BMAG_FID,
                         B3nl=ZEROS5, BK=ZEROS5)
    ci.set_point_mass(PMV=PM_FID)
    ci.reset_bary_struct()
    # tmp = the BB spectra, not returned
    (C_ss, tmp) = ci.C_ss_tomo_limber(l=ell)
    C_gs = ci.C_gs_tomo_limber(l=ell)
    (xip, xim) = ci.xi_pm_tomo()
    gt = ci.w_gammat_tomo()
    theta = ci.get_binning_real_space()
    dv = np.array(ci.compute_data_vector_masked())
    chi2 = ci.compute_chi2(dv)
    return {"ell": ell, "C_ss": C_ss, "C_gs": C_gs,
            "theta": theta, "xip": xip, "xim": xim, "gammat": gt,
            "dv": dv, "chi2": chi2,
            "z_grid": z_interp_2D,
            # get_baryon_suppression takes k in 1/Mpc, but the CAMB helper
            # returns this grid in h/Mpc: convert here, at the one place
            # that links the two, so S(k) is evaluated at the physical k.
            "log10k_grid": log10k_interp_2D + np.log10(H0/100.0)}
