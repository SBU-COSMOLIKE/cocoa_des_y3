"""Base class of the des_y3 Cobaya likelihoods.

Cobaya is the sampler framework: it asks each likelihood which quantities
it needs from the theory codes (get_requirements), runs those codes at
every parameter point, and then calls the likelihood's logp. The five
likelihoods of this folder (cosmic_shear, combo_2x2pt, combo_3x2pt,
combo_xi_gg, combo_xi_ggl) are thin subclasses that differ only in the
probe string they pass to initialize; this file does the work.

CosmoLike, the C library compiled into cosmolike_des_y3_interface.so
(imported as ci), computes the real-space two-point functions of the DES
data vector: cosmic shear xi_plus and xi_minus (the ss block),
galaxy-galaxy lensing gamma_t (gs) and galaxy clustering w(theta) (gg),
900 entries in that order for the DES Y3 and Y1 data sets. CosmoLike
keeps its configuration in global C variables, so every initialize call
configures the one CosmoLike of the Python process: a process holds one
des_y3 configuration at a time, and a second configuration with
different data-vector dimensions aborts it.

One evaluation (logp):

    Cobaya parameters of the point
      -> set_cosmo_related   P(k, z), growth and distances from the
                             theory codes, converted to CosmoLike's units
      -> set_lens_related    point masses, galaxy bias, lens photo-z
      -> set_source_related  shear calibration, source photo-z, IA
      -> CosmoLike data vector [900], masked entries zero
      -> chi2 = (t - d)^T C^-1 (t - d) over the unmasked entries
      -> logp = -chi2/2

Three theory paths, chosen by the yaml option use_emulator:

    0  CAMB (through Cobaya) supplies P(k, z) and the distances, and
       CosmoLike computes the data vector; optional theory blocks add
       the Python FAST-PT tables and the baryonic suppression.
    1  emulator theories supply xi, gamma_t and w(theta) directly, and
       CosmoLike adds only the fast terms (shear calibration, point
       mass, baryon principal components).
    2  hybrid: emulators replace CAMB for P(k, z) and the background,
       and CosmoLike computes the data vector as with 0.

Units: Cobaya's theory codes give k in 1/Mpc, P(k) in Mpc^3 and
comoving distances in Mpc; CosmoLike works with k in h/Mpc, P(k) in
(Mpc/h)^3 and distances in Mpc/h, and set_cosmo_related converts.
"""

# A __future__ import must precede every other statement (only the
# module docstring and comments may come first). Its three features are
# the default behavior of Python 3, so the line changes nothing here.
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
from scipy.interpolate import interp1d
import sys
import time
import functools
from collections.abc import Mapping

# Cobaya, the sampler framework, and GetDist, whose IniFile reads the
# `key = value` lines of the .dataset files
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

# EuclidEmulator2 (EE2): the emulator of the ratio of nonlinear to
# linear matter power, used when non_linear_emul = 1
import euclidemu2 as ee2
import math

from contextlib import contextmanager
@contextmanager
def timer(label):
  """Print the wall-clock time spent inside a `with timer(label):` block.

  @contextmanager turns this generator function into a context manager:
  the code before yield runs when the with-block starts, and the print
  after it runs when the block ends. Nothing in this project calls it; it
  is a debugging aid.

  Arguments:
    label = text printed before the elapsed seconds.
  """
  t0 = time.perf_counter()
  yield
  print(f"{label}: {time.perf_counter() - t0:.4f}s")

# The compiled CosmoLike interface (interface/cosmolike_des_y3_interface.so)
import cosmolike_des_y3_interface as ci

# The OpenMP thread count with_omp_threads restores before every hot
# call: OMP_NUM_THREADS as it was when this module was imported, 1 when
# the variable is unset.
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))

def with_omp_threads(fn):
    """Wrap a method so CosmoLike's OpenMP thread count is reset first.

    CosmoLike's hot loops run in OpenMP parallel regions sized by
    omp_get_max_threads(), which starts at OMP_NUM_THREADS. A Python
    library that shares the OpenMP runtime and calls
    omp_set_num_threads(1) changes that count for the whole process, and
    every later CosmoLike parallel region would then run on one core. The
    wrapper calls ci.set_omp_threads(COSMOLIKE_OMP_THREADS) before every
    call of the wrapped method, so the count is restored whatever ran in
    between; ci.set_omp_threads also keeps BLAS, the linear-algebra
    library, at one thread.

    Used as a decorator: the line @with_omp_threads above a method
    replaces the method by the wrapper when the class is defined, and
    functools.wraps copies the method's name and docstring onto the
    wrapper.

    Arguments:
      fn = the method to wrap.

    Returns:
      wrapper, a function that takes the same arguments as fn and returns
      fn's result.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        """Reset CosmoLike's OpenMP thread count, then call fn unchanged."""
        ci.set_omp_threads(COSMOLIKE_OMP_THREADS)
        return fn(*args, **kwargs)
    return wrapper

# Prefix of every nuisance-parameter name in the parameter files
# (DES_M1, DES_DZ_S1, DES_B1_1, ...); the subclass modules import it too.
survey = "DES"

class _cosmolike_prototype_base(DataSetLikelihood):
  """Cobaya likelihood of the DES real-space data vector, through CosmoLike.

  DataSetLikelihood is Cobaya's base class for a likelihood configured by
  a .dataset file. Cobaya sets every option of the likelihood yaml
  (cosmic_shear.yaml and its siblings, merged with the user's yaml) as an
  attribute of the instance before it calls initialize. The options this
  class reads:

    data_file                  the .dataset file of the data set (in data/)
    accuracyboost              refines the z and k grids and CosmoLike's tables
    internal_accuracyboost     C FAST-PT convolution grid / output grid
    nonlimber_accuracyboost    refines the non-Limber FFTLog chi grid
    pk_z_refinement            further z refinement of the P(k, z) tables
    integration_accuracy       CosmoLike's quadrature level
    photoz_interpolation_type  n(z) interpolation: 0 cubic spline, 1 linear,
                               2+ Steffen
    photoz_zmid_convention     n(z) z column: 0 left cell edges, 1 centers
    adopt_limber_gs, adopt_limber_gg
                               1 Limber at every multipole, 0 non-Limber
                               below l = 150 (ggl and clustering)
    include_HOD_GX, include_halo_IA
                               halo-model galaxy power and halo-model IA
    lmax                       multipole limit of CosmoLike's C_l tables
    kmax_boltzmann             k_max [1/Mpc] asked of CAMB, times accuracyboost
    growth_k                   k [1/Mpc] of the growth table (default 0.05)
    non_linear_emul            nonlinear P(k): 1 EuclidEmulator2, 2 the
                               theory code's own
    external_nz_modeling       send the n(z) arrays at every evaluation
    use_emulator               0 CAMB, 1 data-vector emulators, 2 hybrid
    IA_model                   intrinsic alignment: 0 NLA, 1 TATT
    IA_redshift_evolution      redshift model of the IA amplitudes
    IA_code                    one-loop tables from 0 C FAST-PT, 1 Python
    bias_model                 redshift model of each galaxy-bias term
    use_baryon_pca, create_baryon_pca, add_baryons_on_dv,
    external_baryon_suppression
                               baryonic-feedback treatments (initialize)
    print_datavector           write the theory vector to
                               print_datavector_file at every evaluation
    debug                      CosmoLike log level debug instead of info

  The subclasses pass the probe string to initialize; the module docstring
  lists the steps of one evaluation.
  """

  @classmethod
  def get_modified_defaults(cls, defaults, input_options={}):
    """Apply the yaml option `fixed_params` to the default parameters.

    cobaya calls this class method when it reads the defaults of a
    combination (its yaml file, e.g. combo_xi_gg.yaml), before it merges
    them with the user's yaml. The parameters of a combination come from
    `params: !defaults [params_lens, params_source]`, and the `!defaults`
    tag builds the whole `params` mapping from those files, so the same
    yaml cannot change one entry of it. A combination that fixes some of
    these parameters lists them under `fixed_params` instead (combo_xi_gg
    fixes the point masses, which act only on galaxy-galaxy lensing).
    Each entry replaces the parameter's default info with the cobaya merge
    rule: a value drops prior, ref and proposal and keeps the other keys
    (the latex label). A user yaml can override `fixed_params` like any
    other option of the likelihood. Combinations without `fixed_params`
    keep their defaults unchanged.

    Arguments:
      defaults = the combination's default options (dict, `params`
                 included), changed in place
      input_options = the user's options for this likelihood (dict)

    Returns:
      defaults, with each parameter of `fixed_params` replaced.
    """
    fixed = input_options.get("fixed_params", defaults.get("fixed_params"))
    params = defaults.get("params") or {}
    for p, info in (fixed or {}).items():
      old = params.get(p)
      new = {}
      if isinstance(old, Mapping):
        # keep every key of the default info except the sampling ones
        for key, value in old.items():
          if key not in ("prior", "ref", "proposal"):
            new[key] = value
      if isinstance(info, Mapping):
        new.update(info)
      else:
        new["value"] = info
      params[p] = new
    if params:
      defaults["params"] = params
    return defaults

  def initialize(self, probe):
    """Read the .dataset file, build the grids and configure CosmoLike.

    Cobaya calls initialize once, through the subclass, after it has set
    the yaml options as attributes. The steps: read the file names and the
    binning from the .dataset file (an ini file of `key = value` lines,
    read with GetDist's IniFile); build the redshift and wavenumber grids
    of the tables handed to CosmoLike; configure CosmoLike (probes,
    angular binning, accuracy, n(z), data vector with covariance and mask,
    intrinsic alignment, galaxy bias); then set up the baryon treatment.

    Arguments:
      probe = the blocks to compute: "xi" (cosmic shear), "2x2pt",
              "3x2pt", "xi_gg" or "xi_ggl".

    Side effects:
      sets the file names, the grids (z_interp_1D, z_interp_2D,
      z_interp_2D_camb, log10k_interp_2D) and, with external_nz_modeling,
      the n(z) arrays lens_nz and source_nz; reconfigures the CosmoLike
      library of this process. With IA_model = 0 it resets IA_code to 0,
      and the baryon options override one another (comments in the body).

    Raises:
      LoggedError when pk_z_refinement is not a positive integer.
    """
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")

    # ------------------------------------------------------------------------   
    # z_interp_1D: the redshift grid of the comoving distances and of the
    # growth table, in three uniform blocks (endpoint=False leaves out the
    # upper end of a block): 0 <= z < 3 with 0.8 tmp nodes, 3 <= z < 50.1
    # with 0.4 tmp nodes, and 1070 <= z <= 1100 with 0.1 tmp nodes, around
    # the last-scattering redshift z ~ 1090 (the source of CMB lensing).
    # tmp = 1000 + 250 accuracyboost, so boost 1 gives 1000, 500 and 125
    # nodes; the max() calls set floors of 100, 100 and 50 nodes.
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp)),endpoint=False),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp)),endpoint=False),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    # z_interp_2D: the z nodes of the 2D power-spectrum tables handed to
    # cosmolike, which interpolates linearly in z between exactly these
    # nodes (its piecewise-uniform direct indexing uses the handed grid;
    # there is no internal regridding). Linear interpolation leaves a
    # sawtooth-shaped O(dz^2) residual that vanishes at the nodes, so two
    # grids that do not share nodes disagree by the full residual
    # amplitude. A node count that moves the nodes at every boost value
    # re-phases the sawtooth instead of refining it: min(120 + 20 boost,
    # 250) nodes produce order-unity chi2 jitter in roman_kl's clustering
    # vector, and smaller non-convergent shifts in this project. The
    # dyadic factor m = 2^ceil(log2(boost)) below refines each uniform
    # block by an integer factor with the same endpoints, so (a) every
    # block stays uniform (cosmolike keeps its two-segment direct
    # indexing, no search), (b) every coarser grid's nodes are a subset of
    # every finer grid's nodes, making a boost increase a true refinement
    # (the error falls like 1/m^2), and (c) boost 1 (m = 1) gives
    # 105 + 35 = 140 nodes, the grid handed to CAMB (z_interp_2D_camb
    # below). The low block multiplies its node count (endpoint=False,
    # spacing 3/n); the high block multiplies its interval count
    # (endpoint=True: 35 nodes = 34 intervals -> 34 m + 1 nodes). The
    # boost alone gives at most m = 16 (any boost above 8), that is
    # 105*16 + 34*16 + 1 = 2225 z nodes, before pk_z_refinement multiplies
    # m. The grid ends at z = 49.99, below
    # z = 50, the upper end of the hybrid emulator; the des_y3 projections
    # need only z < 3, where the source n(z) tables end, and the higher
    # nodes serve CMB lensing, which these likelihoods do not compute.
    #
    # pk_z_refinement (a yaml option, a positive integer, default 1)
    # multiplies m further without changing the other uses of
    # accuracyboost. A Fourier-space data vector reads P(k, z) at fixed
    # multipoles, where the residual of the linear z interpolation does
    # not average out as it does in real space: roman_fourier's 3x2pt chi2
    # moves by 0.25, 0.030 and 0.002 from m = 1 to 2, 4 and 8, and
    # roman_real's and lsst_y1's by at most 0.004 from 1 to 2.
    zref = getattr(self, "pk_z_refinement", 1)
    if not (float(zref) == int(zref) and int(zref) >= 1):
      raise LoggedError(self.log, "pk_z_refinement = %s: must be a positive "
                        "integer", zref)
    m = int(min(2**np.ceil(np.log2(max(1.0, self.accuracyboost))), 16))
    m = m*int(zref)
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,105*m,endpoint=False), 
                                       np.linspace(3.0,49.99,34*m + 1)),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    # CAMB's transfer module caps the number of requested redshifts at
    # 256, so the list handed to CAMB through the Pk_interpolator
    # requirement stays at this boost-independent 140-node grid (the
    # m = 1 grid above). The denser nested nodes only re-evaluate the
    # smooth z-spline CAMB builds from these transfer redshifts when the
    # cosmolike tables are filled, so raising the boost refines exactly
    # the resampling of that spline onto the cosmolike table, and the
    # CAMB side never exceeds its cap.
    self.z_interp_2D_camb = np.concatenate((np.linspace(0,3.0,105,endpoint=False), 
                                            np.linspace(3.0,49.99,35)),axis=0)
    
    # log10 of k in 1/Mpc: 1250 + 250 accuracyboost nodes from
    # k = 1.02e-5 to 100 /Mpc; set_cosmology receives log10 k in h/Mpc.
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    # ci.initial_setup resets CosmoLike's global configuration to its
    # defaults, and each init_* call below sets one part of it.
    # init_binning: ntheta logarithmic angular bins between
    # theta_min_arcmin and theta_max_arcmin.
    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(int(self.ntheta), self.theta_min_arcmin, self.theta_max_arcmin)

    if self.debug:
      ci.set_log_level_debug()
    else:
      ci.set_log_level_info()

    # An option the yaml does not set takes the default of its getattr
    # call (getattr(self, name, default) returns default when the attribute
    # is absent). The photo-z conventions of the n(z) files: interpolation
    # type (0 cubic spline, 1 linear, 2+ Steffen) and z column (0 left cell
    # edges, 1 cell centers).
    ci.init_photoz_conventions(
        interpolation_type=int(getattr(self, "photoz_interpolation_type", 0)),
        zmid_convention=int(getattr(self, "photoz_zmid_convention", 0)))

    # C FAST-PT: its internal convolution grid relative to its output table
    # (1.0 = the two grids are equal)
    ci.init_fpt_internal_boost(
        internal_boost=float(getattr(self, "internal_accuracyboost", 1.0)))

    # the non-Limber FFTLog chi grid, refined on top of the accuracy boost
    # (narrow lens bins need it: see init_nonlimber_accuracy_boost)
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(getattr(self, "nonlimber_accuracyboost", 1.0)))

    # galaxy-galaxy lensing and clustering: 1 = Limber at every multipole,
    # 0 = non-Limber below l = 150
    ci.init_adopt_limber_gs(
        adopt_limber_gs=int(getattr(self, "adopt_limber_gs", 1)))

    ci.init_adopt_limber_gg(
        adopt_limber_gg=int(getattr(self, "adopt_limber_gg", 0)))
    # 0 = perturbative galaxy bias, 1 = halo-model (HOD) galaxy power;
    # always set, so a model never inherits the previous model's value
    ci.init_include_HOD_GX(
        include_HOD_GX=int(getattr(self, "include_HOD_GX", 0)))
    # 0 = the init_IA model, 1 = halo-model IA (Fortuna et al. 2021)
    ci.init_include_halo_IA(
        include_halo_IA=int(getattr(self, "include_halo_IA", 0)))
    # Halo statistics use the cold dark matter + baryon spectrum P_cb. The
    # hybrid emulators (use_emulator = 2) have no separate cb spectrum, so
    # P_cb comes from the small-scale ratio of get_neutrino_inputs.
    if self.use_emulator == 2:
      self.log.info("Halo P_cb uses P_lin/(1 - f_nu)^2 because the "
                    "emulators have no cb spectrum (an approximation; "
                    "see get_neutrino_inputs)")

    # use_emulator = 1: emulator theories supply xi, gamma_t and w(theta),
    # so CosmoLike needs only the n(z) and the data set (data vector,
    # covariance, mask), for the fast terms it adds (shear calibration,
    # point mass, baryon PCs) and for the chi2. The low accuracy_boost =
    # 0.35 serves only the point-mass term (the inline note).
    if self.use_emulator == 1:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)  
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      # use_emulator = 0 or 2: CosmoLike computes the data vector. lmax is
      # the multipole limit of its C_l tables; is_linear=False keeps the
      # nonlinear matter power.
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      # external_nz_modeling: read the n(z) tables into the arrays
      # self.lens_nz and self.source_nz, which set_lens_related and
      # set_source_related send at every evaluation, so a user function can
      # change them per point; otherwise CosmoLike reads the files once.
      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file = self.lens_file,
            lens_ntomo = int(self.lens_ntomo), 
            source_multihisto_file = self.source_file,
            source_ntomo = int(self.source_ntomo)
          ) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file = self.lens_file,
          lens_ntomo = int(self.lens_ntomo), 
          source_multihisto_file = self.source_file,
          source_ntomo = int(self.source_ntomo)) 

      ci.init_data_real(self.cov_file, self.mask_file, self.data_vector_file)

      if (int(self.IA_model) == 0) and (int(self.IA_code) == 1):
        # NLA (IA_model = 0) takes its one-loop tables from the C FAST-PT
        # port: IA_code is reset to 0, silently, so get_requirements does not
        # ask for the Python fastpt theory block (IA_PS, bias_PS).
        self.IA_code = 0
      # IA_model: 0 NLA, 1 TATT. IA_redshift_evolution: the redshift model of
      # the amplitudes (3 = a power law in (1+z)/(1+z0), slot 1 the amplitude
      # and slot 2 the exponent). IA_code: 0 C FAST-PT, 1 Python FAST-PT.
      ci.init_IA(ia_model = int(self.IA_model), 
                ia_redshift_evolution = int(self.IA_redshift_evolution),
                ia_code = int(self.IA_code))

      # Galaxy bias exists only with lens galaxies, so cosmic shear alone
      # ("xi") skips it. 3x2pt_ss_sk_sk and 2x2pt_ss_sk are probe strings of
      # other Cosmolike projects and never occur here.
      if self.probe not in ("xi", "3x2pt_ss_sk_sk", "2x2pt_ss_sk"):
        # bias_model: one redshift-evolution code per galaxy-bias term,
        # [b1, b2, bs2, b3, bmag, bK] (cosmolike bias.h): 0 = one free
        # amplitude per lens bin; for b2, bs2 and b3, 1 = derived from b1.
        ci.init_bias(bias_model=self.bias_model)

      # non_linear_emul = 1: build the EuclidEmulator2 object once;
      # set_cosmo_related applies its boost at every evaluation
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()

      # Baryonic-feedback options. external_baryon_suppression (the bfmt
      # theory block supplies the suppression S(k, z) of the nonlinear
      # power) excludes the two data-vector treatments: use_baryon_pca
      # (marginalize over the amplitudes DES_BARYON_Q* of principal
      # components added to the data vector) and add_baryons_on_dv
      # (contaminate the matter power with the hydro simulation
      # which_bsims_add_on_dv). create_baryon_pca (compute the PCs from the
      # simulations baryon_pca_select_sims and write them to
      # filename_baryon_pca) switches off external_baryon_suppression and
      # use_baryon_pca. These overrides apply silently, without a message.
      if self.external_baryon_suppression:
          self.use_baryon_pca = False
          self.add_baryons_on_dv = False

      if self.create_baryon_pca:
        self.external_baryon_suppression = False
        self.use_baryon_pca = False
        self.allsims = ini.relativeFileName('all_sims_hdf5_file')
      else:
        if self.add_baryons_on_dv:
          self.external_baryon_suppression = False
          sim = self.which_bsims_add_on_dv
          self.allsims = ini.relativeFileName('all_sims_hdf5_file')
          ci.init_baryons_contamination(sim = sim, allsims=self.allsims)

    # use_baryon_pca applies to every use_emulator value: CosmoLike loads
    # the principal components from baryon_pca_file (data/pca.txt, one row
    # per data-vector entry, one column per component).
    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      # 4 components, one amplitude each (DES_BARYON_Q1 ... Q4 of
      # params_source.yaml); pca.txt holds more columns, and CosmoLike adds
      # only the first npcs.
      self.npcs = 4
      ci.set_baryon_pcs(eigenvectors = np.loadtxt(baryon_pca_file))
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    """Return the quantities this likelihood needs from the theory codes.

    Cobaya calls get_requirements after initialize and has the theory
    codes (CAMB, the emulators, fastpt, bfmt) compute what the returned
    dict names at every parameter point. A key with the value None asks
    for the quantity alone; a dict value carries its options (the z
    nodes and k_max [1/Mpc] of the P(k, z) interpolator, the z nodes of
    the comoving distances, which come back in Mpc).

      use_emulator = 1: the emulated blocks of the probe, plus H0 and the
        distances for the probes with gamma_t (the point-mass term).
      use_emulator = 2: the cosmology the hybrid emulators take, the linear
        and nonlinear P(k, z) of total matter ("delta_tot") and the
        distances.
      otherwise (CAMB): As, H0, omegam and omegab, the same P(k, z) and
        distances, a CMB TT spectrum (a CAMB workaround, see the inline
        note), omnuh2 and the linear P(k, z) of cold dark matter +
        baryons ("delta_nonu"); the fastpt tables when IA_code = 1, the
        baryonic suppression with external_baryon_suppression, and mnu,
        w and wa when EuclidEmulator2 needs them.

    Returns:
      dict {quantity name: None or options}.
    """
    if self.use_emulator == 1:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    elif self.use_emulator == 2:
      return {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "mnu": None,
        "w": None,
        "wa": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D 
        }, # in Mpc
      }
    else:
      _requirements_ = {
        "As": None,
        "H0": None,
        "omegam": None,
        "omegab": None,
        "Pk_interpolator": {
          "z": self.z_interp_2D_camb,
          "k_max": self.kmax_boltzmann * self.accuracyboost,
          "nonlinear": (True,False),
          "vars_pairs": ([("delta_tot", "delta_tot")])
        },
        "comoving_radial_distance": {
          "z": self.z_interp_1D
        }, # in Mpc
        "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
          'tt': 0
        }
      }
      # The baryon theory block computes S(k, z) at the nodes requested
      # here: z_interp_2D and k = 10^log10k_interp_2D in 1/Mpc (a theory that
      # works in h/Mpc converts).
      if self.external_baryon_suppression:
          _requirements_["baryon_suppression"] = {
              "z": self.z_interp_2D,
              "k": np.power(
                  10.0, self.log10k_interp_2D
              ),
          }
      # IA_code = 1: the Python fastpt theory block supplies the one-loop IA
      # and galaxy-bias tables (IA_PS, bias_PS) instead of CosmoLike's C port
      if (self.IA_code == 1):
        _requirements_["IA_PS"] = None
        _requirements_["bias_PS"] = None
      if self.non_linear_emul == 1:
        _requirements_["omegab"] = None
        _requirements_["mnu"] = None
        _requirements_["w"] = None
        _requirements_["wa"] = None
      # Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2) and, for
      # the cold dark matter + baryon halo field, the linear P_cb
      # (get_neutrino_inputs)
      _requirements_["omnuh2"] = None
      # Keep both fields available to the likelihood and direct halo readers.
      # CAMB obtains them from the same transfer-function calculation.
      _requirements_["Pk_interpolator"]["vars_pairs"] = [
        ("delta_tot", "delta_tot"),
        ("delta_nonu", "delta_nonu")]
      return _requirements_

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_cosmo_related(self):
    """Hand the cosmology of the current point to CosmoLike.

    Runs at every evaluation (a hot path), after Cobaya has run the theory
    codes; self.provider is Cobaya's access to their results. For
    use_emulator = 0 or 2 it reads the linear and nonlinear P(k, z) on the
    (z_interp_2D, k) grid, converts them to CosmoLike's units, builds the
    growth table and the neutrino inputs, applies the optional baryonic
    suppression and calls ci.set_cosmology; with IA_code = 1 it then
    installs the fastpt tables. For use_emulator = 1 it hands over only
    the comoving distances, which the point-mass term needs.

    Shape flow of the ln P tables (Fortran order, the z index fastest, as
    ci.set_cosmology expects):

      PKL.logP(z_interp_2D, k) [n_z, n_k]   ln P in Mpc^3, k in 1/Mpc
        -> .flatten(order='F') [n_z*n_k]     entry iz + n_z*ik
        -> + ln(h^3)                         ln P in (Mpc/h)^3
      log10k_interp_2D - log10(h)            log10 k in h/Mpc
      comoving distance [Mpc] * h            chi in Mpc/h

      legend: n_z = len(z_interp_2D), n_k = len(log10k_interp_2D)

    Raises:
      an error for a non_linear_emul other than 1 or 2: a LoggedError is
      meant, but its message names an undefined variable, so a NameError
      surfaces.
    """
    h = self.provider.get_param("H0")/100.0
    if not (self.use_emulator == 1):
      # PKL interpolates the linear total-matter P(k, z): a cubic spline in z
      # through CAMB's transfer redshifts; extrap_kmin and extrap_kmax
      # [1/Mpc] extend it in k by log extrapolation.
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmin=1e-6,
                                               extrap_kmax=2.5e2*self.accuracyboost)
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # EuclidEmulator2 covers z < 10 only. get_boost2 returns the boost
        # B = P_nonlinear/P_linear at the z < 10 nodes, on its own k grid
        # from 10^-2.0589 = 0.0087 to 10^0.973 = 9.4 h/Mpc.
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        # ln B is interpolated linearly in log10 k onto the likelihood's k grid
        # (converted to h/Mpc) and extrapolated beyond EE2's range; below
        # k = 8.73e-3 h/Mpc, EE2's lower limit, ln B = 0 (B = 1, linear
        # scales). lnbt [n_z, n_k] stays zero at z >= 10.
        tmp = interp1d(np.log10(kbt), 
                        np.log(bt), 
                        axis=1,
                        kind='linear', 
                        fill_value='extrapolate', 
                        assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # First the theory code's own nonlinear P(k, z) (Halofit or HMcode, as
        # the CAMB block sets), which covers every redshift
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # below z = 10 replace it by ln P_lin + ln B (EE2): np.where picks per
        # row, since the condition (z < 10)[:, None] has shape [n_z, 1] and
        # broadcasts across k
        lnPNL = np.where((self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')+lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        # non_linear_emul = 2: the theory code's nonlinear P(k, z) everywhere
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, 
          extrap_kmin=1e-6,
          extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", self.non_linear_emul)

      # G(z) = D(z) (1+z), with D(z) = sqrt(P_lin(z, k)/P_lin(0, k)) the
      # linear growth factor (D(0) = 1) at k = growth_k.
      # G on the dense 1D z grid (clipped to the P(k) interpolator range):
      # cosmolike reads G linearly in z, and on the coarse 2D grid
      # (dz ~ 0.03) the linear read misses D by up to 9e-5 and the
      # growth rate f = 1 - (1+z) dlnG/dz (the slope of the table) by
      # 1%; on the 1D grid (dz = 0.003) by 1e-6 and 0.2%. PKL is a cubic
      # spline in z through CAMB's transfer redshifts, so this asks CAMB
      # for no extra redshifts (about 0.1 ms per evaluation). The table
      # stays divided by G at the last z_2D node (z_growth ends below
      # it); cosmolike's growfac divides by G(0), so D(z=0) = 1.
      z_growth = self.z_interp_1D[self.z_interp_1D <= self.z_interp_2D[-1]]
      # G is sampled at growth_k (default 0.05/Mpc), a sub-horizon scale.
      # At k = 5e-4/Mpc (about 2 H0/c) CAMB's dark-energy perturbations
      # change the growth by 0.5-0.9% at w != -1 (z = 0.5 to 2), while every
      # reader of G (IA amplitudes, one-loop D^4, sigma(M, z), the growth
      # rate f) describes sub-horizon modes; with 0.06 eV neutrinos the
      # growth varies by 0.03% above 0.05/Mpc (cosmolike_core skill,
      # references/growth_factor_measurements.md)
      growth_k = float(getattr(self, "growth_k", 0.05))
      G_growth = np.sqrt(PKL.P(z_growth,growth_k)/PKL.P(0,growth_k))*(1+z_growth)
      z_norm = self.z_interp_2D[-1]
      G_growth /= np.sqrt(PKL.P(z_norm,growth_k)/PKL.P(0,growth_k))*(1+z_norm)
      # external_baryon_suppression: the bfmt theory block returns
      # {z: S(k) array} at the requested z nodes (by default S = 1 outside
      # its calibration range), and ln S is added to ln P_nonlinear at each
      # node. lnPNL[i :: n_z] takes every n_z-th entry starting at i: all k
      # of redshift node i (the z index runs fastest). A node missing from
      # the dict is skipped with a warning, and an exception while reading
      # the result is logged as an error and the evaluation continues
      # without baryonic suppression.
      if self.external_baryon_suppression:
        try:
          supp_dict = self.provider.get_result("baryon_suppression")
          self.log.info(
            "Applying baryon suppression: %d redshifts from theory block",
            len(supp_dict),
          )

          for i, z_val in enumerate(self.z_interp_2D):
            if z_val in supp_dict:
              sup_array = supp_dict[z_val]
              lnbt_baryon = np.log(sup_array)
              lnPNL[i :: self.len_z_interp_2D] += lnbt_baryon
              self.log.debug(
                  "Applied baryon suppression at z=%.3f: "
                  "min_sup=%.6f, max_sup=%.6f",
                  z_val,
                  sup_array.min(),
                  sup_array.max(),
              )
            else:
              self.log.warning(
                  "baryon_suppression dict does not contain z=%.3f; skipping",
                  z_val,
              )
        except Exception as e:
            self.log.error(
                "Failed to retrieve baryon suppression from theory block: %s; "
                "skipping baryon suppression",
                str(e),
            )

      # the massive neutrinos: Omega_nu h^2 and, for the cold dark matter
      # + baryon halo field, the linear P_cb (get_neutrino_inputs)
      (omegan2, lnPL_cb) = self.get_neutrino_inputs(lnPL=lnPL, h=h)

      # Everything in CosmoLike's h units (docstring); set_cosmology redraws
      # CosmoLike's cosmology tag when an input changed, so the cached tables
      # rebuild.
      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        omegab=self.provider.get_param("omegab"),
        omegan2=omegan2,
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_linear_cb=lnPL_cb,
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_G=z_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
      
      # IA_code = 1: install the IA and galaxy-bias tables of the Python
      # fastpt theory block. This must follow ci.set_cosmology, which redraws
      # cosmology.random, the tag CosmoLike's caches compare to detect a new
      # cosmology. FPTIA has 12 rows (10 spectra, the k row at index 10 =
      # FPTIA[-2], and P_lin) with k in h/Mpc; the first and last k are the
      # k range of both tables.
      if int(self.IA_code) == 1:
        FPTIA, FPTIA_kcut  = self.provider.get_IA_PS()
        FPTbias, sigma4    = self.provider.get_bias_PS()
        FPT_kmin, FPT_kmax = FPTIA[-2,0], FPTIA[-2,-1]
        
        ci.set_IA_PS(PS=FPTIA.flatten(order='C'), 
                     kmin=FPT_kmin, 
                     kmax=FPT_kmax, 
                     cutoff=FPTIA_kcut, 
                     N=len(FPTIA[0]))
        
        ci.set_bias_PS(PS=FPTbias.flatten(order='C'), 
                       kmin=FPT_kmin, 
                       kmax=FPT_kmax, 
                       cutoff=FPTIA_kcut, 
                       sigma4=sigma4, 
                       N=len(FPTIA[0]))
    else:
      # use_emulator = 1: only the comoving distances, in Mpc/h, which the
      # point-mass term needs
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  def get_neutrino_inputs(self, lnPL, h):
    """Return the massive-neutrino inputs of ci.set_cosmology.

    omegan2 is Omega_nu h^2 of massive neutrinos today, part of omegam.
    Halo variances use the cold dark matter + baryon spectrum P_cb at
    each redshift. Their mass-radius relation and mass-function density
    use rho_crit (Omega_m - Omega_nu). Total matter remains available
    for lensing and for the separate total-matter variance.

    lnPL_cb is ln P_cb on the same (k,z) grid and in the same units as
    lnPL. Both spectra are provided so direct halo readers can be used
    even after a likelihood evaluation that did not count halos.

    The two theory paths:
      CAMB (use_emulator = 0): omegan2 is CAMB's omnuh2 and P_cb its
        ("delta_nonu", "delta_nonu") linear spectrum, read like P_lin
        (get_requirements asks for both).
      emulators (use_emulator = 2): the emulators take no neutrino
        parameter (they were trained at mnu = 0.06 eV) and have no cb
        spectrum. omegan2 = mnu (3.046/3)^0.75/94.0708, the neutrino
        density the yaml's omegach2 subtracts, and
        P_cb = P_lin/(1 - f_nu)^2 with f_nu = omegan2/(omegam h^2): the
        ratio of the two spectra at wavenumbers far above the neutrino
        free-streaming wavenumber, where the neutrinos do not cluster; an
        approximation on cluster scales whose error on cluster counts has
        not been measured (projects/des_cluster/README.md).

    Arguments:
      lnPL = ln P_lin [(Mpc/h)^3], flattened as set_cosmology's
             lnP_linear (Fortran order: k index slow, z index fast)
      h    = H0/100

    Returns:
      (omegan2, lnPL_cb): a float and a numpy array of lnPL's shape.
    """
    if self.use_emulator == 2:
      mnu = self.provider.get_param("mnu")
      omegan2 = mnu*(3.046/3.0)**0.75/94.0708
    else:
      omegan2 = self.provider.get_param("omnuh2")

    if self.use_emulator == 2:
      # P_cb/P_lin = 1/(1 - f_nu)^2 where the neutrinos no longer
      # cluster (delta_m = (1 - f_nu) delta_cb)
      f_nu = omegan2/(self.provider.get_param("omegam")*h*h)
      lnPL_cb = lnPL - 2.0*np.log(1.0 - f_nu)
    else:
      # the same k extrapolation, (z, k) grid, flattening and units as
      # lnPL in set_cosmo_related
      PKL_cb = self.provider.get_Pk_interpolator(("delta_nonu", "delta_nonu"),
                                                 nonlinear=False,
                                                 extrap_kmin=1e-6,
                                                 extrap_kmax=2.5e2*self.accuracyboost)
      k_grid = np.power(10.0, self.log10k_interp_2D)
      lnPL_cb = PKL_cb.logP(self.z_interp_2D, k_grid).flatten(order='F')
      lnPL_cb = lnPL_cb + np.log(h**3)
    return (omegan2, lnPL_cb)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_source_related(self, **params):
    """Hand the source-sample nuisance parameters of the point to CosmoLike.

    Runs at every evaluation. Each list below has one entry per source bin
    and is read by name: the nested list comprehension
    [params.get(p, 0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    first builds the names DES_M1 ... DES_M4, then reads each value from
    params, 0 (the neutral value) when the name is absent.

      DES_M<i>      multiplicative shear calibration m (the shear of bin i
                    scales by 1 + m)
      DES_DZ_S<i>   source photo-z shift
      DES_A1_<i>, DES_A2_<i>, DES_BTA_<i>
                    IA parameters, whose meaning depends on
                    IA_redshift_evolution (with 3, slot 1 of A1 and A2 is
                    an amplitude and slot 2 its redshift exponent)

    With use_emulator = 1 only the shear calibration is sent. With
    external_nz_modeling the source n(z) array is also sent at every
    evaluation (comments in the body).

    Arguments:
      params = {parameter name: value} of the current point, the keyword
               arguments Cobaya passes to logp.
    """
    ntomo = self.source_ntomo
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      if self.external_nz_modeling: 
        # The n(z) array is sent at every point of the chain, so a user
        # function can change it per point (for example, add outlier
        # populations). To change it: (1) copy the array, which keeps the
        # fiducial self.source_nz intact; (2) modify the copy; (3) send it
        # with ci.set_source_sample.
        source_nz_local = self.source_nz.copy()

        # Extension point: a user function replaces the copy here, e.g.
        # source_nz_local = f(source_nz_local, nuisance parameters).

        ci.set_source_sample(source_nz_local)

        # The photo-z shifts still apply on top of the sent n(z); a user model
        # that already includes them would skip this call.
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def set_lens_related(self, **params):
    """Hand the lens-sample nuisance parameters of the point to CosmoLike.

    Runs at every evaluation; the lists are read by name as in
    set_source_related, one entry per lens bin, with the neutral value 0
    for an absent name except B1, whose neutral value is 1.

      DES_PM<i>     point-mass amplitude of galaxy-galaxy lensing (the
                    unmodeled mass enclosed within the smallest scales)
      DES_B1_<i>, DES_B2_<i>, DES_BMAG_<i>, DES_B3NL_<i>, DES_BK_<i>
                    linear, quadratic, magnification, third-order nonlocal
                    and nonlocal (K) galaxy-bias parameters
      DES_DZ_L<i>   lens photo-z shift

    The point masses are sent for every use_emulator value (the emulator
    path adds the point-mass term too); the rest only for use_emulator = 0
    or 2. With external_nz_modeling the lens n(z) array is also sent.

    Arguments:
      params = {parameter name: value} of the current point.
    """
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not (self.use_emulator == 1):
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]],
        B3nl=[params.get(p,0) for p in [survey+"_B3NL_"+str(i+1) for i in range(ntomo)]],
        BK=[params.get(p,0) for p in [survey+"_BK_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # The n(z) array is sent at every point of the chain, so a user
        # function can change it per point (for example, add outlier
        # populations). To change it: (1) copy the array, which keeps the
        # fiducial self.lens_nz intact; (2) modify the copy; (3) send it with
        # ci.set_lens_sample.
        lens_nz_local = self.lens_nz.copy()

        # Extension point: a user function replaces the copy here, e.g.
        # lens_nz_local = f(lens_nz_local, nuisance parameters).

        ci.set_lens_sample(lens_nz_local)

        # The photo-z shifts still apply on top of the sent n(z); a user model
        # that already includes them would skip this call.
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]]
        )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    """Return ln L = -chi2/2 for a theory data vector.

    ci.compute_chi2 computes chi2 = (t - d)^T C^-1 (t - d) over the entries
    the mask keeps, with the data vector d, the covariance C and the mask
    that init_data_real loaded from the .dataset file.

    Arguments:
      datavector = theory vector t at full length (900), masked entries
                   zero.
    """
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params):
    """Return the log-likelihood of the current point (Cobaya's entry point).

    Arguments:
      params = {parameter name: value}, the values Cobaya passes for the
               parameters this likelihood declares.
    """
    return self.compute_logp(self.get_datavector(**params))

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  @with_omp_threads
  def get_datavector(self, **params):        
    """Return the theory data vector of the current point as float64.

    Dispatches on use_emulator: 1 to internal_get_datavector_emulator,
    otherwise to internal_get_datavector. The decorator resets CosmoLike's
    OpenMP thread count first.

    Arguments:
      params = {parameter name: value} of the current point.

    Returns:
      numpy array [900]: the full DES data vector, with the masked entries
      and the blocks the probe does not compute set to zero.
    """
    if self.use_emulator == 1:
      dv = self.internal_get_datavector_emulator(**params)
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector_emulator(self, **params):
    """Assemble the data vector from the emulated blocks (use_emulator = 1).

    The emulator theories return the blocks of the probe (cosmic shear,
    gamma_t, w(theta)); each is copied to its offset in the full vector,
    whose block sizes [n_ss, n_gs, n_gg] come from CosmoLike. CosmoLike
    then adds the fast terms the emulators leave out (the shear
    calibration, the point mass when some DES_PM<i> is nonzero, and the
    baryon PCs with use_baryon_pca) and sets the masked entries to zero.

    Arguments:
      params = {parameter name: value} of the current point.

    Returns:
      numpy float64 array [n_ss + n_gs + n_gg] (900 for DES).

    Raises:
      ValueError when an emulated block has the wrong length or the probe
      is unknown.

    Side effects:
      with print_datavector, writes (index, value) rows to
      print_datavector_file at every evaluation.
    """
    # ---------------------------------------------------------------
    # fast parameters: the shear calibrations m and the point masses PM are
    # never emulated; CosmoLike adds their terms below. all(...) is True
    # when every PM is zero; only a probe with gamma_t and a nonzero PM
    # needs the lens set-up and the distances.
    PM = [params.get(p,0) for p in [survey+"_PM"+str(i+1) for i in range(self.lens_ntomo)]]
    if self.probe not in ("xi", "xi_gg") and not all(v == 0 for v in PM):
      self.set_lens_related(**params)
      self.set_cosmo_related()
    self.set_source_related(**params)
    # ---------------------------------------------------------------

    # block sizes [n_ss, n_gs, n_gg] of the full vector; every block the
    # probe lacks stays zero
    sizes = ci.compute_data_vector_3x2pt_real_sizes()
    total_size = int(np.sum(sizes))
    dv = np.zeros(total_size, dtype='float64') 
    
    if self.probe == "xi":
      tmp = self.provider.get_cosmic_shear()
      if (len(tmp) != sizes[0]):
        raise ValueError(f'Incompatible Sizes (Emulator Cosmic Shear)')
      dv[0:sizes[0]] = tmp[0:sizes[0]]
    elif self.probe == "xi_ggl":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1]):
        raise ValueError(f'Incompatible Sizes (Emulator xi_ggl)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
    elif self.probe == "3x2pt":
      tmp1 = self.provider.get_cosmic_shear()
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "xi_gg":
      tmp1 = self.provider.get_cosmic_shear()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp1) != sizes[0] or 
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = 0
      iend = sizes[0]
      dv[istart:iend] = tmp1[0:sizes[0]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    elif self.probe == "2x2pt": 
      tmp2 = self.provider.get_ggl()
      tmp3 = self.provider.get_wtheta()
      if (len(tmp2) != sizes[1] or
          len(tmp3) != sizes[2]):
        raise ValueError(f'Incompatible Sizes (Emulator 3x2pt)')
      istart = sizes[0]
      iend = sizes[0]+sizes[1]
      dv[istart:iend] = tmp2[0:sizes[1]]
      
      istart = sizes[0]+sizes[1]
      iend = sizes[0]+sizes[1]+sizes[2]
      dv[istart:iend] = tmp3[0:sizes[2]]
    else:
      raise ValueError(f'Unknown probe')

    # CosmoLike adds the shear calibration, the point mass
    # (force_exclude_pm = 1 skips it when every DES_PM<i> is zero) and,
    # with use_baryon_pca, the baryon PCs, and zeroes the masked entries
    if not self.use_baryon_pca: 
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order(datavector=dv,
                                                     force_exclude_pm=1)
    else:
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]
      if not all(v == 0 for v in PM):
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=0)
      else:
        dv = ci.compute_add_fpm_3x2pt_real_any_order_with_pcs(datavector=dv,
                                                              Q=Q,
                                                              force_exclude_pm=1)
    dv = np.array(dv, dtype='float64')
    
    if self.print_datavector:
      size = len(dv)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = dv
      # one format per column: the index as an integer, the value with eight
      # decimals (the comma makes fmt a tuple)
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return dv

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    """Compute the data vector with CosmoLike (use_emulator = 0 or 2).

    Sends the cosmology, the lens parameters (not needed for cosmic shear
    alone) and the source parameters, then asks CosmoLike for the masked
    data vector. create_baryon_pca computes the principal components from
    the simulations of baryon_pca_select_sims and writes them to
    filename_baryon_pca at every evaluation; use_baryon_pca adds
    sum_i Q_i PC_i with the amplitudes DES_BARYON_Q1 ... Q4.

    Arguments:
      params = {parameter name: value} of the current point.

    Returns:
      the data vector [900] as the interface returns it (a list of
      floats, which get_datavector converts to numpy), masked entries
      zero.

    Side effects:
      with print_datavector, writes (index, value) rows to
      print_datavector_file at every evaluation.
    """
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_select_sims, allsims=self.allsims)
      np.savetxt(self.filename_baryon_pca, pcs)
      datavector = ci.compute_data_vector_masked()
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      # one format per column: the index as an integer, the value with eight
      # decimals (the comma makes fmt a tuple)
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
