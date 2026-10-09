"""Cobaya likelihood des_y3.combo_xi_ggl: DES cosmic shear plus galaxy-galaxy lensing.

Cobaya finds this class through cobaya/likelihoods/des_y3, a link to this
folder that start_cocoa.sh creates, and reads its default options from
combo_xi_ggl.yaml next to this file (the user's yaml overrides them and sets
data_file, the .dataset file of the DES data set to use). All the work is
done by the base class _cosmolike_prototype_base; this class only passes the
probe string "xi_ggl", which selects the blocks of the 900-entry DES data
vector that CosmoLike computes: cosmic shear (ss, entries 0-399) and
galaxy-galaxy lensing (gs, entries 400-799). The entries of the other blocks
are zero in the theory vector and removed from the covariance.
"""

from cobaya.likelihoods.des_y3._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y3_interface as ci
import numpy as np

class combo_xi_ggl(_cosmolike_prototype_base):
  """The base class with probe "xi_ggl": cosmic shear and galaxy-galaxy lensing."""
  def initialize(self):
    """Run the shared set-up of the base class for this probe.

    Cobaya calls initialize once, after it has copied the yaml options onto
    the instance. super(...).initialize runs
    _cosmolike_prototype_base.initialize, which reads the .dataset file,
    builds the redshift and wavenumber grids and configures CosmoLike for
    the selected probe.
    """
    super(combo_xi_ggl,self).initialize(probe="xi_ggl")
