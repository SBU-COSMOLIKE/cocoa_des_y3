"""Cobaya likelihood des_y3.cosmic_shear: DES cosmic shear.

Cobaya finds this class through cobaya/likelihoods/des_y3, a link to this
folder that start_cocoa.sh creates, and reads its default options from
cosmic_shear.yaml next to this file (the user's yaml overrides them and sets
data_file, the .dataset file of the DES data set to use). All the work is
done by the base class _cosmolike_prototype_base; this class only passes the
probe string "xi", which selects the blocks of the 900-entry DES data vector
that CosmoLike computes: cosmic shear xi_plus and xi_minus (the ss block,
entries 0-399). The entries of the other blocks are zero in the theory
vector and removed from the covariance.
"""

from cobaya.likelihoods.des_y3._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_des_y3_interface as ci
import numpy as np

class cosmic_shear(_cosmolike_prototype_base):
  """The base class with probe "xi": cosmic shear."""
  def initialize(self):
    """Run the shared set-up of the base class for this probe.

    Cobaya calls initialize once, after it has copied the yaml options onto
    the instance. super(...).initialize runs
    _cosmolike_prototype_base.initialize, which reads the .dataset file,
    builds the redshift and wavenumber grids and configures CosmoLike for
    the selected probe.
    """
    super(cosmic_shear,self).initialize(probe="xi")
