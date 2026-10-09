"""Triangle plot 5: DES-Y1 cosmic shear vs DES-Y1 3x2pt, IA parameters.

The chains, read from ../chains/ (relative to the folder the script runs
in), are EXAMPLE_MCMC3 (labeled DES-Y1 cosmic shear) and EXAMPLE_MCMC4
(labeled DES-Y1 3x2pt). No EXAMPLE_MCMC3.yaml ships with the project, so
the user provides that chain, for example by running EXAMPLE_MCMC1.yaml
with data_file: des_y1_real.dataset and its output renamed to
EXAMPLE_MCMC3. No EXAMPLE_MCMC4.yaml ships with the project, so the user
provides that chain, for example by running EXAMPLE_MCMC2.yaml with
data_file: des_y1_real.dataset and its output renamed to EXAMPLE_MCMC4.

Each chain is loaded with GetDist (the first half discarded as burn-in),
receives two derived parameters, gamma = Omega_m h and SS8 = S_8 = sigma_8
(Omega_m/0.3)^0.5, and is saved with them as the hidden text files
.VM_P5_TMP1 and .VM_P5_TMP2 in the current folder. The figure shows
omegam, sigma8, w and wa together with the intrinsic-alignment parameters
(DES_A1_1, DES_A1_2); g.export() writes it as plot5.pdf, named after the
script, in the current folder.

Run from the plots/ folder, with the cocoa environment active:

    cd ./projects/des_y3/plots
    python plot5.py

plot_all.sh runs plot1.py, plot2.py and plot3.py the same way.
"""

import getdist.plots as gplot
from getdist import MCSamples
from getdist import loadMCSamples
import os
import matplotlib
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Figure style: STIX fonts for text and mathematics, ticks on the
# bottom and left axes, a grid of zero line width (invisible), and a
# tight bounding box with PDF as the default output format.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

parameter = [u'omegam',u'sigma8', u'w', u'wa',u'DES_A1_1', u'DES_A1_2']
# The folder the script runs in; the chains are read from ../chains
# relative to it, and the temporary chain files are written into it.
chaindir=os.getcwd()

# GetDist analysis settings. ignore_rows 0.5 discards the first half of
# each chain as burn-in; 0.35 is the width of the smoothing kernel of
# the 1D and 2D density estimates; range_confidence sets the confidence
# limit that fixes each parameter's plotted range. The second set reads
# the saved files, whose burn-in is already removed (ignore_rows 0).
analysissettings={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.5',
'range_confidence' : u'0.01'}

analysissettings2={'smooth_scale_1D':0.35,'smooth_scale_2D':0.35,'ignore_rows': u'0.0',
'range_confidence' : u'0.01'}

# The two chains compared, by the output prefix Cobaya gave them
root_chains = (
  'EXAMPLE_MCMC3',
  'EXAMPLE_MCMC4'
)


# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[0],settings=analysissettings)
p = samples.getParams()
# gamma = Omega_m h, the shape parameter of the matter power spectrum.
# s8omegamp5 = sigma_8 Omega_m^0.5 is a derived parameter of the
# chain, and 0.5477225575 = sqrt(0.3), so SS8 = sigma_8 (Omega_m/0.3)^0.5,
# the S_8 of weak-lensing analyses.
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.saveAsText(chaindir + '/.VM_P5_TMP1')
# --------------------------------------------------------------------------------
samples=loadMCSamples(chaindir + '/../chains/' + root_chains[1],settings=analysissettings)
p = samples.getParams()
samples.addDerived(p.omegam*p.H0/100.,name='gamma',label='{\\Omega_m h}')
samples.addDerived(p.s8omegamp5/0.5477225575,name='SS8',label='{S_8}')
samples.saveAsText(chaindir + '/.VM_P5_TMP2')
# --------------------------------------------------------------------------------

# GetDist triangle-plot settings: tick rotation, contour and legend
# style, font sizes and the opacity of filled contours; chain_dir is
# where GetDist looks for the saved chain files.
g=gplot.getSubplotPlotter(chain_dir=chaindir,analysis_settings=analysissettings2,width_inch=9.5)
g.settings.axis_tick_x_rotation=65
g.settings.lw_contour = 1.2
g.settings.legend_rect_border = False
g.settings.figure_legend_frame = False
g.settings.axes_fontsize = 13.0
g.settings.legend_fontsize = 13.5
g.settings.alpha_filled_add = 0.85
g.settings.lab_fontsize=15.5
g.legend_labels=False

# None: no third parameter colors the 2D panels. The plot draws the
# saved chains in order with one style each: the first filled in light
# coral, the second as dashed black lines (the third style entry is
# unused with two chains); legend_loc is in figure coordinates.
param_3d = None
g.triangle_plot([chaindir + '/.VM_P5_TMP1', 
                 chaindir + '/.VM_P5_TMP2'],
parameter,
plot_3d_with_param=param_3d,line_args=[
{'lw': 2.4,'ls': 'solid', 'color':'lightcoral'},
{'lw': 1.4,'ls': '--', 'color':'black'},
{'lw': 1.0,'ls': 'solid', 'color': 'indigo'},
],
contour_colors=['lightcoral','black','indigo'],
contour_ls=['solid','--','solid'], 
contour_lws=[2.4,1.4,1.0],
filled=[True,False,False],
shaded=False,
legend_labels=[
'DES-Y1 CS',
'DES-Y1 3x2pt'
],
legend_loc=(0.48, 0.80))

# export() without a file name writes <script name>.pdf in the
# current folder
g.export()