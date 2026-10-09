# Likelihoods and their parameters <a name="des_y3_likelihood_readme"></a>

This folder holds the likelihoods of the project (one yaml file with the defaults and one python file per likelihood), the parameter files the likelihoods include, and the base class `_cosmolike_prototype_base.py`.

1. [The likelihoods](#des_y3_likelihoods)
2. [The parameter files and `fixed_params`](#des_y3_parameter_files)
3. [The masks](#des_y3_masks)
4. [What each combination fixes](#des_y3_fixed_params)
5. [Parameters that should not be varied](#des_y3_not_varied)
6. [Changing the mask, the scale cuts, or the probes](#des_y3_changing)

# The likelihoods <a name="des_y3_likelihoods"></a>

| likelihood | blocks | parameter files |
|---|---|---|
| `cosmic_shear` | `ss` | `params_source.yaml` |
| `combo_2x2pt` | `gs`, `gg` | `params_lens.yaml`, `params_source.yaml` |
| `combo_3x2pt` | `ss`, `gs`, `gg` | `params_lens.yaml`, `params_source.yaml` |
| `combo_xi_gg` | `ss`, `gg` | `params_lens.yaml`, `params_source.yaml` |
| `combo_xi_ggl` | `ss`, `gs` | `params_lens.yaml`, `params_source.yaml` |

The blocks are cosmic shear $\xi_\pm$ (`ss`), galaxy-galaxy lensing $\gamma_t$ (`gs`), and galaxy clustering $w(\theta)$ (`gg`). Every likelihood reads the data vector, covariance, and mask of one data set and computes only the blocks it holds; the entries of the other blocks are zero in the theory vector and removed from the covariance. The likelihood yamls name `data_file: DES_Y3.dataset`, which the `data` folder does not hold, so the user's yaml sets `data_file`; the examples use `des_y3_real.dataset` (DES Y3) or `des_y1_real.dataset` (DES Y1).

# The parameter files and `fixed_params` <a name="des_y3_parameter_files"></a>

There is one parameter file per galaxy sample, shared by every likelihood that uses the sample:

| file | sample | parameters |
|---|---|---|
| `params_source.yaml` | sources, 4 bins | photo-z shift `DES_DZ_S1` ... `DES_DZ_S4`, shear calibration `DES_M1` ... `DES_M4`, intrinsic alignment `DES_A1_*`, `DES_A2_*`, `DES_BTA_*`, baryon PC amplitudes `DES_BARYON_Q1` ... `DES_BARYON_Q4` |
| `params_lens.yaml` | lenses, 5 bins | photo-z shift `DES_DZ_L*`, linear bias `DES_B1_*`, nonlinear biases `DES_B2_*`, `DES_B3NL_*`, `DES_BK_*`, magnification `DES_BMAG_*`, point mass `DES_PM*` |

Each likelihood yaml includes its parameter files with cobaya's `!defaults` tag:

    [Adapted from likelihood/combo_xi_gg.yaml]
    params: !defaults [params_lens, params_source]

The tag builds the whole `params` mapping from the files, so the likelihood yaml cannot change one of its entries. A combination that fixes some of these parameters lists them in its `fixed_params` block instead:

    [Adapted from likelihood/combo_xi_gg.yaml]
    fixed_params:
      DES_PM1:
        value: 0.0
      (...)
      DES_PM5:
        value: 0.0

The base class applies this block to the defaults before cobaya merges them with the user's yaml. Each entry replaces the entry of the parameter file with cobaya's merge rule: a `value` drops the `prior`, the `ref`, and the `proposal`, and the `latex` label stays. The parameter stays declared, as a constant. The combinations therefore share the parameter files and differ only in their `fixed_params`.

# The masks <a name="des_y3_masks"></a>

The two data sets name these masks, which keep an entry of the data vector when its second column is 1:

    des_y3_real.dataset   mask_file = 3x2pt_baseline.mask
    des_y1_real.dataset   mask_file = mask_Y1_mcal

Both data vectors have 900 entries, in this order:

    ss  entries   0 - 399   xi+ then xi-, 10 source pairs (i <= j) x 20 angles each
    gs  entries 400 - 799   20 (lens, source) pairs, lens bin outer, x 20 angles
    gg  entries 800 - 899   5 lens bins x 20 angles

The entries each mask keeps, per block and per lens bin:

| mask | block | entries | kept | lens bin 1 | lens bin 2 | lens bin 3 | lens bin 4 | lens bin 5 |
|---|---|---|---|---|---|---|---|---|
| `3x2pt_baseline.mask` | `ss` | 400 | 227 | - | - | - | - | - |
| `3x2pt_baseline.mask` | `gs` | 400 | 252 | 40 | 48 | 52 | 56 | 56 |
| `3x2pt_baseline.mask` | `gg` | 100 | 54 | 8 | 10 | 11 | 12 | 13 |
| `mask_Y1_mcal` | `ss` | 400 | 227 | - | - | - | - | - |
| `mask_Y1_mcal` | `gs` | 400 | 176 | 24 | 32 | 36 | 40 | 44 |
| `mask_Y1_mcal` | `gg` | 100 | 54 | 8 | 10 | 11 | 12 | 13 |

Both masks keep entries of every lens bin and of every source bin, so no parameter is inert because of the mask.

# What each combination fixes <a name="des_y3_fixed_params"></a>

| likelihood | `fixed_params` | why |
|---|---|---|
| `cosmic_shear` | none | |
| `combo_2x2pt` | none | |
| `combo_3x2pt` | none | |
| `combo_xi_gg` | `DES_PM1`, `DES_PM2`, `DES_PM3`, `DES_PM4`, `DES_PM5` | no `gs` block: the point masses act on galaxy-galaxy lensing only |
| `combo_xi_ggl` | none | |

The fixed value of the point masses is 0, the center of their prior and of their `ref` in `params_lens.yaml`.

# Parameters that should not be varied <a name="des_y3_not_varied"></a>

The table lists, for every likelihood of this folder, the parameters that do not change its masked data vector, why, and who fixes them. The columns:

    Applies when   the likelihood option (or parameter value) under which the row holds
    Response       `0`: the masked data vector does not change at all
    Fixed by       `fixed_params` of the likelihood yaml; a parameter file, where the
                   parameter is a constant; user: the parameter files sample it, and the
                   user's yaml must fix it; not declared: the likelihood does not include
                   the parameter's file, so the parameter does not exist there

A `*` in a parameter name stands for every bin number.

| Likelihood | Parameters | Reason | Applies when | Response | Fixed by |
|---|---|---|---|---|---|
| `cosmic_shear` | `DES_DZ_L*`, `DES_B1_*`, `DES_B2_*`, `DES_BMAG_*`, `DES_B3NL_*`, `DES_BK_*`, `DES_PM*` | no lens block | always | `0` | not declared (the yaml includes `params_source.yaml` only) |
| `cosmic_shear` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `cosmic_shear` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `cosmic_shear` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_2x2pt` | `DES_B3NL_1`, `DES_B3NL_2`, `DES_B3NL_3`, `DES_B3NL_4`, `DES_B3NL_5`, `DES_BK_1`, `DES_BK_2`, `DES_BK_3`, `DES_BK_4`, `DES_BK_5` | these biases enter only the one-loop bias terms, which stay off while every `DES_B2_*` is 0 | `DES_B2_*: 0` | `0` | `params_lens.yaml` |
| `combo_2x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_2x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_2x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_3x2pt` | `DES_B3NL_1`, `DES_B3NL_2`, `DES_B3NL_3`, `DES_B3NL_4`, `DES_B3NL_5`, `DES_BK_1`, `DES_BK_2`, `DES_BK_3`, `DES_BK_4`, `DES_BK_5` | these biases enter only the one-loop bias terms, which stay off while every `DES_B2_*` is 0 | `DES_B2_*: 0` | `0` | `params_lens.yaml` |
| `combo_3x2pt` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_3x2pt` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_3x2pt` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_xi_gg` | `DES_PM1`, `DES_PM2`, `DES_PM3`, `DES_PM4`, `DES_PM5` | no `gs` block: the point masses act on galaxy-galaxy lensing only | always | `0` | `fixed_params` of `combo_xi_gg.yaml` |
| `combo_xi_gg` | `DES_B3NL_1`, `DES_B3NL_2`, `DES_B3NL_3`, `DES_B3NL_4`, `DES_B3NL_5`, `DES_BK_1`, `DES_BK_2`, `DES_BK_3`, `DES_BK_4`, `DES_BK_5` | these biases enter only the one-loop bias terms, which stay off while every `DES_B2_*` is 0 | `DES_B2_*: 0` | `0` | `params_lens.yaml` |
| `combo_xi_gg` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_xi_gg` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_xi_gg` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |
| `combo_xi_ggl` | `DES_B3NL_1`, `DES_B3NL_2`, `DES_B3NL_3`, `DES_B3NL_4`, `DES_B3NL_5`, `DES_BK_1`, `DES_BK_2`, `DES_BK_3`, `DES_BK_4`, `DES_BK_5` | these biases enter only the one-loop bias terms, which stay off while every `DES_B2_*` is 0 | `DES_B2_*: 0` | `0` | `params_lens.yaml` |
| `combo_xi_ggl` | `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | NLA has no tidal-torque terms | `IA_model: 0` | `0` | user (`params_source.yaml` samples them) |
| `combo_xi_ggl` | `DES_A1_3`, `DES_A1_4`, `DES_A2_3`, `DES_A2_4`, `DES_BTA_2`, `DES_BTA_3`, `DES_BTA_4` | the redshift-evolution IA model reads only `DES_A1_1`, `DES_A1_2`, `DES_A2_1`, `DES_A2_2`, `DES_BTA_1` | `IA_redshift_evolution: 3` | `0` | `params_source.yaml` |
| `combo_xi_ggl` | `DES_BARYON_Q1`, `DES_BARYON_Q2`, `DES_BARYON_Q3`, `DES_BARYON_Q4` | the baryon PC amplitudes enter only when the likelihood applies the PCs | `use_baryon_pca: False` | `0` | `params_source.yaml` |

`DES_B2_*` are constants (0) of `params_lens.yaml` but are not in the table: a nonzero value in any lens bin switches on the one-loop bias terms of every lens bin, and with them `DES_B3NL_*` and `DES_BK_*`.

The examples that run `combo_3x2pt` (`EXAMPLE_EVALUATE2.yaml`, `EXAMPLE_EVALUATE4.yaml`, `EXAMPLE_MCMC2.yaml`, and the EMUL2 examples numbered 2) fix `DES_PM1` ... `DES_PM5` at 0 in their own `params` block, which overrides the parameter files, so they do not sample the point masses although `combo_3x2pt` holds `gs`. Every example except `EXAMPLE_EVALUATE3.yaml`, `EXAMPLE_EVALUATE4.yaml`, and `EXAMPLE_EMUL2_EVALUATE2.yaml` runs TATT (`IA_model: 1`), where `DES_A2_1`, `DES_A2_2`, and `DES_BTA_1` act on the data vector. Those three run NLA: the first two fix the three parameters at 0, while `EXAMPLE_EMUL2_EVALUATE2.yaml` (and the hybrid scripts that read it) still samples them, although they are inert there.

# Changing the mask, the scale cuts, or the probes <a name="des_y3_changing"></a>

> [!Warning]
> The `fixed_params` blocks encode the blocks of each combination (and, in general, the mask of the data set). A user who changes the mask or the scale cuts, or the probes of a combination, must revisit that combination's `fixed_params`: a parameter it fixes may then act on the data vector, and the fixed value hides that dependence without any error.

A `fixed_params` block in the likelihood block of the user's yaml replaces the combination's block as a whole: `fixed_params: null` samples every parameter again, and a shorter block keeps only the entries it repeats. Only `combo_xi_gg.yaml` declares `fixed_params`; with the other likelihoods, fix parameters in the `params` block of the user's yaml.

We assume users are in the Conda cocoa environment from a previous `conda activate cocoa` command, that the shell is bash, and that the current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: in the likelihood block of the user's yaml (below, a copy of `EXAMPLE_EVALUATE2.yaml` named `MY_EVALUATE2.yaml`, with `combo_xi_gg` selected), set `fixed_params`

    likelihood:
      des_y3.combo_xi_gg:
        path: ./external_modules/data/des_y3
        data_file: des_y3_real.dataset
        fixed_params: null

**Step :three:**: declare every parameter sampled again with a `prior` in the `params` block, or delete its `value` entry there (the examples declare the point masses as constants), and add a value for it to the `sampler: evaluate: override` block (cobaya's evaluate sampler refuses an override of a parameter that is not sampled, and draws a sampled parameter that the block omits at random from its `ref`)

    sampler:
      evaluate:
        override:
          (...)
          DES_PM1: 0.0

**Step :four:**: run the evaluation

    cobaya-run ./projects/des_y3/MY_EVALUATE2.yaml -f

> [!NOTE]
> A parameter declared in the `params` block of the user's yaml follows that declaration whatever `fixed_params` holds: the user's `params` entry takes precedence over the likelihood defaults.
