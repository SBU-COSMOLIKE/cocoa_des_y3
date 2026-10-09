"""Unit test: sector-wise cache invalidation (the parameter ladder).

cosmolike caches every expensive stage behind its own key: cosmology
(distances, growth, power-spectrum tables), intrinsic alignment
(FAST-PT tables under TATT), photo-z shifts (n(z) splines, lens
efficiencies), and the shear calibrations (a pure data-vector
rescale). A sector is one such group of parameters with its own cache
key. A partial-invalidation bug (one sector's update path failing to
rebuild a cached table that another sector reads, typically a C static
variable kept between calls) produces silently wrong data vectors only
in mixed update sequences, which the per-point suites never exercise.

The test walks a deterministic ladder in one process, evaluating the
model after every step (each sector's later steps keep the earlier
sectors at their last values, so the ladder ends at one well-defined
point):

    3 x cosmology-only steps   (omegam, H0, As_1e9)
    3 x IA-only steps          (A1; under TATT also A2 and BTA)
    3 x source-photo-z steps   (every DZ_S shift)
    3 x lens-photo-z steps     (every DZ_L shift)
    3 x shear-calibration steps (every M)

(a sector the configuration samples no parameters in drops out of the
ladder: separate DZ_L shifts when the lenses are the source sample, IA
amplitudes where the configuration fixes them)

It records the final data vector, then evaluates one scramble point
(every sector moved at once, galaxy bias included; the chi2 is
discarded) and returns to the ladder's final point: the pipeline must
reproduce the recorded vector bit for bit. A second model instance
walks the mirrored ladder (M -> DZ_L -> DZ_S -> IA -> cosmology) to
the same final point: the answer must depend on the point, never on
the order in which the caches were invalidated.

Assertions, in each intrinsic-alignment model (NLA and TATT; the
TATT ladder exercises the FAST-PT rebuild machinery NLA never
touches):
  1. every ladder step changes the data vector (a dead sector flag
     would pass the later checks vacuously);
  2. each M-only step rescales the masked vector by the analytic
     (1+m_i)(1+m_j) block factors to 1e-12 relative: cosmic shear by
     both bins' factors, gamma_t by the source factor, w by nothing;
  3. a no-op update (re-sending the current M values) leaves the
     vector bitwise unchanged;
  4. after the scramble, returning to the ladder's final point
     reproduces the recorded vector and chi2 bit for bit;
  5. the mirrored-order instance lands on the same final vector bit
     for bit.

Every evaluation forces a full recomputation (cobaya's cache, which
would return a stored result for a repeated point, is bypassed), so
each assertion tests cosmolike's own invalidation, not cobaya's.

To run (from the Cocoa/ folder, cocoa environment active,
start_cocoa.sh sourced):

    python -m pytest ./projects/des_y3/tests/data_vector/test_cache_consistency.py
"""

import os

# OpenMP reads OMP_NUM_THREADS when the compiled libraries load, so
# this must run before any cobaya or cosmolike import in the process.
# 4 is REQUIRED_OMP_THREADS of cocoa_testing.py: a race between
# OpenMP threads can only show up when several threads run.
os.environ["OMP_NUM_THREADS"] = "4"

import re
import sys
import unittest

# The harness stays in the parent tests/ folder; put that folder first
# on sys.path, the list of folders Python searches on import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cocoa_test_utils as u

EXAMPLE = "example2"

# Sector membership by sampled-parameter name: a parameter belongs to
# the first sector whose regular expression occurs in its name
# (_sector_of), so to exactly one. ^(a|b)$ matches exactly the names
# listed; _A1_|_A2_|_BTA_ matches a name containing any of the three;
# _M[0-9]+$ a name ending in _M and digits (DES_M1 ... DES_M4); and "."
# (any character) makes "other" catch every remaining name. The bias
# sector moves only in the scramble step (its B1 entries alone);
# "other" never moves, since DELTAS has no entry for it.
SECTORS = (
    ("cosmo", re.compile(r"^(As_1e9|H0|ns|omegab|omegam|mnu|w|w0pwa)$")),
    ("ia", re.compile(r"_A1_|_A2_|_BTA_")),
    ("dz_source", re.compile(r"_DZ_S")),
    ("dz_lens", re.compile(r"_DZ_L")),
    ("m", re.compile(r"_M[0-9]+$")),
    ("bias", re.compile(r"_B1_|_B2_|_BMAG_")),
    ("other", re.compile(r".")),
)

# Ladder phases (sector order of the forward walk) and the per-step
# offsets: parameter value = fiducial + step * delta, deterministic.
# A string key names one parameter; a compiled pattern applies its
# delta to every sampled name it occurs in. The offsets are omegam
# 0.002, H0 0.2 km/s/Mpc, As_1e9 0.02 (units of 1e-9), IA amplitudes
# and exponents 0.05, photo-z shifts 0.001 in redshift, shear
# calibrations 0.005, and the linear bias B1 0.05.
PHASES = ("cosmo", "ia", "dz_source", "dz_lens", "m")
DELTAS = {
    "cosmo": {"omegam": 0.002, "H0": 0.2, "As_1e9": 0.02},
    "ia": {re.compile(r"_A1_1$"): 0.05, re.compile(r"_A1_2$"): 0.05,
           re.compile(r"_A2_1$"): 0.05, re.compile(r"_A2_2$"): 0.05,
           re.compile(r"_BTA_1$"): 0.05},
    "dz_source": {re.compile(r"_DZ_S"): 0.001},
    "dz_lens": {re.compile(r"_DZ_L"): 0.001},
    "m": {re.compile(r"_M[0-9]+$"): 0.005},
    "bias": {re.compile(r"_B1_"): 0.05},
}
# NSTEP: three steps per sector (the "3 x" of the module docstring).
# SCRAMBLE_STEP: one step beyond the ladder's last, so the scramble
# point differs from the final point in every sector that moves.
# RESCALE_RTOL: the M rescale is exact algebra, so only float64
# rounding (about 1e-16 per operation) separates the two vectors.
NSTEP = 3
SCRAMBLE_STEP = 4  # every sector at step 4, bias included
RESCALE_RTOL = 1.0e-12


def _sector_of(name):
    """Return the sector of one sampled parameter.

    Arguments:
      name = a sampled-parameter name, for example "DES_DZ_S1".

    Returns:
      the label of the first SECTORS entry whose pattern occurs in the
      name (pat.search finds a match anywhere in it); "other" matches
      every nonempty name.
    """
    for sector, pat in SECTORS:
        if pat.search(name):
            return sector
    return "other"


def _deltas_for(sector, names):
    """Return {parameter: per-step delta} for one sector's sampled names.

    Arguments:
      sector = a sector label (a SECTORS entry).
      names  = the sampled-parameter names of that sector.

    Returns:
      dict {name: delta} for the names DELTAS gives an offset; a sector
      without a DELTAS entry gives an empty dict and never moves.
    """
    table = DELTAS.get(sector, {})
    out = {}
    for n in names:
        for key, d in table.items():
            # a string key must equal the name; a compiled pattern only
            # has to occur in it (key.search returns None otherwise)
            if (key == n) if isinstance(key, str) else key.search(n):
                out[n] = d
                break
    return out


class TestCacheConsistency(unittest.TestCase):
    """Sector-ladder cache-invalidation check on the frozen fiducial."""

    # the classmethod decorator hands the method the class itself
    # (cls), not an instance; unittest calls setUpClass once before
    # the first test of the class
    @classmethod
    def setUpClass(cls):
        """Check the environment and the frozen files before any test."""
        u.require_cocoa_environment()
        u.verify_frozen()

    def _point_at(self, fid, steps):
        """Return the ladder point with each sector at its given step count.

        Arguments:
          fid   = the frozen fiducial point, {parameter name: value}.
          steps = {sector label: step count}.

        Returns:
          a new dict {parameter name: value}; fid is not changed. The
          offsets come from self.sector_deltas, which _run_ladder sets
          for the model in use.
        """
        point = dict(fid)
        for sector, step in steps.items():
            for n, d in self.sector_deltas[sector].items():
                point[n] = fid[n] + step * d
        return point

    def _mpairs(self, like, np_):
        """Return the shear-calibration source bins of every vector entry.

        Cosmic shear scales by both source bins' factors
        (1+m_i)(1+m_j), gamma_t (and the ks cross where a project has
        it) by the source bin's factor alone, and clustering, gk and kk
        by none. The layout follows the data vector: Fourier data
        vectors use ncl entries per bin pair, real ones ntheta, and in
        real space the shear block appears twice (xi_plus, then
        xi_minus).

        Arguments:
          like = the likelihood instance (its ntheta or ncl,
                 source_ntomo, lens_ntomo and optional ggl_exclude).
          np_  = the numpy module, passed in by the caller.

        Returns:
          int array [n_data, 2] of (i, j) per entry: the source bins
          whose factors multiply that entry, -1 meaning no factor (a
          gamma_t entry holds (-1, source bin)).

        legend: n_data = data-vector length, the sum of the block sizes
        """
        import cosmolike_des_y3_interface as ci
        # a real-space interface exports the _real_sizes function, a
        # Fourier-space one the _fourier_sizes function; sizes lists the
        # block lengths [ss, gs, gg, ...] from whichever exists
        real = hasattr(ci, "compute_data_vector_3x2pt_real_sizes")
        sizes = [int(x) for x in
                 (ci.compute_data_vector_3x2pt_real_sizes() if real else
                  ci.compute_data_vector_3x2pt_fourier_sizes())]
        nlen = int(like.ntheta) if real else int(like.ncl)
        nsrc = int(like.source_ntomo)
        # the source pairs i <= j in data-vector order, then the
        # (lens, source) pairs removed from gamma_t by the likelihood
        # option ggl_exclude (none when the option is absent, as in
        # des_y3), then the gamma_t pairs kept, lens bin outer
        sspairs = [(i, j) for i in range(nsrc) for j in range(i, nsrc)]
        excluded = {(int(a), int(b)) for a, b in
                    (getattr(like, "ggl_exclude", None) or [])}
        gglpairs = [(zl, zs) for zl in range(int(like.lens_ntomo))
                    for zs in range(nsrc) if (zl, zs) not in excluded]
        # every entry starts at (-1, -1), no factor
        fac = np_.zeros((sum(sizes), 2), dtype=int) - 1
        k = 0
        ssrep = sspairs + sspairs if real else sspairs  # xi_plus + xi_minus
        for (i, j) in ssrep:
            for t in range(nlen):
                fac[k] = (i, j)
                k += 1
        for (zl, zs) in gglpairs:
            for t in range(nlen):
                fac[k] = (-1, zs)
                k += 1
        k = sizes[0] + sizes[1] + sizes[2]  # skip clustering
        if len(sizes) > 3:  # 6x2pt: gk (no factor), ks (source), kk (none)
            k += sizes[3]
            for zs in range(nsrc):
                for t in range(nlen):
                    fac[k] = (-1, zs)
                    k += 1
        return fac

    def _run_ladder(self, tatt):
        """Walk the forward and the mirrored ladder and assert checks 1-5.

        Arguments:
          tatt = True runs the TATT configuration, False the NLA one.

        Returns:
          nothing; a violated check fails the test.

        Side effects:
          builds one model per ladder order in this process and prints
          the final chi2 of each ladder.
        """
        import numpy as np
        import cosmolike_des_y3_interface as ci

        name = u.EXAMPLES[EXAMPLE]["likelihood"]
        results = {}
        for order in ("forward", "mirrored"):
            info = u.load_frozen_info(EXAMPLE, tatt=tatt)
            model = u.make_model(info)
            fid = dict(u.build_point(model, EXAMPLE, tatt=tatt))
            # for each sector: the per-step deltas of the sampled
            # parameters that belong to it (the inner list collects
            # their names)
            self.sector_deltas = {
                s: _deltas_for(s, [n for n in fid if _sector_of(n) == s])
                for s, _ in SECTORS}
            for s in ("cosmo", "dz_source", "m"):
                self.assertTrue(self.sector_deltas[s],
                                f"no sampled parameters in sector {s}")
            # a sector nothing samples drops out of the ladder: lenses
            # that are the source sample carry no separate DZ_L shifts,
            # and roman_kl fixes every IA amplitude; active keeps the
            # phases whose sector has a parameter to move
            active = tuple(s for s in PHASES if self.sector_deltas[s])

            phases = active if order == "forward" else tuple(reversed(active))
            # every sector starts at step 0, the fiducial point; prev is
            # the full-length theory vector of the last evaluation, with
            # masked entries set to zero
            steps = {s: 0 for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, steps))
            prev = np.array(ci.compute_data_vector_masked())
            mfac = self._mpairs(model.likelihood[name], np)

            for sector in phases:
                for r in range(1, NSTEP + 1):
                    # the shear calibrations before this step
                    m_prev = {n: fid[n] + steps["m"] * d
                              for n, d in self.sector_deltas["m"].items()}
                    steps[sector] = r
                    point = self._point_at(fid, steps)
                    u.evaluate_chi2(model, point)
                    dv = np.array(ci.compute_data_vector_masked())
                    self.assertFalse(
                        np.array_equal(dv, prev),
                        f"{order}: {sector} step {r} left the data vector "
                        "unchanged (dead sector flag or stale cache)")
                    if sector == "m":
                        m_now = {n: point[n]
                                 for n in self.sector_deltas["m"]}
                        mp = sorted(m_prev)  # M1..M5 in bin order
                        ratio = np.ones(dv.size)
                        for k in range(dv.size):
                            i, j = mfac[k]
                            if j >= 0:
                                ratio[k] *= ((1 + m_now[mp[j]]) /
                                             (1 + m_prev[mp[j]]))
                            if i >= 0:
                                ratio[k] *= ((1 + m_now[mp[i]]) /
                                             (1 + m_prev[mp[i]]))
                        # nz marks the nonzero entries of the previous
                        # vector (not a redshift distribution): masked
                        # entries are zero and drop out of the ratio
                        nz = prev != 0
                        rel = np.abs(dv[nz]/(prev[nz]*ratio[nz]) - 1.0)
                        self.assertLess(
                            rel.max(), RESCALE_RTOL,
                            f"{order}: M step {r} is not the analytic "
                            f"(1+m_i)(1+m_j) rescale (max {rel.max():.2e})")
                    prev = dv

            final_point = self._point_at(fid, steps)
            final_chi2 = u.evaluate_chi2(model, final_point)
            final_dv = np.array(ci.compute_data_vector_masked())

            # no-op probe: identical point again, bitwise
            u.evaluate_chi2(model, dict(final_point))
            self.assertTrue(
                np.array_equal(np.array(ci.compute_data_vector_masked()),
                               final_dv),
                f"{order}: a no-op re-evaluation changed the data vector")

            # scramble: every sector at once, bias included
            scr = {s: SCRAMBLE_STEP for s in self.sector_deltas}
            u.evaluate_chi2(model, self._point_at(fid, scr))

            # return: the ladder's final point must reproduce bitwise
            back_chi2 = u.evaluate_chi2(model, final_point)
            back_dv = np.array(ci.compute_data_vector_masked())
            self.assertTrue(
                np.array_equal(back_dv, final_dv),
                f"{order}: returning after the scramble did not reproduce "
                "the data vector bit for bit (stale sector cache)")
            self.assertEqual(
                back_chi2, final_chi2,
                f"{order}: chi2 after the scramble return differs")
            results[order] = final_dv
            print(f"  {order} ladder ({'TATT' if tatt else 'NLA'}): "
                  f"final chi2 = {final_chi2:.6f}", flush=True)

        self.assertTrue(
            np.array_equal(results["forward"], results["mirrored"]),
            "the mirrored-order ladder landed on a different data vector: "
            "the answer depends on the invalidation history")

    def test_cache_consistency_nla(self):
        """The ladder checks with the NLA intrinsic-alignment model."""
        self._run_ladder(tatt=False)

    def test_cache_consistency_tatt(self):
        """The ladder checks with TATT, which adds the FAST-PT tables."""
        self._run_ladder(tatt=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
