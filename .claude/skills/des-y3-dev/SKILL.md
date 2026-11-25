---
name: des-y3-dev
description: >-
  Maintain the DES Y3 3×2pt likelihood in Cocoa. Use for emulator accuracy, cheap FoM
  checks and numerical convergence in Cocoa/projects/des_y3.
---

# DES Y3

## Emulator accuracy and cheap FoM tests

Follow Cocoa's [emulator accuracy and cheap FoM guidance](../../../../../../.claude/skills/cocoa-maintenance/SKILL.md#emulator-accuracy-and-cheap-fom-tests-keep-their-purposes-separate).

- Evaluate each candidate on the saved benchmark configuration, mask,
  covariance, nuisance values, and fixed 50 broad and 25 focused points.
  Reuse the saved approximate nuisance-marginalized Fisher reference.
  Check file readability and dimensions. Keep source hashes, automatic
  invalidation, and reference rebuilds outside the benchmark. Use ordinary
  MCMC/chi-squared settings for both predictions. Verify the effective
  CAMB `AccuracyBoost`, `k_per_logint` and native settings after overrides.
  Score `delta_d.T @ C_inverse @ delta_d` with
  `delta_d = d_emulator - d_CAMB` on the retained rows. Use the shared plot
  with a distinct failure marker. Evaluate all valid points even when the
  contours are approximate.
- For the cheap FoM test, compute one full set of derivatives at the
  project's default accuracy, including its non-Limber tolerances.
  Marginalize nuisance parameters with the shared Fisher machinery.
  Use a fixed relative step of 2.5% and documented absolute steps at zero.
  The shared LSST Y1 × DESI noise estimate uses higher accuracy only for
  the selected sensitive columns. Do not compute the remaining columns
  afterward. Mark their corrections as unknown. For nuisance-only columns,
  reuse theory through shared batching while refreshing the native vector.
  Record the estimated shift and selected columns. Full refinements and
  half-step checks belong to separate publication or convergence studies.
- For the laptop benchmark schedule, run one project at a time with two MPI
  ranks, four native OpenMP threads per rank and one BLAS thread. Verify
  effective settings and the native thread count after initialization.
  Time the benchmark and FoM test separately, recording wall time,
  evaluation counts and fresh versus reused predictions. Keep drivers thin
  and reuse the shared calculation and plotting functions.
