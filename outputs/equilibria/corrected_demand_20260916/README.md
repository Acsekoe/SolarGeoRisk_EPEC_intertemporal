# Reported corrected-demand equilibria

This folder is the self-contained reporting package for the three accepted
corrected-demand profiles. The original source artifacts remain in their run
directories so that the recorded provenance paths and hashes continue to work.
No additional model run was used to create this package.

## Algorithm chronology

1. The corrected-demand Stage-1 algorithm was run from fresh primitive states
   for three manuscript player orders. Stage 1 used sequential Gauss--Seidel
   updates with adaptive damping and proximal stabilization.
2. Iterations 23, 20, and 24 of the three respective runs were retained as
   Stage-1 candidate anchors. Their exact-profile one-start, common-frozen,
   zero-proximal audits failed, with maximum relative gains of 6.0681%,
   7.6310%, and 7.0497%.
3. Starting profiles were then constructed from these anchors and released
   with fixed-damping, all-player, zero-proximal Gauss--Seidel sweeps. These
   release sweeps used no move cap, no gain filter, and no player freezing.
4. After each sweep, the common profile was audited by solving all six
   zero-proximal unilateral best-response problems from one local solver start.
   A profile was accepted when all six solves succeeded and the maximum
   normalized relative gain was at most 1%.

The three accepted profiles did **not** arise one-for-one from the three
Stage-1 anchors:

- **E1** was derived from Stage-1 iteration 23 for order
  `CH-AF-APAC-EU-ROW-US`. The Stage-1 capacity path was retained, bilateral
  offers were reset to manufacturing cost, and fixed-damping (`alpha=0.30`)
  zero-proximal sweeps were run. The transformed starting profile failed its
  audit at 9.3280%; sweep 6 passed at 0.621395% (CH).
- **E2** was derived from Stage-1 iteration 24 for order
  `CH-ROW-APAC-US-EU-AF`. The net-capacity-change path was weighted by 0.50 and
  bilateral offers were initialized at 1.20 times manufacturing cost. The
  transformed starting profile failed at 6.7573%; sweep 5 passed at 0.835510%
  (APAC).
- **E3** was derived from the same Stage-1 iteration 24 and player order as E2.
  The full net-capacity-change path was retained and bilateral offers were
  initialized at 1.20 times manufacturing cost. The transformed starting
  profile failed at 8.0545%; sweep 7 passed at 0.910692% (APAC).

The Stage-1 iteration-20 anchor for order `CH-AF-EU-US-ROW-APAC` did not
produce one of the three reported equilibria.

## Files

Each `E1`, `E2`, and `E3` directory contains:

- `equilibrium_profile.json`: accepted common strategy and market profile;
- `audit_one_start.json`: six-player frozen-profile acceptance audit;
- `initialization.json`: starting profile for the zero-proximal release; and
- `status.json`: branch specification and terminal status.

The `provenance` directory contains the fixed protocols and complete source
manifests. The full comparison report remains one directory above as
`three_corrected_demand_equilibria_20260916_232146.md`.

`convergence_paths_24_reported_runs.json` is the plot-ready consolidation of
the 21 completed factorial branches that did not pass and the three accepted
paths E1--E3. It stores the initial and after-sweep audit values, player-level
relative gains, sequential-update gains, and strategy-change metrics. E1 is
identified as coming from the earlier 3x3 experiment; the factorial branch
that failed during sweep 12 is retained separately inside the JSON under
`excluded_partial_runs`.

## Claim boundary

These files support the description **one-start local computational
1%-equilibria**. They do not establish globally optimal unilateral best
responses or exact/global Nash equilibria.
