# Curated equilibrium-candidate view

This is the simplest entry point to the candidate results. It collects the 16
profiles that met the common frozen-profile, zero-proximal **one-start** 1%
criterion and arranges them uniformly as:

`update-order/branch/`

Each branch contains:

- `profile.json`: selected strategy, capacity, and market profile;
- `audit_one_start.json`: the one-start audit that accepted the profile;
- `audit_three_start.json`: available only for the six older CH-first profiles;
- `plots/`: four figure types in PNG and vector PDF form.

`candidate_index.csv` gives the complete candidate list and audit status.

## Important interpretation

- All 16 profiles pass the declared one-start local 1% criterion.
- Six older `ch-af-apac-eu-row-us` profiles were also subjected to a
  three-start audit. All six exceed 1% in that audit.
- The two `pf080` CH-first profiles and the eight AF-first and EU-first profiles
  have not yet received the three-start audit.

The folder therefore contains 16 **one-start local candidates**, not 16
multistart-robust or globally certified Nash equilibria.

## Branch codes

- `pf080` / `pf100` / `pf120`: bilateral export offers initialized at
  0.80 / 1.00 / 1.20 times period-specific manufacturing cost;
- `k050` / `k100`: anchor capacity-change path weighted by 0.50 / 1.00;
- `a030` / `a040`: fixed Gauss--Seidel damping of 0.30 / 0.40.

These files are convenience copies. The complete raw sweeps, logs, statuses,
and authoritative manifests remain in the parent archive and retain their
original paths.
