# Equilibrium-candidate archive

## Start here

Use `candidates/` for the simplified result view. It contains all 16 profiles
that met the one-start local 1% criterion, organized consistently by update
order and branch. Its `candidate_index.csv` records the selected sweep, audit
gain, limiting player, and multistart status.

The other directories preserve the complete raw searches and supporting
analysis. They remain in their original locations so saved paths in manifests
and audit records are not broken.

## Audit outcome

The common frozen-profile, zero-proximal one-start audit was applied to both sweep-40 continuation profiles. All six best-response solves succeeded in both audits, but neither profile met the local 1% maximum unilateral-gain criterion:

| Update order | Maximum gain | Player attaining maximum |
|---|---:|---|
| AF–EU–US–APAC–ROW–CH | 11.088% | APAC |
| EU–US–AF–ROW–APAC–CH | 10.445% | APAC |

The conditional price/capacity/damping grid was therefore run for both profiles. Eight profiles met the one-start criterion with all six solves successful: four from each update order. Together with the six older CH-first profiles, that first curated view contained 14 one-start candidates. A later 0.80 price-factor experiment added two accepted CH-first profiles, bringing the current curated view to 16. The six older profiles all fail the later three-start audit; the ten newer profiles have not yet received it. These are local computational candidates, not multistart-robust or global-equilibrium claims.

## Directory map

- `candidates/`: simplified candidate-first view; use this for ordinary inspection.
- `penalized_profiles/`: complete penalized sweep histories for all seven update orders, including discarded nonconvergent paths.
- `comparison_new_vs_old/`: comparison tables and candidate-level metrics.
- `existing_equilibria/`: preserved source package for the six older CH-first candidates.
- `continuation_outputs/`: complete copy of the sweep-30-to-40 continuation run.
- `new_profile_workflow_20260920_143353/direct_audits/`: direct sweep-40 audit JSON files.
- `new_profile_workflow_20260920_143353/basin_search/`: all reinitialization branches, intermediate profiles, audits, logs, and status files.
- `new_profile_workflow_20260920_143353/manifest.json`: authoritative workflow manifest and accepted-profile index.
- `new_profile_workflow_20260920_143353/results_summary.csv`: compact branch status table.
- `../pf080_low_start_20260920_171507/`: raw 0.80 price-factor experiment, including all 12 branches and their audits.

## Interpretation note

Damping is an algorithmic parameter rather than an economic parameter. Profiles that differ only in damping were retained because their economic outcome vectors are not duplicates. Among the original 14-candidate damping pairs, capacity-vector distances are about 1.9%–7.8%, price-vector distances 1.5%–6.0%, and bilateral-flow distances 19.1%–36.7%. The large flow differences may partly reflect alternative trade allocations, so economic interpretation should emphasize capacities and prices before individual bilateral flows.
