# 0.8 price-factor low-start experiment

This directory is an isolated experiment for the three penalized-search orders that reached movement convergence. Nothing here has been promoted into `outputs/new_equilibria`.

## Design

- Orders: `ch-af-apac-eu-row-us`, `af-eu-us-apac-row-ch`, and `eu-us-af-row-apac-ch`
- Export-offer initialization: `0.8` times period-specific manufacturing cost
- Capacity-path weights: `0.5` and `1.0`
- Fixed damping values: `0.3` and `0.4`
- Maximum: 15 Gauss--Seidel sweeps per branch
- Proximal penalties, move caps, gain filters, and player freezing: disabled
- Acceptance test: one-start, common frozen-profile, zero-proximal audit with all six solves successful and maximum relative gain at or below 1%

The grid contains 12 branches. Two passed the local 1% criterion, both for the CH-first order and full (`k100`) capacity path.

## Accepted branches

| Order | Branch | Selected sweep | Maximum gain | Limiting player |
|---|---:|---:|---:|---|
| `ch-af-apac-eu-row-us` | `pf080_k100_a030` | 13 | 0.927165% | APAC |
| `ch-af-apac-eu-row-us` | `pf080_k100_a040` | 12 | 0.605465% | CH |

These are local computational 1%-equilibrium results from one audit start; they are not multistart or global-equilibrium claims.

## Best audited result for non-passing branches

| Order | Branch | Selected sweep | Maximum gain | Limiting player |
|---|---:|---:|---:|---|
| `ch-af-apac-eu-row-us` | `pf080_k050_a030` | 12 | 1.850165% | APAC |
| `ch-af-apac-eu-row-us` | `pf080_k050_a040` | 10 | 2.893658% | APAC |
| `af-eu-us-apac-row-ch` | `pf080_k050_a030` | 14 | 1.786657% | CH |
| `af-eu-us-apac-row-ch` | `pf080_k050_a040` | 1 | 2.460607% | APAC |
| `af-eu-us-apac-row-ch` | `pf080_k100_a030` | 10 | 1.236975% | APAC |
| `af-eu-us-apac-row-ch` | `pf080_k100_a040` | 15 | 2.284852% | APAC |
| `eu-us-af-row-apac-ch` | `pf080_k050_a030` | 15 | 1.942110% | APAC |
| `eu-us-af-row-apac-ch` | `pf080_k050_a040` | 1 | 2.560571% | CH |
| `eu-us-af-row-apac-ch` | `pf080_k100_a030` | 1 | 5.017533% | APAC |
| `eu-us-af-row-apac-ch` | `pf080_k100_a040` | 11 | 2.493222% | APAC |

## Layout

- `results_summary.csv`: consolidated 12-branch result index
- `ch_first_anchor/`: CH-first profiles, audits, statuses, and runner manifest
- `continued_anchors/`: AF-first and EU-first profiles, audits, direct anchor audits, statuses, and runner manifest

Each branch status selects its best valid audited profile when no sweep passes. Raw per-sweep profiles and audits are retained.
