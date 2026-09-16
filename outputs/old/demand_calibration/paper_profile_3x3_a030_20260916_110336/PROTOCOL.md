# Fixed 3 x 3 paper-profile release experiment

Protocol fixed before launching the run on 2026-09-16 (Europe/Vienna).

## Research question

For each of the three corrected-demand terminal profiles corresponding to the
manuscript player orders, can a fixed zero-proximal Gauss--Seidel release reach
a profile satisfying the local relative 1% unilateral-deviation criterion
within 20 complete sweeps?

## Player orders and source endpoints

1. `ch-af-apac-eu-row-us`, source iteration 23.
2. `ch-af-eu-us-row-apac`, source iteration 20.
3. `ch-row-apac-us-eu-af`, source iteration 24.

The source endpoints are replayed from the corresponding corrected-demand
workbooks under
`outputs/demand_calibration/cold_start_full_20260915_132217/`.

## Starting scenarios

Each source endpoint is used to construct exactly three starting profiles:

1. `stage1`: retain the exact Stage-1 capacity and offer-price strategy.
2. `halfcost`: retain Stage-1 capacities and move every cross-regional offer
   halfway from its Stage-1 value to the exporter's period-specific
   manufacturing cost.
3. `cost`: retain Stage-1 capacities and set every cross-regional offer to the
   exporter's period-specific manufacturing cost.

This produces nine branches. No additional starting scenario will be added
after outcomes are observed.

## Release and audit rule

- Corrected demand input:
  `outputs/demand_calibration/correction_20260915_131409/input_data_intertemporal_corrected_20260915_131409.xlsx`.
- Fixed damping: `alpha = 0.30`.
- Algorithmic proximal coefficients: zero from the first released sweep.
- Every player is updated in every sweep; no selective freezing.
- No move cap.
- One optimizer start per sequential best response.
- Player order remains the source manuscript order.
- Maximum 20 complete sweeps per branch.
- The unchanged starting profile is audited at sweep 0.
- A one-start, common-frozen-profile, zero-proximal audit is performed after
  every complete sweep.
- A branch is accepted at its first profile for which all six audit solves
  succeed and the maximum relative unilateral economic gain is at most 1%.
- A branch terminates at its first accepted profile, at the 20-sweep limit, or
  on a best-response solver failure.
- All successes, failures, and non-passes are retained and reported.

## Execution

The nine branches are executed with five parallel workers and a per-solve
iteration limit of 600 using `scripts/run_corrected_a030_search.py`.

The existing partial alpha-0.30 Stage-1-only trajectories in
`a030_one_start_corrected_20260915_192458` are prior pilot evidence and are not
part of this run. Accordingly, this experiment is prospectively fixed after
pilot analysis, not represented as outcome-blind preregistration.
