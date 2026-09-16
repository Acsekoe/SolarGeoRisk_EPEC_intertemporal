# Corrected-demand paper-profile factorial protocol

Created 2026-09-16T18:50:30+02:00.

## Fixed design

- `ch-af-apac-eu-row-us`: Stage-1 iteration 23; order `ch-af-apac-eu-row-us`
- `ch-af-eu-us-row-apac`: Stage-1 iteration 20; order `ch-af-eu-us-row-apac`
- `ch-row-apac-us-eu-af`: Stage-1 iteration 24; order `ch-row-apac-us-eu-af`

- Corrected input: `outputs\old\demand_calibration\correction_20260915_131409\input_data_intertemporal_corrected_20260915_131409.xlsx`
- Archived Stage-1 source root: `outputs\old\demand_calibration\cold_start_full_20260915_132217`
- Output root: `outputs\demand_calibration\paper_profile_factorial_20260916_182347`
- Export-offer manufacturing-cost factors: `[1.0, 1.2]`
- Stage-1 net-capacity-change weights: `[0.5, 1.0]`
- Fixed damping factors: `[0.3, 0.4]`
- Maximum sweeps per branch: `20`
- Best-response maximum iterations: `600`
- Parallel workers: `5`

The capacity weight multiplies the complete Stage-1 `dK_net` path and the
corresponding capacity path is reconstructed from the model's accounting
identity. A weight of zero would be the observed-capacity, zero-change path;
the present design uses the predeclared weights listed above.

Every off-diagonal bilateral offer is initialized to its exporter's
period-specific manufacturing cost times the branch price factor. Domestic
offers remain at manufacturing cost. Values are clipped only to the model's
feasible offer bounds during initialization.

## Candidate generation

- all-player sequential Gauss--Seidel best responses;
- fixed branch-specific damping;
- zero algorithmic proximal penalties;
- no move cap;
- no gain filter or player freezing;
- one solver start per sequential best response.

## Acceptance

After the initialization and every completed sweep, all six players receive
an independent zero-proximal best-response solve against one common frozen
profile. Damping, the capacity-path weight, and the initialization price factor
do not enter this audit. A profile is accepted when all six audit solves
succeed and the maximum relative unilateral gain is at most 1%. Candidate
generation stops at the first accepted sweep; otherwise the lowest-gain audited
profile is retained as a diagnostic. Multistart is not used by this run.
