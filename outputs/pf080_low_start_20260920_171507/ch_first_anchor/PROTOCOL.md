# Corrected-demand paper-profile factorial protocol

Created 2026-09-20T17:15:39+02:00.

## Fixed design

- `ch-af-apac-eu-row-us`: Stage-1 iteration 26; order `ch-af-apac-eu-row-us`

- Corrected input: `_MOVE\demand_calibration\correction_20260915_131409\input_data_intertemporal_corrected_20260915_131409.xlsx`
- Archived Stage-1 source root: `_MOVE\demand_calibration\terminal_state_salvage_cold_start_20260917_141134`
- Output root: `outputs\pf080_low_start_20260920_171507\ch_first_anchor`
- Export-offer manufacturing-cost factors: `[0.8]`
- Stage-1 net-capacity-change weights: `[0.5, 1.0]`
- Fixed damping factors: `[0.3, 0.4]`
- Maximum sweeps per branch: `15`
- Best-response maximum iterations: `600`
- Parallel workers: `2`
- Terminal salvage fraction: `0.5`
- Operating market periods: `2025, 2030, 2035, 2040`
- Terminal state: `2045` installed capacity only; no 2045 market or operating payoff
- Terminal salvage definition: one credit on 2045 installed capacity, discounted
  to 2045; no all-period investment subsidy
- Stage-1 source rule: last contiguous checkpoint for which every player solve
  was acceptable; any state at or after an interrupted solve is excluded

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
