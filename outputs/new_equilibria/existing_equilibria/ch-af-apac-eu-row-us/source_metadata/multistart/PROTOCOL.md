# Terminal-salvage multistart audit protocol

Created: 2026-09-17T15:45:46+02:00.

- Source factorial: `outputs\demand_calibration\terminal_state_salvage_factorial_20260917_145309\manifest.json`
- Output root: `outputs\demand_calibration\terminal_state_salvage_multistart_20260917_154543`
- Source profiles: all 15 profiles accepted by the source one-start
  local 1% criterion
- Starts per player: 3
- Maximum optimizer iterations per start: 600
- Parallel candidate workers: 6
- Algorithmic proximal penalties: zero
- Common profile frozen during every player audit
- No equilibrium profile is updated by this diagnostic

The three starts are the candidate strategy, the standard zero-capacity-change
and manufacturing-cost-price initialization, and a feasible 0.95-times
candidate-price perturbation. A candidate passes the stronger diagnostic only
when all 18 optimizer attempts succeed and the largest relative unilateral
gain over all six players is at most 1%. This is a stronger local computational
test, not a global Nash-equilibrium certificate.
