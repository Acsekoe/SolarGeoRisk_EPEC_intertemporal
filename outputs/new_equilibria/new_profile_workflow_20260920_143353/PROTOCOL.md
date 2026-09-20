# Continued-profile equilibrium workflow

Created 2026-09-20T14:33:54+02:00.

## Direct check

- Profiles: `['af-eu-us-apac-row-ch', 'eu-us-af-row-apac-ch']`
- Source checkpoint: sweep 30
- Continued checkpoint: absolute sweep 40
- Common frozen profile during every audit
- One candidate-initialized best-response solve per player
- Zero algorithmic proximal penalties in the audit
- Acceptance: all six solves successful and maximum normalized unilateral gain <= 1%

## Conditional basin search

The established reinitialization grid is run only for profiles that fail the
direct check. Off-diagonal offer prices start at 1.0 or 1.2 times the
exporter's manufacturing cost; the continued net-capacity-change path receives
weight 0.5 or 1.0; and sequential Gauss--Seidel damping is fixed at 0.3 or 0.4.
All updates and audits use zero proximal penalties. Each branch is audited at
initialization and after every sweep, stops at its first accepted profile, and
runs for at most 15 sweeps.

- Corrected input: `outputs\demand_calibration\correction_20260915_131409\input_data_intertemporal_corrected_20260915_131409.xlsx`
- Original sweep-30 root: `outputs\demand_calibration\terminal_state_salvage_figure2_orders_20260918_195503`
- Penalized continuation root: `outputs\demand_calibration\penalized_sweep30_continuation_20260920_140510`
- Output root: `outputs\new_equilibria\new_profile_workflow_20260920_143353`
- Direct/grid best-response maximum iterations: 600
- Parallel workers: 6
- Terminal salvage fraction: 0.5
