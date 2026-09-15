# Corrected diversified equilibrium search

Started 2026-09-15 15:54:02 CEST after stopping the unproductive adaptive-O6
continuation. The stopped O6 tree remains intact at
`outputs/demand_calibration/equilibrium_search_corrected_20260915_141124/`.

Every branch starts from the terminal profile of its corrected Stage-1
paper-algorithm run. No stalled O6 state or previously certified profile is
used as an initialization.

## Branches per player order

1. `stage1_selective`: exact Stage-1 profile; selective zero-proximal updates
   with `omega=0.10` and normalized move cap `0.01`.
2. `halfcost_selective`: Stage-1 capacities with offers interpolated 50% toward
   time-specific manufacturing cost; the same selective update rule.
3. `cost_staged`: Stage-1 capacities with offers reset to manufacturing cost;
   four sweeps capped at `0.002`, one sweep capped at `0.06`, then selective
   `omega=0.10`, cap-`0.01` refinement.

Only players with sequential relative economic gain above 1% are updated.
Every complete sweep is saved and audited again at one common frozen profile.
The branch stops at the first successful one-start audit with maximum gain at
or below 1%, then records a separate three-start diagnostic.

The cost-staged branches for all three orders are first in the three-worker
queue because the earlier strongest equilibrium came from this basin.

## Launch

- Windows launcher PID: `31184`
- Python orchestrator PID: `29484`
- Initial cost-staged worker PIDs:
  - CH-AF-APAC-EU-ROW-US: `28864`
  - CH-AF-EU-US-ROW-APAC: `428`
  - CH-ROW-APAC-US-EU-AF: `32148`
- Corrected input SHA-256:
  `5E3E392B695AB917D8FF445203C9A04F1308083391BBF3C64D35E7ED3C9AD234`

```powershell
python -u scripts/run_corrected_diversified_search.py `
  --input outputs/demand_calibration/correction_20260915_131409/input_data_intertemporal_corrected_20260915_131409.xlsx `
  --cold-start-root outputs/demand_calibration/cold_start_full_20260915_132217 `
  --output-root outputs/demand_calibration/diversified_search_corrected_20260915_155249 `
  --workers 3 --maxiter 500 --gain-tolerance 0.01
```

`manifest.json` contains the complete branch specifications and launch-time
SHA-256 hashes. Live per-branch state is recorded in each `status.json`.
