# Planner discount-rate verification

The old benchmark was optimized at 0% annual discounting; the EPEC profiles and the published welfare evaluation use 2%. The planner was re-solved with the identical calibrated input (SHA-256 `F426E480A20565286F6CD6678A5EEC0DE6E7376382826983E7FC3D39C2DEA5E0`), 2% annual discounting, and 50% terminal salvage. IPOPT reported `ModelStatus.OptimalLocal`.

## Result

- The production, trade, and capacity allocation is unchanged to numerical precision (largest capacity/output difference 1.09e-08 GW; largest bilateral flow difference 1.17e-08 GW).
- Discounted global welfare excluding salvage is 10,114.701051848 billion USD for the old allocation and 10,114.701051850 billion USD after re-optimization. Both round to **10,114.7 billion USD**.
- The new optimization objective including terminal salvage is 10,127.116532 billion USD. The manuscript welfare table excludes salvage for both planner and EPEC.
- Planner dual prices differ by at most 1.471 USD/kW. The physical allocation and aggregate welfare are the same, but regional CS/PS and welfare shift slightly because these prices enter regional transfers.

| Region | Old planner welfare | 2% planner welfare |
|---|---:|---:|
| CH | 3,963.8 | 3,965.5 |
| EU | 2,117.5 | 2,117.0 |
| US | 812.6 | 812.3 |
| APAC | 1,080.1 | 1,079.6 |
| AF | 493.7 | 493.5 |
| ROW | 1,647.1 | 1,646.7 |

| Year | Old global planner price | 2% global planner price |
|---|---:|---:|
| 2025 | 168.757 | 168.757 |
| 2030 | 114.139 | 114.139 |
| 2035 | 103.307 | 103.090 |
| 2040 | 96.238 | 97.709 |

## Reproduce

```powershell
python -m model.model_llp_planner --input 'D:\Alexander\Studium\EEG\Complementarity Modelling\_MOVE\demand_calibration\correction_20260915_131409\input_data_intertemporal_corrected_20260915_131409.xlsx' --output-dir 'outputs/planner_discount_check_20260929/cli' --terminal-salvage-fraction 0.5 --discount-rate 0.02 --base-year 2025
```

The original workbook and log are preserved in `original_0pct/`. The corrected workbook is also installed in `outputs/llp_planner/`; the paper price and welfare summaries and figures were regenerated from it.
