# Corrected cold-start runs

Started 2026-09-15 13:22:51 CEST. These are algorithm-development candidates,
not paper-ready equilibria. Each requires a frozen-profile unilateral-deviation
audit after completion.

All runs use:

- input: `outputs/demand_calibration/correction_20260915_131409/input_data_intertemporal_corrected_20260915_131409.xlsx`
- input SHA-256: `5E3E392B695AB917D8FF445203C9A04F1308083391BBF3C64D35E7ED3C9AD234`
- parameter sheet: `params_region_new`
- initialization: `fresh_cost_capacity_zero_change`
- initialization SHA-256: `6A2AAE43C9900F51AD07616D52B319868C20B8F1E6ABC9521A07CD98ACD7E977`
- solver: IPOPT
- maximum sweeps: 30

## CH, AF, APAC, EU, ROW, US

- Worker PID: `17024`
- Launcher PID: `29436`
- Output: `ch-af-apac-eu-row-us/`
- Logs: `logs/ch-af-apac-eu-row-us.stdout.log` and `logs/ch-af-apac-eu-row-us.stderr.log`

```powershell
python -u scripts/run_corrected_cold_start.py --input outputs/demand_calibration/correction_20260915_131409/input_data_intertemporal_corrected_20260915_131409.xlsx --output-dir outputs/demand_calibration/cold_start_full_20260915_132217/ch-af-apac-eu-row-us --order ch,af,apac,eu,row,us --iters 30 --log-path outputs/demand_calibration/cold_start_full_20260915_132217/logs/ch-af-apac-eu-row-us.stdout.log
```

## CH, AF, EU, US, ROW, APAC

- Worker PID: `10784`
- Launcher PID: `23200`
- Output: `ch-af-eu-us-row-apac/`
- Logs: `logs/ch-af-eu-us-row-apac.stdout.log` and `logs/ch-af-eu-us-row-apac.stderr.log`

```powershell
python -u scripts/run_corrected_cold_start.py --input outputs/demand_calibration/correction_20260915_131409/input_data_intertemporal_corrected_20260915_131409.xlsx --output-dir outputs/demand_calibration/cold_start_full_20260915_132217/ch-af-eu-us-row-apac --order ch,af,eu,us,row,apac --iters 30 --log-path outputs/demand_calibration/cold_start_full_20260915_132217/logs/ch-af-eu-us-row-apac.stdout.log
```

## CH, ROW, APAC, US, EU, AF

- Worker PID: `28948`
- Launcher PID: `5908`
- Output: `ch-row-apac-us-eu-af/`
- Logs: `logs/ch-row-apac-us-eu-af.stdout.log` and `logs/ch-row-apac-us-eu-af.stderr.log`

```powershell
python -u scripts/run_corrected_cold_start.py --input outputs/demand_calibration/correction_20260915_131409/input_data_intertemporal_corrected_20260915_131409.xlsx --output-dir outputs/demand_calibration/cold_start_full_20260915_132217/ch-row-apac-us-eu-af --order ch,row,apac,us,eu,af --iters 30 --log-path outputs/demand_calibration/cold_start_full_20260915_132217/logs/ch-row-apac-us-eu-af.stdout.log
```

The `provenance.json` in each order directory is authoritative for live status,
the actual worker PID, complete loaded Dmax/a/b matrices, code hashes, settings,
and post-run result verification.
