# Demand calibration audit

Generated 2026-09-15 13:24 CEST.

## Finding

The demand-calibration error is confirmed and material. The three historical
`outputs/sens/converged/sens_*.xlsx` workbooks used the active Dmax values from
`params_region_new` together with explicit `b_dem` values carried over from the
old demand calibration. Of the 24 explicit 2025-2040 slope coefficients, 23 are
inconsistent with the workbook's documented formula. Africa 2025 is the only
matching coefficient. The inherited 2045 coefficients repeat the erroneous
2040 values.

The active `a_dem` coefficients are consistent with the documented formula.

## Backup and source integrity

- Verified backup: `workflow/backups/inputs_20260915_125947/inputs/`
- Backup contents: one file, `input_data_intertemporal.xlsx` (83,889 bytes)
- Backup workbook SHA-256: `B42A2C12CEEC35C0CFB199774EE5DB2FAC10F83321C215FE63B6AFEC0D4D0A0D`
- Original active workbook SHA-256: `B42A2C12CEEC35C0CFB199774EE5DB2FAC10F83321C215FE63B6AFEC0D4D0A0D`
- The backup and active original match exactly. The original workbook was not modified.

## Historical run provenance

| Historical workbook | Run ID | Player order | SHA-256 |
| --- | --- | --- | --- |
| `sens_ch-af-apac-eu-row-us.xlsx` | `20260409_025320_8ab798` | CH, AF, APAC, EU, ROW, US | `F96E3154A20B87B33626FBC6EA2FA7D92470C9F4C7AC0E2722C2782B1D911E15` |
| `sens_ch-af-eu-us-row-apac.xlsx` | `20260409_021059_b4cc51` | CH, AF, EU, US, ROW, APAC | `6FBB730928F314B0D2BC48B8DE004281359CFC2576837A8D9F0C3294217C4DFC` |
| `sens_ch-row-apac-us-eu-af.xlsx` | `20260409_000435_df8017` | CH, ROW, APAC, US, EU, AF | `D515AB2E4CFD1E6894E76EFBFF88724B373CF7D48677FC7AAA8BDAB0DD00E8A9` |

Each historical `meta` sheet records the same original input path. All three
record IPOPT, feasibility and optimality tolerances of `1e-4`, 30 sweeps,
initial damping `0.8`, adaptive damping with minimum `0.4`, a 1% reported
tolerance, and three stable iterations. Each `regions` sheet contains the same
`a_dem_used` and `b_dem_used` matrices shown below.

The input workbook is the Git blob present at commits
`a3cb86f574a967f1883d43422298782463a35c66` (2026-04-08 20:41 CEST) and
`70a51a9b0da91e1ce517672dcadc8ed21de268cb` (2026-04-09 07:44 CEST), with the
same SHA-256 as the active original and backup. The relevant historical
`model/data_prep.py` SHA-256 is
`F8CBD04FBF94D96FC84B64136AFAB6ACD6BCCEC0D7A3859D4125AE30262BDDF2`.
That loader reads the requested parameter sheet, prioritizes explicit
time-indexed `b_dem` columns, and the run configuration defaults to
`params_region_new`.

The old output metadata does not store an input hash or parameter-sheet name.
Therefore the old files are not self-contained cryptographic provenance. The
input path, unchanged Git-era workbook hash, code path, timestamps, player-order
sweep, and exact demand-parameter signature collectively identify the source as
the SHA-matched original workbook and `params_region_new`.

## Dmax used by all three historical runs

The implementation intentionally inherits 2040 into the terminal 2045 period.

| Region | 2025 | 2030 | 2035 | 2040 | 2045 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CH | 280 | 350 | 333 | 316 | 316 |
| EU | 75 | 90 | 95 | 99 | 99 |
| US | 45 | 65 | 70 | 75 | 75 |
| APAC | 50 | 85 | 103 | 126 | 126 |
| AF | 5 | 10 | 14 | 20 | 20 |
| ROW | 45 | 70 | 79 | 89 | 89 |

## a_dem used by all three historical runs

These values are constant over 2025-2045 and are correctly calibrated as
`p_full * (1 + 1 / 0.90)`.

| Region | a_dem in every period |
| --- | ---: |
| CH | 1100.775555555556 |
| EU | 2042.816666666667 |
| US | 1184.481111111111 |
| APAC | 1127.628888888889 |
| AF | 2397.377777777778 |
| ROW | 2046.764444444445 |

## b_dem used by all three historical runs

| Region | 2025 | 2030 | 2035 | 2040 | 2045 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CH | 2.084012789768 | 1.413062330623 | 1.072880658436 | 0.965592592593 | 0.965592592593 |
| EU | 18.226253037238 | 15.811274509804 | 7.414942528736 | 4.300666666667 | 4.300666666667 |
| US | 16.289812153413 | 8.905873015873 | 6.562222222222 | 5.420966183575 | 5.420966183575 |
| APAC | 12.092275649733 | 2.046513409962 | 1.582637037037 | 1.290193236715 | 1.290193236715 |
| AF | 252.355555555556 | 16.699017704841 | 9.705982905983 | 7.009876543210 | 7.009876543210 |
| ROW | 39.016459414866 | 71.816296296296 | 17.374910394265 | 11.969382716049 | 11.969382716049 |

## Correct coefficients

Independent recalculation uses:

`a_i = p_full_i * (1 + 1 / epsilon_i)`

`b_i,t = p_full_i / (epsilon_i * Dmax_i,t)`

with `epsilon_i = 0.90` for every region.

| Region | 2025 | 2030 | 2035 | 2040 | 2045 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CH | 2.069126984127 | 1.655301587302 | 1.739806473140 | 1.833403656821 | 1.833403656821 |
| EU | 14.335555555556 | 11.946296296296 | 11.317543859649 | 10.860269360269 | 10.860269360269 |
| US | 13.853580246914 | 9.590940170940 | 8.905873015873 | 8.312148148148 | 8.312148148148 |
| APAC | 11.869777777778 | 6.982222222222 | 5.762028047465 | 4.710229276896 | 4.710229276896 |
| AF | 252.355555555556 | 126.177777777778 | 90.126984126984 | 63.088888888889 | 63.088888888889 |
| ROW | 23.938765432099 | 15.389206349206 | 13.636005625879 | 12.103870162297 | 12.103870162297 |

Rounded to three decimals, these values exactly match the matrix in the
investigation request.

## Origin of the error

For CH, EU, US, and APAC, the explicit values in `params_region_new!T:W`
match slopes calibrated against the older Dmax trajectory rather than the
active Dmax trajectory.

The Africa/ROW problem is more severe. In `params_region_OLD`, Africa's
2030-2040 formulas reference ROW's Dmax row (`D7:F7`), while ROW's formulas
reference Africa's Dmax row (`D6:F6`). For example:

- `16.699017704841 = 1135.6 / (0.9 * 75.56)`, using old ROW 2030 Dmax for Africa.
- `71.816296296296 = 969.52 / (0.9 * 15)`, using old Africa 2030 Dmax for ROW.

Those erroneous evaluated numbers were then carried as explicit values into
`params_region_new`.

## Loader correction and validation

`model/data_prep.py` now:

1. calculates the documented expected intercept and slope from `p_full`,
   `eps_abs_base`, and each period's Dmax;
2. uses that formula when explicit coefficients are absent; and
3. rejects explicit `a_dem` or `b_dem` values whose absolute difference from
   the documented formula exceeds `1e-9`.

The previous fallback `b_dem = a_dem / Dmax` was inconsistent with the workbook
README by a factor of `1 + epsilon` and has been removed.

Integration checks:

- The original workbook fails with 29 reported slope mismatches across
  2025-2045.
- The corrected workbook loads successfully.
- The one-sweep smoke result reproduces the loaded `a_dem` and `b_dem`
  matrices with maximum absolute differences of `4.55e-13` and `4.26e-14`.
- Three unit tests for the documented formula and invalid calibration inputs pass.

## Corrected workbook and cold-start status

Corrected workbook:
`input_data_intertemporal_corrected_20260915_131409.xlsx`

Corrected workbook SHA-256:
`5E3E392B695AB917D8FF445203C9A04F1308083391BBF3C64D35E7ED3C9AD234`

Only `params_region_new!T2:W7` and the explanatory README text differ from the
original. The 24 active slope cells are formulas linked directly to `p_full`,
`eps_abs_base`, and the matching period's Dmax. All unrelated values, formulas,
styles, comments, hyperlinks, embedded images, sheet order, and dimensions were
preserved. The exporter intentionally rebuilt the calculation chain and
normalized document properties, comment-person metadata, media filenames, and
one legacy printer-settings package; none of these changes affects model data
or formulas.

The representative one-sweep smoke test completed successfully under
`outputs/demand_calibration/cold_start_smoke_20260915_132035/`.

Three 30-sweep corrected cold-start calculations were launched under
`outputs/demand_calibration/cold_start_full_20260915_132217/`. Each constructs
its initial state from input primitives only: existing capacity, zero capacity
change, exporter manufacturing-cost offers, and calibrated true-demand bids.
No historical result, checkpoint, certified profile, paper strategy, or
workbook initial-state sheet is read.

These new runs are algorithm-development candidates. They are not paper-ready
equilibria until the same frozen-profile unilateral-deviation audit used in the
equilibrium-repair workflow has been completed.
