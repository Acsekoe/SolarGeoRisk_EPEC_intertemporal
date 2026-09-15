# SolarGeoRisk EPEC workflow handoff

Last updated: 2026-09-14 19:12 CEST (UTC+02:00)

## Current objective

Continue the selected CH -> ROW -> APAC -> US -> EU -> AF Gauss-Seidel case from its accepted iteration-21 state while gradually removing the algorithmic proximal penalties. The final target is a zero-proximal-penalty profile with small strategy changes and no material profitable unilateral deviation.

## Selected source

- Workbook: `outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx`
- Source run ID: `20260409_000435_df8017`
- Accepted restart point: iteration 21, the first endpoint with three consecutive relative strategy residuals below 1%.
- Player order: `ch`, `row`, `apac`, `us`, `eu`, `af`.
- Model data sheet: `params_region_new` in `inputs/input_data_intertemporal.xlsx`.
- Solver: IPOPT using the installed academic GAMSPy license. No license access code is stored in this repository.

## Continuation design

The restart script is `scripts/continue_selected_equilibrium.py`. It reconstructs internal accepted Gauss-Seidel values, including the final player's damped state, internal `Q_offer`, and domestic-price references. It verifies the reconstruction against every recorded `r_strat` before solving.

Penalty stages use the original terminal coefficients `(q=2, p=3, a=2, dk=2)` multiplied by:

1. 50%, with fixed damping `omega=0.20`.
2. 25%, with `omega=0.15`.
3. 10%, with `omega=0.12`.
4. 2%, with `omega=0.10`.
5. 0%, with `omega=0.08`.

Each stage repeats checkpoint blocks until `r_strat <= 1e-3` for three consecutive sweeps. A stage does not advance merely because its maximum block length was reached. Economic quadratic terms remain at `c_quad_q=c_quad_p=c_quad_a=0.1`; only the algorithmic proximal terms are reduced.

## Persistent state and outputs

- Machine-readable progress: `workflow/continuation_manifest.json` (created after the first successful block).
- Checkpoint workbooks: `outputs/continuation/ch-row-apac-us-eu-af/stage_*/results_*.xlsx`.
- Every checkpoint is replayed from its actual starting state and checked against its recorded residual history before it is accepted into the manifest.

## Latest checkpoint

- Finished: 2026-09-14 15:43:38 CEST (UTC+02:00).
- Stage: 1, `half_penalty` (50% of the original proximal penalties), block 2.
- Workbook: `outputs/continuation/ch-row-apac-us-eu-af/stage_01_half_penalty/results_20260914_153101_5cbce7.xlsx`.
- Completed: 10 sweeps using the required `ch -> row -> apac -> us -> eu -> af` order and fixed `omega=0.20`.
- Ending strategy residual: `0.0863639155250246`; ending stable count: `0`.
- Block-2 residual path: `0.034057, 0.033771, 0.032237, 0.046043, 0.027622, 0.022317, 0.022463, 0.053477, 0.074209, 0.086364`.
- Interpretation: APAC continued moving along its new branch. The US changed capacity branch in global sweep 14, and AF changed branch in global sweeps 17--20. AF's 2035 capacity path dominates the final residual. This is not a converged half-penalty state, so the manifest correctly keeps stage 1 active.
- Restart verification passed after writing the checkpoint. Maximum residual-replay errors are `6.938893903907228e-18` for block 2, `1.2923689896027213e-16` for block 1, and `5.169475958410885e-16` for the source workbook.

## Welfare findings from the first 10 continuation sweeps

The workbook's recorded `obj` is not pure welfare. It contains the consumer-surplus, producer-surplus, and capacity-cost terms, minus both the economic quadratic penalties and the algorithmic proximal penalties. The analysis therefore reports:

- `no-proximal objective = recorded obj + algorithmic proximal cost`;
- `core welfare = recorded obj + algorithmic proximal cost + economic quadratic cost`.

Because `Q_offer=Kcap` and `a_bid` is fixed to true demand in this run, the economic `q` and `a` quadratic costs are zero. The remaining economic quadratic adjustment is the `c_quad_p=0.1` price-markup penalty. Capacity-policy incentives are zero. Values below compare each player's core welfare at its turn in the Gauss-Seidel sweep with the same player's reconstructed value at source iteration 21.

| Sweep | CH | ROW | APAC | US | EU | AF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -2.308% | +0.029% | +0.255% | -0.641% | -0.053% | -0.016% |
| 2 | -1.412% | +0.037% | +0.159% | -1.149% | -0.103% | -0.028% |
| 3 | -1.376% | +0.045% | +0.059% | -1.569% | -0.156% | -0.042% |
| 4 | -2.121% | +0.053% | -0.109% | -1.982% | -0.243% | -0.067% |
| 5 | -2.042% | +0.061% | -0.307% | -2.363% | -0.346% | -0.075% |
| 6 | -1.971% | +0.069% | -0.515% | -2.706% | -0.453% | -0.083% |
| 7 | -1.901% | +0.077% | +0.676% | -3.019% | -0.565% | -0.091% |
| 8 | -1.858% | +0.086% | +1.412% | -3.305% | -0.679% | -0.098% |
| 9 | -1.832% | +0.094% | +1.234% | -3.570% | -0.792% | -0.106% |
| 10 | -1.804% | +0.102% | +1.059% | -3.818% | -0.905% | -0.114% |

Sweep-10 changes and proximal costs are discounted multi-period model values, expressed below in approximately billion USD:

| Player | Core-welfare change (bn USD) | Core-welfare change (%) | No-proximal objective change (%) | Sweep-10 proximal cost (bn USD) |
| --- | ---: | ---: | ---: | ---: |
| CH | -104.3 bn | -1.804% | -2.096% | 4.59 bn |
| ROW | +1.5 bn | +0.102% | +10.061% | 19.70 bn |
| APAC | +19.4 bn | +1.059% | +1.308% | 4.74 bn |
| US | -36.0 bn | -3.818% | -3.171% | 1.24 bn |
| EU | -22.5 bn | -0.905% | -0.691% | 0.98 bn |
| AF | -0.6 bn | -0.114% | +0.838% | 0.90 bn |

Main interpretation:

- US has the largest relative core-welfare loss and deteriorates in every sweep. EU also deteriorates monotonically.
- APAC's regime shift is economically visible: its deviation changes from `-0.515%` in sweep 6 to `+0.676%` in sweep 7, peaks at `+1.412%` in sweep 8, and then recedes.
- CH remains roughly 1.8–2.3% below the source welfare level. ROW's core welfare is nearly unchanged, and AF has a small welfare loss.
- ROW's apparent `+10.061%` improvement in the no-proximal solver objective is mostly a reduction in the economic price-markup penalty; its core-welfare improvement is only `+0.102%`. AF likewise has a higher no-proximal objective but slightly lower core welfare.
- Total reconstructed proximal cost falls from approximately `40.26` bn in sweep 1 to `32.15` bn in sweep 10, but APAC's component rises from `0.67` bn to `4.74` bn during its regime change. ROW still has the largest sweep-10 proximal cost at `19.70` bn.
- Summing the player-specific turn values gives a diagnostic change of approximately `-142.6` bn (`-1.09%`) by sweep 10. This is not a publishable global-welfare result because the six objectives are evaluated sequentially at different within-sweep states. A valid regional and global comparison requires evaluating all players at one common accepted strategy profile.

### Drift-versus-cycle diagnosis

The first 20 sweeps show directed drift through several response-regime changes, not a two-cycle:

- The normalized Euclidean distance from the source state increases monotonically from `0.0171` after sweep 1 to `0.1433` after sweep 10 and `0.5119` after sweep 20. The state does not return toward an earlier sweep.
- Consecutive update directions are strongly aligned through sweep 6, with cosine similarities of approximately `0.9945` to `0.9996`. The APAC regime change makes the sweep-7 direction nearly orthogonal to sweep 6 (`0.0634`), but it is not a reversal. Alignment then rises to `0.9100`, `0.9888`, and `0.9973` in sweeps 8–10.
- Update directions remain positively aligned in block 2. Cosine similarity drops at the US and AF branch changes but never reverses; it is `0.9840` between sweeps 19 and 20. This remains inconsistent with an observed period-two cycle.
- ROW capacity offers continue to decline: its 2040 offer falls from `190.934` at the source state to `178.434` by sweep 10 and `165.935` by sweep 20. China's 2040 offer price to the US rises from `176.271` at the source to `224.333` by sweep 10 and `250.708` by sweep 20.
- APAC's 2030 capacity reaches `164.070` by sweep 20, while its 2030 net capacity change falls to `2.194`. The US temporarily re-enters positive post-2030 capacity during sweep 14. AF then moves from zero 2035 capacity toward `3.815`, producing the largest final updates.

This evidence is consistent with directed movement through active-set or best-response branch changes, but it does not yet show approach to a stationary point. The residual accelerated to a new high in sweeps 18--20. Do not lower the penalty and do not launch another block at `omega=0.20` without revising the damping choice. The next experiment should remain at the 50% penalty stage with lower fixed damping, preferably `omega=0.10`; interpret any smaller damped residual together with the underlying raw best-response gap rather than as convergence by itself.

## Commands

Verify the source/checkpoint reconstruction without solving:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts\continue_selected_equilibrium.py --check-only
```

Run one continuation block:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts\continue_selected_equilibrium.py --blocks 1
```

Run several blocks sequentially:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts\continue_selected_equilibrium.py --blocks 3
```

## Equilibrium verification requirement

Small step changes alone are insufficient. After the zero-penalty stage converges, hold the other five strategies fixed and solve each player's objective with all `c_pen_*` values equal to zero. Record the reference objective, best-response objective, relative gain, feasibility residual, and strategy distance. A feasible profitable deviation rejects equilibrium even when a solver stops before proving local optimality.

## Status

The continuation runner, manifest, two stage-1 checkpoints, and replay verification are complete. The saved state is the end of global sweep 20. Stage 1 remains unconverged, with residual `0.086364` and stable count `0`; later penalty stages have not started. Before running again, change stage-1 damping from `omega=0.20` to `omega=0.10` and retain the 50% proximal penalties. The zero-penalty unilateral-deviation verification remains outstanding.

## Objective-based equilibrium search (2026-09-14 evening update)

The equilibrium criterion has been corrected: strategy changes, including undamped best-response distance, are diagnostics only. Convergence and the equilibrium claim are based on profitable unilateral objective deviations at a common frozen profile:

`relative gain_i = max(BR objective_i - reference objective_i, 0) / max(abs(reference objective_i), 1)`.

The candidate is accepted as a computational 1%-equilibrium only if every player's zero-algorithmic-proximal best-response solve is successful and `max_i(relative gain_i) <= 0.01`. Damping changes the amount applied after a best response; it does not change this test.

The primary implementation is now `scripts/search_nested_equilibrium.py`, using the model's submitted economic objective with algorithmic proximal penalties set to zero. The lower-level market is solved at every objective evaluation, and each best response uses three starts. Within a sweep, players whose computed gain is already at or below 1% are not updated. Strategy-distance values are retained solely to diagnose update size.

The latest completed two-sweep checkpoint is `outputs/equilibrium_search/ch-row-apac-us-eu-af/nested_zero_prox/checkpoint_20260914_184148.json`. Its final within-sweep maximum relative objective gain was `14.4839%`, so it is not an equilibrium. At the start of that second sweep, notable gains were CH `8.578%`, ROW `14.484%`, APAC `1.092%`, and US `4.809%`; EU and AF were already below 1%.

One earlier checkpoint, `checkpoint_20260914_182903.json`, is marked invalid because an implementation bug forced all effective update weights to zero. It is ignored when reconstructing the active state. Failed optimizer attempts are no longer allowed to drive a profile update.

A checkpointed long run was launched with `scripts/run_objective_equilibrium_search.py`: up to ten blocks of five sweeps (50 sweeps total), fixed `omega=0.10`, maximum normalized applied move `0.01`, three starts, 500 optimizer iterations, and a 1% objective-gain threshold. Three consecutive within-sweep objective passes are required for early stopping. Each five-sweep block is followed by a multistart audit at one common frozen profile; only that audit can set `equilibrium_verified=true` in `workflow/nested_equilibrium_search_manifest.json`.

Long-run logs:

- `outputs/equilibrium_search/ch-row-apac-us-eu-af/nested_zero_prox/long_objective_search.stdout.log`
- `outputs/equilibrium_search/ch-row-apac-us-eu-af/nested_zero_prox/long_objective_search.stderr.log`

## Certified 1%-epsilon equilibrium (2026-09-14 final update)

The paper's iteration-21 profile is not the final claimable equilibrium. The certified profile is pinned independently of the mutable search manifest:

- Candidate: `outputs/equilibrium_search/ch-row-apac-us-eu-af/nested_zero_prox/checkpoint_20260914_184749.json`.
- Frozen-profile three-start audit: `outputs/equilibrium_search/ch-row-apac-us-eu-af/nested_zero_prox/audit_20260914_184948.json`.
- Stable machine-readable pointer: `workflow/certified_equilibrium_manifest.json`.
- Source/candidate objective, capacity, and offer comparison: `outputs/verification/ch-row-apac-us-eu-af/certified_candidate_comparison.json`.

All algorithmic proximal penalties are zero. The economic quadratic terms remain `c_quad_q=c_quad_p=c_quad_a=0.1`; with `Q_offer=Kcap` and true demand fixed, only the price-markup term is active. All 18 upper best-response attempts terminated successfully. The exact convex lower market has maximum balance residual `1.47e-13`, zero capacity violation, and maximum stationarity residual `4.43e-8`.

Frozen common-profile unilateral economic gains are:

| Player | Reference objective | Best-response objective | Relative gain |
| --- | ---: | ---: | ---: |
| CH | 5,465,883.9 | 5,493,787.4 | 0.511% |
| ROW | 1,530,904.9 | 1,538,400.3 | 0.490% |
| APAC | 1,651,013.3 | 1,661,934.2 | 0.661% |
| US | 989,142.0 | 993,612.3 | 0.452% |
| EU | 2,631,406.7 | 2,631,495.2 | 0.003% |
| AF | 566,389.3 | 567,440.5 | 0.186% |

The maximum is APAC at `0.6615%`, below the prespecified 1% tolerance. This is the defensible equilibrium claim: a computational 1%-epsilon equilibrium under zero algorithmic proximal penalties.

Relative to source iteration 21, common-profile economic objectives change by CH `-0.991%`, ROW `+42.691%`, APAC `-5.083%`, US `+7.831%`, EU `+6.679%`, and AF `+8.251%`. ROW's large increase is chiefly the removal of inherited offer-price markups rather than a comparable core-welfare increase.

Candidate capacity paths for 2025/2030/2035/2040/2045 (GW) are:

| Player | 2025 | 2030 | 2035 | 2040 | 2045 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CH | 931.00 | 721.84 | 719.97 | 719.90 | 624.18 |
| ROW | 293.00 | 98.82 | 72.40 | 66.33 | 61.96 |
| APAC | 110.00 | 134.69 | 166.28 | 191.30 | 198.61 |
| US | 42.00 | 32.06 | 30.23 | 32.64 | 24.11 |
| EU | 22.00 | 1.15 | 0.78 | 0.73 | 0.42 |
| AF | 3.40 | 5.34 | 4.84 | 5.11 | 5.10 |

The corresponding regional market-clearing prices, reconstructed by solving one common lower-level market with the candidate strategies frozen, are:

| Region | 2025 | 2030 | 2035 | 2040 | 2045 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CH | 163.28 | 107.79 | 95.91 | 86.65 | 80.38 |
| ROW | 284.61 | 194.02 | 174.91 | 159.71 | 150.48 |
| APAC | 265.67 | 175.01 | 155.81 | 140.73 | 130.52 |
| US | 279.43 | 188.85 | 169.69 | 155.49 | 145.42 |
| EU | 282.03 | 191.45 | 172.36 | 158.22 | 156.49 |
| AF | 222.34 | 188.27 | 168.65 | 152.12 | 144.77 |

Capacities are in GW and market-clearing prices `lambda` are in USD/kW. The 2045 values belong to the terminal buffer period rather than the main 2025--2040 reporting horizon. The reconstructed market has maximum balance residual `1.47e-13`, zero capacity violation, and maximum stationarity residual `4.43e-8`.

Economically, this candidate implies strong restructuring: EU capacity falls from `22.00` GW in 2025 to `0.73` GW in 2040; ROW falls from `293.00` to `66.33` GW; China falls from `931.00` to `719.90` GW; and APAC expands from `110.00` to `191.30` GW. Regional clearing prices decline over time everywhere, while EU, ROW, and the US remain among the highest-price regions.

### Candidate provenance and methodological caveat

This candidate did not emerge from the detached 50-sweep continuation. It was obtained from a separate equilibrium basin. Starting from the then-current capacity profile, all bilateral offer prices were reset to each exporter's time-specific manufacturing cost. This was an initialization only, not a constraint on the final candidate. An objective-based, zero-algorithmic-proximal best-response search then updated only players whose unilateral relative objective gain exceeded 1%. Four tightly capped sweeps were followed by one larger capped sweep: China accepted its full profitable response, APAC received a partial response update, and players already below 1% were frozen. The resulting saved profile is `checkpoint_20260914_184749.json`.

The frozen-profile audit subsequently optimized each player's unilateral deviation with the other five strategies fixed and used three starts per player. Its maximum relative objective gain was `0.6615%` for APAC. The cost-price restart is a feasible way to explore a different equilibrium basin, but it is harder to defend as a direct continuation of the submitted solution. It should therefore be disclosed explicitly and treated as supplementary equilibrium-search evidence unless the initialization strategy is formalized and applied systematically across multiple starts.

Mean non-domestic offer prices fall from source to candidate as follows: CH `215.39 -> 171.39`, ROW `513.32 -> 230.85`, APAC `235.95 -> 174.48`, US `276.46 -> 209.86`, EU `252.07 -> 195.49`, and AF `171.55 -> 118.20`. Mean markups are approximately zero for ROW, US, EU, and AF; `0.87` for APAC; and `64.91` for CH.

### Undamped verification

`scripts/verify_nested_undamped_sweep.py` ran one full `omega=1`, zero-proximal, three-start Gauss-Seidel sweep from the certified profile. Every immediate sequential objective gain remained below 1%: CH `0.511%`, ROW `0.490%`, APAC `0.692%`, US `0.452%`, EU `0.002%`, and AF `0.083%`. The output is `outputs/verification/ch-row-apac-us-eu-af/nested_undamped/verification_20260914_190951.json`.

Do not use the post-sweep strategy profile as the equilibrium. Because the undamped algorithm accepts large strategy moves worth less than 1%, the post-sweep profile crosses an active-set boundary and China's re-audited gain becomes `18.225%`; the other five post-sweep gains are effectively zero. This is why the convergence rule should freeze any player already within the 1% payoff tolerance. The equilibrium object is the starting frozen common profile, not the endpoint of a literal undamped update sweep.

The original GAMS MPEC sweep is not a valid rejection of this candidate. A cold-lower-level CONOPT sweep reported raw residual `1.570`; a fully warm-started IPOPT sweep reported `0.570`; and warm CONOPT stalled on AF. In both completed runs, first-moving China's returned objective was below the feasible incumbent (`-1.42%` for CONOPT and `-2.52%` for IPOPT). A maximization best response cannot be worse than the do-nothing incumbent, so these are locally inferior complementarity branches. Their workbooks are retained under `outputs/verification/ch-row-apac-us-eu-af/original_mpec_undamped_zero_prox/` for diagnosis, not equilibrium evidence.

The detached long objective-search branch was stopped after it overwrote the shared manifest with a worse pre-cost branch (last within-sweep maximum gain `8.72%`). Its checkpoints and logs remain available, but it is not the certified result.

## Local search from the reported paper profile (2026-09-14 late update)

A separate, reproducible experiment now starts from the fully replayed accepted iteration-21 paper profile and from six prescribed nearby initializations. Its implementation is `scripts/run_local_paper_equilibrium_experiment.py`; all results are isolated under `outputs/equilibrium_search/ch-row-apac-us-eu-af/local_paper_profile_zero_prox/`, with manifest `workflow/local_paper_profile_zero_prox_manifest.json` and final report `report.md` in the output directory. Existing continuation and certified-candidate results were not modified.

The exact paper profile's three-start frozen audit has a maximum zero-proximal relative unilateral gain of `47.5777%` for ROW. With fixed `alpha=0.65`, none of Branches A--F reached a verified 1%-equilibrium. The exact, +/-5%, +/-10%, and price-only starts persistently oscillate or drift; the 25% cost-price interpolation produces a detected two-cycle led by APAC. The most promising new profile is the 50% paper-price/50% manufacturing-cost interpolation with `alpha=0.50` after six sweeps, but its frozen audit remains above tolerance: CH `3.8441%`, APAC `1.5424%`, and all other players below `0.025%`.

Damping robustness does not alter the conclusion. At `alpha=0.65`, the best Branch-F frozen audit is `3.9592%` for CH; at `alpha=0.50` it is `3.8441%`; and the optional `alpha=0.30` run gives `4.3407%` before crossing a large ROW/APAC response branch. Since no new local candidate passed the 1% frozen audit, alternative player-order testing was not triggered. The separately preserved manufacturing-cost-initialized candidate remains the only solver-verified computational relative 1%-equilibrium found, with maximum gain `0.6615%` for APAC, and must retain its disclosed basin-initialization caveat.

## Overnight O1--O6 search launched (2026-09-14 22:45 CEST)

The new checkpointed runner is `scripts/run_overnight_equilibrium_experiment.py`. It implements the requested O1--O6 design in an isolated timestamped output tree and does not modify the certified-candidate or local-paper-profile outputs. `scripts/nested_market_audit.py` was extended backward-compatibly so O4 can optimize a temporarily proximal objective while returning and recording the original unpenalized economic objective. Frozen audits always use zero algorithmic proximal penalties.

The runner provides:

- exact iteration-21 paper replay for O1--O4;
- exact `branch_F_alpha_0p50/sweep_006.json` restart for O5--O6;
- fixed-alpha, adaptive-alpha, and seven-stage O4 homotopy schedules;
- sequential `ch -> row -> apac -> us -> eu -> af` updates inside every GS sweep;
- complete immutable JSON checkpoints after every sweep;
- one-start frozen audits for every fifth sweep, followed by process-parallel three-start audits of the best five candidates;
- 10--20-start stress tests for the two critical players of every three-start 1% pass;
- per-task PID/status/runtime logging, failure isolation, trajectory plots, comparison tables, and a final report;
- resumability from existing sweep and audit task files.

A smoke run at `outputs/equilibrium_search/ch-row-apac-us-eu-af/overnight/smoke_20260914_overnight/` verified checkpoint resume and the parallel 18-solve final-audit path. All 18 higher-limit solves succeeded. A direct positive-proximal response check also confirmed that the O4 optimizer records a nonzero proximal cost separately from the unpenalized economic payoff.

The full run is active:

- Run ID: `overnight_20260914_224500`.
- Output root: `outputs/equilibrium_search/ch-row-apac-us-eu-af/overnight/overnight_20260914_224500/`.
- Machine-readable manifest: `workflow/overnight_20260914_224500_manifest.json` (mirrored as `manifest.json` in the output root).
- Orchestrator log: `outputs/equilibrium_search/ch-row-apac-us-eu-af/overnight/overnight_20260914_224500/orchestrator.log`.
- Captured stdout/stderr: `launcher.stdout.log` and `launcher.stderr.log` in the same directory.
- Launcher PID: `30672`; Python orchestrator PID recorded in the manifest: `30384`.
- Worker count: 3 process workers on 8 physical/16 logical CPUs, selected conservatively because only about 6.1 GB physical RAM was available at launch. BLAS/OpenMP thread counts are capped at one per worker.
- First wave: O1 PID `7396`, O2 PID `24456`, O3 PID `29924`; O4--O6 are queued and will start as workers become free.
- Startup verification: no stderr output; O3 wrote its first full sweep checkpoint with all six player solves successful while O1 and O2 were progressing normally.

## Overnight recovery after Windows Update (2026-09-15 08:48 CEST)

Windows initiated two reboot transitions at approximately 04:38--04:40 CEST, including a `TrustedInstaller.exe` restart explicitly recorded as a planned operating-system update. This terminated the experiment processes while O4 was partway through global sweep 111. The atomic checkpoint design worked: O4 sweep 110 is complete and valid, the incomplete sweep 111 was not written, and the other five branches had already completed their full budgets. No frozen-profile audits had started before the reboot.

Saved branch coverage is O1 `150/150`, O2 `200/200`, O3 `150/150`, O4 `110/122`, O5 `150/150`, and O6 `150/150`. Across the one-start within-sweep diagnostics, the most promising saved state is O6 sweep 94 with maximum sequential gain `1.07085%` (CH); this is not yet a frozen-profile equilibrium result. O4's best such diagnostic is `2.21656%` at sweep 86. None of the six trajectories recorded a within-sweep maximum below 1%.

The restart path in `scripts/run_overnight_equilibrium_experiment.py` now preserves completed-branch summaries, resumes incomplete branches from their last checkpoint, and adds every branch's best diagnostic profile and terminal profile to the fifth-sweep audit set. Audit targets are sorted by diagnostic promise, so O6 sweep 94 is first. The same run ID `overnight_20260914_224500` was resumed at 08:48 CEST with three process workers. Completed O1/O2/O3/O5/O6 work was skipped, and O4 successfully reconstructed sweep 111 from checkpoint 110 and wrote a new complete sweep-111 checkpoint. After O4 reaches sweep 122, the runner will execute 188 lightweight frozen-profile audits, select the best five for three-start audits, and stress-test any 1% passes. Windows reported no pending update reboot at restart.
