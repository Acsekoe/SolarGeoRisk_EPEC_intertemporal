# Terminal-state salvage equilibrium search

Created: 2026-09-17 15:39:12 Europe/Vienna

## Outcome

The corrected-input search completed successfully. Stage 2 evaluated all 24
predeclared branches with six parallel workers and a maximum of 15
Gauss--Seidel sweeps per branch. Fifteen branches satisfy the declared local
one-percent equilibrium criterion; nine completed without passing. There were
no failed tasks or executor exceptions.

Acceptance means that, at one common frozen profile, all six zero-proximal
one-start best-response solves succeeded and the maximum normalized unilateral
profit gain was at most 1%. These are local computational 1%-equilibria, not
global Nash certificates. No multistart solve was used.

The accepted maximum gains range from 0.209492% to 0.995711%. The nine
non-passing branches have best audited gains from 1.048995% to 1.916042%.

## Model and search configuration

- Market and operating-payoff periods: 2025, 2030, 2035, and 2040.
- Terminal capacity state: 2045.
- The 2040 net-capacity decision determines 2045 capacity.
- Salvage payoff: `beta_2045 * 0.5 * c_inv[r] * Kcap[r,2045]`, credited once.
- The obsolete capacity-retention reward/subsidy is zero.
- Corrected input SHA-256:
  `F426E480A20565286F6CD6678A5EEC0DE6E7376382826983E7FC3D39C2DEA5E0`.
- Stage 2: 24 branches, six workers, at most 15 sweeps, one solver start,
  zero proximal penalties, no move cap, no gain filter, and no player freezing.

Stage 1 supplied clean-prefix source iterations 26, 25, and 30 for the three
player orders. The second Stage-1 run produced solver warnings after iteration
25; those later states were excluded automatically. The first source converged
under the Stage-1 movement test, while the third remained clean but had not met
that movement test by iteration 30.

## Stage-2 branch results

`pf100` and `pf120` denote initial bilateral offers at 1.00 and 1.20 times
manufacturing cost. `k050` and `k100` denote Stage-1 net-capacity-path weights
0.50 and 1.00. `a030` and `a040` denote damping 0.30 and 0.40.

| Player order | Branch | Selected sweep | Maximum audited gain | Outcome |
|---|---|---:|---:|---|
| CH-AF-APAC-EU-ROW-US | pf100_k050_a030 | 14 | 0.518140% | accepted |
| CH-AF-APAC-EU-ROW-US | pf100_k050_a040 | 10 | 0.533668% | accepted |
| CH-AF-APAC-EU-ROW-US | pf100_k100_a030 | 13 | 1.481065% | no pass |
| CH-AF-APAC-EU-ROW-US | pf100_k100_a040 | 10 | 0.960737% | accepted |
| CH-AF-APAC-EU-ROW-US | pf120_k050_a030 | 13 | 0.790283% | accepted |
| CH-AF-APAC-EU-ROW-US | pf120_k050_a040 | 15 | 0.802529% | accepted |
| CH-AF-APAC-EU-ROW-US | pf120_k100_a030 | 12 | 1.916042% | no pass |
| CH-AF-APAC-EU-ROW-US | pf120_k100_a040 | 6 | 0.626525% | accepted |
| CH-AF-EU-US-ROW-APAC | pf100_k050_a030 | 13 | 0.756636% | accepted |
| CH-AF-EU-US-ROW-APAC | pf100_k050_a040 | 9 | 0.209492% | accepted |
| CH-AF-EU-US-ROW-APAC | pf100_k100_a030 | 13 | 1.152049% | no pass |
| CH-AF-EU-US-ROW-APAC | pf100_k100_a040 | 9 | 0.468494% | accepted |
| CH-AF-EU-US-ROW-APAC | pf120_k050_a030 | 10 | 1.344570% | no pass |
| CH-AF-EU-US-ROW-APAC | pf120_k050_a040 | 10 | 0.871959% | accepted |
| CH-AF-EU-US-ROW-APAC | pf120_k100_a030 | 1 | 1.390009% | no pass |
| CH-AF-EU-US-ROW-APAC | pf120_k100_a040 | 7 | 1.130480% | no pass |
| CH-ROW-APAC-US-EU-AF | pf100_k050_a030 | 10 | 0.969406% | accepted |
| CH-ROW-APAC-US-EU-AF | pf100_k050_a040 | 7 | 0.995711% | accepted |
| CH-ROW-APAC-US-EU-AF | pf100_k100_a030 | 12 | 0.627435% | accepted |
| CH-ROW-APAC-US-EU-AF | pf100_k100_a040 | 9 | 0.538139% | accepted |
| CH-ROW-APAC-US-EU-AF | pf120_k050_a030 | 10 | 1.714690% | no pass |
| CH-ROW-APAC-US-EU-AF | pf120_k050_a040 | 6 | 1.194706% | no pass |
| CH-ROW-APAC-US-EU-AF | pf120_k100_a030 | 9 | 1.048995% | no pass |
| CH-ROW-APAC-US-EU-AF | pf120_k100_a040 | 5 | 0.956513% | accepted |

The cost-level offer initialization (`pf100`) was most robust: both
`pf100_k050` branches and `pf100_k100_a040` passed for every player order.
The higher 1.20 offer initialization was more order-sensitive.

## Validation and distinctness

- All 24 selected audits have six successful best-response solves.
- Maximum source replay error is `5.27e-16`.
- Selected markets contain no 2045 prices, demand, or trade.
- Net-capacity decisions occur only in 2025--2040, while capacity states include
  2045.
- All recorded input, protocol, and code hashes reproduce exactly.
- No two accepted profiles are numerical duplicates. Across the 105 pairwise
  comparisons, the smallest pairwise maximum differences were 1.08343 in net
  capacity change, 8.70604 in active-period offers, 6.52480 in capacity, and
  5.18563 in clearing price (model units).

Saved strategy payloads retain six nonzero domestic-offer placeholders for
2045 as a legacy full-state serialization detail. All terminal bilateral
off-diagonal offers are zero, and the 2045 placeholders are absent from market
clearing, profit, strategy updates, convergence distances, and frozen-profile
audits. They therefore do not affect the results. The original artifacts are
left unchanged to preserve run provenance.

## Interpretation and next step

This run shows that the corrected data, 2040 operating horizon, and explicit
2045 salvage formulation admit numerous locally certified profiles within the
15-sweep diagnostic budget. It does not isolate a salvage-only causal effect,
because both the input data and the horizon/terminal treatment differ from the
earlier equilibrium set.

The next useful step is to select a small, economically diverse subset of the
15 accepted profiles and build a common comparison table for capacities,
offers, prices, trade, regional payoffs, and the salvage contribution. Extra
sweeps are not necessary to establish existence, although the 1.048995%
near-miss is the natural continuation candidate if another accepted branch is
needed.

## Subsequent three-start diagnostic

On 2026-09-17, all 15 one-start-accepted profiles received a separate
three-start frozen-profile audit. All 270 optimizer attempts succeeded, but no
profile remained below the 1% gain threshold. The strongest relative candidates
were `ch-row-apac-us-eu-af/pf100_k100_a030` at 1.602905% and
`ch-row-apac-us-eu-af/pf120_k100_a040` at 1.889689%.

The original one-start results above remain valid under their declared local
criterion, but none should be described as multistart-robust. Full results and
per-candidate audits are in
`outputs/demand_calibration/terminal_state_salvage_multistart_20260917_154543/`.
