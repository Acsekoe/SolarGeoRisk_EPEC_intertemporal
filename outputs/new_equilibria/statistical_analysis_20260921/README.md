# Statistical ranges across curated equilibrium candidates

## Scope

The analysis covers all 16 profiles in the curated candidate index. Each profile passed the common one-start, frozen-profile, zero-proximal 1% audit. The profiles are deterministic, selection-conditioned computational outcomes rather than independent statistical observations. Correlations, clusters, and matched contrasts are descriptive associations and must not be interpreted as economic causal effects.

## Main ranges

- Total installed capacity in 2040 ranges from **782.2 to 1,246.2 GW**.
- The demand-weighted 2040 clearing price ranges from **106.8 to 160.3 USD/kW**.
- Across candidates, 2040 total capacity and the demand-weighted price have **Spearman ρ = -0.54** and **Pearson r = -0.63**. This is an equilibrium-set association, not an estimated demand or supply effect.
- The widest regional 2040 price IQR occurs in **ROW** (22.7 USD/kW). The widest 2040 capacity IQR occurs in **China** (40.3 GW).
- All period-based tables and figures use the common market years 2025, 2030, 2035, and 2040. The 2025 installed capacities are fixed by initialization and therefore have no cross-candidate dispersion.
- All boxplots use minimum–maximum whiskers. Charcoal dots show individual candidate values and use small horizontal jitter only to reveal overlaps; the boxes still show the interquartile range and median.

## Search success across the complete 36-branch design

These rates describe the algorithm's ability to locate a one-start candidate within 15 sweeps. They do not rank economic equilibria.

| Initialization factor | Accepted | Branches | Pass rate |
|---:|---:|---:|---:|
| 0.80 | 2 | 12 | 16.7% |
| 1.00 | 6 | 12 | 50.0% |
| 1.20 | 8 | 12 | 66.7% |

| Update-order anchor | Accepted | Branches | Pass rate |
|---|---:|---:|---:|
| CH-first | 8 | 12 | 66.7% |
| AF-first | 4 | 12 | 33.3% |
| EU-first | 4 | 12 | 33.3% |

## Matched candidate contrasts

Each comparison holds the other encoded search settings fixed and includes only pairs for which both profiles were accepted. This conditioning makes the contrasts useful diagnostics but prevents causal interpretation.

| Contrast | Outcome | Pairs | Median difference | Range |
|---|---|---:|---:|---:|
| PF 1.20 minus PF 1.00 | total capacity in 2040 | 5 | -1.29% | -8.21% to +9.19% |
| PF 1.20 minus PF 1.00 | demand-weighted price in 2040 | 5 | -0.33% | -17.26% to +2.16% |
| PF 0.80 minus PF 1.00 | total capacity in 2040 | 1 | +3.75% | +3.75% to +3.75% |
| PF 0.80 minus PF 1.00 | demand-weighted price in 2040 | 1 | +3.83% | +3.83% to +3.83% |
| Capacity weight 1.00 minus 0.50 | total capacity in 2040 | 4 | +0.39% | -2.47% to +40.50% |
| Capacity weight 1.00 minus 0.50 | demand-weighted price in 2040 | 4 | -6.49% | -16.91% to +0.18% |
| Damping 0.40 minus 0.30 | total capacity in 2040 | 5 | +1.82% | -5.62% to +10.37% |
| Damping 0.40 minus 0.30 | demand-weighted price in 2040 | 5 | -0.92% | -15.83% to +0.55% |
| AF-first minus CH-first | total capacity in 2040 | 4 | -1.19% | -8.99% to +3.07% |
| AF-first minus CH-first | demand-weighted price in 2040 | 4 | +9.51% | -0.07% to +21.59% |
| EU-first minus CH-first | total capacity in 2040 | 4 | +1.92% | -2.71% to +43.53% |
| EU-first minus CH-first | demand-weighted price in 2040 | 4 | -1.61% | -15.11% to +0.75% |

## Outcome families

A PCA of regional capacity and price paths over 2025–2040 explains **64.3%** of standardized variation in its first two components. Constant features, including initialized 2025 capacities, are omitted automatically. Ward clustering selected **2 descriptive families** by the highest silhouette score among two to five clusters (0.648). Cluster membership is exploratory and is provided to support later economic interpretation.

## Files

- `equilibrium_statistical_analysis.xlsx`: compact workbook with summary and analysis tables.
- `candidate_metrics.csv`, `price_summary.csv`, `capacity_summary.csv`: principal analysis tables.
- `associations.csv`, `matched_contrasts.csv`, `cluster_assignments.csv`: exploratory relationship and family diagnostics.
- `algorithm_convergence_diagnostics`: PNG/PDF figure separating penalized movement convergence, direct frozen-profile audits, and the 36-branch reinitialization outcomes.
- `penalized_movement_history.csv` and `direct_anchor_audits.csv`: convergence data underlying that diagnostic figure.
- PNG and vector PDF figures provide combined and region-specific price/capacity boxplots, the 2040 capacity-price relationship, matched contrasts, search pass rates, and PCA families.

## Interpretation limits

1. The accepted profiles are not a random sample and several share the same basin anchor.
2. Damping is an algorithmic setting, not an economic primitive.
3. The accepted design is unbalanced because failed branches are absent from economic-outcome comparisons.
4. Only the six older CH-first PF100/PF120 profiles have three-start audit results, and all six fail that stricter 1% test. The remaining ten candidates, including both PF080 profiles, have not yet received it.
5. Convergence paths stop at the first clean three-sweep movement pass: sweep 26 for CH-first and sweep 33 for AF-first and EU-first. A separate forced CH-first restart through sweep 36 is retained as a diagnostic artifact but excluded from the official stopping path.
6. Statistical associations therefore support statements about observed computational equilibrium ranges and search sensitivity, not causal economic claims or global-equilibrium probabilities.
