# Clean-objective Stage 2 statistical analysis

## Scope and selection

This analysis uses all **28 accepted branch outcomes** from the 36-run clean-objective Stage 2 factorial. The objective retains the full market-price producer margin (no `-mu_offer` subtraction) and removes both economic-quadratic and algorithmic-proximal penalties. There were **28 accepted**, **7 no-pass**, and **1 failed** branches.

Accepted branch outcomes are deterministic, selection-conditioned computational results. They are not independent observations and may represent nearby points in the same equilibrium basin. Consequently, percentiles, correlations, clusters, and matched contrasts are descriptive rather than inferential or causal.

## Main clean-objective ranges

- Total installed capacity in 2040 ranges from **778.7 to 973.2 GW** (median 855.0 GW).
- The demand-weighted 2040 clearing price ranges from **124.0 to 216.9 USD/kW** (median 143.2 USD/kW).
- Across accepted branch outcomes, capacity and price have **Spearman rho = -0.51** and **Pearson r = -0.59**.
- The widest regional 2040 price IQR is in **Africa** (82.9 USD/kW); the widest capacity IQR is in **China** (94.4 GW).

## Comparison with the previous objective

- Median horizon-average total capacity increases from **861.8 to 905.5 GW**, while the median horizon-average demand-weighted price increases from **155.3 to 169.7 USD/kW**.
- The former EU/US exit is not robust to the clean objective. Median 2040 EU capacity changes from **0.1 to 54.8 GW** and US capacity from **0.4 to 75.1 GW**.
- Median 2040 cross-border trade changes from **283.4 to 149.4 GW**. The median flow-weighted bilateral offer on realized cross-border trade changes from **174.9 to 159.1 USD/kW**.
- These shifts are descriptive: the objective and the accepted branch set both changed.

## Search success

| Dimension | Level | Accepted | Branches | Pass rate |
|---|---|---:|---:|---:|
| price_factor | 0.8 | 7 | 12 | 58.3% |
| price_factor | 1.0 | 11 | 12 | 91.7% |
| price_factor | 1.2 | 10 | 12 | 83.3% |
| update_order | AF-first | 9 | 12 | 75.0% |
| update_order | CH-first | 10 | 12 | 83.3% |
| update_order | EU-first | 9 | 12 | 75.0% |

## Matched branch contrasts

Contrasts retain only accepted pairs that match on the other encoded search settings. They diagnose sensitivity to initialization and damping; they do not identify economic treatment effects.

| Contrast | Outcome | Pairs | Median difference | Range |
|---|---|---:|---:|---:|
| PF 1.20 minus PF 1.00 | total capacity in 2040 | 9 | +3.69% | -8.61% to +17.45% |
| PF 1.20 minus PF 1.00 | demand-weighted price in 2040 | 9 | -11.73% | -29.94% to +47.35% |
| PF 0.80 minus PF 1.00 | total capacity in 2040 | 7 | -2.67% | -4.91% to +12.88% |
| PF 0.80 minus PF 1.00 | demand-weighted price in 2040 | 7 | -3.37% | -19.55% to +8.80% |
| Capacity weight 1.00 minus 0.50 | total capacity in 2040 | 10 | +0.61% | -5.64% to +16.42% |
| Capacity weight 1.00 minus 0.50 | demand-weighted price in 2040 | 10 | -0.27% | -34.62% to +29.36% |
| Damping 0.40 minus 0.30 | total capacity in 2040 | 12 | -7.17% | -14.78% to +1.05% |
| Damping 0.40 minus 0.30 | demand-weighted price in 2040 | 12 | +4.33% | -17.58% to +73.12% |
| AF-first minus CH-first | total capacity in 2040 | 7 | -2.17% | -11.11% to +1.43% |
| AF-first minus CH-first | demand-weighted price in 2040 | 7 | -5.11% | -28.46% to +10.55% |
| EU-first minus CH-first | total capacity in 2040 | 7 | -1.61% | -12.10% to +2.99% |
| EU-first minus CH-first | demand-weighted price in 2040 | 7 | -5.64% | -18.56% to +52.38% |

## Figure assessment

- **Main-text candidates:** `equilibrium_bands_capacity_by_region`, `equilibrium_bands_prices_by_region`, and `capacity_price_pathway_equilibria`. These communicate the pathways and dispersion without implying a sampling distribution.
- **Direct formulation comparison:** `comparison_previous_vs_clean` uses identical horizon metrics for the prior and clean accepted sets. It remains descriptive because acceptance changes with the objective.
- **Regional mechanism comparison:** `comparison_regional_capacity_paths` makes the disappearance of systematic EU/US exit visible; `comparison_regional_price_paths` shows the associated price dispersion.
- **Trade mechanism comparison:** `comparison_trade_flows_and_offers` compares total cross-border trade and the flow-weighted bilateral offer price on realized trade routes.
- **Algorithm evidence:** `algorithm_convergence_diagnostics` uses only clean-objective Stage 2 audits. The historical mixed-objective convergence figure was deliberately not copied unchanged.
- **Supplementary figures:** boxplots, pass-rate heatmap, matched contrasts, and PCA. PCA families are exploratory and should not be presented as structurally distinct equilibria.

## Outcome families

The first two standardized PCA components explain **50.9%** of pathway variation. Ward clustering selects **4 descriptive families** with silhouette score 0.261.

## Interpretation limits

1. The unit of analysis is an accepted algorithm branch, not a random draw and not necessarily a unique equilibrium.
2. Damping, price factor, capacity weight, and update order are search settings rather than economic primitives.
3. Failed and no-pass branches enter pass-rate diagnostics but are excluded from economic-outcome ranges.
4. All acceptance audits are one-start local best-response checks; no global or multistart equilibrium claim is made.
5. Differences from `new_equilibria` combine a changed objective with a changed accepted set, so they should not be interpreted as a clean causal decomposition.
