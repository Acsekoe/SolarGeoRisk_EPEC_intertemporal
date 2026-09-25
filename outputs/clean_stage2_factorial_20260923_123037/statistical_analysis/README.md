# Clean-objective Stage 2 statistical analysis

Figures are stored in this folder; CSV tables are in [`csv/`](csv/).

## Scope and selection

This analysis reports **27 branch outcomes** from the 36-run clean-objective Stage 2 factorial. The objective retains the full market-price producer margin (no `-mu_offer` subtraction) and removes both economic-quadratic and algorithmic-proximal penalties. The run produced **28 accepted**, **7 no-pass**, and **1 failed** branches.

The accepted `ch-af-apac-eu-row-us/pf100_k100_a040` branch is excluded from reported economic results because its Europe 2035 clearing price reaches 641.45 USD/kW. Its source profile and acceptance audit remain in the run record; search-success diagnostics still count it as accepted. The exclusion is recorded in `results_selection.json` at the run root.

Accepted branch outcomes are deterministic, selection-conditioned computational results. They are not independent observations and may represent nearby points in the same equilibrium basin. Consequently, percentiles, correlations, clusters, and matched contrasts are descriptive rather than inferential or causal.

## Main clean-objective ranges

- Total installed capacity in 2040 ranges from **778.7 to 973.2 GW** (median 857.9 GW).
- The demand-weighted 2040 clearing price ranges from **124.0 to 216.9 USD/kW** (median 142.4 USD/kW).
- Across accepted branch outcomes, capacity and price have **Spearman rho = -0.49** and **Pearson r = -0.59**.
- The widest regional 2040 price IQR is in **EU** (77.0 USD/kW); the widest capacity IQR is in **China** (97.6 GW).

## Welfare components by region

Discounted billion USD over 2025–2040. The strategic values are component-wise medians across the 27 reported equilibria. Producer surplus is shown before capacity costs, which enter welfare separately; terminal salvage is excluded.

| Region | Planner CS | Median strategic CS | Planner PS | Median strategic PS |
|---|---:|---:|---:|---:|
| China | 3,990.4 | 3,991.5 | 6.6 | 82.0 |
| Europe | 2,118.5 | 1,891.7 | 0.0 | 41.5 |
| United States | 814.5 | 666.3 | 0.0 | 44.0 |
| Asia-Pacific | 1,082.5 | 1,019.1 | 0.4 | 195.8 |
| Africa | 493.8 | 448.2 | 0.0 | 18.8 |
| Rest of World | 1,656.2 | 1,522.7 | 0.0 | 5.6 |

The relative CS/PS boxplots use `100 × (strategic component − planner component) / planner regional welfare` for each branch and region, where planner regional welfare is CS + PS − capacity costs. Thus both components share one denominator; a positive value means the strategic outcome raises that component relative to global welfare maximization. Boxes span the 25th–75th percentiles, the dark mark is the median, and whiskers span all 27 reported outcomes.

## Comparison with the previous objective

- Median horizon-average total capacity increases from **861.8 to 905.6 GW**, while the median horizon-average demand-weighted price increases from **155.3 to 168.8 USD/kW**.
- The former EU/US exit is not robust to the clean objective. Median 2040 EU capacity changes from **0.1 to 54.8 GW** and US capacity from **0.4 to 72.4 GW**.
- Median 2040 cross-border trade changes from **283.4 to 152.2 GW**. The median flow-weighted bilateral offer on realized cross-border trade changes from **174.9 to 155.7 USD/kW**.
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
| PF 1.20 minus PF 1.00 | total capacity in 2040 | 8 | +3.31% | -8.61% to +17.45% |
| PF 1.20 minus PF 1.00 | demand-weighted price in 2040 | 8 | -9.88% | -17.11% to +47.35% |
| PF 0.80 minus PF 1.00 | total capacity in 2040 | 6 | -1.38% | -4.91% to +12.88% |
| PF 0.80 minus PF 1.00 | demand-weighted price in 2040 | 6 | -4.14% | -19.55% to +4.38% |
| Capacity weight 1.00 minus 0.50 | total capacity in 2040 | 9 | +0.68% | -5.64% to +16.42% |
| Capacity weight 1.00 minus 0.50 | demand-weighted price in 2040 | 9 | -2.90% | -34.62% to +29.36% |
| Damping 0.40 minus 0.30 | total capacity in 2040 | 11 | -7.09% | -14.78% to +1.05% |
| Damping 0.40 minus 0.30 | demand-weighted price in 2040 | 11 | +1.72% | -17.58% to +73.12% |
| AF-first minus CH-first | total capacity in 2040 | 6 | -3.63% | -11.11% to +1.43% |
| AF-first minus CH-first | demand-weighted price in 2040 | 6 | -3.70% | -19.70% to +10.55% |
| EU-first minus CH-first | total capacity in 2040 | 6 | -1.88% | -12.10% to +1.86% |
| EU-first minus CH-first | demand-weighted price in 2040 | 6 | -1.52% | -14.28% to +52.38% |

## Figure assessment

- **Main-text candidates:** `equilibrium_bands_capacity_by_region`, `equilibrium_bands_prices_by_region`, and `capacity_price_pathway_equilibria`. These communicate the pathways and dispersion without implying a sampling distribution.
- **Band-figure selection:** `equilibrium_bands_capacity_by_region` and `equilibrium_bands_prices_by_region` show all 27 reported branches in every region and year; individual outcome dots are hidden and median markers remain. The price panels use a common 0–600 USD/kW scale.
- **Welfare components:** `welfare_cs_ps_distributions` shows the observed ranges and percentiles of planner-relative discounted consumer surplus and producer surplus less capacity costs across the same 27 branches. The adjacent welfare-level table reports producer surplus before capacity costs, consistent with the paper's CS + PS − CC accounting. Values are in discounted billion USD and exclude terminal salvage.
- **Relative welfare components:** `welfare_cs_ps_relative_distributions` divides each branch's CS and PS change by the corresponding region's total planner welfare. This common denominator makes the opposing changes comparable even where planner PS is close to zero. The plotted CS and PS are the paper's separate terms; capacity costs remain a separate welfare term.
- **Relative welfare boxplots:** `welfare_cs_ps_relative_boxplots` uses the same 27 branch-level percentages. Each box spans the 25th–75th percentiles, the dark line is the median, and whiskers span the full observed range; no outlier points are suppressed. Consumer and producer effects share one symmetric percentage axis.
- **Absolute welfare boxplots:** `welfare_cs_ps_absolute_boxplots` uses the same strategic-minus-planner CS and PS differences and branch set without percentage normalization. The x-axis is discounted billion USD over 2025–2040, and the box/whisker definitions match the relative version.
- **Planner reference:** The green Global Welfare maximization line uses `lam` from `outputs/llp_planner/llp_planner_results.xlsx` (`regions` sheet). The planner rerun uses the same corrected input workbook as Stage 2.
- **Direct formulation comparison:** `comparison_previous_vs_clean` uses identical horizon metrics for the prior and clean accepted sets. It remains descriptive because acceptance changes with the objective.
- **Regional mechanism comparison:** `comparison_regional_capacity_paths` makes the disappearance of systematic EU/US exit visible; `comparison_regional_price_paths` shows the associated price dispersion.
- **Trade mechanism comparison:** `comparison_trade_flows_and_offers` compares total cross-border trade and the flow-weighted bilateral offer price on realized trade routes.
- **Algorithm evidence:** `algorithm_convergence_diagnostics` uses only clean-objective Stage 2 audits. The historical mixed-objective convergence figure was deliberately not copied unchanged.
- **Supplementary figures:** boxplots, pass-rate heatmap, matched contrasts, and PCA. PCA families are exploratory and should not be presented as structurally distinct equilibria.

## Outcome families

The first two standardized PCA components explain **51.9%** of pathway variation. Ward clustering selects **4 descriptive families** with silhouette score 0.288.

## Interpretation limits

1. The unit of analysis is an accepted algorithm branch, not a random draw and not necessarily a unique equilibrium.
2. Damping, price factor, capacity weight, and update order are search settings rather than economic primitives.
3. Failed and no-pass branches enter pass-rate diagnostics but are excluded from economic-outcome ranges.
4. All acceptance audits are one-start local best-response checks; no global or multistart equilibrium claim is made.
5. Differences from `new_equilibria` combine a changed objective with a changed accepted set, so they should not be interpreted as a clean causal decomposition.
