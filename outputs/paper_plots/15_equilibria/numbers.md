# Results from 15 provisionally retained profiles

The 15 IDs in retained_profiles.txt belong to the passing deviation-screen verdicts; the full initial list of 15 was also checked for exact agreement. Multi-start verification is pending. Profile statistics use filtered observation-level CSV records, with selected-sweep JSON used for the offer/cost checks at the end. Planner prices use the regions/lam column of outputs/llp_planner/llp_planner_results.xlsx.

## Multiple equilibria

OLS slope of horizon-average demand-weighted price on horizon-average capacity: **-9.4 USD/kW per 100 GW**, R² **0.28**, n = **15**. Horizon weights: 2025 and 2040 = 1/6 each; 2030 and 2035 = 1/3 each.

| Highlight | Profile | Capacity GW | Price USD/kW | Price factor | Capacity weight | Damping | Welfare rank |
|---|---|---|---|---|---|---|---|
| Eq 1: highest price | ch-af-apac-eu-row-us/pf080_k100_a030 | 941 | 180 | 0.80 | 1.00 | 0.30 | 15/15 |
| Eq 2: highest capacity | ch-af-apac-eu-row-us/pf120_k100_a030 | 998 | 155 | 1.20 | 1.00 | 0.30 | 1/15 |

## Market-clearing prices

Price-band shared y-axis ceiling: **450 USD/kW**, the next multiple of 50 above the retained maximum.

Global demand-weighted price by year (USD/kW; markup in percent):

| Year | Min | Median | Max | Planner | Median gap | Markup |
|---|---|---|---|---|---|---|
| 2025 | 210 | 221 | 242 | 169 | 53 | 31% |
| 2030 | 143 | 159 | 194 | 114 | 45 | 40% |
| 2035 | 134 | 154 | 178 | 103 | 50 | 49% |
| 2040 | 124 | 140 | 148 | 96 | 44 | 45% |

Regional prices (USD/kW; ratio is median/planner):

| Region | Year | Min | Median | P90 | Max | Planner | Median/planner | Min - planner |
|---|---|---|---|---|---|---|---|---|
| CH | 2025 | 163 | 163 | 163 | 163 | 163 | 1.00 | 0 |
| CH | 2030 | 108 | 108 | 108 | 108 | 107 | 1.00 | 0 |
| CH | 2035 | 96 | 96 | 96 | 96 | 96 | 1.00 | 0 |
| CH | 2040 | 87 | 87 | 87 | 87 | 88 | 0.98 | -2 |
| EU | 2025 | 256 | 311 | 365 | 373 | 179 | 1.74 | 77 |
| EU | 2030 | 191 | 241 | 307 | 398 | 124 | 1.95 | 67 |
| EU | 2035 | 175 | 226 | 261 | 306 | 112 | 2.02 | 63 |
| EU | 2040 | 166 | 201 | 254 | 275 | 104 | 1.92 | 62 |
| US | 2025 | 285 | 307 | 350 | 366 | 174 | 1.76 | 110 |
| US | 2030 | 192 | 246 | 292 | 349 | 119 | 2.08 | 73 |
| US | 2035 | 178 | 218 | 259 | 328 | 107 | 2.03 | 71 |
| US | 2040 | 169 | 185 | 222 | 227 | 100 | 1.86 | 69 |
| APAC | 2025 | 198 | 256 | 303 | 330 | 177 | 1.45 | 22 |
| APAC | 2030 | 124 | 161 | 224 | 234 | 121 | 1.33 | 2 |
| APAC | 2035 | 110 | 158 | 172 | 191 | 110 | 1.44 | 0 |
| APAC | 2040 | 99 | 133 | 154 | 159 | 102 | 1.30 | -3 |
| AF | 2025 | 295 | 338 | 389 | 446 | 173 | 1.95 | 121 |
| AF | 2030 | 199 | 274 | 320 | 369 | 118 | 2.32 | 81 |
| AF | 2035 | 186 | 250 | 279 | 304 | 107 | 2.34 | 79 |
| AF | 2040 | 167 | 208 | 240 | 309 | 99 | 2.10 | 68 |
| ROW | 2025 | 302 | 326 | 353 | 353 | 178 | 1.83 | 124 |
| ROW | 2030 | 212 | 227 | 275 | 295 | 122 | 1.86 | 90 |
| ROW | 2035 | 188 | 207 | 207 | 298 | 111 | 1.87 | 77 |
| ROW | 2040 | 171 | 187 | 203 | 209 | 103 | 1.81 | 68 |

China and APAC median prices versus planner (USD/kW; gaps use unrounded values):

| Year | China median | China planner | China gap | APAC median | APAC planner | APAC gap |
|---|---|---|---|---|---|---|
| 2025 | 163.3 | 163.0 | 0.3 | 256.4 | 176.8 | 79.6 |
| 2030 | 107.8 | 107.4 | 0.4 | 161.2 | 121.2 | 40.0 |
| 2035 | 95.9 | 96.0 | -0.1 | 158.4 | 109.8 | 48.6 |
| 2040 | 86.7 | 88.4 | -1.7 | 132.6 | 102.2 | 30.5 |

Median regional exports (GW):

| Year | China | APAC |
|---|---|---|
| 2025 | 89.6 | 60.0 |
| 2030 | 71.8 | 83.2 |
| 2035 | 22.5 | 119.0 |
| 2040 | 16.7 | 133.1 |

Median 2040 cross-border trade: **169.3 GW**.

## Welfare distribution

Planner total: **10,114.7 billion USD** (CS + PS - capacity costs).

| Region | Planner | Minimum | Median | Maximum |
|---|---|---|---|---|
| CH | 3,963.8 | 3,973.0 | 4,075.2 | 4,145.2 |
| EU | 2,117.5 | 1,901.2 | 1,952.3 | 1,986.5 |
| US | 812.6 | 694.5 | 702.7 | 713.5 |
| APAC | 1,080.1 | 1,152.7 | 1,208.8 | 1,263.0 |
| AF | 493.7 | 458.1 | 467.2 | 473.7 |
| ROW | 1,647.1 | 1,504.3 | 1,520.2 | 1,548.6 |
| Total | 10,114.7 | 9,789.1 | 9,929.3 | 10,029.9 |

Global welfare loss: min **0.8%**, median **1.8%**, max **3.2%**.

Number of profiles with regional welfare above planner:

| Region | Gaining profiles |
|---|---|
| CH | 15/15 |
| EU | 0/15 |
| US | 0/15 |
| APAC | 15/15 |
| AF | 0/15 |
| ROW | 0/15 |

Median joint EU+US+AF+ROW loss: **429.7 billion USD**. Median joint CH+APAC gain: **254.0 billion USD**. These are medians of sums within each profile.
Spearman rho(China welfare, other-four welfare) = **0.79**. rho(global welfare, China 2040 capacity share) = **0.87**.

Relative CS and PS change as a percentage of each region's planner welfare:

| Region | CS min | CS median | CS max | PS min | PS median | PS max |
|---|---|---|---|---|---|---|
| CH | 0.0 | 0.0 | 0.0 | -0.1 | 2.6 | 4.5 |
| EU | -11.8 | -8.5 | -6.2 | 0.2 | 1.2 | 2.6 |
| US | -20.4 | -15.5 | -12.3 | 0.1 | 3.5 | 9.3 |
| APAC | -10.7 | -6.4 | -0.9 | 8.7 | 20.3 | 26.6 |
| AF | -11.3 | -8.5 | -6.7 | 2.7 | 3.4 | 4.4 |
| ROW | -10.6 | -7.6 | -6.9 | 0.0 | 0.0 | 5.2 |

## Manufacturing capacity

2040 regional capacity (GW):

| Region | Minimum | Median | Maximum |
|---|---|---|---|
| CH | 342.0 | 435.5 | 590.3 |
| EU | 27.8 | 45.9 | 61.5 |
| US | 4.8 | 52.7 | 102.0 |
| APAC | 199.3 | 260.0 | 289.4 |
| AF | 10.2 | 10.9 | 10.9 |
| ROW | 21.2 | 62.7 | 134.2 |

Median total capacity minus demand, and median China capacity (GW):

| Year | Capacity - demand | China capacity |
|---|---|---|
| 2025 | 852.4 | 931.0 |
| 2030 | 127.0 | 518.4 |
| 2035 | 137.7 | 437.1 |
| 2040 | 157.0 | 435.5 |

The stacked capacity figure sums regional medians and does not represent one profile. Capacity-minus-demand above is computed within each profile before taking the median.

## Highlight-specific checks for existing Results prose

The old Eq 1 scarcity narrative must be reassessed because the highlighted profile changed. Prices and flows below come from the filtered observation CSVs; bilateral offers and domestic costs come from the selected sweep JSON.

Eq 1 and retained-profile median trade and China exports (GW):

| Year | Eq 1 trade | Median trade | Eq 1 China exports | Median China exports |
|---|---|---|---|---|
| 2025 | 193.6 | 153.0 | 47.6 | 89.6 |
| 2030 | 174.0 | 159.8 | 0.0 | 71.8 |
| 2035 | 185.0 | 171.6 | 0.0 | 22.5 |
| 2040 | 291.0 | 169.3 | 99.0 | 16.7 |

Eq 1 Chinese bilateral export offers relative to its domestic cost:

| Year | Domestic cost USD/kW | Lowest offer/cost | Highest offer/cost | Active Chinese export destinations |
|---|---|---|---|---|
| 2035 | 95.6 | 2.80 | 3.86 | none |
| 2040 | 86.3 | 1.74 | 2.42 | EU |

Eq 1 regional 2040 prices versus domestic cost (USD/kW); the last column is its price rank among retained profiles, with 1 highest:

| Region | Eq 1 price | Domestic cost | Price-cost gap | Retained max price | Eq 1 price rank |
|---|---|---|---|---|---|
| CH | 86.7 | 86.3 | 0.3 | 86.7 | 3/15 |
| EU | 166.2 | 158.5 | 7.7 | 275.2 | 15/15 |
| US | 175.3 | 170.2 | 5.1 | 226.9 | 11/15 |
| APAC | 99.4 | 99.3 | 0.1 | 159.5 | 14/15 |
| AF | 186.9 | 95.8 | 91.1 | 309.3 | 13/15 |
| ROW | 181.5 | 187.2 | -5.7 | 209.5 | 10/15 |

2040 capacity not used for demand (GW):

| Profile | Global capacity - demand | China unused | APAC unused |
|---|---|---|---|
| Eq 1 | 153.4 | 20.5 | 9.1 |
| Eq 2 | 226.4 | 163.7 | 0.0 |