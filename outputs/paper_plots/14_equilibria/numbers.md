# Results from 14 provisionally retained profiles

The 14 IDs in retained_profiles.txt are the 15 passing deviation-screen verdicts without the former Eq 1 (ch-af-apac-eu-row-us/pf080_k100_a030). Profile statistics use filtered observation-level CSV records, with selected-sweep JSON used for the offer/cost checks at the end. Planner prices use the regions/lam column of outputs/llp_planner/llp_planner_results.xlsx.

## Multiple equilibria

OLS slope of horizon-average demand-weighted price on horizon-average capacity: **-10.3 USD/kW per 100 GW**, R² **0.44**, n = **14**. Horizon weights: 2025 and 2040 = 1/6 each; 2030 and 2035 = 1/3 each.

| Highlight | Profile | Capacity GW | Price USD/kW | Price factor | Capacity weight | Damping | Welfare rank |
|---|---|---|---|---|---|---|---|
| Eq 1: highest price | af-eu-us-apac-row-ch/pf100_k050_a040 | 888 | 179 | 1.00 | 0.50 | 0.40 | 14/14 |
| Eq 2: highest capacity | ch-af-apac-eu-row-us/pf120_k100_a030 | 998 | 155 | 1.20 | 1.00 | 0.30 | 1/14 |

## Market-clearing prices

Price-band shared y-axis ceiling: **450 USD/kW**, the next multiple of 50 above the retained maximum.

Global demand-weighted price by year (USD/kW; markup in percent):

| Year | Min | Median | Max | Planner | Median gap | Markup |
|---|---|---|---|---|---|---|
| 2025 | 211 | 221 | 242 | 169 | 53 | 31% |
| 2030 | 143 | 159 | 192 | 114 | 45 | 39% |
| 2035 | 134 | 151 | 161 | 103 | 48 | 47% |
| 2040 | 132 | 140 | 148 | 98 | 42 | 43% |

Regional prices (USD/kW; ratio is median/planner):

| Region | Year | Min | Median | P90 | Max | Planner | Median/planner | Min - planner |
|---|---|---|---|---|---|---|---|---|
| CH | 2025 | 163 | 163 | 163 | 163 | 163 | 1.00 | 0 |
| CH | 2030 | 108 | 108 | 108 | 108 | 107 | 1.00 | 0 |
| CH | 2035 | 96 | 96 | 96 | 96 | 96 | 1.00 | 0 |
| CH | 2040 | 87 | 87 | 87 | 87 | 90 | 0.96 | -3 |
| EU | 2025 | 256 | 312 | 365 | 373 | 179 | 1.74 | 77 |
| EU | 2030 | 191 | 235 | 280 | 324 | 124 | 1.90 | 67 |
| EU | 2035 | 175 | 219 | 255 | 261 | 112 | 1.96 | 63 |
| EU | 2040 | 177 | 202 | 255 | 275 | 106 | 1.90 | 71 |
| US | 2025 | 285 | 313 | 352 | 366 | 174 | 1.79 | 110 |
| US | 2030 | 192 | 244 | 270 | 349 | 119 | 2.06 | 73 |
| US | 2035 | 178 | 214 | 251 | 262 | 107 | 2.00 | 71 |
| US | 2040 | 169 | 193 | 223 | 227 | 101 | 1.91 | 68 |
| APAC | 2025 | 209 | 259 | 305 | 330 | 177 | 1.46 | 32 |
| APAC | 2030 | 124 | 162 | 225 | 234 | 121 | 1.34 | 2 |
| APAC | 2035 | 110 | 160 | 172 | 191 | 110 | 1.46 | 0 |
| APAC | 2040 | 99 | 137 | 154 | 159 | 104 | 1.32 | -4 |
| AF | 2025 | 295 | 341 | 390 | 446 | 173 | 1.96 | 121 |
| AF | 2030 | 199 | 262 | 307 | 323 | 118 | 2.22 | 81 |
| AF | 2035 | 186 | 249 | 273 | 280 | 106 | 2.34 | 80 |
| AF | 2040 | 167 | 210 | 242 | 309 | 100 | 2.09 | 67 |
| ROW | 2025 | 302 | 326 | 353 | 353 | 178 | 1.84 | 124 |
| ROW | 2030 | 212 | 226 | 246 | 291 | 122 | 1.86 | 90 |
| ROW | 2035 | 188 | 206 | 207 | 207 | 110 | 1.87 | 77 |
| ROW | 2040 | 171 | 187 | 203 | 209 | 104 | 1.79 | 66 |

Regional median price versus own manufacturing cost (USD/kW); counts are profiles within 5 USD/kW, below and above cost:

| Region | Year | Median price | Cost | Median gap | Min gap | Max gap | Within 5 | Below | Above |
|---|---|---|---|---|---|---|---|---|---|
| CH | 2025 | 163.3 | 163.0 | 0.3 | 0.3 | 0.3 | 14 | 0 | 0 |
| CH | 2030 | 107.8 | 107.4 | 0.4 | 0.4 | 0.4 | 14 | 0 | 0 |
| CH | 2035 | 95.9 | 95.6 | 0.3 | 0.3 | 0.3 | 14 | 0 | 0 |
| CH | 2040 | 86.7 | 86.3 | 0.3 | 0.3 | 0.3 | 14 | 0 | 0 |
| EU | 2025 | 311.9 | 299.2 | 12.6 | -43.5 | 73.5 | 1 | 5 | 8 |
| EU | 2030 | 235.1 | 197.2 | 37.9 | -6.3 | 126.3 | 1 | 1 | 12 |
| EU | 2035 | 219.0 | 175.5 | 43.6 | -0.5 | 85.7 | 1 | 0 | 13 |
| EU | 2040 | 201.8 | 158.5 | 43.3 | 18.7 | 116.7 | 0 | 0 | 14 |
| US | 2025 | 312.6 | 321.2 | -8.7 | -36.7 | 44.5 | 4 | 7 | 3 |
| US | 2030 | 244.5 | 211.8 | 32.7 | -20.2 | 136.9 | 1 | 1 | 12 |
| US | 2035 | 214.2 | 188.4 | 25.8 | -10.3 | 73.6 | 2 | 1 | 11 |
| US | 2040 | 193.4 | 170.2 | 23.2 | -1.4 | 56.8 | 4 | 0 | 10 |
| APAC | 2025 | 258.7 | 187.4 | 71.3 | 21.1 | 142.5 | 0 | 0 | 14 |
| APAC | 2030 | 162.1 | 123.6 | 38.5 | 0.1 | 110.8 | 2 | 0 | 12 |
| APAC | 2035 | 160.1 | 109.9 | 50.2 | 0.1 | 80.9 | 1 | 0 | 13 |
| APAC | 2040 | 136.9 | 99.3 | 37.6 | 0.1 | 60.2 | 2 | 0 | 12 |
| AF | 2025 | 340.7 | 180.9 | 159.8 | 113.8 | 265.2 | 0 | 0 | 14 |
| AF | 2030 | 261.6 | 119.3 | 142.3 | 80.1 | 203.7 | 0 | 0 | 14 |
| AF | 2035 | 248.8 | 106.1 | 142.7 | 79.7 | 174.3 | 0 | 0 | 14 |
| AF | 2040 | 209.6 | 95.8 | 113.7 | 71.1 | 213.5 | 0 | 0 | 14 |
| ROW | 2025 | 325.9 | 353.4 | -27.4 | -51.8 | 0.0 | 4 | 10 | 0 |
| ROW | 2030 | 226.5 | 232.9 | -6.5 | -21.3 | 58.0 | 4 | 8 | 2 |
| ROW | 2035 | 206.2 | 207.2 | -1.0 | -19.5 | 0.1 | 9 | 5 | 0 |
| ROW | 2040 | 186.9 | 187.2 | -0.2 | -16.6 | 22.3 | 6 | 5 | 3 |

China and APAC median prices versus planner (USD/kW; gaps use unrounded values):

| Year | China median | China planner | China gap | APAC median | APAC planner | APAC gap |
|---|---|---|---|---|---|---|
| 2025 | 163.3 | 163.0 | 0.3 | 258.7 | 176.8 | 82.0 |
| 2030 | 107.8 | 107.4 | 0.4 | 162.1 | 121.2 | 40.8 |
| 2035 | 95.9 | 95.8 | 0.1 | 160.1 | 109.6 | 50.5 |
| 2040 | 86.7 | 89.9 | -3.2 | 136.9 | 103.6 | 33.3 |

Median regional exports (GW):

| Year | China | APAC |
|---|---|---|
| 2025 | 89.6 | 60.0 |
| 2030 | 72.9 | 83.7 |
| 2035 | 27.0 | 119.7 |
| 2040 | 11.2 | 133.9 |

Median 2040 cross-border trade: **168.4 GW**.

## Welfare distribution

Planner total: **10,114.7 billion USD** (CS + PS - capacity costs).

| Region | Planner | Minimum | Median | Maximum |
|---|---|---|---|---|
| CH | 3,965.5 | 3,973.0 | 4,084.4 | 4,145.2 |
| EU | 2,117.0 | 1,916.4 | 1,955.6 | 1,986.5 |
| US | 812.3 | 694.5 | 700.1 | 713.5 |
| APAC | 1,079.6 | 1,185.1 | 1,210.0 | 1,263.0 |
| AF | 493.5 | 458.1 | 468.1 | 473.7 |
| ROW | 1,646.7 | 1,504.3 | 1,520.1 | 1,529.5 |
| Total | 10,114.7 | 9,799.5 | 9,940.9 | 10,029.9 |

Global welfare loss: min **0.8%**, median **1.7%**, max **3.1%**.

Number of profiles with regional welfare above planner:

| Region | Gaining profiles |
|---|---|
| CH | 14/14 |
| EU | 0/14 |
| US | 0/14 |
| APAC | 14/14 |
| AF | 0/14 |
| ROW | 0/14 |

Median joint EU+US+AF+ROW loss: **427.4 billion USD**. Median joint CH+APAC gain: **256.2 billion USD**. These are medians of sums within each profile.
Spearman rho(China welfare, other-four welfare) = **0.77**. rho(global welfare, China 2040 capacity share) = **0.96**.

Relative CS and PS change as a percentage of each region's planner welfare:

| Region | CS min | CS median | CS max | PS min | PS median | PS max |
|---|---|---|---|---|---|---|
| CH | 0.1 | 0.1 | 0.1 | -0.1 | 2.7 | 4.4 |
| EU | -11.6 | -8.3 | -6.2 | 0.2 | 1.1 | 2.6 |
| US | -20.3 | -15.4 | -12.2 | 0.1 | 2.6 | 9.3 |
| APAC | -10.6 | -6.7 | -1.3 | 12.8 | 20.5 | 26.5 |
| AF | -11.2 | -8.3 | -6.7 | 2.7 | 3.3 | 4.4 |
| ROW | -9.1 | -7.5 | -6.8 | 0.0 | 0.0 | 0.7 |

## Manufacturing capacity

2040 regional capacity (GW):

| Region | Minimum | Median | Maximum |
|---|---|---|---|
| CH | 342.0 | 440.6 | 590.3 |
| EU | 27.8 | 46.7 | 61.5 |
| US | 4.8 | 51.9 | 102.0 |
| APAC | 199.3 | 260.4 | 289.4 |
| AF | 10.2 | 10.8 | 10.9 |
| ROW | 21.2 | 61.3 | 134.2 |

Median total capacity minus demand, and median China capacity (GW):

| Year | Capacity - demand | China capacity |
|---|---|---|
| 2025 | 852.4 | 931.0 |
| 2030 | 129.2 | 519.0 |
| 2035 | 135.8 | 442.1 |
| 2040 | 157.4 | 440.6 |

The stacked capacity figure sums regional medians and does not represent one profile. Capacity-minus-demand above is computed within each profile before taking the median.
