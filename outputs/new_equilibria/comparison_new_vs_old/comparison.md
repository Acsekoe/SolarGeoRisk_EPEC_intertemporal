# New versus old accepted equilibrium outcomes

All entries are simple averages across accepted profiles in the named group. Damping is treated as an algorithmic parameter, not an economic input. Objectives are the regional reference objectives recorded by the common frozen-profile zero-proximal audit; their sum is descriptive and is not a social-welfare measure.

## Group-level comparison

| Group | Accepted | Total capacity 2040 | Total capacity 2045 | Mean regional price 2040 | Cross-border trade 2040 | Sum of objectives |
|---|---:|---:|---:|---:|---:|---:|
| Old CH-first (6) | 6 | 835.1 | 834.7 | 156.43 | 284.1 | 9,994,278.1 |
| New AF-first (4) | 4 | 801.3 | 801.7 | 178.50 | 279.9 | 9,970,265.9 |
| New EU-first (4) | 4 | 946.6 | 946.2 | 146.78 | 287.5 | 10,015,922.5 |

Relative to the old accepted set:

| New group | Capacity 2040 | Capacity 2045 | Mean price 2040 | Cross-border trade 2040 | Sum of objectives |
|---|---:|---:|---:|---:|---:|
| New AF-first (4) | -4.05% | -3.95% | +14.11% | -1.49% | -0.24% |
| New EU-first (4) | +13.35% | +13.35% | -6.17% | +1.19% | +0.22% |

## Regional endpoint comparison (capacity 2045; market outcomes 2040)

| Region | Old capacity | AF-first capacity | EU-first capacity | Old output | AF-first output | EU-first output |
|---|---:|---:|---:|---:|---:|---:|
| China | 520.9 | 497.9 | 611.2 | 432.3 | 442.0 | 447.0 |
| Africa | 2.5 | 2.6 | 3.4 | 2.5 | 2.6 | 3.4 |
| EU | 1.6 | 1.0 | 0.5 | 1.6 | 1.0 | 0.5 |
| US | 0.5 | 0.7 | 0.4 | 0.5 | 0.7 | 0.4 |
| APAC | 293.6 | 291.7 | 282.0 | 293.5 | 278.8 | 281.7 |
| ROW | 15.7 | 7.8 | 48.8 | 2.6 | 7.8 | 0.0 |

## Within-group spread

| Group | Capacity 2045 range | Mean price 2040 range | Trade 2040 range |
|---|---:|---:|---:|
| Old CH-first (6) | 794.0 to 874.7 | 150.16 to 160.19 | 275.6 to 291.0 |
| New AF-first (4) | 782.3 to 818.1 | 158.35 to 203.59 | 275.4 to 285.4 |
| New EU-first (4) | 811.5 to 1,245.8 | 118.35 to 157.70 | 284.5 to 290.3 |

## Matched-branch update-order comparison

Each row holds the price factor, capacity-path weight, and damping fixed and changes only the update order. Percentages are the new outcome relative to its old CH-first counterpart.

| New sequence | Branch | Capacity 2045 | Mean price 2040 | Trade 2040 | Sum of objectives |
|---|---|---:|---:|---:|---:|
| af-eu-us-apac-row-ch | pf100_k050_a040 | -0.93% | +30.33% | +0.73% | -0.339% |
| af-eu-us-apac-row-ch | pf100_k100_a040 | +2.90% | +14.31% | -4.58% | -0.280% |
| af-eu-us-apac-row-ch | pf120_k050_a030 | -1.48% | +7.42% | +1.19% | -0.508% |
| af-eu-us-apac-row-ch | pf120_k050_a040 | -8.56% | -0.23% | -1.00% | +0.489% |
| eu-us-af-row-apac-ch | pf100_k050_a030 | -2.41% | +1.19% | +1.65% | -0.010% |
| eu-us-af-row-apac-ch | pf120_k050_a030 | +2.20% | -2.15% | +4.19% | +0.022% |
| eu-us-af-row-apac-ch | pf120_k050_a040 | +1.40% | -2.76% | +2.70% | +0.494% |
| eu-us-af-row-apac-ch | pf120_k100_a040 | +43.38% | -21.18% | -2.25% | +0.427% |

## Regional prices and import reliance (2040)

| Region | Old price | AF-first price | EU-first price | Old import share | AF-first import share | EU-first import share |
|---|---:|---:|---:|---:|---:|---:|
| China | 86.65 | 90.27 | 86.65 | 0.0% | 0.0% | 0.0% |
| Africa | 186.22 | 218.13 | 171.45 | 92.2% | 90.8% | 90.6% |
| EU | 193.22 | 223.20 | 175.32 | 98.4% | 98.9% | 99.5% |
| US | 186.60 | 223.44 | 172.59 | 99.4% | 99.1% | 99.5% |
| APAC | 100.84 | 100.20 | 101.56 | 0.0% | 0.0% | 0.0% |
| ROW | 185.03 | 215.75 | 173.10 | 97.0% | 92.3% | 100.0% |

## Damping-pair outcome distances

Relative L1 distance is sum(abs(A-B)) divided by the average absolute mass of the two outcome vectors. Zero means identical. Only accepted 0.30/0.40 pairs with the same sequence, price factor, and capacity-path weight are shown.

| Sequence/group | Economic branch | Capacity distance | Price distance | Flow distance |
|---|---|---:|---:|---:|
| af-eu-us-apac-row-ch | pf120_k050 | 1.914% | 3.118% | 27.394% |
| ch-af-apac-eu-row-us | pf100_k050 | 4.417% | 1.669% | 36.704% |
| ch-af-apac-eu-row-us | pf120_k050 | 7.814% | 5.951% | 30.367% |
| eu-us-af-row-apac-ch | pf120_k050 | 6.649% | 1.540% | 19.103% |

## Audit quality

| Group | Max gain range | Limiting players |
|---|---:|---|
| Old CH-first (6) | 0.518% to 0.961% | APAC, China |
| New AF-first (4) | 0.786% to 0.975% | APAC, China |
| New EU-first (4) | 0.809% to 0.895% | APAC |
