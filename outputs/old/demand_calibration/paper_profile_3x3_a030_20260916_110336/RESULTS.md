# Fixed 3 x 3 paper-profile release: results

The run completed on 2026-09-16 at 12:42:35 CEST. All nine declared
branches reached a terminal status. Five parallel workers were used. One
branch met the one-start, common-frozen-profile, zero-proximal relative 1%
criterion; eight branches exhausted the 20-sweep schedule without passing.
No sequential best-response failure terminated a branch and `stderr.log` is
empty. Some non-selected intermediate frozen audits contained an unsuccessful
player solve (one in the first-order cost branch, one in the second-order cost
branch, three in the third-order cost branch, and one in the third-order
halfway branch); those sweeps were ineligible for acceptance. The accepted
sweep had all six audit solves successful.

| Manuscript player order | Start | Best sweep | Best frozen maximum gain | Binding player | Result |
|---|---|---:|---:|---|---|
| CH-AF-APAC-EU-ROW-US | exact Stage 1 | 8 | 4.0773% | CH | no pass |
| CH-AF-APAC-EU-ROW-US | halfway to cost | 14 | 1.4509% | CH | no pass |
| CH-AF-APAC-EU-ROW-US | manufacturing cost | 6 | **0.6214%** | CH | **accepted** |
| CH-AF-EU-US-ROW-APAC | exact Stage 1 | 9 | 6.7158% | CH | no pass |
| CH-AF-EU-US-ROW-APAC | halfway to cost | 8 | 1.6059% | APAC | no pass |
| CH-AF-EU-US-ROW-APAC | manufacturing cost | 2 | 4.5531% | CH | no pass |
| CH-ROW-APAC-US-EU-AF | exact Stage 1 | 0 | 7.0497% | CH | no pass |
| CH-ROW-APAC-US-EU-AF | halfway to cost | 11 | 1.5365% | APAC | no pass |
| CH-ROW-APAC-US-EU-AF | manufacturing cost | 2 | 4.6279% | CH | no pass |

The accepted branch retained the corrected Stage-1 capacities from source
iteration 23, reset cross-regional offers to period-specific manufacturing
cost, and then used six zero-proximal Gauss--Seidel sweeps with fixed
`alpha=0.30`, all players updated, no player freezing, and no move cap. Its
final audit had all six solves successful.

The experiment exactly reproduces the previously observed 0.6213949% result.
It does not support the claim that the three exact proximal endpoints are
already equilibria or that a direct zero-proximal release from those endpoints
reliably reaches the 1% criterion within 20 sweeps. The halfway transformations
were consistent near-misses but did not pass. The positive result is specific
to the full cost reset and the CH-AF-APAC-EU-ROW-US order.

The defensible algorithmic interpretation is therefore that the proximal
stage supplied a useful capacity basin for one declared order, while the
manufacturing-cost price restart and zero-proximal release were necessary
parts of the successful candidate-generation pipeline. A matched run from
primitive capacities plus the same cost-price start is still required to
isolate the incremental value of the proximal Stage-1 capacity basin.

Before using the accepted profile as the manuscript's final equilibrium
candidate, it should receive a predeclared multistart frozen-profile audit.
