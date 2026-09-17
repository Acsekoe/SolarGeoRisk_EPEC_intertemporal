# Three corrected-demand computational equilibria

Created: 2026-09-16 23:21:46 Europe/Vienna

## Result

Three distinct profiles now satisfy the declared corrected-demand local
one-percent equilibrium criterion. At each accepted common profile, all six
players received an independent, zero-proximal best-response solve with the
other five strategies frozen. Every solve reported success and the largest
normalized unilateral profit gain was at most 1%.

These are computational local 1%-equilibria under the recorded one-start audit.
They are not globally certified Nash equilibria and have not yet received the
planned multistart diagnostic.

| ID | Stage-1 anchor and initialization | Gauss--Seidel order | Accepted sweep | Limiting player | Maximum gain |
|---|---|---|---:|---|---:|
| E1 | Iteration 23; Stage-1 capacity path; bilateral offers reset to manufacturing cost | CH--AF--APAC--EU--ROW--US | 6 | CH | **0.621395%** |
| E2 | Iteration 24; Stage-1 net-capacity-change path scaled by 0.50; bilateral offers initialized at 1.20 times manufacturing cost | CH--ROW--APAC--US--EU--AF | 5 | APAC | **0.835510%** |
| E3 | Iteration 24; full Stage-1 net-capacity-change path; bilateral offers initialized at 1.20 times manufacturing cost | CH--ROW--APAC--US--EU--AF | 7 | APAC | **0.910692%** |

All three candidate-generation paths used fixed damping \(\alpha=0.30\),
sequential all-player Gauss--Seidel updates, zero algorithmic proximal
penalties, no move cap, no gain filter, no player freezing, and one local solver
start for every sequential best response. The capacity weight and price factor
only define the initial profile; they do not constrain subsequent player best
responses. Damping and initialization transformations are also absent from the
frozen-profile acceptance audit.

## Six-player frozen-profile audits

Relative unilateral profit gains are reported in percent.

| Player | E1 | E2 | E3 |
|---|---:|---:|---:|
| CH | **0.621395** | 0.767688 | 0.321814 |
| AF | 0.050806 | 0.073144 | 0.013361 |
| APAC | 0.036649 | **0.835510** | **0.910692** |
| EU | 0.000000 | 0.022282 | 0.002578 |
| ROW | 0.024729 | 0.238040 | 0.101018 |
| US | 0.000000 | 0.089406 | 0.008220 |

All 18 reported player solves succeeded. The audit denominator is
\(\max\{|\Pi_i|,1\}\), and negative improvements are truncated at zero.

## Evidence that the profiles are distinct

The table reports maximum absolute pairwise differences in model units. The
coordinates in parentheses identify where each maximum occurs.

| Pair | Net capacity change | Bilateral offer | Clearing price | Demand |
|---|---:|---:|---:|---:|
| E1--E2 | 23.804647 (APAC, 2035) | 60.005740 (CH to ROW, 2045) | 59.930717 (ROW, 2045) | \(2.7440\times10^{-8}\) |
| E1--E3 | 33.107959 (APAC, 2035) | 54.607997 (CH to ROW, 2045) | 54.509308 (ROW, 2045) | \(2.4544\times10^{-8}\) |
| E2--E3 | 9.303312 (APAC, 2035) | 9.078213 (APAC to EU, 2030) | 8.066464 (AF, 2030) | \(2.8809\times10^{-8}\) |

The corrected demand quantities are effectively identical across these
profiles. The economically relevant multiplicity is instead visible in
capacity decisions, strategic offers, market-clearing prices, and consequently
trade and regional payoffs. The subsequent paper comparison should therefore
focus on those outcomes rather than describing the solutions as materially
different demand profiles.

## Factorial search that produced E2 and E3

The completed experiment crossed three corrected Stage-1 anchors with offer
factors \(\{1.0,1.2\}\), capacity-path weights \(\{0.5,1.0\}\), and damping
factors \(\{0.30,0.40\}\), for 24 predeclared branches. Every branch allowed at
most 20 sweeps and was audited after initialization and after every completed
sweep.

Outcome: two branches accepted, 21 completed without passing, and one failed
when China's one-start best-response solve failed at sweep 12. No additional
process remains active.

| Order / Stage-1 anchor | Branch | Best audited gain | Best sweep | Outcome |
|---|---|---:|---:|---|
| CH--AF--APAC--EU--ROW--US | pf100_k050_a030 | 2.2367% | 6 | failed later at sweep 12 |
| CH--AF--APAC--EU--ROW--US | pf100_k050_a040 | 3.5698% | 2 | no pass |
| CH--AF--APAC--EU--ROW--US | pf100_k100_a030 | 1.8514% | 9 | no pass |
| CH--AF--APAC--EU--ROW--US | pf100_k100_a040 | 4.9267% | 1 | no pass |
| CH--AF--APAC--EU--ROW--US | pf120_k050_a030 | 1.3495% | 4 | no pass |
| CH--AF--APAC--EU--ROW--US | pf120_k050_a040 | 1.8705% | 2 | no pass |
| CH--AF--APAC--EU--ROW--US | pf120_k100_a030 | 1.3332% | 11 | no pass |
| CH--AF--APAC--EU--ROW--US | pf120_k100_a040 | 1.6262% | 2 | no pass |
| CH--AF--EU--US--ROW--APAC | pf100_k050_a030 | 3.0965% | 8 | no pass |
| CH--AF--EU--US--ROW--APAC | pf100_k050_a040 | 1.9682% | 3 | no pass |
| CH--AF--EU--US--ROW--APAC | pf100_k100_a030 | 2.7223% | 2 | no pass |
| CH--AF--EU--US--ROW--APAC | pf100_k100_a040 | 3.4085% | 2 | no pass |
| CH--AF--EU--US--ROW--APAC | pf120_k050_a030 | 1.4889% | 15 | no pass |
| CH--AF--EU--US--ROW--APAC | pf120_k050_a040 | 1.9625% | 18 | no pass |
| CH--AF--EU--US--ROW--APAC | pf120_k100_a030 | 1.7085% | 3 | no pass |
| CH--AF--EU--US--ROW--APAC | pf120_k100_a040 | 2.4870% | 2 | no pass |
| CH--ROW--APAC--US--EU--AF | pf100_k050_a030 | 3.7908% | 3 | no pass |
| CH--ROW--APAC--US--EU--AF | pf100_k050_a040 | 1.7013% | 3 | no pass |
| CH--ROW--APAC--US--EU--AF | pf100_k100_a030 | 1.4563% | 11 | no pass |
| CH--ROW--APAC--US--EU--AF | pf100_k100_a040 | 3.5065% | 1 | no pass |
| CH--ROW--APAC--US--EU--AF | pf120_k050_a030 | **0.8355%** | 5 | **accepted as E2** |
| CH--ROW--APAC--US--EU--AF | pf120_k050_a040 | 1.7291% | 2 | no pass |
| CH--ROW--APAC--US--EU--AF | pf120_k100_a030 | **0.9107%** | 7 | **accepted as E3** |
| CH--ROW--APAC--US--EU--AF | pf120_k100_a040 | 1.5393% | 4 | no pass |

The common successful configuration for E2 and E3 was the iteration-24 anchor,
the CH--ROW--APAC--US--EU--AF order, offers initialized at 1.20 times cost, and
\(\alpha=0.30\). Their distinct starting capacity-path weights led to distinct
accepted profiles. The matched \(\alpha=0.40\) branches did not pass, showing
that damping affected basin selection even though it does not enter the final
equilibrium test.

## Primary artifacts and hashes

### E1

- Profile: `outputs/equilibria/corrected_demand_20260916/E1/equilibrium_profile.json`
- Profile SHA-256: `2944B91566F6C2804225D71C3FE3A5C7A30F786F20C4713D903A449B3952E7B5`
- Audit: `outputs/equilibria/corrected_demand_20260916/E1/audit_one_start.json`
- Audit SHA-256: `0BD58B56465AA3D765CFA29A92A7B1F941283CB8329E7CDEC700EAE72AB9CA9F`

### E2

- Profile: `outputs/equilibria/corrected_demand_20260916/E2/equilibrium_profile.json`
- Profile SHA-256: `2163A7D210F3BE8592B9C08538DDCEC3E92B1652F1C657B81D26C07B1C2A72E5`
- Audit: `outputs/equilibria/corrected_demand_20260916/E2/audit_one_start.json`
- Audit SHA-256: `4480C66D931211F5AB20E18C4DB2191C441908F4ED30EB72392E2ED55EADB4D6`

### E3

- Profile: `outputs/equilibria/corrected_demand_20260916/E3/equilibrium_profile.json`
- Profile SHA-256: `08CD5D82638EAE3475D122054926C7EAC2F2E1232A8B54F4B7BE63B197A23FD3`
- Audit: `outputs/equilibria/corrected_demand_20260916/E3/audit_one_start.json`
- Audit SHA-256: `DF8F7E849DCE457512EB4C5B4DB428426CBF7355D20DAB5D8F8DDE7C49BBE674`

### Factorial experiment

- Protocol: `outputs/equilibria/corrected_demand_20260916/provenance/E2_E3_factorial_protocol.md`
- Protocol SHA-256: `7598031B637135769BD837F6EC78F47E59A249D214E1B97689E4B65564DD3CAB`
- Manifest: `outputs/equilibria/corrected_demand_20260916/provenance/E2_E3_factorial_manifest.json`
- Manifest SHA-256: `856C680C4D1F2141E3F07621806BB31D8CEFAF7448C5E663F69B8445EDBD08D1`
- Corrected input SHA-256: `5E3E392B695AB917D8FF445203C9A04F1308083391BBF3C64D35E7ED3C9AD234`

## Interpretation and limitations

1. The minimum target of three distinct corrected-demand equilibria has been
   reached without using the proposed post-hoc warm-start recovery shortcut.
2. E2 and E3 came from the same Stage-1 anchor and update order. They demonstrate
   initialization-dependent multiplicity, but they do not give one accepted
   equilibrium for every manuscript order.
3. The nominal branch intended to reproduce E1 did not return to E1 in the new
   factorial. Previous diagnosis traced the divergence to tiny initialization
   and lower-market differences that sent China's first nonconvex local
   best-response solve to a different local solution. E1 remains an accepted
   archived profile, but its generation is numerically basin-sensitive.
4. All acceptance results use one local solver start per player. A multistart
   audit is a stronger diagnostic and must be reported separately rather than
   silently changing the declared acceptance rule.
5. The 1% threshold establishes approximate local computational equilibrium,
   not uniqueness, an exact Nash equilibrium, or global optimality of any
   unilateral best response.

## Immediate next steps

1. Run a separate multistart frozen-profile diagnostic on E1--E3.
2. Compute a common comparison table covering regional and aggregate objective
   values, capacities, strategic offers, clearing prices, and trade flows.
3. Use those comparisons to answer Reviewer 2 by discussing commonalities,
   differences, and the sensitivity of equilibrium selection to initialization
   and damping.
4. Preserve the current profiles and audits unchanged; derive any paper tables
   and figures from these hashed artifacts.
