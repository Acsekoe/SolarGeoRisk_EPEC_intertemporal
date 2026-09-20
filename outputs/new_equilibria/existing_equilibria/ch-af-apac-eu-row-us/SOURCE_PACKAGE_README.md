# Fifteen one-start local 1%-equilibrium candidates

This folder packages the 15 distinct profiles accepted by the one-start,
common-frozen-profile, zero-proximal 1% criterion in
`terminal_state_salvage_factorial_20260917_145309`.

The source artifacts remain in their original locations. These files are
copies, so the recorded source paths and hashes are not broken.

## Important claim scope

All 15 profiles passed the declared **one-start local 1%** test. A later
three-start frozen-profile audit successfully completed all 270 optimizer
attempts, but none of the 15 remained below 1%. These files should therefore
be described as one-start local equilibrium candidates, not multistart-robust
or globally certified Nash equilibria.

The economic market horizon is 2025--2040. The recorded 2045 capacity is a
terminal stock used for salvage valuation; there is no 2045 market, demand,
trade, price, or operating payoff.

## Folder layout

- `profiles/`: the 15 selected Stage-2 sweep files containing the complete
  strategy, capacity, and market profiles.
- `one_start_audits/`: the frozen-profile audit that accepted each profile.
- `three_start_audits/`: the later robustness audit for each profile.
- `status/`: copied factorial and multistart status records.
- `source_metadata/`: copied manifests, protocols, and result summaries for
  the factorial search and three-start audit.

Each artifact type retains the original `player-order/branch/` hierarchy.

## Result figures

Run the following command from the project root to regenerate the packaged
candidate figures:

```bash
python plots/plot_fifteen_equilibria.py
```

The script writes PNG and vector PDF versions to `plots/`:

- `candidate_audit_robustness`: one-start versus three-start maximum unilateral
  welfare gains for all 15 candidates, with the 1% criterion marked.
- `candidate_price_paths`: 2025--2040 regional market-price paths, separated by
  the `pf100` and `pf120` offer-price initialization families.
- `candidate_capacity_paths`: 2025--2040 regional manufacturing-capacity paths,
  showing every candidate, the median, and the full range.

It also writes `candidate_summary.csv`, the compact candidate/audit table used
for the robustness figure. The plots intentionally exclude 2045 market results;
2045 capacity is a terminal salvage stock rather than an economic market period.

## Candidate index

| Player order | Branch | Selected sweep | One-start max gain | Three-start max gain |
|---|---|---:|---:|---:|
| CH-AF-APAC-EU-ROW-US | `pf100_k050_a030` | 14 | 0.518140% | 4.072027% |
| CH-AF-APAC-EU-ROW-US | `pf100_k050_a040` | 10 | 0.533668% | 3.563641% |
| CH-AF-APAC-EU-ROW-US | `pf100_k100_a040` | 10 | 0.960737% | 2.991591% |
| CH-AF-APAC-EU-ROW-US | `pf120_k050_a030` | 13 | 0.790283% | 4.464629% |
| CH-AF-APAC-EU-ROW-US | `pf120_k050_a040` | 15 | 0.802529% | 5.451140% |
| CH-AF-APAC-EU-ROW-US | `pf120_k100_a040` | 6 | 0.626525% | 2.305518% |
| CH-AF-EU-US-ROW-APAC | `pf100_k050_a030` | 13 | 0.756636% | 3.739909% |
| CH-AF-EU-US-ROW-APAC | `pf100_k050_a040` | 9 | 0.209492% | 3.117086% |
| CH-AF-EU-US-ROW-APAC | `pf100_k100_a040` | 9 | 0.468494% | 3.474185% |
| CH-AF-EU-US-ROW-APAC | `pf120_k050_a040` | 10 | 0.871959% | 4.857112% |
| CH-ROW-APAC-US-EU-AF | `pf100_k050_a030` | 10 | 0.969406% | 3.686453% |
| CH-ROW-APAC-US-EU-AF | `pf100_k050_a040` | 7 | 0.995711% | 2.472313% |
| CH-ROW-APAC-US-EU-AF | `pf100_k100_a030` | 12 | 0.627435% | 1.602905% |
| CH-ROW-APAC-US-EU-AF | `pf100_k100_a040` | 9 | 0.538139% | 3.185265% |
| CH-ROW-APAC-US-EU-AF | `pf120_k100_a040` | 5 | 0.956513% | 1.889689% |
