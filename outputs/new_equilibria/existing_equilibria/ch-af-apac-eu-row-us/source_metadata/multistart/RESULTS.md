# Terminal-salvage three-start audit results

Created: 2026-09-17T15:49:40+02:00.

## Outcome

- Audited candidates: 15
- Three-start local 1% survivors: 0
- Candidates failing the stronger diagnostic: 15
- Failed audit tasks: 0
- Each complete candidate has 18 frozen-profile optimizer attempts.
- No source profile was changed.

Lower maximum gain is stronger within this diagnostic, provided all
attempts succeeded. It is not a global optimality ranking.

| Rank | Candidate | One-start max gain | Three-start max gain | Worst player | Non-candidate starts selected | Result |
|---:|---|---:|---:|---|---|---|
| 1 | `ch-row-apac-us-eu-af/pf100_k100_a030` | 0.627435% | 1.602905% | CH | CH, ROW, APAC, US, EU, AF | does not survive |
| 2 | `ch-row-apac-us-eu-af/pf120_k100_a040` | 0.956513% | 1.889689% | CH | CH, ROW, APAC | does not survive |
| 3 | `ch-af-apac-eu-row-us/pf120_k100_a040` | 0.626525% | 2.305518% | CH | CH, APAC, ROW, US | does not survive |
| 4 | `ch-row-apac-us-eu-af/pf100_k050_a040` | 0.995711% | 2.472313% | APAC | CH, ROW, APAC, US, EU, AF | does not survive |
| 5 | `ch-af-apac-eu-row-us/pf100_k100_a040` | 0.960737% | 2.991591% | APAC | CH, APAC, ROW, US | does not survive |
| 6 | `ch-af-eu-us-row-apac/pf100_k050_a040` | 0.209492% | 3.117086% | CH | CH, AF, EU, US, ROW, APAC | does not survive |
| 7 | `ch-row-apac-us-eu-af/pf100_k100_a040` | 0.538139% | 3.185265% | CH | CH, ROW, APAC, US, EU, AF | does not survive |
| 8 | `ch-af-eu-us-row-apac/pf100_k100_a040` | 0.468494% | 3.474185% | CH | CH, AF, EU, US, ROW, APAC | does not survive |
| 9 | `ch-af-apac-eu-row-us/pf100_k050_a040` | 0.533668% | 3.563641% | CH | CH, AF, APAC, EU, ROW, US | does not survive |
| 10 | `ch-row-apac-us-eu-af/pf100_k050_a030` | 0.969406% | 3.686453% | APAC | CH, ROW, APAC, US, EU, AF | does not survive |
| 11 | `ch-af-eu-us-row-apac/pf100_k050_a030` | 0.756636% | 3.739909% | CH | CH, EU, US, ROW, APAC | does not survive |
| 12 | `ch-af-apac-eu-row-us/pf100_k050_a030` | 0.518140% | 4.072027% | CH | CH, AF, APAC, EU, ROW, US | does not survive |
| 13 | `ch-af-apac-eu-row-us/pf120_k050_a030` | 0.790283% | 4.464629% | CH | CH, APAC, ROW, US | does not survive |
| 14 | `ch-af-eu-us-row-apac/pf120_k050_a040` | 0.871959% | 4.857112% | CH | CH, AF, US, ROW, APAC | does not survive |
| 15 | `ch-af-apac-eu-row-us/pf120_k050_a040` | 0.802529% | 5.451140% | CH | CH, EU, ROW, US | does not survive |

## Claim scope

A survivor is a stronger local computational candidate than a profile
that passes only the one-start audit. The diagnostic still samples a
finite set of local optimizer initializations and therefore does not
establish a global Nash equilibrium.
