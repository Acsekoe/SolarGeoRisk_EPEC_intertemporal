# Eq 1 nonlocal deviation check

Input SHA-256: `F426E480A20565286F6CD6678A5EEC0DE6E7376382826983E7FC3D39C2DEA5E0`. The corrected workbook and clean Stage-2 objective reproduced China's reference objective (4,119,175.729360), using destination market-clearing price for export revenue and charging the exporter manufacturing and shipping costs. All other players' strategies were frozen. Each trial was re-cleared through the existing Clarabel lower-level solver.

**China:** best tested gain 3.420% from a low-offer start with all offers and capacity re-optimized. Fixed-capacity offer tests reached 0.777%. Verdict: **not an equilibrium**.
**APAC:** best tested gain 11.643% from a low-offer start with all offers and capacity re-optimized. A 2040 APAC-to-EU offer cut alone yielded 1.999% with capacity fixed. Verdict: **not an equilibrium**.

The original one-start SLSQP audit initialized at the candidate's high offers. APAC's fixed-capacity undercut directly shows a profitable move across the zero-flow entry threshold. China's larger gain was found from broad low-offer starts with capacity allowed to adjust, which the original start did not find. These feasible counterexamples reject Eq 1 under the 1% rule; they do not require a globally solved best response.

**Screening:** All other 25 profiles have an idle-capacity, near-zero-flow route with a potential margin above 50 USD/kW. All 403 flagged route-periods received three undercuts and an 11-point grid. 10 profiles fail the 1% test: CH-first: `pf080_k050_a040` (1.96%), `pf100_k050_a040` (3.70%); AF-first: `pf080_k050_a030` (1.06%), `pf100_k050_a030` (1.54%), `pf100_k100_a040` (1.85%); EU-first: `pf080_k050_a040` (1.09%), `pf100_k050_a030` (1.72%), `pf100_k050_a040` (1.02%), `pf100_k100_a030` (2.02%), `pf100_k100_a040` (1.63%). The prefixes refer to the player update orders in the run manifest.

Thus at least 11 of the 26 retained profiles fail the stated equilibrium tolerance. Verdicts for all 25 other profiles are in `screening_summary.csv`; every test is in `results.csv`. A passing label means only that these tests found no deviation above 1%, not that global Nash equilibrium has been proved. The earlier exclusion of `eu-us-af-row-apac-ch/pf120_k050_a040` is noted in `notes/notes_2026-09-27.txt` and `IEEE Paper/revision.tex`, but its exact additional-test script was not present in `scripts/` or `notes/`.
