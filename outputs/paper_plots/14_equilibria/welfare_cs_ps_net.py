"""The paper's CS/PS boxplot with PS net of capacity costs.

PS includes fixed and investment capacity costs, as in the paper's PS
definition, so CS + PS equals the regional welfare change of the welfare table.
Two variants: absolute (billion USD) and relative (% of regional planner welfare).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts import analyze_clean_stage2_ranges as clean  # noqa: E402

CSV = ROOT / "outputs/clean_stage2_factorial_20260923_123037/statistical_analysis/csv"


def observations():
    ids = [line.strip() for line in (OUT / "retained_profiles.txt").read_text(
        encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    w = pd.read_csv(CSV / "welfare_level_observations.csv")
    w = w[w.candidate.isin(ids)]
    planner = pd.read_csv(CSV / "welfare_level_comparison.csv").set_index("region")
    cs0 = planner.planner_cs_billion_usd_pv
    ps0 = planner.planner_ps_billion_usd_pv - planner.planner_capacity_cost_billion_usd_pv
    rows = []
    for r in w.itertuples(index=False):
        ps = r.producer_surplus_billion_usd_pv - r.capacity_cost_billion_usd_pv
        base = cs0[r.region] + ps0[r.region]
        for component, change in (("Consumer surplus", r.consumer_surplus_billion_usd_pv - cs0[r.region]),
                                  ("Producer surplus", ps - ps0[r.region])):
            rows.append({"candidate": r.candidate, "region": r.region, "component": component,
                         "absolute": change, "relative": 100 * change / base})
    frame = pd.DataFrame(rows)
    if frame.candidate.nunique() != len(ids):
        raise ValueError("Welfare observations do not match the retained list")
    return frame


def main():
    obs = observations()
    clean.plot_welfare_difference_boxplots(
        obs, OUT, value_column="absolute", xlabel="[billion USD]",
        legend_prefix="Absolute", stem="welfare_cs_ps_absolute_boxplots",
        symmetric_axis=False, abbreviate_components=True, compact=True)
    clean.plot_welfare_difference_boxplots(
        obs, OUT, value_column="relative", xlabel="[%]",
        legend_prefix="Relative", stem="welfare_cs_ps_relative_net_boxplots",
        symmetric_axis=True, abbreviate_components=True, compact=True)


if __name__ == "__main__":
    main()
