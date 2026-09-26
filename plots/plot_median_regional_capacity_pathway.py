"""Plot regional median capacities for the reported clean-objective outcomes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs/clean_stage2_factorial_20260923_123037/statistical_analysis"
OUTPUT = ROOT / "outputs/paper_plots"
YEARS = (2025, 2030, 2035, 2040)
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
REGION_LABELS = {"ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "af": "AF", "row": "ROW"}
REGION_COLORS = {
    "ch": "#CA6180",
    "eu": "#FEFD99",
    "us": "#FCB7C7",
    "apac": "#B7A6D8",
    "af": "#B8D99E",
    "row": "#9ED3DC",
}
COLOR_DEMAND = "#222222"
COLOR_PLAN = "#2E6F40"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
    }
)


def main() -> None:
    summary = pd.read_csv(ANALYSIS / "csv/capacity_summary.csv")
    summary = summary[summary["year"].isin(YEARS) & summary["region"].isin(REGIONS)]
    if len(summary) != len(YEARS) * len(REGIONS) or not (summary["n"] == 27).all():
        raise ValueError("Expected one 27-outcome capacity median per region and year")
    regional_medians = summary.pivot(index="year", columns="region", values="median")
    regional_medians = regional_medians.reindex(index=YEARS, columns=REGIONS)
    if regional_medians.isna().any().any():
        raise ValueError("Missing a regional capacity median")

    system = pd.read_csv(ANALYSIS / "csv/system_metrics.csv")
    system = system[system["year"].isin(YEARS)]
    demand = system.groupby("year")["total_demand_gw"].median().reindex(YEARS)
    demand_spread = system.groupby("year")["total_demand_gw"].agg(
        lambda values: values.max() - values.min()
    )
    if (demand_spread > 1e-6).any() or demand.isna().any():
        raise ValueError("Global demand is missing or differs among reported outcomes")

    planner = pd.read_excel(ROOT / "outputs/llp_planner/llp_planner_results.xlsx", sheet_name="regions")
    planner["t"] = planner["t"].astype(int)
    planner_capacity = planner.groupby("t")["Kcap"].sum().reindex(YEARS)
    if planner_capacity.isna().any():
        raise ValueError("Missing planner capacity for one or more model years")

    x = np.arange(len(YEARS), dtype=float)
    bottom = np.zeros(len(YEARS), dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for region in REGIONS:
        values = regional_medians[region].to_numpy(float)
        ax.bar(
            x, values, width=0.46, bottom=bottom,
            color=REGION_COLORS[region], alpha=0.78,
            edgecolor="white", linewidth=1.0, zorder=2,
        )
        bottom += values

    ax.plot(
        x, demand.to_numpy(float), color=COLOR_DEMAND, linestyle="-.",
        marker="^", markerfacecolor=COLOR_DEMAND,
        markeredgecolor=COLOR_DEMAND, markeredgewidth=1.6,
        markersize=6.8, linewidth=1.8, zorder=5,
    )
    ax.plot(
        x, planner_capacity.to_numpy(float), color=COLOR_PLAN, linestyle="--",
        marker="*", markerfacecolor=COLOR_PLAN,
        markeredgecolor=COLOR_PLAN, markeredgewidth=1.4,
        markersize=9.0, linewidth=1.8, zorder=5,
    )

    top = max(float(bottom.max()), float(demand.max()), float(planner_capacity.max()))
    for xpos, total in zip(x, bottom):
        ax.text(
            xpos, total + top * 0.018, f"{total:.0f}",
            ha="center", va="bottom", fontsize=12, color=COLOR_DEMAND,
        )
    ax.set_xticks(x, [str(year) for year in YEARS], fontsize=15)
    ax.set_ylabel("GW", fontsize=15)
    ax.set_ylim(0, top * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", length=0)

    handles = [
        Patch(facecolor=REGION_COLORS[region], edgecolor="white", linewidth=0.8,
              label=REGION_LABELS[region])
        for region in REGIONS
    ]
    handles.extend(
        [
            Line2D([0], [0], color=COLOR_DEMAND, linestyle="-.", marker="^",
                   linewidth=1.8, markersize=6.2, label="Global demand"),
            Line2D([0], [0], color=COLOR_PLAN, linestyle="--", marker="*",
                   linewidth=1.8, markersize=8.0, label="Planner capacity"),
        ]
    )
    ax.legend(
        handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.31),
        frameon=True, fontsize=11.5, framealpha=0.9,
        handlelength=1.4, handletextpad=0.45, columnspacing=0.9,
        borderpad=0.45, labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.31)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "median_regional_capacity_pathway"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
