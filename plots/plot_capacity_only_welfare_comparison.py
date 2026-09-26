"""Compare the three capacity-only equilibria with the reported strategic set and planner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_clean_stage2_ranges import (
    relative_welfare_component_tables,
    welfare_component_tables,
)


CAPACITY_RUN = ROOT / "outputs" / "capacity_only_chain_20260924_120009" / "stage2"
STRATEGIC_RUN = ROOT / "outputs" / "clean_stage2_factorial_20260923_123037"
PLANNER_PATH = ROOT / "outputs" / "llp_planner" / "llp_planner_results.xlsx"
OUTPUT_DIR = CAPACITY_RUN / "comparison"
FIGURE_DIR = OUTPUT_DIR / "figures"
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
REGION_NAMES = {
    "ch": "China", "eu": "EU", "us": "US", "apac": "APAC",
    "af": "Africa", "row": "ROW",
}
CS_COLOR = "#B43C38"
PS_COLOR = "#2E6F40"


def load_welfare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    capacity_manifest = json.loads((CAPACITY_RUN / "manifest.json").read_text())
    strategic_manifest = json.loads((STRATEGIC_RUN / "manifest.json").read_text())
    for setting in ("input_sha256", "terminal_salvage_fraction"):
        if capacity_manifest["protocol"][setting] != strategic_manifest["protocol"][setting]:
            raise ValueError(f"Capacity-only and strategic runs differ in {setting}")
    accepted = [
        {
            "candidate": f"{row['sequence']}/{row['branch']}",
            "sequence": row["sequence"],
            "branch": row["branch"],
        }
        for row in capacity_manifest["results"]
        if row["status"] == "accepted"
    ]
    if len(accepted) != 3:
        raise ValueError("Expected exactly three accepted capacity-only profiles")
    _, _, capacity_levels, planner = welfare_component_tables(
        accepted, capacity_manifest, CAPACITY_RUN, PLANNER_PATH
    )
    capacity_relative, _ = relative_welfare_component_tables(capacity_levels, planner)

    strategic_levels = pd.read_csv(
        STRATEGIC_RUN / "statistical_analysis" / "csv" / "welfare_level_observations.csv"
    )
    strategic_summary = pd.read_csv(STRATEGIC_RUN / "summary.csv")
    exclusions = json.loads((STRATEGIC_RUN / "results_selection.json").read_text())
    excluded = {row["candidate"] for row in exclusions["excluded_candidates"]}
    selected = {
        f"{row.sequence}/{row.branch}"
        for row in strategic_summary.itertuples(index=False)
        if row.status == "accepted"
    } - excluded
    if len(selected) != 27 or set(strategic_levels["candidate"]) != selected:
        raise ValueError("Strategic welfare data do not match the 27 reported profiles")

    for label, levels, count in (
        ("capacity-only", capacity_levels, 3),
        ("strategic", strategic_levels, 27),
    ):
        if len(levels) != count * len(REGIONS):
            raise ValueError(f"Incomplete {label} regional welfare observations")
        if levels.duplicated(["candidate", "region"]).any():
            raise ValueError(f"Duplicate {label} candidate-region observations")
    return capacity_levels, capacity_relative, strategic_levels, planner


def total_welfare(levels: pd.DataFrame, scenario: str) -> pd.DataFrame:
    frame = levels.assign(
        total_welfare_billion_usd_pv=(
            levels["consumer_surplus_billion_usd_pv"]
            + levels["producer_surplus_billion_usd_pv"]
            - levels["capacity_cost_billion_usd_pv"]
        )
    )
    totals = frame.groupby("candidate", as_index=False)[
        "total_welfare_billion_usd_pv"
    ].sum()
    totals.insert(0, "scenario", scenario)
    return totals


def save_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_relative_components(relative: pd.DataFrame) -> None:
    value_column = "change_percent_of_planner_regional_welfare"
    limit = max(5.0, 5.0 * np.ceil(np.abs(relative[value_column]).max() / 5.0))
    colors = {"Consumer surplus": CS_COLOR, "Producer surplus": PS_COLOR}
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(6.0, 4.25))
        positions = np.arange(1, len(REGIONS) + 1, dtype=float)
        for component, color in colors.items():
            groups = [
                relative.loc[
                    (relative["region"] == region)
                    & (relative["component"] == component),
                    value_column,
                ].to_numpy(float)
                for region in REGIONS
            ]
            if any(len(group) != 3 for group in groups):
                raise ValueError("Every CS/PS boxplot group must have three observations")
            ax.boxplot(
                groups, positions=positions, vert=False, widths=0.36,
                whis=(0, 100), showfliers=False, patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": color,
                          "alpha": 0.48, "linewidth": 1.2},
                whiskerprops={"color": color, "linewidth": 1.25, "alpha": 0.85},
                capprops={"color": color, "linewidth": 1.25, "alpha": 0.85},
                medianprops={"color": "#202020", "linewidth": 2.1},
            )
            for position, values in zip(positions, groups):
                ax.scatter(
                    values, position + np.array([-0.06, 0.0, 0.06]),
                    s=14, facecolor=color, edgecolor="white", linewidth=0.35,
                    zorder=5,
                )
        ax.axvline(0, color="#333333", linewidth=1.15, zorder=1)
        ax.set_yticks(positions, [REGION_NAMES[region] for region in REGIONS])
        ax.invert_yaxis()
        ax.set_xlim(-limit, limit)
        ax.set_xlabel("Change [% of regional planner welfare]", fontsize=11.5)
        ax.grid(axis="x", linestyle=":", color="#D5D5D5")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.legend(
            handles=[
                Patch(facecolor=CS_COLOR, edgecolor=CS_COLOR, alpha=0.48,
                      label="CS difference to planner"),
                Patch(facecolor=PS_COLOR, edgecolor=PS_COLOR, alpha=0.48,
                      label="PS difference to planner"),
                Line2D([0], [0], marker="o", linestyle="none", color="#555555",
                       markersize=4, label="Individual outcomes (n=3)"),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=1,
            frameon=True, fontsize=10.5, handlelength=1.7,
            labelspacing=0.35, borderpad=0.55,
        )
        fig.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.29)
        save_figure(fig, "capacity_only_welfare_cs_ps_relative_boxplots")


def plot_total_welfare(totals: pd.DataFrame, planner_total: float) -> None:
    strategic = totals.loc[
        totals["scenario"] == "Strategic offers (27)",
        "total_welfare_billion_usd_pv",
    ].to_numpy(float)
    capacity = totals.loc[
        totals["scenario"] == "Capacity only (3)",
        "total_welfare_billion_usd_pv",
    ].to_numpy(float)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(7.2, 2.9))
        boxes = ax.boxplot(
            [strategic, capacity], positions=[1, 2], vert=False, widths=0.42,
            whis=(0, 100), showfliers=False, patch_artist=True,
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
            medianprops={"color": "#202020", "linewidth": 2.0},
        )
        for patch, color in zip(boxes["boxes"], ("#7570B3", CS_COLOR)):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.43)
        for part in ("whiskers", "caps"):
            for artist, color in zip(boxes[part], ("#7570B3",) * 2 + (CS_COLOR,) * 2):
                artist.set_color(color)
        ax.scatter(
            capacity, 2 + np.array([-0.09, 0.0, 0.09]),
            s=25, color=CS_COLOR, edgecolor="white", linewidth=0.5, zorder=5,
        )
        ax.axvline(planner_total, color=PS_COLOR, linestyle="--", linewidth=1.8,
                   label="Planner benchmark", zorder=3)
        ax.set_yticks([1, 2], ["Strategic offers (27)", "Capacity only (3)"])
        ax.invert_yaxis()
        ax.set_xlabel("Total welfare [discounted billion USD]", fontsize=12)
        ax.set_xlim(min(strategic.min(), capacity.min()) - 30, planner_total + 28)
        ax.grid(axis="x", linestyle=":", color="#D5D5D5")
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower left", fontsize=10.5, frameon=True)
        fig.subplots_adjust(left=0.30, right=0.98, top=0.94, bottom=0.26)
        save_figure(fig, "capacity_only_total_welfare_comparison")


def write_summary(
    capacity_levels: pd.DataFrame,
    strategic_levels: pd.DataFrame,
    planner: pd.DataFrame,
    totals: pd.DataFrame,
) -> None:
    planner = planner.set_index("region")
    planner_region_total = (
        planner["planner_cs_billion_usd_pv"]
        + planner["planner_ps_billion_usd_pv"]
        - planner["planner_capacity_cost_billion_usd_pv"]
    )
    capacity_region = capacity_levels.assign(
        welfare=lambda frame: frame["consumer_surplus_billion_usd_pv"]
        + frame["producer_surplus_billion_usd_pv"]
        - frame["capacity_cost_billion_usd_pv"]
    ).groupby("region")["welfare"].agg(["min", "median", "max"])
    strategic_region = strategic_levels.assign(
        welfare=lambda frame: frame["consumer_surplus_billion_usd_pv"]
        + frame["producer_surplus_billion_usd_pv"]
        - frame["capacity_cost_billion_usd_pv"]
    ).groupby("region")["welfare"].median()
    regional_rows = []
    for region in REGIONS:
        regional_rows.append({
            "region": region,
            "planner_billion_usd": planner_region_total[region],
            "capacity_only_min_billion_usd": capacity_region.loc[region, "min"],
            "capacity_only_median_billion_usd": capacity_region.loc[region, "median"],
            "capacity_only_max_billion_usd": capacity_region.loc[region, "max"],
            "strategic_27_median_billion_usd": strategic_region[region],
        })
    regional = pd.DataFrame(regional_rows)
    regional.to_csv(OUTPUT_DIR / "capacity_only_regional_welfare_summary.csv", index=False)

    planner_total = float(planner_region_total.sum())
    lines = [
        "# Capacity-only welfare comparison", "",
        "All values are discounted billion USD over 2025–2040 and exclude terminal salvage.",
        "Total welfare equals CS + PS − capacity costs. The CS/PS boxplot shows PS",
        "before capacity costs, matching the 27-equilibrium response plot; both component",
        "differences use regional planner welfare as their denominator.", "",
        "## Total welfare", "",
        "| Scenario | Minimum | Median | Maximum |",
        "|---|---:|---:|---:|",
        f"| Planner | {planner_total:,.1f} | {planner_total:,.1f} | {planner_total:,.1f} |",
    ]
    for scenario in ("Capacity only (3)", "Strategic offers (27)"):
        values = totals.loc[
            totals["scenario"] == scenario, "total_welfare_billion_usd_pv"
        ]
        lines.append(
            f"| {scenario} | {values.min():,.1f} | {values.median():,.1f} | "
            f"{values.max():,.1f} |"
        )
    lines += [
        "", "## Regional welfare", "",
        "| Region | Planner | Capacity-only median [min, max] | Strategic median |",
        "|---|---:|---:|---:|",
    ]
    for row in regional.itertuples(index=False):
        lines.append(
            f"| {REGION_NAMES[row.region]} | {row.planner_billion_usd:,.1f} | "
            f"{row.capacity_only_median_billion_usd:,.1f} "
            f"[{row.capacity_only_min_billion_usd:,.1f}, "
            f"{row.capacity_only_max_billion_usd:,.1f}] | "
            f"{row.strategic_27_median_billion_usd:,.1f} |"
        )
    lines += [
        "", "These are accepted algorithm branches, not independent samples. The",
        "capacity-only and strategic-offer sets come from different games and are not",
        "matched treatment pairs. The boxplots are descriptive; all three capacity-only",
        "outcomes are plotted individually.", "",
    ]
    (OUTPUT_DIR / "capacity_only_welfare_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    capacity_levels, capacity_relative, strategic_levels, planner = load_welfare()
    totals = pd.concat(
        [
            total_welfare(capacity_levels, "Capacity only (3)"),
            total_welfare(strategic_levels, "Strategic offers (27)"),
        ],
        ignore_index=True,
    )
    planner_total = float((
        planner["planner_cs_billion_usd_pv"]
        + planner["planner_ps_billion_usd_pv"]
        - planner["planner_capacity_cost_billion_usd_pv"]
    ).sum())
    capacity_levels.to_csv(OUTPUT_DIR / "capacity_only_welfare_components.csv", index=False)
    capacity_relative.to_csv(OUTPUT_DIR / "capacity_only_cs_ps_relative.csv", index=False)
    totals.to_csv(OUTPUT_DIR / "capacity_only_total_welfare_by_profile.csv", index=False)
    plot_relative_components(capacity_relative)
    plot_total_welfare(totals, planner_total)
    write_summary(capacity_levels, strategic_levels, planner, totals)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
