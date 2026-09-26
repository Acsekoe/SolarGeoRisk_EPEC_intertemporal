"""Regional capacity and price paths: three capacity-only equilibria versus 27 strategic medians and the LLP benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
STRATEGIC_ROOT = ROOT / "outputs" / "clean_stage2_factorial_20260923_123037"
CAPACITY_ROOT = ROOT / "outputs" / "capacity_only_chain_20260924_120009"
REGIONS = (
    ("ch", "China"),
    ("eu", "Europe"),
    ("us", "United States"),
    ("apac", "Asia-Pacific"),
    ("af", "Africa"),
    ("row", "Rest of World"),
)
YEARS = (2025, 2030, 2035, 2040)
PLANNER_PATH = ROOT / "outputs" / "llp_planner" / "llp_planner_results.xlsx"
CAPACITY_COLOR = "#B64550"


def accepted_branches(root: Path) -> set[str]:
    summary = pd.read_csv(root / "summary.csv", dtype={"selected_sweep": "string"})
    accepted = summary.loc[summary["status"] == "accepted"]
    return set(accepted["sequence"] + "/" + accepted["branch"])


def validate_and_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategic_manifest = json.loads((STRATEGIC_ROOT / "manifest.json").read_text())
    capacity_manifest = json.loads((CAPACITY_ROOT / "stage2" / "manifest.json").read_text())
    strategic_protocol = strategic_manifest["protocol"]
    capacity_protocol = capacity_manifest["protocol"]
    if strategic_protocol["input_sha256"] != capacity_protocol["input_sha256"]:
        raise ValueError("The two sets use different input workbooks")
    if strategic_protocol["terminal_salvage_fraction"] != capacity_protocol["terminal_salvage_fraction"]:
        raise ValueError("The two sets use different salvage fractions")
    if not capacity_protocol["fix_offers_to_cost"]:
        raise ValueError("The capacity-only run did not fix bilateral offers to cost")

    selection = json.loads((STRATEGIC_ROOT / "results_selection.json").read_text())
    excluded = {item["candidate"] for item in selection["excluded_candidates"]}
    reported_strategic = accepted_branches(STRATEGIC_ROOT) - excluded
    reported_capacity = accepted_branches(CAPACITY_ROOT / "stage2")
    if len(reported_strategic) != 27 or len(reported_capacity) != 3:
        raise ValueError("Expected 27 strategic and 3 capacity-only accepted profiles")

    analysis_root = STRATEGIC_ROOT / "statistical_analysis"
    capacity_rows = pd.read_csv(analysis_root / "csv" / "capacity_observations.csv")
    price_rows = pd.read_csv(analysis_root / "csv" / "price_observations.csv")
    strategic = capacity_rows[["candidate", "region", "year", "capacity_gw"]].merge(
        price_rows[["candidate", "region", "year", "price_usd_per_kw"]],
        on=["candidate", "region", "year"],
        validate="one_to_one",
    )
    if set(strategic["candidate"]) != reported_strategic:
        raise ValueError("The strategic observations do not match the reported 27")

    comparison = pd.read_csv(CAPACITY_ROOT / "stage2" / "comparison" / "regional_metrics.csv")
    fixed = comparison.loc[comparison["scenario"] == "fixed_cost_offers"].copy()
    fixed = fixed.rename(
        columns={
            "branch": "candidate",
            "period": "year",
            "clearing_price_usd_per_kw": "price_usd_per_kw",
        }
    )
    if set(fixed["candidate"]) != reported_capacity:
        raise ValueError("The capacity-only observations do not match the accepted three")

    expected = {(region, year) for region, _ in REGIONS for year in YEARS}
    for label, frame, count in (("strategic", strategic, 27), ("fixed", fixed, 3)):
        if len(frame) != count * len(expected):
            raise ValueError(f"Wrong number of {label} observations")
        for candidate, group in frame.groupby("candidate"):
            actual = set(zip(group["region"], group["year"]))
            if actual != expected:
                raise ValueError(f"Missing or duplicate region-year in {candidate}")
    planner = pd.read_excel(
        PLANNER_PATH, sheet_name="regions", usecols=["r", "t", "Kcap", "lam"]
    ).rename(
        columns={
            "r": "region",
            "t": "year",
            "Kcap": "capacity_gw",
            "lam": "price_usd_per_kw",
        }
    )
    planner["region"] = planner["region"].astype(str).str.lower().str.strip()
    planner["year"] = pd.to_numeric(planner["year"], errors="raise").astype(int)
    if planner.duplicated(["region", "year"]).any():
        raise ValueError("Duplicate LLP benchmark region-year records")
    planner = planner.set_index(["region", "year"]).reindex(
        pd.MultiIndex.from_tuples(sorted(expected), names=["region", "year"])
    )
    if not np.isfinite(planner.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite LLP benchmark values")
    return strategic, fixed, planner.reset_index()


def draw(
    strategic: pd.DataFrame,
    fixed: pd.DataFrame,
    planner: pd.DataFrame,
    column: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
) -> None:
    x = np.arange(len(YEARS), dtype=float)
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.5))
        for index, (axis, (region, name)) in enumerate(zip(axes.flat, REGIONS)):
            strategic_values = np.array(
                [
                    strategic.loc[
                        (strategic["region"] == region) & (strategic["year"] == year), column
                    ].median()
                    for year in YEARS
                ],
                dtype=float,
            )
            fixed_matrix = (
                fixed.loc[fixed["region"] == region]
                .pivot(index="candidate", columns="year", values=column)
                .reindex(columns=YEARS)
                .sort_index()
                .to_numpy(dtype=float)
            )
            lower = fixed_matrix.min(axis=0)
            upper = fixed_matrix.max(axis=0)
            axis.fill_between(
                x, lower, upper, color=CAPACITY_COLOR, alpha=0.15, linewidth=0,
                zorder=2,
            )
            for values in fixed_matrix:
                axis.plot(
                    x, values, color=CAPACITY_COLOR,
                    linewidth=1.35, alpha=0.9, zorder=4,
                )
            axis.plot(
                x, strategic_values, color="#222222", linewidth=1.8,
                marker="o", markersize=4.5, zorder=6,
            )
            if column == "price_usd_per_kw":
                benchmark_values = (
                    planner.loc[planner["region"] == region]
                    .set_index("year")
                    .loc[list(YEARS), column]
                    .to_numpy(dtype=float)
                )
                axis.plot(
                    x, benchmark_values, color="#2E6F40", linewidth=2.0,
                    marker="s", markersize=4.5, zorder=5,
                )
            axis.set_title(name, fontsize=15)
            axis.set_xticks(x, [str(year) for year in YEARS])
            if column == "price_usd_per_kw":
                axis.set_ylim(0, 600)
            if index % 2 == 0:
                axis.set_ylabel(ylabel, fontsize=15)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.tick_params(axis="both", labelsize=12.5)

        handles = [
            Patch(facecolor=CAPACITY_COLOR, alpha=0.15, edgecolor="none",
                  label="Capacity-only observed range"),
            Line2D([0], [0], color=CAPACITY_COLOR, linewidth=1.35,
                   label="Three capacity-only equilibria"),
            Line2D([0], [0], color="#222222", marker="o", markersize=4.5,
                   linewidth=1.8, label="Median of 27 strategic equilibria"),
        ]
        if column == "price_usd_per_kw":
            handles.append(
                Line2D([0], [0], color="#2E6F40", marker="s", markersize=4.5,
                       linewidth=2.0, label="LLP benchmark")
            )
        fig.legend(
            handles=handles, loc="lower center", ncol=1,
            bbox_to_anchor=(0.5, 0.02), fontsize=14.5,
            frameon=True, framealpha=0.9, handlelength=1.6,
            handletextpad=0.5, borderpad=0.55, labelspacing=0.45,
        )
        fig.subplots_adjust(
            left=0.11, right=0.98, top=0.96,
            bottom=0.24,
            wspace=0.43, hspace=0.50,
        )
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=CAPACITY_ROOT / "stage2" / "comparison" / "figures",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strategic, fixed, planner = validate_and_load()
    draw(
        strategic, fixed, planner, "capacity_gw", "Capacity [GW]",
        "capacity_only_vs_strategic_capacity_bands", output_dir,
    )
    draw(
        strategic, fixed, planner, "price_usd_per_kw", "Price [$/kW]",
        "capacity_only_vs_strategic_price_bands", output_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
