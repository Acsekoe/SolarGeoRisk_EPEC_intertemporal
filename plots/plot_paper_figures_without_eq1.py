"""Regenerate the five active Results figures after excluding Eq. 1.

The published analysis CSVs remain intact. Figures are rendered together in a
temporary directory and copied to the paths already used by SolarGeoRisk_V2.tex.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plots.plot_median_regional_capacity_pathway import plot_from_observations
from scripts import analyze_clean_stage2_ranges as clean
from scripts import analyze_equilibrium_ranges as base


CSV_DIR = (
    ROOT
    / "outputs"
    / "clean_stage2_factorial_20260923_123037"
    / "statistical_analysis"
    / "csv"
)
PAPER_PLOTS = ROOT / "outputs" / "paper_plots"
EXCLUDED = "eu-us-af-row-apac-ch/pf120_k050_a040"
EQ1 = "ch-af-apac-eu-row-us/pf080_k100_a040"
EQ2 = "ch-af-apac-eu-row-us/pf120_k100_a030"
STEMS = (
    "capacity_price_pathway_equilibria",
    "equilibrium_bands_prices_by_region",
    "equilibrium_bands_capacity_by_region",
    "median_regional_capacity_pathway",
    "welfare_cs_ps_relative_boxplots_response",
)


def load_filtered(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(CSV_DIR / filename)
    candidates = set(frame["candidate"])
    if EXCLUDED not in candidates or len(candidates) != 27:
        raise ValueError(f"Expected the original 27 profiles, including Eq. 1, in {filename}")
    filtered = frame[frame["candidate"] != EXCLUDED].copy()
    if filtered["candidate"].nunique() != 26:
        raise ValueError(f"Expected 26 retained profiles in {filename}")
    return filtered


def main() -> None:
    metrics = load_filtered("candidate_metrics.csv")
    prices = load_filtered("price_observations.csv")
    capacities = load_filtered("capacity_observations.csv")
    system = load_filtered("system_metrics.csv")
    welfare = load_filtered("welfare_relative_observations.csv")
    retained = set(metrics["candidate"])
    if any(set(frame["candidate"]) != retained for frame in (prices, capacities, system, welfare)):
        raise ValueError("The figure inputs do not contain the same 26 profiles")
    if not {EQ1, EQ2} <= retained:
        raise ValueError("Eq. 1 or Eq. 2 is missing from the retained profiles")

    PAPER_PLOTS.mkdir(parents=True, exist_ok=True)
    base.apply_plot_style()
    with tempfile.TemporaryDirectory(prefix="paper_figures_26_", dir=PAPER_PLOTS.parent) as temp:
        staged = Path(temp)
        base.plot_horizon_capacity_price_equilibria(
            metrics, staged, highlighted={EQ1: ("Eq 1", (0, 14)), EQ2: ("Eq 2", (-4, 16))}
        )
        clean.plot_regional_bands(
            prices,
            "price_usd_per_kw",
            base.MARKET_YEARS,
            "Price [$/kW]",
            "#A83232",
            staged,
            "equilibrium_bands_prices_by_region",
            display_candidates=retained,
            shared_ymax=600.0,
            show_individual_outcomes=False,
            planner_prices=clean.load_planner_prices(
                clean.DEFAULT_PLANNER_RESULTS, base.MARKET_YEARS
            ),
        )
        clean.plot_regional_bands(
            capacities,
            "capacity_gw",
            base.CAPACITY_YEARS,
            "Capacity [GW]",
            "#7570B3",
            staged,
            "equilibrium_bands_capacity_by_region",
            display_candidates=retained,
            show_individual_outcomes=False,
            stacked_legend=True,
        )
        plot_from_observations(capacities, system, staged)
        clean.plot_welfare_difference_boxplots(
            welfare,
            staged,
            value_column="change_percent_of_planner_regional_welfare",
            xlabel="[%]",
            legend_prefix="Relative",
            stem="welfare_cs_ps_relative_boxplots_response",
            symmetric_axis=True,
            abbreviate_components=True,
            compact=True,
        )
        for stem in STEMS:
            for extension in ("pdf", "png"):
                source = staged / f"{stem}.{extension}"
                if not source.is_file() or source.stat().st_size == 0:
                    raise RuntimeError(f"Figure was not generated: {source}")
        for stem in STEMS:
            for extension in ("pdf", "png"):
                shutil.copyfile(staged / f"{stem}.{extension}", PAPER_PLOTS / f"{stem}.{extension}")
    print(f"Updated {len(STEMS)} paper figures using {len(retained)} profiles; excluded {EXCLUDED}")


if __name__ == "__main__":
    main()
