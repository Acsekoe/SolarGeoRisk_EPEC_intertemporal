"""Scatter the average price against China's import share, APAC's exports and both export offers.

The 14 retained equilibria share the price axis of the capacity-price figure
(demand-weighted horizon average), and the x variables use the same horizon
weights. China's import share is its share of all cross-border flows into the
EU, the US, Africa, and ROW. The export offer of a region is the median offer over its five foreign
destinations in each period, since offers above the destination price receive no
flow and can take arbitrarily high values. Eq 1 and Eq 2 are selected as in
outputs/paper_plots/14_equilibria/regenerate.py. Spearman rho is descriptive
only, since the equilibria are not independent draws.

Run ``python analysis/china_apac_competition.py`` first so that the route
observations exist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs/clean_stage2_factorial_20260923_123037/statistical_analysis/csv"
ROUTES = ROOT / "outputs/china_apac_competition/csv/route_observations.csv"
FOLDER = ROOT / "outputs/paper_plots/14_equilibria"
STEM = "price_supply_drivers"
WEIGHTS = {2025: 1 / 6, 2030: 1 / 3, 2035: 1 / 3, 2040: 1 / 6}
IMPORTERS = ["eu", "us", "af", "row"]
CAPACITY = "average_total_capacity_2025_2040_gw"
PRICE = "average_demand_weighted_price_2025_2040_usd_per_kw"
# Rows: exports, offers. Columns: China, APAC.
PANELS = (
    ("ch_import_share", "China share of importer imports [%]"),
    ("apac_exports", "APAC exports [GW]"),
    ("ch_offer", "China export offer [$/kW]"),
    ("apac_offer", "APAC export offer [$/kW]"),
)
# Label offsets in points per panel, chosen to keep labels clear of the fit line.
OFFSETS = {
    "ch_import_share": {"Eq 1": (0, 12), "Eq 2": (-22, 0)},
    "apac_exports": {"Eq 1": (0, 12), "Eq 2": (0, 12)},
    "ch_offer": {"Eq 1": (0, 12), "Eq 2": (22, 0)},
    "apac_offer": {"Eq 1": (0, 12), "Eq 2": (22, 0)},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def retained() -> list[str]:
    lines = (FOLDER / "retained_profiles.txt").read_text(encoding="utf-8").splitlines()
    ids = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError(f"Expected 14 unique retained profiles, got {len(ids)}")
    return ids


def horizon_data(ids: list[str]) -> pd.DataFrame:
    metrics = pd.read_csv(CSV / "candidate_metrics.csv").set_index("candidate").loc[ids]
    regional = pd.read_csv(CSV / "regional_market_metrics.csv")
    regional = regional[regional.candidate.isin(ids) & regional.year.isin(WEIGHTS)]
    regional = regional.assign(weight=regional.year.map(WEIGHTS))
    exports = (regional.exports_gw * regional.weight).groupby(
        [regional.candidate, regional.region]).sum().unstack()

    all_routes = pd.read_csv(ROUTES)
    all_routes = all_routes[all_routes.candidate.isin(ids) & all_routes.year.isin(WEIGHTS)
                            & (all_routes.exporter != all_routes.importer)]
    imports = all_routes[all_routes.importer.isin(IMPORTERS)]
    weighted_flow = imports.flow_gw * imports.year.map(WEIGHTS)
    total_imports = weighted_flow.groupby(imports.candidate).sum()
    china_imports = weighted_flow[imports.exporter.eq("ch")].groupby(imports.candidate).sum()

    routes = all_routes[all_routes.exporter.isin(["ch", "apac"])]
    if routes.groupby(["candidate", "exporter", "year"]).size().ne(5).any():
        raise ValueError("Expected five foreign destinations per exporter and period")
    offers = routes.groupby(["candidate", "exporter", "year"]).offer_usd_per_kw.median()
    offers = offers.reset_index()
    offers["weighted"] = offers.offer_usd_per_kw * offers.year.map(WEIGHTS)
    offers = offers.groupby(["candidate", "exporter"]).weighted.sum().unstack()

    data = pd.DataFrame({
        "price": metrics[PRICE],
        "capacity": metrics[CAPACITY],
        "ch_exports": exports["ch"],
        "ch_import_share": 100 * china_imports / total_imports,
        "apac_exports": exports["apac"],
        "ch_offer": offers["ch"],
        "apac_offer": offers["apac"],
    })
    if data.isna().any().any():
        raise ValueError("Missing horizon averages")
    return data


def highlights(data: pd.DataFrame) -> dict[str, str]:
    ordered = data.reset_index(names="candidate")
    eq1 = ordered.sort_values(["price", "candidate"], ascending=[False, True]).iloc[0]
    eq2 = ordered.sort_values(["capacity", "candidate"], ascending=[False, True]).iloc[0]
    if eq1.candidate == eq2.candidate:
        eq2 = ordered.sort_values(["capacity", "candidate"], ascending=[True, True]).iloc[0]
    return {eq1.candidate: "Eq 1", eq2.candidate: "Eq 2"}


def plot(data: pd.DataFrame, marked: dict[str, str]) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), sharey=True)
    is_marked = data.index.isin(marked)
    rows = []
    for ax, (column, label) in zip(axes.flat, PANELS):
        x = data[column].to_numpy(float)
        y = data.price.to_numpy(float)
        rho = spearmanr(x, y).statistic
        rows.append({"variable": column, "spearman_rho_price": rho})
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, slope * grid + intercept, color="#343434", linestyle="--",
                linewidth=1.8, zorder=2)
        ax.scatter(x[~is_marked], y[~is_marked], s=58, marker="x", color="#A83232",
                   linewidth=1.8, zorder=3)
        for candidate, name in marked.items():
            point = (data.at[candidate, column], data.at[candidate, "price"])
            ax.scatter(*point, s=58, marker="x", color="#2E7D32", linewidth=2.2, zorder=4)
            ax.annotate(name, point, xytext=OFFSETS[column][name], textcoords="offset points",
                        ha="center", va="center", fontsize=13, color="#2E7D32", zorder=5)
        ax.set_title(rf"$\rho = {rho:+.2f}$", fontsize=14, pad=6)
        ax.set_xlabel(label, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Price [$/kW]", fontsize=14)
    lo, hi = data.price.min(), data.price.max()
    axes[0, 0].set_ylim(lo - 3, hi + 4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.1,
                        wspace=0.12, hspace=0.55)
    for extension in ("pdf", "png"):
        fig.savefig(FOLDER / f"{STEM}.{extension}", dpi=300, bbox_inches="tight",
                    pad_inches=0.04)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    data = horizon_data(retained())
    marked = highlights(data)
    rho = plot(data, marked)
    data.assign(label=data.index.map(marked)).to_csv(FOLDER / f"{STEM}.csv")
    print(rho.round(2).to_string(index=False))
    print(data.describe().loc[["min", "50%", "max"]].round(1).to_string())
    print(marked)


if __name__ == "__main__":
    main()
