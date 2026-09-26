"""Plot regional supplier shares and bilateral offer/dispatch outcomes.

Run ``python analysis/china_apac_competition.py`` first, then run this script
from the repository root. The merit-order snapshots use one accepted profile;
the other figures describe the 27 reported accepted profiles.
"""

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
DATA = ROOT / "outputs/china_apac_competition"
DATA_CSV = DATA / "csv"
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
MARKETS = ("eu", "us", "row", "af")
YEARS = (2025, 2030, 2035, 2040)
LABELS = {"ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "af": "AF", "row": "ROW"}
COLORS = {
    "ch": "#CA6180", "eu": "#FEFD99", "us": "#FCB7C7",
    "apac": "#B7A6D8", "af": "#B8D99E", "row": "#9ED3DC",
}
# Same hues as the capacity-pathway bars, darkened for legible thin lines.
LINE_COLORS = {
    "ch": "#A34262", "eu": "#B4A900", "us": "#C77389",
    "apac": "#8064A2", "af": "#679748", "row": "#478D9A",
}
ACTIVE_GW = 0.1
EXAMPLE_PROFILE = "eu-us-af-row-apac-ch/pf080_k050_a040"  # C20; EU co-supply in 2030.

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(DATA / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(DATA / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def region_legend() -> list[Patch]:
    return [Patch(facecolor=COLORS[region], edgecolor="#888888", linewidth=0.5,
                  label=LABELS[region]) for region in REGIONS]


def plot_supplier_shares(routes: pd.DataFrame) -> None:
    grouped = (routes.groupby(["importer", "year", "exporter"], as_index=False)
               .agg(mean_share=("share_of_market", "mean"),
                    mean_flow_gw=("flow_gw", "mean"),
                    active_profiles=("active", "sum")))
    grouped.to_csv(DATA_CSV / "all_supplier_market_shares.csv", index=False)
    assert np.allclose(grouped.groupby(["importer", "year"]).mean_share.sum(), 1, atol=1e-7)

    fig, axes = plt.subplots(3, 2, figsize=(10.8, 10.2), sharex=True, sharey=True)
    x = np.arange(len(YEARS), dtype=float)
    for ax, importer in zip(axes.flat, REGIONS):
        bottom = np.zeros(len(YEARS))
        for exporter in REGIONS:
            values = np.array([
                grouped[(grouped.importer.eq(importer)) & grouped.year.eq(year)
                        & grouped.exporter.eq(exporter)].mean_share.iloc[0] * 100
                for year in YEARS
            ])
            ax.bar(x, values, width=0.58, bottom=bottom,
                   color=COLORS[exporter], alpha=0.78,
                   edgecolor="white", linewidth=1.0, zorder=2)
            bottom += values
        ax.set_title(LABELS[importer], fontsize=15, pad=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x, YEARS)
        ax.tick_params(axis="both", labelsize=12)
        ax.tick_params(axis="x", length=0, labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of served demand [%]", fontsize=13)
    fig.legend(handles=region_legend(), ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.95,
               fontsize=12, columnspacing=1.2, handlelength=1.35)
    fig.text(0.5, 0.079, "Arithmetic mean across the 27 reported profiles; every supplier is shown separately.",
             ha="center", fontsize=10.5, color="#444444")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.13,
                        wspace=0.23, hspace=0.38)
    save(fig, "supplier_shares_all_regions")


def plot_offer_paths(routes: pd.DataFrame) -> None:
    active = routes[routes.flow_gw.gt(ACTIVE_GW)]
    offers = (active.groupby(["importer", "year", "exporter"], as_index=False)
              .agg(median_delivered_offer=("delivered_offer_usd_per_kw", "median"),
                   active_profiles=("candidate", "nunique")))
    prices = (routes.groupby(["importer", "year"], as_index=False)
              .agg(median_market_price=("market_price_usd_per_kw", "median")))
    offers.to_csv(DATA_CSV / "active_offer_paths.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), sharex=True, sharey=True)
    x = np.arange(len(YEARS), dtype=float)
    for ax, importer in zip(axes.flat, MARKETS):
        for exporter in REGIONS:
            route = offers[(offers.importer.eq(importer)) & offers.exporter.eq(exporter)]
            series = route.set_index("year").reindex(YEARS)
            values = series.median_delivered_offer.to_numpy(float)
            counts = series.active_profiles.to_numpy(float)
            valid = np.isfinite(values)
            if not valid.any():
                continue
            solid = values.copy()
            solid[counts < 3] = np.nan
            ax.plot(x, solid,
                    color=LINE_COLORS[exporter], linewidth=1.75,
                    marker="o", markersize=5.2,
                    markerfacecolor=COLORS[exporter], markeredgecolor="#333333",
                    markeredgewidth=0.6, zorder=3)
            sparse = valid & (counts < 3)
            ax.scatter(x[sparse], values[sparse], s=36, facecolor="white",
                       edgecolor=LINE_COLORS[exporter], linewidth=1.25, zorder=4)
        local = prices[prices.importer.eq(importer)].set_index("year").reindex(YEARS)
        ax.plot(x, local.median_market_price.to_numpy(float), color="#222222",
                linestyle="-.", linewidth=2.2, marker="^", markersize=6,
                zorder=5)
        ax.set_title(LABELS[importer], fontsize=15, pad=8)
        ax.set_xticks(x, YEARS)
        ax.set_ylim(0, 550)
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        ax.tick_params(axis="both", labelsize=12)
        ax.tick_params(axis="x", length=0, labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Price [USD/kW]", fontsize=13)
    legend = [Line2D([0], [0], color=LINE_COLORS[r], marker="o",
                     markerfacecolor=COLORS[r], markeredgecolor="#333333",
                     linewidth=1.7, label=LABELS[r]) for r in REGIONS]
    legend.append(Line2D([0], [0], color="#222222", linestyle="-.",
                         marker="^", linewidth=2.0, label="Market price"))
    legend.append(Line2D([0], [0], color="#666666", linestyle="none",
                         marker="o", markerfacecolor="white", markeredgecolor="#666666",
                         label="1–2 active profiles"))
    fig.legend(handles=legend, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), frameon=True, framealpha=0.95,
               fontsize=11.5, columnspacing=1.0, handlelength=1.8)
    fig.text(0.5, 0.105, "Delivered offers conditional on flow > 0.1 GW; market-price medians use all 27 profiles",
             ha="center", fontsize=10.5, color="#444444")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.19,
                        wspace=0.23, hspace=0.34)
    save(fig, "active_delivered_offer_paths")


def plot_cleared_offer_stack(routes: pd.DataFrame, regions: pd.DataFrame, year: int) -> None:
    sample_routes = routes[routes.candidate.eq(EXAMPLE_PROFILE) & routes.year.eq(year)]
    sample_regions = regions[regions.candidate.eq(EXAMPLE_PROFILE) & regions.year.eq(year)]
    assert len(sample_routes) == 36 and len(sample_regions) == 6
    capacities = sample_regions.set_index("region").capacity_gw.to_dict()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.9), sharey=True)
    max_price = max(sample_routes.market_price_usd_per_kw.max(),
                    sample_routes[sample_routes.flow_gw.gt(ACTIVE_GW)].delivered_offer_usd_per_kw.max())
    y_max = max(250, np.ceil(max_price / 50) * 50 + 30)
    detail_rows = []
    for ax, importer in zip(axes.flat, REGIONS):
        market = sample_routes[sample_routes.importer.eq(importer)].copy()
        demand = float(market.demand_gw.iloc[0])
        clearing = float(market.market_price_usd_per_kw.iloc[0])
        cleared = market[market.flow_gw.gt(0.01)].sort_values("delivered_offer_usd_per_kw")
        left = 0.0
        for row in cleared.itertuples(index=False):
            width = max(0.0, float(row.flow_gw))
            ax.bar(left, row.delivered_offer_usd_per_kw, width=width,
                   align="edge", color=COLORS[row.exporter], alpha=0.78,
                   edgecolor="white", linewidth=1.0, zorder=2)
            if width / demand > 0.14:
                ax.text(left + width / 2, row.delivered_offer_usd_per_kw / 2,
                        LABELS[row.exporter], ha="center", va="center", fontsize=10.5,
                        color="#222222", fontweight="bold", zorder=4)
            left += width
            detail_rows.append(dict(year=year, profile=EXAMPLE_PROFILE,
                                    importer=importer, exporter=row.exporter,
                                    cleared_flow_gw=width,
                                    delivered_offer_usd_per_kw=row.delivered_offer_usd_per_kw,
                                    market_price_usd_per_kw=clearing,
                                    demand_gw=demand,
                                    exporter_capacity_gw=capacities[row.exporter],
                                    importer_capacity_gw=capacities[importer]))
        assert abs(left - demand) < 0.01, (year, importer, left, demand)
        ax.axhline(clearing, color="#222222", linestyle="-.", linewidth=1.8, zorder=5)
        ax.axvline(demand, color="#222222", linestyle=":", linewidth=1.0, zorder=4)
        ax.text(demand * 1.025, clearing, f"{clearing:.0f}",
                ha="left", va="center", fontsize=10.5, color="#222222")
        ax.set_xlim(0, demand * 1.20)
        ax.set_ylim(0, y_max)
        ax.set_title(f"{LABELS[importer]}   demand {demand:.0f} GW",
                     fontsize=13, pad=9)
        ax.tick_params(axis="both", labelsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Delivered offer [USD/kW]", fontsize=11.5)
    for ax in axes[1, :]:
        ax.set_xlabel("Cumulative cleared flow [GW]", fontsize=11)
    handles = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.5,
                     label=f"{LABELS[r]}  K={capacities[r]:.0f} GW") for r in REGIONS]
    handles.append(Line2D([0], [0], color="#222222", linestyle="-.",
                          linewidth=1.8, label="Market price"))
    fig.suptitle(f"Cleared offer stacks in {year} — illustrative accepted profile C20",
                 fontsize=15, y=0.995)
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), frameon=True, framealpha=0.95,
               fontsize=10.5, columnspacing=0.9, handlelength=1.5)
    fig.text(0.5, 0.115,
             "Bars show realized flows; K is each exporter's installed capacity shared across all markets.",
             ha="center", fontsize=10.5, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.915, bottom=0.20,
                        wspace=0.28, hspace=0.38)
    save(fig, f"cleared_offer_stacks_C20_{year}")
    pd.DataFrame(detail_rows).to_csv(DATA_CSV / f"cleared_offer_stacks_C20_{year}.csv", index=False)


def main() -> None:
    DATA_CSV.mkdir(exist_ok=True)
    routes = pd.read_csv(DATA_CSV / "route_observations.csv")
    regions = pd.read_csv(DATA_CSV / "regional_observations.csv")
    assert routes.candidate.nunique() == regions.candidate.nunique() == 27
    plot_supplier_shares(routes)
    plot_offer_paths(routes)
    for year in (2030, 2040):
        plot_cleared_offer_stack(routes, regions, year)
    print("Created all-supplier shares, active offer paths, and C20 cleared-offer stacks for 2030 and 2040.")


if __name__ == "__main__":
    main()
