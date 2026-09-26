"""Cleared-supply merit diagrams using the model's period-specific unit costs.

For each destination and period, use the complete accepted profile selected at
the median clearing price by plot_price_composition.py. Order active suppliers
by manufacturing marginal cost plus shipping; width is their actual cleared
flow divided by destination demand. The companion price-level chart instead
uses actual cleared GW and delivered offers. These are cleared-flow diagrams,
not counterfactual supply curves or plots of unused production capacity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/china_apac_competition"
DATA_CSV = DATA / "csv"
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
YEARS = (2025, 2030, 2035, 2040)
LABELS = {"ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "af": "AF", "row": "ROW"}
COLORS = {
    "ch": "#CA6180", "eu": "#FEFD99", "us": "#FCB7C7",
    "apac": "#B7A6D8", "af": "#B8D99E", "row": "#9ED3DC",
}
DUAL_COLOR = "#B8D99E"
SHIP_COLOR = "#C7D1D5"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def prepare() -> pd.DataFrame:
    data = pd.read_csv(DATA_CSV / "median_price_supplier_components.csv")
    data["landed_marginal_cost_usd_per_kw"] = (
        data.manufacturing_cost_usd_per_kw + data.shipping_usd_per_kw)
    data["strategic_offer_wedge_usd_per_kw"] = (
        data.offer_usd_per_kw - data.manufacturing_cost_usd_per_kw)
    assert data.groupby(["importer", "year"]).ngroups == 24
    assert (data.groupby(["importer", "year"]).gw_per_1gw_demand.sum()
            .sub(1).abs().max() < 0.01)
    assert (data.offer_usd_per_kw + data.shipping_usd_per_kw
            + data.offer_capacity_dual - data.market_price_usd_per_kw).abs().max() < 0.5
    data.to_csv(DATA_CSV / "median_merit_order_routes.csv", index=False)
    return data


def draw_market(ax: plt.Axes, market: pd.DataFrame, *, detailed: bool) -> None:
    market = market.sort_values(["landed_marginal_cost_usd_per_kw", "exporter"])
    price = float(market.market_price_usd_per_kw.iloc[0])
    left = 0.0
    for row in market.itertuples(index=False):
        width = float(row.gw_per_1gw_demand)
        offer = float(row.offer_usd_per_kw)
        shipping = float(row.shipping_usd_per_kw)
        mu_offer = max(0.0, float(row.offer_capacity_dual))
        mc = float(row.manufacturing_cost_usd_per_kw)
        landed_mc = float(row.landed_marginal_cost_usd_per_kw)
        ax.bar(left, offer, width=width, align="edge", color=COLORS[row.exporter],
               alpha=0.80, edgecolor="white", linewidth=0.75, zorder=2)
        if shipping > 0.02:
            ax.bar(left, shipping, width=width, bottom=offer, align="edge",
                   color=SHIP_COLOR, hatch="xx", edgecolor="#67767C",
                   linewidth=0.3, zorder=3)
        if mu_offer > 0.02:
            ax.bar(left, mu_offer, width=width, bottom=offer + shipping,
                   align="edge", color=DUAL_COLOR, alpha=0.48, hatch="///",
                   edgecolor="#6D986E", linewidth=0.4, zorder=4)
        inset = min(0.004, width * 0.08)
        ax.hlines(mc, left + inset, left + width - inset,
                  color="#252525", linewidth=1.15, zorder=6)
        if shipping > 0.2:
            ax.hlines(landed_mc, left + inset, left + width - inset,
                      color="#4A4A4A", linestyle="--", linewidth=1.1, zorder=6)
        if width > (0.14 if detailed else 0.19):
            label = f"{LABELS[row.exporter]}\n{width:.0%}"
            if detailed:
                label += f"\nMC {mc:.0f}"
            ax.text(left + width / 2, min(offer * 0.31, 66), label,
                    ha="center", va="center", fontsize=9.2 if detailed else 8.5,
                    fontweight="bold", color="#202020", zorder=7)
        left += width
    assert abs(left - 1.0) < 0.01
    ax.hlines(price, 0, 1.0, color="#202020", linestyle="-.",
              linewidth=1.6, zorder=8)
    ax.axvline(1.0, color="#333333", linestyle=":", linewidth=0.9, zorder=8)
    ax.text(1.025, price, f"{price:.0f}", ha="left", va="center", fontsize=9,
            zorder=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 425)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.spines[["top", "right"]].set_visible(False)


def legends() -> tuple[list[Patch], list[Patch | Line2D]]:
    regions = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                     label=LABELS[r]) for r in REGIONS]
    components: list[Patch | Line2D] = [
        Patch(facecolor=SHIP_COLOR, edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=DUAL_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#252525", linewidth=1.15,
               label="Manufacturing marginal cost"),
        Line2D([0], [0], color="#4A4A4A", linestyle="--", linewidth=1.1,
               label="Cost + shipping"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.6,
               label="Market price"),
    ]
    return regions, components


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(DATA / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(DATA / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overview(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(REGIONS), len(YEARS), figsize=(16.5, 14.2),
                             sharex=True, sharey=True)
    for i, importer in enumerate(REGIONS):
        for j, year in enumerate(YEARS):
            ax = axes[i, j]
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            draw_market(ax, market, detailed=False)
            demand = float(market.demand_gw.iloc[0])
            import_share = float(market.loc[
                market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
            ax.text(0.01, 0.98, f"D {demand:.0f} GW  ·  imports {import_share:.0%}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8.6, color="#444444")
            ax.tick_params(axis="both", labelsize=8.5)
            if i == 0:
                ax.set_title(str(year), fontsize=13, pad=7)
            if j == 0:
                ax.set_ylabel(f"{LABELS[importer]}\nUSD/kW", fontsize=11,
                              rotation=0, labelpad=36, va="center")
    fig.supxlabel("Cumulative cleared supply per 1 GW of destination demand [GW]",
                  y=0.12, fontsize=11)
    regions, components = legends()
    fig.legend(handles=regions, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.005), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.1)
    fig.legend(handles=components, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.04), frameon=True, framealpha=0.95,
               fontsize=9.1, columnspacing=0.9)
    fig.suptitle("Cleared-supply merit order: marginal cost, offers and market price",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.095,
             "Suppliers run left to right by cost + shipping; width = cleared share. Colored top = offer; hatching adds shipping and dual.",
             ha="center", fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.16,
                        hspace=0.43, wspace=0.22)
    save(fig, "median_merit_order_24_markets")


def plot_region(data: pd.DataFrame, importer: str, pages: PdfPages) -> None:
    subset = data[data.importer.eq(importer)]
    fig, axes = plt.subplots(2, 2, figsize=(11.9, 8.4), sharex=True, sharey=True)
    for ax, year in zip(axes.flat, YEARS):
        market = subset[subset.year.eq(year)]
        draw_market(ax, market, detailed=True)
        demand = float(market.demand_gw.iloc[0])
        import_share = float(market.loc[
            market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
        ax.set_title(f"{year}   demand {demand:.0f} GW   imports {import_share:.0%}",
                     fontsize=12, pad=7)
        ax.tick_params(axis="both", labelsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("USD/kW", fontsize=11)
    fig.supxlabel("Cumulative cleared supply per 1 GW of demand [GW]",
                  y=0.15, fontsize=11)
    regions, components = legends()
    fig.legend(handles=regions, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.005), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.1)
    fig.legend(handles=components, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.045), frameon=True, framealpha=0.95,
               fontsize=9.3, columnspacing=0.9)
    fig.suptitle(f"{LABELS[importer]}: cleared-supply merit order at the median market price",
                 fontsize=15, y=0.995)
    fig.text(0.5, 0.115,
             "Blocks are ordered by landed marginal cost; width is actual cleared flow / demand. MC labels are manufacturing cost.",
             ha="center", fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.21,
                        hspace=0.43, wspace=0.25)
    fig.savefig(DATA / f"median_merit_order_{importer}.png", dpi=220, bbox_inches="tight")
    fig.savefig(DATA / f"median_merit_order_{importer}.pdf", bbox_inches="tight")
    pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_demand_price_levels(data: pd.DataFrame) -> None:
    """Show actual cleared GW at each supplier's delivered offer, by market."""
    fig, axes = plt.subplots(len(REGIONS), len(YEARS), figsize=(17.0, 15.0),
                             sharey=True)
    block_rows = []
    for i, importer in enumerate(REGIONS):
        for j, year in enumerate(YEARS):
            ax = axes[i, j]
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            market = market.sort_values(["delivered_offer_usd_per_kw", "exporter"])
            assert market.candidate.nunique() == 1
            demand = float(market.demand_gw.iloc[0])
            price = float(market.market_price_usd_per_kw.iloc[0])
            left = 0.0
            for row in market.itertuples(index=False):
                width = float(row.flow_gw)
                delivered = float(row.delivered_offer_usd_per_kw)
                assert abs(delivered - row.offer_usd_per_kw - row.shipping_usd_per_kw) < 1e-4
                ax.bar(left, delivered, width=width, align="edge",
                       color=COLORS[row.exporter], alpha=0.83,
                       edgecolor="white", linewidth=0.75, zorder=2)
                if width / demand >= 0.16:
                    ax.text(left + width / 2, min(delivered * 0.43, 145),
                            f"{LABELS[row.exporter]}\n{width:.0f} GW",
                            ha="center", va="center", fontsize=8.3,
                            fontweight="bold", color="#202020", zorder=3)
                block_rows.append({
                    "importer": importer, "year": year, "candidate": row.candidate,
                    "exporter": row.exporter, "x_left_gw": left,
                    "x_right_gw": left + width, "flow_gw": width,
                    "demand_gw": demand, "offer_usd_per_kw": row.offer_usd_per_kw,
                    "shipping_usd_per_kw": row.shipping_usd_per_kw,
                    "delivered_offer_usd_per_kw": delivered,
                    "market_price_usd_per_kw": price,
                })
                left += width
            assert abs(left - demand) < 0.01
            ax.hlines(price, 0, demand, color="#202020", linestyle="-.",
                      linewidth=1.6, zorder=4)
            ax.text(demand * 1.025, price, f"{price:.0f}", va="center",
                    fontsize=8.5, color="#202020",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
            ax.set_xlim(0, demand * 1.17)
            ax.set_ylim(0, 400)
            ax.set_xticks([0, demand / 2, demand],
                          ["0", f"{demand / 2:g}", f"{demand:.0f}"])
            ax.set_yticks([0, 100, 200, 300, 400])
            ax.tick_params(axis="both", labelsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(str(year), fontsize=13, pad=7)
            if j == 0:
                ax.set_ylabel(f"{LABELS[importer]}\nUSD/kW", fontsize=10.5,
                              rotation=0, labelpad=34, va="center")
    pd.DataFrame(block_rows).to_csv(DATA_CSV / "cleared_demand_price_levels.csv",
                                    index=False)
    fig.suptitle("Cleared demand by supplier delivered-offer level",
                 fontsize=16, y=0.987)
    fig.supxlabel("Cumulative served demand [GW] (each panel has its own scale)",
                  y=0.108, fontsize=11)
    fig.text(0.5, 0.077,
             "One accepted equilibrium per market, selected at the median price across 27 profiles.\n"
             "Block width = cleared supplier flow; block height = offer + shipping. Dash-dot line = uniform market price.",
             ha="center", fontsize=9.5, color="#444444")
    fig.legend(handles=[Patch(facecolor=COLORS[r], edgecolor="#888888",
                              linewidth=0.4, label=LABELS[r]) for r in REGIONS],
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.018),
               frameon=True, framealpha=0.95, fontsize=9.5, columnspacing=1.1)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.145,
                        hspace=0.47, wspace=0.24)
    save(fig, "cleared_demand_price_levels_24_markets")


def main() -> None:
    DATA_CSV.mkdir(exist_ok=True)
    data = prepare()
    plot_overview(data)
    plot_demand_price_levels(data)
    with PdfPages(DATA / "median_merit_order_all_regions.pdf") as pages:
        for importer in REGIONS:
            plot_region(data, importer, pages)
    print("Created normalized merit-order figures and the actual-GW delivered-offer chart.")


if __name__ == "__main__":
    main()
