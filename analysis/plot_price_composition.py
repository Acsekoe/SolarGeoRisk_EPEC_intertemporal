"""Visualize one-GW supply mix and market-price accounting.

The aggregate view averages exact within-profile price identities across the
27 reported outcomes. Supplier-level panels use one internally consistent
accepted profile (C20), with flows normalized to 1 GW of destination demand.
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
OFFER_COLOR = "#929FA6"
SHADOW_COLOR = "#B8D99E"
PROFILE = "eu-us-af-row-apac-ch/pf080_k050_a040"  # C20
EPS_X = 1e-3  # run_clean_stage2_factorial.py
ACTIVE_GW = 0.1

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


def price_accounting(routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = routes.copy()
    data["share"] = data.flow_gw / data.demand_gw
    data["weighted_cost"] = data.share * data.manufacturing_cost_usd_per_kw
    data["weighted_ship"] = data.share * data.shipping_usd_per_kw
    data["weighted_markup"] = data.share * data.offer_minus_cost_usd_per_kw
    data["weighted_offer"] = data.share * data.delivered_offer_usd_per_kw
    data["weighted_flow_term"] = data.share * EPS_X * data.flow_gw
    grouped = (data.groupby(["candidate", "importer", "year"], as_index=False)
               .agg(market_price=("market_price_usd_per_kw", "first"),
                    manufacturing=("weighted_cost", "sum"),
                    shipping=("weighted_ship", "sum"),
                    offer_markup=("weighted_markup", "sum"),
                    delivered_offer=("weighted_offer", "sum"),
                    flow_term=("weighted_flow_term", "sum")))
    grouped["offer_capacity_dual"] = grouped.market_price - grouped.delivered_offer - grouped.flow_term
    grouped["offer_to_price_gap"] = grouped.market_price - grouped.delivered_offer
    error = (grouped.manufacturing + grouped.shipping + grouped.offer_markup
             + grouped.offer_capacity_dual + grouped.flow_term - grouped.market_price).abs().max()
    assert error < 1e-5, error
    assert grouped.offer_capacity_dual.min() > -1e-3, grouped.offer_capacity_dual.min()
    grouped.to_csv(DATA_CSV / "market_price_accounting_profiles.csv", index=False)
    means = (grouped.groupby(["importer", "year"], as_index=False)
             .agg(n_profiles=("candidate", "nunique"),
                  market_price=("market_price", "mean"),
                  manufacturing=("manufacturing", "mean"),
                  shipping=("shipping", "mean"),
                  offer_markup=("offer_markup", "mean"),
                  delivered_offer=("delivered_offer", "mean"),
                  offer_capacity_dual=("offer_capacity_dual", "mean"),
                  flow_term=("flow_term", "mean"),
                  offer_to_price_gap=("offer_to_price_gap", "mean")))
    assert (means.n_profiles == 27).all()
    means.to_csv(DATA_CSV / "market_price_accounting_mean.csv", index=False)
    return grouped, means


def plot_aggregate_accounting(routes: pd.DataFrame, means: pd.DataFrame) -> None:
    shares = (routes.groupby(["importer", "year", "exporter"], as_index=False)
              .agg(mean_share=("share_of_market", "mean")))
    fig = plt.figure(figsize=(13.0, 9.0))
    outer = fig.add_gridspec(2, 3, left=0.08, right=0.985, top=0.93,
                             bottom=0.18, wspace=0.30, hspace=0.37)
    for n, importer in enumerate(REGIONS):
        sub = outer[n // 3, n % 3].subgridspec(2, 1, height_ratios=[4.4, 1.0], hspace=0.04)
        ax = fig.add_subplot(sub[0])
        mix = fig.add_subplot(sub[1], sharex=ax)
        subset = means[means.importer.eq(importer)].set_index("year").reindex(YEARS)
        x = np.arange(len(YEARS), dtype=float)
        delivered = subset.delivered_offer.to_numpy(float)
        price = subset.market_price.to_numpy(float)
        ax.bar(x, delivered, width=0.57, color=OFFER_COLOR, alpha=0.86,
               edgecolor="white", linewidth=0.9, zorder=2)
        ax.bar(x, price - delivered, width=0.57, bottom=delivered,
               color=SHADOW_COLOR, alpha=0.80, hatch="///",
               edgecolor="#6D986E", linewidth=0.6, zorder=3)
        for i, year in enumerate(YEARS):
            cost = float(subset.loc[year, "manufacturing"])
            landed_cost = cost + float(subset.loc[year, "shipping"])
            ax.hlines(cost, i - 0.28, i + 0.28, color="#222222", linewidth=1.15, zorder=5)
            ax.hlines(landed_cost, i - 0.28, i + 0.28, color="#333333",
                      linestyle="--", linewidth=1.15, zorder=5)
            ax.text(i, price[i] + 4, f"{price[i]:.0f}", ha="center", va="bottom",
                    fontsize=9.5, color="#222222")
        ax.set_title(LABELS[importer], fontsize=15, pad=6)
        ax.set_ylim(0, 425)
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.spines[["top", "right"]].set_visible(False)
        if n % 3 == 0:
            ax.set_ylabel("Mean price [USD/kW]", fontsize=11)
        bottoms = np.zeros(len(YEARS))
        for exporter in REGIONS:
            values = np.array([
                shares[(shares.importer.eq(importer)) & shares.year.eq(year)
                       & shares.exporter.eq(exporter)].mean_share.iloc[0]
                for year in YEARS
            ])
            mix.bar(x, values, width=0.57, bottom=bottoms, color=COLORS[exporter],
                    alpha=0.78, edgecolor="white", linewidth=0.7)
            bottoms += values
        assert np.allclose(bottoms, 1, atol=1e-8)
        mix.set_ylim(0, 1)
        mix.set_yticks([])
        mix.set_xticks(x, YEARS)
        mix.tick_params(axis="x", labelsize=10, length=0)
        mix.spines[["top", "right", "left"]].set_visible(False)
        if n % 3 == 0:
            mix.set_ylabel("1 GW mix", fontsize=9)
    component_legend = [
        Patch(facecolor=OFFER_COLOR, label="Flow-weighted delivered offer"),
        Patch(facecolor=SHADOW_COLOR, hatch="///", edgecolor="#6D986E",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#222222", linewidth=1.1, label="Manufacturing cost"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.1,
               label="Manufacturing + shipping"),
    ]
    fig.legend(handles=component_legend, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.075), frameon=True, framealpha=0.95,
               fontsize=10, columnspacing=1.0)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.5,
                           label=LABELS[r]) for r in REGIONS]
    fig.legend(handles=region_legend, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), frameon=True, framealpha=0.95,
               fontsize=10, columnspacing=1.0)
    fig.suptitle("Market-price accounting and the supplier mix per 1 GW of demand",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.143, "Means across 27 reported profiles; line markers within the offer bar locate flow-weighted costs.",
             ha="center", fontsize=10, color="#444444")
    save(fig, "market_price_accounting_1gw_27_profiles")


def detailed_components(routes: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    selected = routes[routes.candidate.eq(PROFILE)].copy()
    selected = selected[selected.flow_gw.gt(ACTIVE_GW)].copy()
    selected["gw_per_1gw_demand"] = selected.flow_gw / selected.demand_gw
    selected["flow_term"] = EPS_X * selected.flow_gw
    selected["offer_capacity_dual"] = (selected.market_price_usd_per_kw
                                       - selected.delivered_offer_usd_per_kw
                                       - selected.flow_term)
    capacities = regions[regions.candidate.eq(PROFILE)][
        ["year", "region", "capacity_gw", "output_gw", "unused_capacity_gw"]
    ]
    selected = selected.merge(capacities, left_on=["year", "exporter"],
                              right_on=["year", "region"])
    assert selected.offer_capacity_dual.min() > -1e-3
    assert (selected.groupby(["year", "importer"])
            .gw_per_1gw_demand.sum().sub(1).abs().max() < 0.01)
    selected.to_csv(DATA_CSV / "normalized_supplier_price_components_C20.csv", index=False)
    return selected


def draw_market_blocks(
    ax: plt.Axes,
    market: pd.DataFrame,
    *,
    label_threshold: float,
    show_manufacturing_cost: bool,
) -> tuple[float, float]:
    """Draw cleared supplier flows, offer, shipping, and mu_offer in one bar."""
    market = market.sort_values("delivered_offer_usd_per_kw")
    price = float(market.market_price_usd_per_kw.iloc[0])
    left = 0.0
    for row in market.itertuples(index=False):
        width = float(row.gw_per_1gw_demand)
        offer = float(row.offer_usd_per_kw)
        shipping = float(row.shipping_usd_per_kw)
        delivered = offer + shipping
        assert abs(delivered - float(row.delivered_offer_usd_per_kw)) < 1e-4
        ax.bar(left, offer, width=width, align="edge", color=COLORS[row.exporter],
               alpha=0.80, edgecolor="white", linewidth=0.75, zorder=2)
        if shipping > 0.02:
            ax.bar(left, shipping, width=width, bottom=offer, align="edge",
                   color="#C7D1D5", hatch="xx", edgecolor="#67767C",
                   linewidth=0.3, zorder=3)
        offer_dual_gap = max(0.0, price - delivered)
        if offer_dual_gap > 0.02:
            ax.bar(left, offer_dual_gap, width=width, bottom=delivered, align="edge",
                   color=SHADOW_COLOR, alpha=0.48, hatch="///",
                   edgecolor="#6D986E", linewidth=0.4, zorder=4)
        if show_manufacturing_cost and width >= 0.025:
            ax.hlines(float(row.manufacturing_cost_usd_per_kw),
                      left + 0.003, left + width - 0.003,
                      color="#252525", linewidth=1.0, zorder=5)
        if width >= label_threshold:
            ax.text(left + width / 2, min(offer * 0.30, 62),
                    f"{LABELS[row.exporter]}\n{width:.2f}", ha="center", va="center",
                    fontsize=8.5 if label_threshold > 0.2 else 10.0,
                    color="#202020", fontweight="bold", zorder=6)
        left += width
    assert abs(left - 1.0) < 0.01
    ax.hlines(price, 0, 1.0, color="#202020", linestyle="-.", linewidth=1.5, zorder=6)
    ax.axvline(1.0, color="#303030", linestyle=":", linewidth=0.85, zorder=6)
    ax.text(1.025, price, f"{price:.0f}", va="center", ha="left", fontsize=9,
            zorder=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    return price, float(market.loc[market.exporter.ne(market.importer), "gw_per_1gw_demand"].sum())


def plot_supplier_offer_mu_overview(data: pd.DataFrame) -> None:
    """A single-equilibrium overview with supplier identity and price in each panel."""
    fig, axes = plt.subplots(len(REGIONS), len(YEARS), figsize=(16.0, 13.7),
                             sharex=True, sharey=True)
    ymax = max(400.0, float(data.market_price_usd_per_kw.max()) * 1.12)
    for i, importer in enumerate(REGIONS):
        for j, year in enumerate(YEARS):
            ax = axes[i, j]
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            _, import_share = draw_market_blocks(
                ax, market, label_threshold=0.19, show_manufacturing_cost=False)
            demand = float(market.demand_gw.iloc[0])
            ax.text(0.01, 0.98, f"{demand:.0f} GW demand  ·  {import_share:.0%} imports",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.8,
                    color="#414141")
            ax.set_xlim(0, 1.15)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_yticks([0, 100, 200, 300, 400])
            ax.tick_params(axis="both", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(str(year), fontsize=13, pad=7)
            if j == 0:
                ax.set_ylabel(f"{LABELS[importer]}\nUSD/kW", fontsize=11,
                              rotation=0, labelpad=34, va="center")
    fig.supxlabel("Cleared supplier flow per 1 GW of destination demand [GW]",
                  y=0.105, fontsize=12)
    fig.suptitle("Supplier offers, shipping and offer-quantity duals behind market prices — profile C20",
                 fontsize=16, y=0.985)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.5,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, 0.02), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.text(0.5, 0.075,
             "Colored height = supplier offer; crosshatching = shipping; diagonal hatching = offer-quantity dual.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.15,
                        hspace=0.30, wspace=0.18)
    save(fig, "supplier_offer_mu_1gw_C20_overview")


def draw_paired_market(ax: plt.Axes, market: pd.DataFrame, *, label_threshold: float) -> None:
    """Pair identical 1-GW supplier mixes, before and after adding mu_offer."""
    market = market.sort_values("delivered_offer_usd_per_kw")
    price = float(market.market_price_usd_per_kw.iloc[0])
    for offset, include_dual in ((0.0, False), (1.25, True)):
        left = offset
        for row in market.itertuples(index=False):
            width = float(row.gw_per_1gw_demand)
            offer = float(row.offer_usd_per_kw)
            shipping = float(row.shipping_usd_per_kw)
            ax.bar(left, offer, width=width, align="edge", color=COLORS[row.exporter],
                   alpha=0.80, edgecolor="white", linewidth=0.65, zorder=2)
            if shipping > 0.02:
                ax.bar(left, shipping, width=width, bottom=offer, align="edge",
                       color="#C7D1D5", hatch="xx", edgecolor="#67767C",
                       linewidth=0.3, zorder=3)
            if include_dual and row.offer_capacity_dual > 0.02:
                ax.bar(left, row.offer_capacity_dual, width=width,
                       bottom=offer + shipping, align="edge",
                       color=SHADOW_COLOR, alpha=0.48, hatch="///",
                       edgecolor="#6D986E", linewidth=0.4, zorder=4)
            if not include_dual and width >= label_threshold:
                ax.text(left + width / 2, min(offer * 0.30, 62),
                        f"{LABELS[row.exporter]}\n{width:.2f}", ha="center", va="center",
                        fontsize=8.5 if label_threshold >= 0.25 else 10,
                        color="#202020", fontweight="bold", zorder=5)
            left += width
        assert abs(left - (offset + 1.0)) < 0.01
        ax.axvline(offset + 1.0, color="#444444", linestyle=":", linewidth=0.7)
    ax.hlines(price, 0, 2.25, color="#202020", linestyle="-.", linewidth=1.5, zorder=6)
    ax.text(2.28, price, f"{price:.0f}", ha="left", va="center", fontsize=9,
            zorder=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_xlim(0, 2.43)
    ax.set_xticks([0.5, 1.75], ["Delivered offer", r"Delivered + $\mu_{\mathrm{offer}}$"])


def plot_paired_overview(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(REGIONS), len(YEARS), figsize=(18.0, 14.2),
                             sharex=True, sharey=True)
    ymax = max(400.0, float(data.market_price_usd_per_kw.max()) * 1.12)
    for i, importer in enumerate(REGIONS):
        for j, year in enumerate(YEARS):
            ax = axes[i, j]
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            draw_paired_market(ax, market, label_threshold=0.26)
            demand = float(market.demand_gw.iloc[0])
            imported = float(market.loc[market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
            ax.text(0.01, 0.98, f"{demand:.0f} GW demand  ·  {imported:.0%} imports",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8.7,
                    color="#414141")
            ax.set_ylim(0, ymax)
            ax.set_yticks([0, 100, 200, 300, 400])
            ax.tick_params(axis="both", labelsize=8)
            ax.tick_params(axis="x", labelbottom=True)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(str(year), fontsize=13, pad=7)
            if j == 0:
                ax.set_ylabel(f"{LABELS[importer]}\nUSD/kW", fontsize=11,
                              rotation=0, labelpad=34, va="center")
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.5,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle("Paired supplier offers and market-price composition per 1 GW — profile C20",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.075,
             "Both bars use the same cleared supplier flows; the second adds each exporter's offer-quantity dual.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.078, right=0.985, top=0.93, bottom=0.15,
                        hspace=0.39, wspace=0.19)
    save(fig, "paired_offer_mu_1gw_C20_overview")


def plot_paired_region(data: pd.DataFrame, importer: str, pages: PdfPages) -> None:
    subset = data[data.importer.eq(importer)]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), sharey=True)
    ymax = max(300.0, float(subset.market_price_usd_per_kw.max()) * 1.16)
    for ax, year in zip(axes.flat, YEARS):
        market = subset[subset.year.eq(year)]
        draw_paired_market(ax, market, label_threshold=0.12)
        demand = float(market.demand_gw.iloc[0])
        imported = float(market.loc[market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
        flows = " · ".join(f"{LABELS[row.exporter]} {row.flow_gw:.1f}"
                           for row in market.sort_values("delivered_offer_usd_per_kw").itertuples(index=False))
        ax.set_title(f"{year}   demand {demand:.0f} GW   imports {imported:.0%}",
                     fontsize=12, pad=7)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, -0.19, "Supplier flow [GW]: " + flows,
                transform=ax.transAxes, fontsize=9.0, color="#444444",
                ha="left", va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price [USD/kW]", fontsize=11)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.5,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle(f"{LABELS[importer]}: paired supplier offers and price composition per 1 GW — C20",
                 fontsize=15, y=0.995)
    fig.text(0.5, 0.11, "Each bar spans 1 GW of demand; supplier widths are identical in the two bars.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.22,
                        hspace=0.55, wspace=0.25)
    fig.savefig(DATA / f"paired_offer_mu_1gw_C20_{importer}.png", dpi=220, bbox_inches="tight")
    fig.savefig(DATA / f"paired_offer_mu_1gw_C20_{importer}.pdf", bbox_inches="tight")
    pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_compact_six_region(data: pd.DataFrame, years: tuple[int, ...], stem: str) -> None:
    """First-draft 2x3 layout with one or two fully decomposed market bars per region."""
    assert 1 <= len(years) <= 2
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), sharey=True)
    offsets = (0.0,) if len(years) == 1 else (0.0, 1.30)
    ymax = 440.0
    for ax, importer in zip(axes.flat, REGIONS):
        for offset, year in zip(offsets, years):
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            market = market.sort_values("delivered_offer_usd_per_kw")
            price = float(market.market_price_usd_per_kw.iloc[0])
            demand = float(market.demand_gw.iloc[0])
            import_share = float(market.loc[
                market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
            left = offset
            for row in market.itertuples(index=False):
                width = float(row.gw_per_1gw_demand)
                offer = float(row.offer_usd_per_kw)
                shipping = float(row.shipping_usd_per_kw)
                ax.bar(left, offer, width=width, align="edge", color=COLORS[row.exporter],
                       alpha=0.80, edgecolor="white", linewidth=0.7, zorder=2)
                if shipping > 0.02:
                    ax.bar(left, shipping, width=width, bottom=offer, align="edge",
                           color="#C7D1D5", hatch="xx", edgecolor="#67767C",
                           linewidth=0.3, zorder=3)
                if row.offer_capacity_dual > 0.02:
                    ax.bar(left, row.offer_capacity_dual, width=width,
                           bottom=offer + shipping, align="edge",
                           color=SHADOW_COLOR, alpha=0.48, hatch="///",
                           edgecolor="#6D986E", linewidth=0.4, zorder=4)
                if width >= 0.16:
                    ax.text(left + width / 2, min(offer * 0.30, 65),
                            f"{LABELS[row.exporter]}\n{width:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color="#202020", zorder=5)
                left += width
            assert abs(left - (offset + 1.0)) < 0.01
            ax.hlines(price, offset, offset + 1.0, color="#202020",
                      linestyle="-.", linewidth=1.55, zorder=6)
            ax.text(offset + 0.5, price + 5.5, f"{price:.0f}",
                    ha="center", va="bottom", fontsize=9.5, color="#202020")
            ax.text(offset + 0.5, ymax - 10,
                    f"D {demand:.0f} GW  ·  imports {import_share:.0%}",
                    ha="center", va="top", fontsize=8.5, color="#444444")
        ax.set_title(LABELS[importer], fontsize=15, pad=6)
        ax.set_xlim(0, offsets[-1] + 1.0)
        ax.set_ylim(0, ymax)
        ax.set_xticks([o + 0.5 for o in offsets], [str(y) for y in years])
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.tick_params(axis="both", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Price [USD/kW]", fontsize=11)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.55,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle("Market-price composition by destination and period — profile C20",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.075,
             "Each bar serves 1 GW: supplier-colored width shows flow; height shows offer + shipping + offer-quantity dual.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.16,
                        hspace=0.32, wspace=0.22)
    save(fig, stem)


def median_price_market_components(routes: pd.DataFrame) -> pd.DataFrame:
    """Choose the median-priced accepted equilibrium separately for each market-period."""
    prices = (routes.groupby(["candidate", "importer", "year"], as_index=False)
              .agg(market_price=("market_price_usd_per_kw", "first")))
    selection: list[dict[str, object]] = []
    for (importer, year), market_prices in prices.groupby(["importer", "year"]):
        assert len(market_prices) == 27
        median_price = float(market_prices.market_price.median())
        distance = (market_prices.market_price - median_price).abs()
        tied = market_prices[distance.le(distance.min() + 1e-4)].copy()
        mix = (routes[routes.importer.eq(importer) & routes.year.eq(year)]
               .pivot(index="candidate", columns="exporter", values="share_of_market")
               .reindex(columns=REGIONS))
        median_mix = mix.median(axis=0)
        mix_distance = mix.sub(median_mix).abs().sum(axis=1)
        tied["mix_distance"] = tied.candidate.map(mix_distance)
        winner = tied.sort_values(["mix_distance", "candidate"]).iloc[0]
        selection.append({
            "importer": importer, "year": year,
            "candidate": winner.candidate,
            "market_price_usd_per_kw": winner.market_price,
            "median_price_usd_per_kw": median_price,
            "supplier_mix_distance": winner.mix_distance,
            "n_equilibria": len(market_prices),
        })
    chosen = pd.DataFrame(selection)
    assert len(chosen) == len(REGIONS) * len(YEARS)
    assert (chosen.market_price_usd_per_kw - chosen.median_price_usd_per_kw).abs().max() < 1e-3
    chosen.to_csv(DATA_CSV / "median_price_market_selection.csv", index=False)
    selected = routes.merge(chosen[["importer", "year", "candidate"]],
                            on=["importer", "year", "candidate"], how="inner")
    selected = selected[selected.flow_gw.gt(ACTIVE_GW)].copy()
    selected["gw_per_1gw_demand"] = selected.flow_gw / selected.demand_gw
    selected["offer_capacity_dual"] = (
        selected.market_price_usd_per_kw - selected.delivered_offer_usd_per_kw
        - EPS_X * selected.flow_gw)
    assert selected.offer_capacity_dual.min() > -1e-3
    assert (selected.groupby(["importer", "year"]).gw_per_1gw_demand.sum()
            .sub(1).abs().max() < 0.01)
    selected.to_csv(DATA_CSV / "median_price_supplier_components.csv", index=False)
    return selected


def plot_median_four_bars(data: pd.DataFrame) -> None:
    """Six first-draft-style panels with four median-price equilibrium bars each."""
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.9), sharey=True)
    bar_width = 0.72
    for ax, importer in zip(axes.flat, REGIONS):
        tick_labels: list[str] = []
        for j, year in enumerate(YEARS):
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            market = market.sort_values("delivered_offer_usd_per_kw")
            assert not market.empty
            price = float(market.market_price_usd_per_kw.iloc[0])
            import_share = float(market.loc[
                market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
            tick_labels.append(f"{year}\nI {import_share:.0%}")
            left = j - bar_width / 2
            for row in market.itertuples(index=False):
                width = float(row.gw_per_1gw_demand) * bar_width
                offer = float(row.offer_usd_per_kw)
                shipping = float(row.shipping_usd_per_kw)
                ax.bar(left, offer, width=width, align="edge", color=COLORS[row.exporter],
                       alpha=0.80, edgecolor="white", linewidth=0.65, zorder=2)
                if shipping > 0.02:
                    ax.bar(left, shipping, width=width, bottom=offer, align="edge",
                           color="#C7D1D5", hatch="xx", edgecolor="#67767C",
                           linewidth=0.3, zorder=3)
                if row.offer_capacity_dual > 0.02:
                    ax.bar(left, row.offer_capacity_dual, width=width,
                           bottom=offer + shipping, align="edge",
                           color=SHADOW_COLOR, alpha=0.48, hatch="///",
                           edgecolor="#6D986E", linewidth=0.4, zorder=4)
                if row.gw_per_1gw_demand >= 0.30:
                    ax.text(left + width / 2, min(offer * 0.30, 62),
                            LABELS[row.exporter], ha="center", va="center",
                            fontsize=8.5, fontweight="bold", color="#202020", zorder=5)
                left += width
            assert abs(left - (j + bar_width / 2)) < 0.01
            ax.hlines(price, j - bar_width / 2, j + bar_width / 2,
                      color="#202020", linestyle="-.", linewidth=1.45, zorder=6)
            ax.text(j, price + 4, f"{price:.0f}", ha="center", va="bottom",
                    fontsize=9.5, color="#202020")
        ax.set_title(LABELS[importer], fontsize=15, pad=6)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0, 435)
        ax.set_xticks(np.arange(4), tick_labels)
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.tick_params(axis="both", labelsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Price [USD/kW]", fontsize=11)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.45,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle("Median-price equilibrium by destination and period",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.075,
             "Each 1 GW bar uses the median-priced outcome among 27 accepted profiles; I = import share.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.17,
                        hspace=0.36, wspace=0.22)
    save(fig, "median_price_composition_six_regions_all_periods")


def plot_median_vertical_components(data: pd.DataFrame) -> None:
    """Align offer, shipping and mu_offer as vertical layers in each market bar."""
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.9), sharey=True)
    records: list[dict[str, object]] = []
    bar_width = 0.64
    for ax, importer in zip(axes.flat, REGIONS):
        tick_labels: list[str] = []
        for j, year in enumerate(YEARS):
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            assert not market.empty
            price = float(market.market_price_usd_per_kw.iloc[0])
            demand = float(market.demand_gw.iloc[0])
            import_share = float(market.loc[
                market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
            tick_labels.append(f"{year}\nI {import_share:.0%}")
            bottom = 0.0
            record: dict[str, object] = {
                "importer": importer, "year": year,
                "candidate": market.candidate.iloc[0], "demand_gw": demand,
                "import_share": import_share,
                "market_price_usd_per_kw": price,
            }
            for exporter in REGIONS:
                route = market[market.exporter.eq(exporter)]
                contribution = float((route.gw_per_1gw_demand * route.offer_usd_per_kw).sum())
                share = float(route.gw_per_1gw_demand.sum())
                record[f"offer_contribution_{exporter}"] = contribution
                record[f"supplier_share_{exporter}"] = share
                if contribution > 0.02:
                    ax.bar(j, contribution, width=bar_width, bottom=bottom,
                           color=COLORS[exporter], alpha=0.80,
                           edgecolor="white", linewidth=0.65, zorder=2)
                    if contribution > 31 and share > 0.15:
                        ax.text(j, bottom + contribution / 2,
                                f"{LABELS[exporter]}\n{share:.0%}", ha="center",
                                va="center", fontsize=8.5, fontweight="bold",
                                color="#202020", zorder=5)
                bottom += contribution
            shipping = float((market.gw_per_1gw_demand * market.shipping_usd_per_kw).sum())
            mu_offer = float((market.gw_per_1gw_demand * market.offer_capacity_dual).sum())
            record["weighted_offer_usd_per_kw"] = bottom
            record["weighted_shipping_usd_per_kw"] = shipping
            record["weighted_mu_offer_usd_per_kw"] = mu_offer
            records.append(record)
            if shipping > 0.02:
                ax.bar(j, shipping, width=bar_width, bottom=bottom,
                       color="#C7D1D5", hatch="xx", edgecolor="#67767C",
                       linewidth=0.35, zorder=3)
            bottom += shipping
            if mu_offer > 0.02:
                ax.bar(j, mu_offer, width=bar_width, bottom=bottom,
                       color=SHADOW_COLOR, alpha=0.55, hatch="///",
                       edgecolor="#6D986E", linewidth=0.4, zorder=4)
            assert abs(bottom + mu_offer - price) < 0.5
            ax.hlines(price, j - bar_width / 2, j + bar_width / 2,
                      color="#202020", linestyle="-.", linewidth=1.45, zorder=6)
            ax.text(j, price + 4, f"{price:.0f}", ha="center", va="bottom",
                    fontsize=9.5, color="#202020")
        ax.set_title(LABELS[importer], fontsize=15, pad=6)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0, 435)
        ax.set_xticks(np.arange(4), tick_labels)
        ax.set_yticks([0, 100, 200, 300, 400])
        ax.tick_params(axis="both", labelsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Contribution to\nprice [USD/kW]", fontsize=11)
    components = pd.DataFrame(records)
    assert len(components) == len(REGIONS) * len(YEARS)
    components.to_csv(DATA_CSV / "median_price_vertical_components.csv", index=False)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.45,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle("Vertically stacked market-price components — median-price equilibria",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.087,
             "Colored height = supplier market share × quoted offer (including domestic supply), not the raw supplier offer.",
             ha="center", fontsize=10, color="#444444")
    fig.text(0.5, 0.066,
             "Shipping and offer-quantity dual are also flow-weighted; I = import share. Each bar uses one of 27 accepted profiles.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.925, bottom=0.17,
                        hspace=0.36, wspace=0.22)
    save(fig, "median_price_components_vertical_six_regions_all_periods")


def draw_median_supplier_bars(ax: plt.Axes, market: pd.DataFrame, *, detailed: bool,
                              bar_width: float = 0.67) -> None:
    """One per-unit offer/shipping/mu_offer bar for each active supplier."""
    order = {region: i for i, region in enumerate(REGIONS)}
    market = market.assign(region_order=market.exporter.map(order)).sort_values("region_order")
    price = float(market.market_price_usd_per_kw.iloc[0])
    positions = np.arange(len(market), dtype=float)
    for x, row in zip(positions, market.itertuples(index=False)):
        offer = float(row.offer_usd_per_kw)
        shipping = float(row.shipping_usd_per_kw)
        mu_offer = max(0.0, float(row.offer_capacity_dual))
        ax.bar(x, offer, width=bar_width, color=COLORS[row.exporter], alpha=0.80,
               edgecolor="white", linewidth=0.7, zorder=2)
        if shipping > 0.02:
            ax.bar(x, shipping, width=bar_width, bottom=offer,
                   color="#C7D1D5", hatch="xx", edgecolor="#67767C",
                   linewidth=0.35, zorder=3)
        if mu_offer > 0.02:
            ax.bar(x, mu_offer, width=bar_width, bottom=offer + shipping,
                   color=SHADOW_COLOR, alpha=0.55, hatch="///",
                   edgecolor="#6D986E", linewidth=0.4, zorder=4)
        assert abs(offer + shipping + mu_offer - price) < 0.5
        if detailed:
            ax.text(x, min(offer * 0.52, 120), f"offer {offer:.0f}",
                    ha="center", va="center", fontsize=9.0,
                    color="#202020", zorder=5)
            if mu_offer > 24:
                ax.text(x, offer + shipping + mu_offer / 2,
                        rf"$\mu$ {mu_offer:.0f}", ha="center", va="center",
                        fontsize=8.5, color="#202020", zorder=5)
            if shipping > 12:
                ax.text(x, offer + shipping / 2, f"{shipping:.0f}",
                        ha="center", va="center", fontsize=7.5,
                        color="#202020", zorder=5)
    ax.hlines(price, -bar_width / 2 - 0.08,
              len(market) - 1 + bar_width / 2 + 0.08,
              color="#202020", linestyle="-.", linewidth=1.5, zorder=6)
    ax.text(len(market) - 0.49, price, f"{price:.0f}", ha="left", va="center",
            fontsize=9, zorder=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set_xlim(-0.55, len(market) - 0.3)
    ax.set_xticks(positions, [f"{LABELS[row.exporter]}\n{row.gw_per_1gw_demand:.0%}"
                              for row in market.itertuples(index=False)])
    ax.set_ylim(0, 425)
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.spines[["top", "right"]].set_visible(False)


def plot_median_supplier_bars_overview(data: pd.DataFrame) -> None:
    """Six regions by four periods, with one vertical component bar per supplier."""
    fig, axes = plt.subplots(len(REGIONS), len(YEARS), figsize=(16.5, 14.0),
                             sharey=True)
    for i, importer in enumerate(REGIONS):
        for j, year in enumerate(YEARS):
            ax = axes[i, j]
            market = data[data.importer.eq(importer) & data.year.eq(year)]
            draw_median_supplier_bars(ax, market, detailed=False, bar_width=0.44)
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
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.5,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=9,
               bbox_to_anchor=(0.5, 0.015), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle("Supplier offers, shipping and offer-quantity duals by market and period",
                 fontsize=16, y=0.985)
    fig.text(0.5, 0.075,
             "Each active supplier has one bar; its height explains the common price. Percentages below bars are market shares.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=0.15,
                        hspace=0.42, wspace=0.22)
    save(fig, "median_supplier_price_bars_24_markets")


def plot_median_supplier_bars_region(data: pd.DataFrame, importer: str,
                                     pages: PdfPages) -> None:
    subset = data[data.importer.eq(importer)]
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.2), sharey=True)
    for ax, year in zip(axes.flat, YEARS):
        market = subset[subset.year.eq(year)]
        draw_median_supplier_bars(ax, market, detailed=True)
        demand = float(market.demand_gw.iloc[0])
        import_share = float(market.loc[
            market.exporter.ne(importer), "gw_per_1gw_demand"].sum())
        ax.set_title(f"{year}   demand {demand:.0f} GW   imports {import_share:.0%}",
                     fontsize=12, pad=7)
        ax.tick_params(axis="both", labelsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Price [USD/kW]", fontsize=11)
    region_legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.4,
                           label=LABELS[r]) for r in REGIONS]
    component_legend = [
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx", label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#202020", linestyle="-.", linewidth=1.5,
               label="Market price"),
    ]
    fig.legend(handles=region_legend + component_legend, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95,
               fontsize=9.5, columnspacing=1.0)
    fig.suptitle(f"{LABELS[importer]}: supplier-level price composition at the median price",
                 fontsize=15, y=0.995)
    fig.text(0.5, 0.10,
             "Active suppliers only; each bar is USD/kW, while the percentage beneath it is its share of served demand.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.20,
                        hspace=0.37, wspace=0.24)
    fig.savefig(DATA / f"median_supplier_price_bars_{importer}.png", dpi=220, bbox_inches="tight")
    fig.savefig(DATA / f"median_supplier_price_bars_{importer}.pdf", bbox_inches="tight")
    pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_region_detail(data: pd.DataFrame, importer: str, pdf_pages: PdfPages) -> None:
    subset = data[data.importer.eq(importer)]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharex=True, sharey=True)
    y_max = max(300.0, subset[["market_price_usd_per_kw", "delivered_offer_usd_per_kw"]].max().max(),
                (subset.manufacturing_cost_usd_per_kw + subset.shipping_usd_per_kw).max()) * 1.16
    for ax, year in zip(axes.flat, YEARS):
        market = subset[subset.year.eq(year)].sort_values("delivered_offer_usd_per_kw")
        _, import_share = draw_market_blocks(
            ax, market, label_threshold=0.12, show_manufacturing_cost=True)
        demand = float(market.demand_gw.iloc[0])
        flow_labels = " · ".join(f"{LABELS[row.exporter]} {row.flow_gw:.1f}"
                                 for row in market.itertuples(index=False))
        cap_labels = " · ".join(f"{LABELS[row.exporter]} {row.capacity_gw:.0f}"
                                for row in market.itertuples(index=False))
        ax.set_xlim(0, 1.13)
        ax.set_ylim(0, y_max)
        ax.set_title(f"{year}   demand {demand:.0f} GW   imports {import_share:.0%}",
                     fontsize=12, pad=8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="both", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.0, -0.16, "Supplier flow [GW]: " + flow_labels + "\nInstalled K [GW]: " + cap_labels,
                transform=ax.transAxes, fontsize=8.8, color="#444444",
                ha="left", va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price components [USD/kW]", fontsize=11.5)
    legend = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.5,
                    label=LABELS[r]) for r in REGIONS]
    legend.extend([
        Patch(facecolor="#C7D1D5", edgecolor="#67767C", hatch="xx",
              label="Shipping"),
        Patch(facecolor=SHADOW_COLOR, edgecolor="#6D986E", hatch="///",
              label=r"Offer-quantity dual $\mu_{\mathrm{offer}}$"),
        Line2D([0], [0], color="#222222", linewidth=1.0, label="Manufacturing cost"),
        Line2D([0], [0], color="#222222", linestyle="-.", linewidth=1.75,
               label="Market price"),
    ])
    fig.legend(handles=legend, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=0.95,
               fontsize=10, columnspacing=1.0)
    fig.suptitle(f"{LABELS[importer]}: supplier price composition per 1 GW of demand — profile C20",
                 fontsize=15, y=0.995)
    fig.text(0.5, 0.10,
             "Width: supplier flow per 1 GW demand. Colored height: offer; crosshatching: shipping; diagonal hatching: offer-quantity dual.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.22,
                        hspace=0.57, wspace=0.26)
    fig.savefig(DATA / f"price_composition_1gw_C20_{importer}.png", dpi=220, bbox_inches="tight")
    fig.savefig(DATA / f"price_composition_1gw_C20_{importer}.pdf", bbox_inches="tight")
    pdf_pages.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DATA_CSV.mkdir(exist_ok=True)
    routes = pd.read_csv(DATA_CSV / "route_observations.csv")
    regions = pd.read_csv(DATA_CSV / "regional_observations.csv")
    _, means = price_accounting(routes)
    plot_aggregate_accounting(routes, means)
    details = detailed_components(routes, regions)
    median_details = median_price_market_components(routes)
    plot_median_four_bars(median_details)
    plot_median_vertical_components(median_details)
    plot_median_supplier_bars_overview(median_details)
    with PdfPages(DATA / "median_supplier_price_bars_all_regions.pdf") as pages:
        for importer in REGIONS:
            plot_median_supplier_bars_region(median_details, importer, pages)
    plot_supplier_offer_mu_overview(details)
    plot_paired_overview(details)
    plot_compact_six_region(details, (2030, 2040), "compact_six_region_price_composition_C20_2030_2040")
    with PdfPages(DATA / "price_composition_1gw_C20_all_regions.pdf") as pages:
        for importer in REGIONS:
            plot_region_detail(details, importer, pages)
    with PdfPages(DATA / "paired_offer_mu_1gw_C20_all_regions.pdf") as pages:
        for importer in REGIONS:
            plot_paired_region(details, importer, pages)
    print("Created supplier-level, aligned, and supplier-width median-price figures, plus C20 and mean figures.")


if __name__ == "__main__":
    main()
