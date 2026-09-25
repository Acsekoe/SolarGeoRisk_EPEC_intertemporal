"""Describe China/APAC trade competition in the 27 reported Stage 2 profiles.

Run from the repository root: python analysis/china_apac_competition.py
All statistics are descriptive across algorithm branches, not sampling estimates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/clean_stage2_factorial_20260923_123037"
STATS = RUN / "statistical_analysis"
OUT = ROOT / "outputs/china_apac_competition"
OUT_CSV = OUT / "csv"
YEARS = (2025, 2030, 2035, 2040)
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
DESTINATIONS = ("eu", "us", "row", "af")
ACTIVE_GW = 0.1  # Treat smaller flows as numerical or economically negligible.


def valmap(rows: list[dict], *keys: str) -> dict[tuple, float]:
    return {tuple(row[key] for key in keys): float(row["value"]) for row in rows}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    OUT_CSV.mkdir(exist_ok=True)
    selected = pd.read_csv(STATS / "csv" / "candidate_metrics.csv")
    assert len(selected) == 27 and selected.candidate.nunique() == 27
    shipping_table = pd.read_excel(ROOT / "inputs/input_data_intertemporal.xlsx", sheet_name="c_ship")
    shipping = {
        (row.exporter, importer): float(getattr(row, importer))
        for row in shipping_table.itertuples(index=False)
        for importer in REGIONS
    }

    route_rows: list[dict] = []
    region_rows: list[dict] = []
    max_balance_error = 0.0
    max_capacity_error = 0.0
    for selected_row in selected.itertuples(index=False):
        path = RUN / selected_row.candidate / f"sweep_{int(selected_row.selected_sweep):03d}.json"
        profile = json.loads(path.read_text(encoding="utf-8"))["ending_profile"]
        flows = valmap(profile["market"]["trade_flows"], "exporter", "importer", "time")
        offers = valmap(profile["strategy"]["p_offer"], "exporter", "importer", "time")
        prices = valmap(profile["market"]["clearing_prices"], "region", "time")
        demand = valmap(profile["market"]["demand"], "region", "time")
        capacity = valmap(profile["capacities"], "player", "time")
        for year in YEARS:
            time = str(year)
            for importer in REGIONS:
                received = sum(flows[(exporter, importer, time)] for exporter in REGIONS)
                max_balance_error = max(max_balance_error, abs(received - demand[(importer, time)]))
            for exporter in REGIONS:
                output = sum(flows[(exporter, importer, time)] for importer in REGIONS)
                cap = capacity[(exporter, time)]
                max_capacity_error = max(max_capacity_error, output - cap)
                region_rows.append(
                    dict(candidate=selected_row.candidate, year=year, region=exporter,
                         capacity_gw=cap, output_gw=output,
                         unused_capacity_gw=cap - output,
                         domestic_gw=flows[(exporter, exporter, time)],
                         exports_gw=output - flows[(exporter, exporter, time)],
                         demand_gw=demand[(exporter, time)],
                         price_usd_per_kw=prices[(exporter, time)],
                         utilization=output / cap if cap > 0 else np.nan)
                )
                for importer in REGIONS:
                    offer = offers[(exporter, importer, time)]
                    route_rows.append(
                        dict(candidate=selected_row.candidate, year=year,
                             exporter=exporter, importer=importer,
                             flow_gw=flows[(exporter, importer, time)],
                             demand_gw=demand[(importer, time)],
                             market_price_usd_per_kw=prices[(importer, time)],
                             offer_usd_per_kw=offer,
                             delivered_offer_usd_per_kw=offer + shipping[(exporter, importer)],
                             manufacturing_cost_usd_per_kw=offers[(exporter, exporter, time)],
                             shipping_usd_per_kw=shipping[(exporter, importer)],
                             operating_margin_usd_per_kw=(prices[(importer, time)]
                                                           - offers[(exporter, exporter, time)]
                                                           - shipping[(exporter, importer)]))
                    )
    assert max_balance_error < 0.01, max_balance_error
    assert max_capacity_error < 0.01, max_capacity_error
    routes = pd.DataFrame(route_rows)
    regions = pd.DataFrame(region_rows)
    routes["share_of_market"] = routes.flow_gw / routes.demand_gw
    routes["active"] = routes.flow_gw > ACTIVE_GW
    routes["offer_minus_cost_usd_per_kw"] = (
        routes.offer_usd_per_kw - routes.manufacturing_cost_usd_per_kw
    )
    routes.to_csv(OUT_CSV / "route_observations.csv", index=False)
    regions.to_csv(OUT_CSV / "regional_observations.csv", index=False)

    # Market shares use arithmetic means so shares of all suppliers add to 100%.
    market = (routes.groupby(["year", "importer", "exporter"])
              .agg(mean_flow_gw=("flow_gw", "mean"), median_flow_gw=("flow_gw", "median"),
                   mean_share=("share_of_market", "mean"),
                   median_share=("share_of_market", "median"),
                   active_profiles=("active", "sum"))
              .reset_index())
    market.to_csv(OUT_CSV / "market_shares.csv", index=False)

    co_rows = []
    regional_by_candidate = regions.set_index(["candidate", "year", "region"])
    for (year, importer), part in routes[routes.importer.isin(DESTINATIONS)].groupby(["year", "importer"]):
        by_candidate = part.pivot(index="candidate", columns="exporter",
                                  values=["flow_gw", "share_of_market", "delivered_offer_usd_per_kw",
                                          "offer_minus_cost_usd_per_kw", "market_price_usd_per_kw"])
        ch = by_candidate[("flow_gw", "ch")]
        apac = by_candidate[("flow_gw", "apac")]
        both = (ch > ACTIVE_GW) & (apac > ACTIVE_GW)
        ch_only = (ch > ACTIVE_GW) & ~(apac > ACTIVE_GW)
        apac_only = (apac > ACTIVE_GW) & ~(ch > ACTIVE_GW)
        gap = (by_candidate[("delivered_offer_usd_per_kw", "apac")]
               - by_candidate[("delivered_offer_usd_per_kw", "ch")])
        apac_slack = pd.Series({candidate: regional_by_candidate.loc[(candidate, year, "apac"), "unused_capacity_gw"]
                                for candidate in by_candidate.index})
        ch_slack = pd.Series({candidate: regional_by_candidate.loc[(candidate, year, "ch"), "unused_capacity_gw"]
                              for candidate in by_candidate.index})
        co_rows.append(dict(
            year=year, importer=importer, ch_active=int((ch > ACTIVE_GW).sum()),
            apac_active=int((apac > ACTIVE_GW).sum()), both_active=int(both.sum()),
            ch_only=int(ch_only.sum()), apac_only=int(apac_only.sum()),
            neither=int((~(ch > ACTIVE_GW) & ~(apac > ACTIVE_GW)).sum()),
            apac_larger_when_both=int(((apac > ch) & both).sum()),
            apac_cheaper_delivered_when_both=int(((gap < -0.01) & both).sum()),
            delivered_apac_minus_ch_median_when_both=float(gap[both].median()) if both.any() else np.nan,
            median_ch_flow_when_both=float(ch[both].median()) if both.any() else np.nan,
            median_apac_flow_when_both=float(apac[both].median()) if both.any() else np.nan,
            apac_capacity_binding_when_both=int(((apac_slack < ACTIVE_GW) & both).sum()),
            ch_capacity_binding_when_both=int(((ch_slack < ACTIVE_GW) & both).sum()),
        ))
    co = pd.DataFrame(co_rows)
    co.to_csv(OUT_CSV / "co_supply.csv", index=False)

    exp = (regions[regions.region.isin(["ch", "apac"])]
           .groupby(["year", "region"])
           .agg(median_capacity_gw=("capacity_gw", "median"),
                median_output_gw=("output_gw", "median"),
                median_domestic_gw=("domestic_gw", "median"),
                median_exports_gw=("exports_gw", "median"),
                mean_exports_gw=("exports_gw", "mean"),
                median_unused_capacity_gw=("unused_capacity_gw", "median"),
                capacity_binding_profiles=("unused_capacity_gw", lambda x: int((x < ACTIVE_GW).sum())),
                median_utilization=("utilization", "median"))
           .reset_index())
    exp.to_csv(OUT_CSV / "exporter_paths.csv", index=False)

    active = routes[(routes.exporter.isin(["ch", "apac"])) &
                    (routes.exporter != routes.importer) & routes.active]
    offer_summary = (active.groupby(["year", "exporter", "importer"])
                     .agg(active_profiles=("candidate", "nunique"),
                          median_flow_gw=("flow_gw", "median"),
                          median_offer_usd_per_kw=("offer_usd_per_kw", "median"),
                          median_delivered_usd_per_kw=("delivered_offer_usd_per_kw", "median"),
                          median_offer_minus_cost=("offer_minus_cost_usd_per_kw", "median"),
                          median_operating_margin=("operating_margin_usd_per_kw", "median"),
                          below_cost_profiles=("offer_minus_cost_usd_per_kw", lambda x: int((x < -0.01).sum())))
                     .reset_index())
    offer_summary.to_csv(OUT_CSV / "active_route_offers.csv", index=False)

    # Four destination panels; averages describe the reported set, not one profile.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    labels = {"eu": "Europe", "us": "United States", "row": "Rest of World", "af": "Africa"}
    colors = {"ch": "#cf4446", "apac": "#278570", "other": "#c1c7cb"}
    for ax, importer in zip(axes.flat, DESTINATIONS):
        subset = market[market.importer.eq(importer)]
        ch = np.array([subset[(subset.year.eq(year)) & subset.exporter.eq("ch")].mean_share.iloc[0] for year in YEARS])
        apac = np.array([subset[(subset.year.eq(year)) & subset.exporter.eq("apac")].mean_share.iloc[0] for year in YEARS])
        other = 1 - ch - apac
        positions = np.arange(len(YEARS))
        ax.bar(positions, 100 * ch, color=colors["ch"])
        ax.bar(positions, 100 * apac, bottom=100 * ch, color=colors["apac"])
        ax.bar(positions, 100 * other, bottom=100 * (ch + apac), color=colors["other"])
        ax.set_title(labels[importer])
        ax.set_xticks(positions, YEARS)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Mean share of served demand (%)")
    axes[1, 0].set_ylabel("Mean share of served demand (%)")
    fig.legend(handles=[Patch(facecolor=colors["ch"], label="China"),
                        Patch(facecolor=colors["apac"], label="APAC"),
                        Patch(facecolor=colors["other"], label="Other suppliers")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("China and APAC shares in four destination markets (27 reported profiles)")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(OUT / "destination_market_shares.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "destination_market_shares.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Profiles: {len(selected)}; balance error: {max_balance_error:.2e} GW; capacity error: {max_capacity_error:.2e} GW")
    print("\nExporter paths (medians):")
    print(exp.round(2).to_string(index=False))
    print("\nDestination mean shares and active profiles (CH/APAC):")
    print(market[(market.importer.isin(DESTINATIONS)) & market.exporter.isin(["ch", "apac"])]
          .sort_values(["importer", "year", "exporter"])
          .assign(mean_share_percent=lambda x: 100 * x.mean_share)
          [["year", "importer", "exporter", "mean_flow_gw", "median_flow_gw", "mean_share_percent", "active_profiles"]]
          .round(1).to_string(index=False))
    print("\nCo-supply:")
    print(co.sort_values(["importer", "year"]).round(1).to_string(index=False))
    print("\nActive route offers:")
    print(offer_summary[offer_summary.importer.isin(DESTINATIONS)].sort_values(["importer", "year", "exporter"])
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
