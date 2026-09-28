"""Regenerate five Results figures and all cited numbers from the 14 retained profiles.

The former Eq 1 (ch-af-apac-eu-row-us/pf080_k100_a030) is excluded. Price bands
include the regional manufacturing-cost lines of the original price figure.

Read-only inputs: existing Stage-2 observation CSVs, screening verdicts, planner
workbook. Outputs are written only beside this script.
"""

from __future__ import annotations

import math
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from plots.plot_median_regional_capacity_pathway import plot_from_observations
from scripts import analyze_clean_stage2_ranges as clean
from scripts import analyze_equilibrium_ranges as base

CSV = ROOT / "outputs/clean_stage2_factorial_20260923_123037/statistical_analysis/csv"
SCREEN = ROOT / "outputs/eq1_deviation_check_20260928_104717/screening_summary.csv"
YEARS = tuple(base.MARKET_YEARS)
REGIONS = tuple(base.PAPER_REGION_ORDER)
STEMS = (
    "capacity_price_pathway_equilibria",
    "equilibrium_bands_prices_by_region",
    "equilibrium_bands_capacity_by_region",
    "median_regional_capacity_pathway",
    "welfare_cs_ps_relative_boxplots_response",
)


def number(value, digits=1):
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:,.{digits}f}"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|"] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows
    ]


def retained():
    ids = [s.strip() for s in (OUT / "retained_profiles.txt").read_text(
        encoding="utf-8").splitlines() if s.strip() and not s.startswith("#")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Retained list is empty or contains duplicates")
    verdicts = pd.read_csv(SCREEN)
    passed = set(verdicts.loc[
        verdicts.verdict == "equilibrium at 1% tolerance", "profile"])
    if len(passed) != 15:
        raise ValueError(f"Expected 15 passing screening verdicts, got {len(passed)}")
    if len(ids) == 15 and set(ids) != passed:
        raise ValueError(f"Provided 15 IDs disagree with screening: missing={passed-set(ids)}, extra={set(ids)-passed}")
    if not set(ids) <= passed:
        raise ValueError(f"Retained list includes rejected or unknown profiles: {set(ids)-passed}")
    return ids


def load(filename, ids, keys, expected):
    source = pd.read_csv(CSV / filename)
    frame = source[source.candidate.isin(ids)].copy()
    if set(frame.candidate) != set(ids):
        raise ValueError(f"{filename}: candidate set differs from retained list")
    if len(frame) != len(ids) * expected:
        raise ValueError(f"{filename}: expected {len(ids)*expected} rows, got {len(frame)}")
    if frame.duplicated(keys).any():
        raise ValueError(f"{filename}: duplicate records")
    return frame


def inputs(ids):
    n = len(REGIONS) * len(YEARS)
    return {
        "metrics": load("candidate_metrics.csv", ids, ["candidate"], 1),
        "prices": load("price_observations.csv", ids, ["candidate", "region", "year"], n),
        "capacity": load("capacity_observations.csv", ids, ["candidate", "region", "year"], n),
        "system": load("system_metrics.csv", ids, ["candidate", "year"], len(YEARS)),
        "regional": load("regional_market_metrics.csv", ids, ["candidate", "region", "year"], n),
        "relative": load("welfare_relative_observations.csv", ids,
                         ["candidate", "region", "component"], len(REGIONS)*2),
        "welfare": load("welfare_level_observations.csv", ids,
                        ["candidate", "region"], len(REGIONS)),
    }


def horizon_data(system, metrics):
    cap = system.pivot(index="candidate", columns="year", values="total_capacity_gw")
    price = system.pivot(index="candidate", columns="year", values="demand_weighted_price_usd_per_kw")
    avg_cap = sum(base.HORIZON_WEIGHTS[y] * cap[y] for y in YEARS)
    avg_price = sum(base.HORIZON_WEIGHTS[y] * price[y] for y in YEARS)
    out = metrics[["candidate", "price_factor", "capacity_weight", "damping",
                   "average_total_capacity_2025_2040_gw",
                   "average_demand_weighted_price_2025_2040_usd_per_kw"]].copy()
    out = out.set_index("candidate")
    if not np.allclose(out["average_total_capacity_2025_2040_gw"].loc[avg_cap.index],
                       avg_cap, atol=1e-5, rtol=0):
        raise ValueError("Capacity horizon averages do not reconcile")
    if not np.allclose(out["average_demand_weighted_price_2025_2040_usd_per_kw"].loc[avg_price.index],
                       avg_price, atol=1e-5, rtol=0):
        raise ValueError("Price horizon averages do not reconcile")
    out["average_total_capacity_2025_2040_gw"] = avg_cap
    out["average_demand_weighted_price_2025_2040_usd_per_kw"] = avg_price
    return out.reset_index()


def highlights(horizon):
    x = "average_total_capacity_2025_2040_gw"
    y = "average_demand_weighted_price_2025_2040_usd_per_kw"
    eq1 = horizon.sort_values([y, "candidate"], ascending=[False, True]).iloc[0]
    eq2 = horizon.sort_values([x, "candidate"], ascending=[False, True]).iloc[0]
    fallback = eq1.candidate == eq2.candidate
    if fallback:
        eq2 = horizon.sort_values([x, "candidate"], ascending=[True, True]).iloc[0]
    return eq1, eq2, fallback


def global_prices(prices, regional, system, planner):
    joined = prices.merge(regional[["candidate", "region", "year", "demand_gw"]],
                          on=["candidate", "region", "year"], validate="one_to_one")
    joined["weighted"] = joined.price_usd_per_kw * joined.demand_gw
    grouped = joined.groupby(["candidate", "year"])[["weighted", "demand_gw"]].sum()
    profile = (grouped.weighted / grouped.demand_gw).rename("price").reset_index()
    check = profile.merge(system[["candidate", "year", "demand_weighted_price_usd_per_kw"]],
                          on=["candidate", "year"], validate="one_to_one")
    if not np.allclose(check.price, check.demand_weighted_price_usd_per_kw,
                       atol=1e-5, rtol=0):
        raise ValueError("Global demand-weighted price does not reconcile")
    spread = regional.groupby(["region", "year"]).demand_gw.agg(lambda x: x.max()-x.min())
    if (spread > 1e-5).any():
        raise ValueError("Regional demand differs across profiles")
    weights = regional.groupby(["region", "year"], as_index=False).demand_gw.median()
    p = planner.merge(weights, on=["region", "year"], validate="one_to_one")
    p["weighted"] = p.planner_price_usd_per_kw * p.demand_gw
    pg = p.groupby("year")[["weighted", "demand_gw"]].sum()
    pg = (pg.weighted / pg.demand_gw).rename("planner")
    summary = profile.groupby("year").price.agg(["min", "median", "max"]).join(pg)
    summary["gap"] = summary["median"] - summary["planner"]
    summary["markup"] = 100 * summary["gap"] / summary["planner"]
    return summary


def welfare_data(welfare, system):
    planner_source = pd.read_csv(CSV / "welfare_level_comparison.csv")
    if set(planner_source.region) != set(REGIONS):
        raise ValueError("Planner welfare comparison has unexpected regions")
    planner_source = planner_source.set_index("region").reindex(REGIONS)
    planner = (planner_source.planner_cs_billion_usd_pv
               + planner_source.planner_ps_billion_usd_pv
               - planner_source.planner_capacity_cost_billion_usd_pv)
    planner_total = float(planner.sum())
    if round(planner_total, 1) != 10114.7:
        raise ValueError(f"Planner total does not round to 10,114.7: {planner_total}")
    planner.loc["total"] = planner_total
    w = welfare.copy()
    w["value"] = (w.consumer_surplus_billion_usd_pv
                  + w.producer_surplus_billion_usd_pv
                  - w.capacity_cost_billion_usd_pv)
    wide = w.pivot(index="candidate", columns="region", values="value").reindex(columns=REGIONS)
    wide["total"] = wide[list(REGIONS)].sum(axis=1)
    share = system.loc[system.year == 2040].set_index("candidate").china_capacity_share
    return wide, planner, share.loc[wide.index]


def selected_profile(candidate):
    status_path = ROOT / "outputs/clean_stage2_factorial_20260923_123037" / candidate / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    sweep_path = ROOT / Path(status["selected_profile"].replace("\\", "/"))
    return json.loads(sweep_path.read_text(encoding="utf-8"))["ending_profile"]


def manufacturing_costs(ids):
    """Domestic offers equal c_man_t and are identical across profiles."""
    frames = []
    for candidate in ids:
        offers = pd.DataFrame(selected_profile(candidate)["strategy"]["p_offer"])
        offers = offers[(offers.exporter == offers.importer)
                        & offers.time.isin([str(y) for y in YEARS])]
        frames.append(offers.assign(candidate=candidate))
    offers = pd.concat(frames)
    spread = offers.groupby(["exporter", "time"]).value.agg(lambda v: v.max() - v.min())
    if (spread > 1e-6).any():
        raise ValueError("Manufacturing costs differ across profiles")
    cost = offers.groupby(["exporter", "time"], as_index=False).value.first()
    return pd.DataFrame({"region": cost.exporter, "year": cost.time.astype(int),
                         "cost_usd_per_kw": cost.value})


def figures(data, horizon, eq1, eq2, planner, costs):
    base.apply_plot_style()
    ids = set(horizon.candidate)
    highest = max(float(data["prices"].price_usd_per_kw.max()),
                  float(planner.planner_price_usd_per_kw.max()),
                  float(costs.cost_usd_per_kw.max()))
    ymax = 50 * (math.floor(highest/50) + 1)
    base.plot_horizon_capacity_price_equilibria(
        horizon, OUT, highlighted={
            eq1.candidate: ("Eq 1", (0, 14)),
            eq2.candidate: ("Eq 2", (-4, 16)),
        })
    clean.plot_regional_bands(
        data["prices"], "price_usd_per_kw", YEARS, "Price [$/kW]",
        "#A83232", OUT, "equilibrium_bands_prices_by_region",
        display_candidates=ids, shared_ymax=ymax,
        show_individual_outcomes=False, planner_prices=planner,
        cost_lines=costs)
    clean.plot_regional_bands(
        data["capacity"], "capacity_gw", base.CAPACITY_YEARS, "Capacity [GW]",
        "#7570B3", OUT, "equilibrium_bands_capacity_by_region",
        display_candidates=ids, show_individual_outcomes=False,
        stacked_legend=True)
    plot_from_observations(data["capacity"], data["system"], OUT)
    clean.plot_welfare_difference_boxplots(
        data["relative"], OUT,
        value_column="change_percent_of_planner_regional_welfare",
        xlabel="[%]", legend_prefix="Relative",
        stem="welfare_cs_ps_relative_boxplots_response",
        symmetric_axis=True, abbreviate_components=True, compact=True)
    for stem in STEMS:
        for ext in ("pdf", "png"):
            path = OUT / f"{stem}.{ext}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing figure: {path}")
    return ymax


def write_numbers(ids, data, horizon, eq1, eq2, fallback, planner, ymax, costs):
    x = horizon.average_total_capacity_2025_2040_gw.to_numpy(float)
    y = horizon.average_demand_weighted_price_2025_2040_usd_per_kw.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (slope*x+intercept))**2) / np.sum((y-y.mean())**2)
    gp = global_prices(data["prices"], data["regional"], data["system"], planner)
    rp = data["prices"].merge(planner, on=["region", "year"], validate="many_to_one")
    welfare, planner_welfare, china_share = welfare_data(data["welfare"], data["system"])
    delta = welfare[list(REGIONS)].subtract(planner_welfare[list(REGIONS)], axis=1)
    loss = 100 * (planner_welfare["total"] - welfare.total) / planner_welfare["total"]
    other_delta = delta[["eu", "us", "af", "row"]].sum(axis=1)
    exporters_delta = delta[["ch", "apac"]].sum(axis=1)
    other_welfare = welfare[["eu", "us", "af", "row"]].sum(axis=1)
    rho1 = spearmanr(welfare.ch, other_welfare).statistic
    rho2 = spearmanr(welfare.total, china_share).statistic
    rank = welfare.total.rank(ascending=False, method="min").astype(int)
    lines = [
        f"# Results from {len(ids)} provisionally retained profiles", "",
        f"The {len(ids)} IDs in retained_profiles.txt are the 15 passing deviation-screen verdicts without the former Eq 1 (ch-af-apac-eu-row-us/pf080_k100_a030). Profile statistics use filtered observation-level CSV records, with selected-sweep JSON used for the offer/cost checks at the end. Planner prices use the regions/lam column of outputs/llp_planner/llp_planner_results.xlsx.", "",
        "## Multiple equilibria", "",
        f"OLS slope of horizon-average demand-weighted price on horizon-average capacity: **{number(slope*100,1)} USD/kW per 100 GW**, R² **{number(r2,2)}**, n = **{len(ids)}**. Horizon weights: 2025 and 2040 = 1/6 each; 2030 and 2035 = 1/3 each.", "",
    ]
    rows = []
    for label, row in (("Eq 1: highest price", eq1),
                       ("Eq 2: lowest capacity" if fallback else "Eq 2: highest capacity", eq2)):
        rows.append([label, row.candidate,
                     number(row.average_total_capacity_2025_2040_gw,0),
                     number(row.average_demand_weighted_price_2025_2040_usd_per_kw,0),
                     number(row.price_factor,2), number(row.capacity_weight,2),
                     number(row.damping,2), f"{rank.loc[row.candidate]}/{len(ids)}"])
    lines += table(["Highlight", "Profile", "Capacity GW", "Price USD/kW",
                    "Price factor", "Capacity weight", "Damping", "Welfare rank"], rows)
    lines += ["", "## Market-clearing prices", "",
              f"Price-band shared y-axis ceiling: **{number(ymax,0)} USD/kW**, the next multiple of 50 above the retained maximum.", "",
              "Global demand-weighted price by year (USD/kW; markup in percent):", ""]
    lines += table(["Year", "Min", "Median", "Max", "Planner", "Median gap", "Markup"],
                   [[str(year), number(v["min"],0), number(v["median"],0),
                     number(v["max"],0), number(v["planner"],0), number(v["gap"],0),
                     number(v["markup"],0)+"%"] for year, v in gp.iterrows()])
    lines += ["", "Regional prices (USD/kW; ratio is median/planner):", ""]
    rows = []
    for region in REGIONS:
        for year in YEARS:
            sub = rp[(rp.region == region) & (rp.year == year)]
            v = sub.price_usd_per_kw
            p = float(sub.planner_price_usd_per_kw.iloc[0])
            rows.append([region.upper(), str(year), number(v.min(),0),
                         number(v.median(),0), number(v.quantile(.9),0),
                         number(v.max(),0), number(p,0), number(v.median()/p,2),
                         number(v.min()-p,0)])
    lines += table(["Region", "Year", "Min", "Median", "P90", "Max",
                    "Planner", "Median/planner", "Min - planner"], rows)
    lines += ["", "Regional median price versus own manufacturing cost (USD/kW); counts are profiles within 5 USD/kW, below and above cost:", ""]
    rows = []
    pc = data["prices"].merge(costs, on=["region", "year"], validate="many_to_one")
    pc["gap"] = pc.price_usd_per_kw - pc.cost_usd_per_kw
    for region in REGIONS:
        for year in YEARS:
            g = pc[(pc.region == region) & (pc.year == year)]
            rows.append([region.upper(), str(year), number(g.price_usd_per_kw.median(),1),
                         number(g.cost_usd_per_kw.iloc[0],1), number(g.gap.median(),1),
                         number(g.gap.min(),1), number(g.gap.max(),1),
                         int((g.gap.abs() <= 5).sum()), int((g.gap < -5).sum()),
                         int((g.gap > 5).sum())])
    lines += table(["Region", "Year", "Median price", "Cost", "Median gap", "Min gap",
                    "Max gap", "Within 5", "Below", "Above"], rows)
    lines += ["", "China and APAC median prices versus planner (USD/kW; gaps use unrounded values):", ""]
    rows = []
    for year in YEARS:
        row = [str(year)]
        for region in ("ch", "apac"):
            sub = rp[(rp.region == region) & (rp.year == year)]
            v = float(sub.price_usd_per_kw.median())
            p = float(sub.planner_price_usd_per_kw.iloc[0])
            row.extend([number(v,1), number(p,1), number(v-p,1)])
        rows.append(row)
    lines += table(["Year", "China median", "China planner", "China gap",
                    "APAC median", "APAC planner", "APAC gap"], rows)
    lines += ["", "Median regional exports (GW):", ""]
    rows = []
    for year in YEARS:
        sub = data["regional"][data["regional"].year == year]
        rows.append([str(year),
                     number(sub.loc[sub.region == "ch", "exports_gw"].median(),1),
                     number(sub.loc[sub.region == "apac", "exports_gw"].median(),1)])
    lines += table(["Year", "China", "APAC"], rows)
    trade = data["system"].loc[data["system"].year == 2040, "cross_border_trade_gw"].median()
    lines += ["", f"Median 2040 cross-border trade: **{number(trade,1)} GW**.", "",
              "## Welfare distribution", "",
              f"Planner total: **{number(planner_welfare['total'],1)} billion USD** (CS + PS - capacity costs).", ""]
    rows = []
    for region in (*REGIONS, "total"):
        v = welfare[region]
        rows.append([region.upper() if region != "total" else "Total",
                     number(planner_welfare[region],1), number(v.min(),1),
                     number(v.median(),1), number(v.max(),1)])
    lines += table(["Region", "Planner", "Minimum", "Median", "Maximum"], rows)
    lines += ["", f"Global welfare loss: min **{number(loss.min(),1)}%**, median **{number(loss.median(),1)}%**, max **{number(loss.max(),1)}%**.", "",
              "Number of profiles with regional welfare above planner:", ""]
    lines += table(["Region", "Gaining profiles"],
                   [[r.upper(), f"{int((delta[r] > 0).sum())}/{len(ids)}"] for r in REGIONS])
    lines += ["", f"Median joint EU+US+AF+ROW loss: **{number(-other_delta.median(),1)} billion USD**. Median joint CH+APAC gain: **{number(exporters_delta.median(),1)} billion USD**. These are medians of sums within each profile.",
              f"Spearman rho(China welfare, other-four welfare) = **{number(rho1,2)}**. rho(global welfare, China 2040 capacity share) = **{number(rho2,2)}**.", "",
              "Relative CS and PS change as a percentage of each region's planner welfare:", ""]
    rows = []
    for region in REGIONS:
        row = [region.upper()]
        for component in ("Consumer surplus", "Producer surplus"):
            v = data["relative"].loc[
                (data["relative"].region == region)
                & (data["relative"].component == component),
                "change_percent_of_planner_regional_welfare"]
            row.extend([number(v.min(),1), number(v.median(),1), number(v.max(),1)])
        rows.append(row)
    lines += table(["Region", "CS min", "CS median", "CS max",
                    "PS min", "PS median", "PS max"], rows)
    lines += ["", "## Manufacturing capacity", "", "2040 regional capacity (GW):", ""]
    rows = []
    for region in REGIONS:
        v = data["capacity"].loc[
            (data["capacity"].region == region) & (data["capacity"].year == 2040),
            "capacity_gw"]
        rows.append([region.upper(), number(v.min(),1), number(v.median(),1),
                     number(v.max(),1)])
    lines += table(["Region", "Minimum", "Median", "Maximum"], rows)
    rows = []
    for year in YEARS:
        s = data["system"][data["system"].year == year]
        idle = (s.total_capacity_gw - s.total_demand_gw).median()
        ch = data["capacity"].loc[
            (data["capacity"].region == "ch") & (data["capacity"].year == year),
            "capacity_gw"].median()
        rows.append([str(year), number(idle,1), number(ch,1)])
    lines += ["", "Median total capacity minus demand, and median China capacity (GW):", ""]
    lines += table(["Year", "Capacity - demand", "China capacity"], rows)
    lines += ["", "The stacked capacity figure sums regional medians and does not represent one profile. Capacity-minus-demand above is computed within each profile before taking the median.", ""]
    (OUT / "numbers.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ids = retained()
    data = inputs(ids)
    horizon = horizon_data(data["system"], data["metrics"])
    eq1, eq2, fallback = highlights(horizon)
    planner = clean.load_planner_prices(clean.DEFAULT_PLANNER_RESULTS, YEARS)
    costs = manufacturing_costs(ids)
    ymax = figures(data, horizon, eq1, eq2, planner, costs)
    write_numbers(ids, data, horizon, eq1, eq2, fallback, planner, ymax, costs)
    print(f"Retained {len(ids)}; Eq 1={eq1.candidate}; Eq 2={eq2.candidate}")
    print(f"Created {len(STEMS)} figure pairs, numbers.md, shared price ymax={ymax:g}")


if __name__ == "__main__":
    main()
