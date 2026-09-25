"""Compare accepted fixed-cost-offer branches with the strategic-offer baseline."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import model_main as mm, run_gs
from model.data_prep import load_data_from_excel
from scripts.nested_market_audit import nested_economic_objective, solve_nested_market
from scripts.run_clean_stage2_factorial import clean_configuration, sha256
from scripts.run_corrected_cold_start import _build_fresh_state
from scripts.search_nested_equilibrium import _deserialize_state


DEFAULT_BASELINE = ROOT / "outputs" / "clean_stage2_factorial_20260923_123037"


def read_accepted(root: Path) -> list[dict]:
    with (root / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    accepted = []
    for row in rows:
        if row["status"] != "accepted":
            continue
        branch_root = root / row["sequence"] / row["branch"]
        sweep = int(row["selected_sweep"])
        path = (branch_root / "initialization.json" if sweep == 0 else
                branch_root / f"sweep_{sweep:03d}.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = payload["profile"] if sweep == 0 else payload["ending_profile"]
        accepted.append({"row": row, "profile": profile, "path": path})
    return accepted


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    baseline_root = args.baseline_root.resolve()
    counter_root = args.counterfactual_root.resolve()
    output_dir = (args.output_dir or (counter_root / "comparison")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text(encoding="utf-8"))
    counter_manifest = json.loads((counter_root / "manifest.json").read_text(encoding="utf-8"))
    base_protocol = baseline_manifest["protocol"]
    counter_protocol = counter_manifest["protocol"]
    if base_protocol["input_sha256"] != counter_protocol["input_sha256"]:
        raise ValueError("Baseline and counterfactual used different inputs")
    if float(base_protocol["terminal_salvage_fraction"]) != float(counter_protocol["terminal_salvage_fraction"]):
        raise ValueError("Baseline and counterfactual used different salvage fractions")
    if not counter_protocol.get("fix_offers_to_cost"):
        raise ValueError("Counterfactual did not fix offers to cost")
    input_path = Path(counter_protocol["input"])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    input_path = input_path.resolve()
    if sha256(input_path) != counter_protocol["input_sha256"]:
        raise ValueError("Corrected input hash mismatch")
    accepted_sets = {
        "strategic_offers": read_accepted(baseline_root),
        "fixed_cost_offers": read_accepted(counter_root),
    }
    branch_rows: list[dict] = []
    regional_rows: list[dict] = []
    welfare_rows: list[dict] = []
    for scenario, accepted in accepted_sets.items():
        if not accepted:
            continue
        fixed = scenario == "fixed_cost_offers"
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_data_from_excel(str(input_path), params_region_sheet="params_region_new")
            cfg = clean_configuration(
                input_path, float(counter_protocol["terminal_salvage_fraction"]),
                fix_offers_to_cost=fixed,
            )
            run_gs._apply_data_overrides(data, cfg)
        template = _build_fresh_state(data, period_specific_cost_offers=fixed)
        for item in accepted:
            row, profile = item["row"], item["profile"]
            state = _deserialize_state(profile["strategy"], data, template)
            market, diagnostics = solve_nested_market(data, state)
            if diagnostics["max_balance_residual"] > 1e-5:
                raise RuntimeError(f"Market imbalance in {item['path']}")
            price_error = max(
                abs(float(market["lam"][(entry["region"], entry["time"])]) - float(entry["value"]))
                for entry in profile["market"]["clearing_prices"]
            )
            if price_error > 1e-4:
                raise RuntimeError(f"Saved and recomputed prices disagree in {item['path']}: {price_error:g}")
            capacities = mm._implied_capacity_path(data, list(data.times or []), state["dK_net"])
            welfare = {
                region: nested_economic_objective(data, state, market, region)
                for region in data.players
            }
            branch_id = f"{row['sequence']}/{row['branch']}"
            for region, value in [*welfare.items(), ("ALL", sum(welfare.values()))]:
                welfare_rows.append({
                    "scenario": scenario, "branch": branch_id, "region": region,
                    "discounted_welfare_musd": float(value),
                })
            for period in mm._operating_times(data):
                demand_total = sum(float(market["x_dem"][(region, period)]) for region in data.regions)
                weighted_price = sum(
                    float(market["lam"][(region, period)]) *
                    float(market["x_dem"][(region, period)])
                    for region in data.regions
                ) / demand_total
                branch_rows.append({
                    "scenario": scenario, "branch": branch_id, "period": period,
                    "total_capacity_gw": sum(float(capacities[(region, period)]) for region in data.regions),
                    "demand_weighted_price_usd_per_kw": weighted_price,
                    "cross_border_trade_gw": sum(
                        float(market["x"][(exporter, importer, period)])
                        for exporter in data.regions for importer in data.regions
                        if exporter != importer
                    ),
                })
                for region in data.regions:
                    regional_rows.append({
                        "scenario": scenario, "branch": branch_id,
                        "period": period, "region": region,
                        "capacity_gw": float(capacities[(region, period)]),
                        "clearing_price_usd_per_kw": float(market["lam"][(region, period)]),
                    })

    write_csv(output_dir / "branch_metrics.csv", branch_rows)
    write_csv(output_dir / "regional_metrics.csv", regional_rows)
    write_csv(output_dir / "welfare_metrics.csv", welfare_rows)
    summaries: list[dict] = []
    for scenario in accepted_sets:
        for period in ("2025", "2030", "2035", "2040"):
            subset = [row for row in branch_rows if row["scenario"] == scenario and row["period"] == period]
            for metric in ("total_capacity_gw", "demand_weighted_price_usd_per_kw", "cross_border_trade_gw"):
                values = [float(row[metric]) for row in subset]
                summaries.append({
                    "scenario": scenario, "period": period, "region": "ALL", "metric": metric,
                    "branches": len(values),
                    "min": min(values) if values else "",
                    "median": median(values) if values else "",
                    "max": max(values) if values else "",
                })
            for region in ("ch", "eu", "us", "apac", "af", "row"):
                regional = [row for row in regional_rows if row["scenario"] == scenario and row["period"] == period and row["region"] == region]
                for metric in ("capacity_gw", "clearing_price_usd_per_kw"):
                    values = [float(row[metric]) for row in regional]
                    summaries.append({
                        "scenario": scenario, "period": period, "region": region, "metric": metric,
                        "branches": len(values),
                        "min": min(values) if values else "",
                        "median": median(values) if values else "",
                        "max": max(values) if values else "",
                    })
    write_csv(output_dir / "summary_metrics.csv", summaries)
    welfare_summaries = []
    for scenario in accepted_sets:
        for region in ("ALL", "ch", "eu", "us", "apac", "af", "row"):
            values = [
                float(row["discounted_welfare_musd"])
                for row in welfare_rows
                if row["scenario"] == scenario and row["region"] == region
            ]
            welfare_summaries.append({
                "scenario": scenario, "region": region, "branches": len(values),
                "min": min(values) if values else "",
                "median": median(values) if values else "",
                "max": max(values) if values else "",
            })
    write_csv(output_dir / "welfare_summary.csv", welfare_summaries)
    baseline_count = len(accepted_sets["strategic_offers"])
    counter_count = len(accepted_sets["fixed_cost_offers"])
    lines = [
        "# Competitive-offer capacity-only counterfactual", "",
        f"Corrected input SHA-256: `{counter_protocol['input_sha256']}`.",
        "Stage 1 retains the economic and proximal penalties. Stage 2 and the frozen-profile audits use the clean objective in both scenarios. Offers in the counterfactual equal exporter-period marginal manufacturing cost on every bilateral route. Market-clearing prices remain endogenous.",
        "",
        f"Accepted branches: {baseline_count} strategic-offer baseline; {counter_count} fixed-cost-offer counterfactual.",
        "",
    ]
    if counter_count:
        lines += [
            "## 2040 outcomes across accepted branches", "",
            "| Metric | Strategic offers: median [range] | Fixed cost offers: median [range] |",
            "|---|---:|---:|",
        ]
        for region, metric, label, unit in [
            ("ALL", "total_capacity_gw", "Total capacity", "GW"),
            ("ALL", "demand_weighted_price_usd_per_kw", "Demand-weighted clearing price", "USD/kW"),
            ("ALL", "cross_border_trade_gw", "Cross-border trade", "GW"),
            *[(region, "capacity_gw", f"{region.upper()} capacity", "GW") for region in ("ch", "eu", "us", "apac", "af", "row")],
            *[(region, "clearing_price_usd_per_kw", f"{region.upper()} clearing price", "USD/kW") for region in ("ch", "eu", "us", "apac", "af", "row")],
        ]:
            entries = [
                next(x for x in summaries if x["scenario"] == scenario and x["period"] == "2040" and x["region"] == region and x["metric"] == metric)
                for scenario in ("strategic_offers", "fixed_cost_offers")
            ]
            cells = [f"{x['median']:,.1f} [{x['min']:,.1f}–{x['max']:,.1f}]" for x in entries]
            lines.append(f"| {label} ({unit}) | {cells[0]} | {cells[1]} |")
        lines += [
            "", "These are distributions across accepted algorithm branches, which may include nearby points in the same basin. Differences are descriptive and depend on which equilibria the search found; they are not a matched causal estimate.",
            "", "## Discounted regional welfare across the full horizon", "",
            "| Region | Strategic offers: median [range] | Fixed cost offers: median [range] |",
            "|---|---:|---:|",
        ]
        for region in ("ALL", "ch", "eu", "us", "apac", "af", "row"):
            entries = [
                next(x for x in welfare_summaries if x["scenario"] == scenario and x["region"] == region)
                for scenario in ("strategic_offers", "fixed_cost_offers")
            ]
            cells = [f"{x['median']:,.1f} [{x['min']:,.1f}–{x['max']:,.1f}]" for x in entries]
            lines.append(f"| {region.upper()} (million USD) | {cells[0]} | {cells[1]} |")
    else:
        lines.append("No fixed-cost-offer branch passed the one-start 1% frozen-profile criterion within the scheduled search. Economic outcome ranges are therefore not reported for that scenario.")
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "comparison.md")


if __name__ == "__main__":
    main()
