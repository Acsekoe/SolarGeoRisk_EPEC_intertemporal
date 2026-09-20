"""Compare accepted equilibria from the new profile workflow with the old package.

The script reads only archived JSON artifacts and writes a compact machine-readable
CSV plus a Markdown summary.  It deliberately compares economic outcomes rather
than treating damping as an economic model parameter.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "new_equilibria"
NEW_RUN = ARCHIVE / "new_profile_workflow_20260920_143353"
OLD_ROOT = ARCHIVE / "existing_equilibria" / "ch-af-apac-eu-row-us"
OUT_DIR = ARCHIVE / "comparison_new_vs_old"

REGION_LABELS = {
    "ch": "China",
    "af": "Africa",
    "eu": "EU",
    "us": "US",
    "apac": "APAC",
    "row": "ROW",
}
REGIONS = ("ch", "af", "eu", "us", "apac", "row")
TIMES = ("2025", "2030", "2035", "2040", "2045")
MARKET_TIMES = ("2025", "2030", "2035", "2040")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_project_path(value: str) -> Path:
    return ROOT / Path(value.replace("\\", "/"))


def rows_to_map(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], float]:
    return {tuple(str(row[key]) for key in keys): float(row["value"]) for row in rows}


def candidate_record(
    group: str,
    sequence: str,
    branch: str,
    profile_path: Path,
    audit_path: Path,
) -> dict:
    profile_document = load_json(profile_path)
    audit = load_json(audit_path)
    profile = profile_document["ending_profile"]
    market = profile["market"]

    capacities = rows_to_map(profile["capacities"], ("player", "time"))
    prices = rows_to_map(market["clearing_prices"], ("region", "time"))
    demand = rows_to_map(market["demand"], ("region", "time"))
    flows = rows_to_map(market["trade_flows"], ("exporter", "importer", "time"))
    objectives = {row["player"]: float(row["reference_objective"]) for row in audit["players"]}

    output = {
        (region, time): sum(flows.get((region, importer, time), 0.0) for importer in REGIONS)
        for region in REGIONS
        for time in MARKET_TIMES
    }
    imports = {
        (region, time): sum(
            flows.get((exporter, region, time), 0.0) for exporter in REGIONS if exporter != region
        )
        for region in REGIONS
        for time in MARKET_TIMES
    }
    exports = {
        (region, time): sum(
            flows.get((region, importer, time), 0.0) for importer in REGIONS if importer != region
        )
        for region in REGIONS
        for time in MARKET_TIMES
    }

    return {
        "group": group,
        "sequence": sequence,
        "branch": branch,
        "profile_path": str(profile_path.relative_to(ROOT)),
        "audit_path": str(audit_path.relative_to(ROOT)),
        "max_gain": float(audit["max_relative_gain"]),
        "max_gain_player": audit["max_gain_player"],
        "capacities": capacities,
        "prices": prices,
        "demand": demand,
        "flows": flows,
        "output": output,
        "imports": imports,
        "exports": exports,
        "objectives": objectives,
    }


def collect_candidates() -> list[dict]:
    candidates: list[dict] = []

    for profile_path in sorted((OLD_ROOT / "profiles" / "ch-af-apac-eu-row-us").glob("*/*.json")):
        branch = profile_path.parent.name
        audit_matches = sorted((OLD_ROOT / "one_start_audits" / "ch-af-apac-eu-row-us" / branch).glob("*.json"))
        if len(audit_matches) != 1:
            raise RuntimeError(f"Expected exactly one old audit for {branch}, found {len(audit_matches)}")
        candidates.append(
            candidate_record(
                "old_ch_first",
                "ch-af-apac-eu-row-us",
                branch,
                profile_path,
                audit_matches[0],
            )
        )

    manifest = load_json(NEW_RUN / "manifest.json")
    for result in manifest["grid_results"]:
        if not result.get("local_one_percent_equilibrium", False):
            continue
        sequence = result["sequence"]
        group = "new_af_first" if sequence.startswith("af-") else "new_eu_first"
        candidates.append(
            candidate_record(
                group,
                sequence,
                result["branch"],
                resolve_project_path(result["selected_profile"]),
                resolve_project_path(result["one_start_audit"]),
            )
        )

    return candidates


def average_metric(candidates: list[dict], metric: str, key: tuple[str, str]) -> float:
    return mean(candidate[metric][key] for candidate in candidates)


def relative_l1(first: dict, second: dict, metric: str) -> float:
    keys = sorted(set(first[metric]) | set(second[metric]))
    numerator = sum(abs(first[metric].get(key, 0.0) - second[metric].get(key, 0.0)) for key in keys)
    denominator = 0.5 * sum(
        abs(first[metric].get(key, 0.0)) + abs(second[metric].get(key, 0.0)) for key in keys
    )
    return numerator / denominator if denominator else 0.0


def percent_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0) if old else math.nan


def write_candidate_csv(candidates: list[dict]) -> None:
    fields = [
        "group",
        "sequence",
        "branch",
        "max_gain_percent",
        "max_gain_player",
        "total_capacity_2025",
        "total_capacity_2040",
        "total_capacity_2045",
        "total_output_2040",
        "cross_border_trade_2040",
        "mean_price_2040",
        "sum_regional_objectives",
    ]
    for region in REGIONS:
        fields.extend(
            [
                f"capacity_2045_{region}",
                f"output_2040_{region}",
                f"price_2040_{region}",
                f"import_share_2040_{region}",
                f"objective_{region}",
            ]
        )

    with (OUT_DIR / "candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            row = {
                "group": candidate["group"],
                "sequence": candidate["sequence"],
                "branch": candidate["branch"],
                "max_gain_percent": 100.0 * candidate["max_gain"],
                "max_gain_player": candidate["max_gain_player"],
                "total_capacity_2025": sum(candidate["capacities"][(r, "2025")] for r in REGIONS),
                "total_capacity_2040": sum(candidate["capacities"][(r, "2040")] for r in REGIONS),
                "total_capacity_2045": sum(candidate["capacities"][(r, "2045")] for r in REGIONS),
                "total_output_2040": sum(candidate["output"][(r, "2040")] for r in REGIONS),
                "cross_border_trade_2040": sum(candidate["exports"][(r, "2040")] for r in REGIONS),
                "mean_price_2040": mean(candidate["prices"][(r, "2040")] for r in REGIONS),
                "sum_regional_objectives": sum(candidate["objectives"].values()),
            }
            for region in REGIONS:
                regional_demand = candidate["demand"][(region, "2040")]
                row.update(
                    {
                        f"capacity_2045_{region}": candidate["capacities"][(region, "2045")],
                        f"output_2040_{region}": candidate["output"][(region, "2040")],
                        f"price_2040_{region}": candidate["prices"][(region, "2040")],
                        f"import_share_2040_{region}": candidate["imports"][(region, "2040")] / regional_demand,
                        f"objective_{region}": candidate["objectives"][region],
                    }
                )
            writer.writerow(row)


def group_summary(candidates: list[dict], group: str) -> dict:
    selected = [candidate for candidate in candidates if candidate["group"] == group]
    return {
        "count": len(selected),
        "capacity": {
            time: sum(average_metric(selected, "capacities", (region, time)) for region in REGIONS)
            for time in TIMES
        },
        "capacity_by_region_2045": {
            region: average_metric(selected, "capacities", (region, "2045")) for region in REGIONS
        },
        "output_by_region_2040": {
            region: average_metric(selected, "output", (region, "2040")) for region in REGIONS
        },
        "price_by_region_2040": {
            region: average_metric(selected, "prices", (region, "2040")) for region in REGIONS
        },
        "import_share_by_region_2040": {
            region: mean(
                candidate["imports"][(region, "2040")] / candidate["demand"][(region, "2040")]
                for candidate in selected
            )
            for region in REGIONS
        },
        "cross_border_trade_2040": mean(
            sum(candidate["exports"][(region, "2040")] for region in REGIONS) for candidate in selected
        ),
        "mean_price_2040": mean(
            mean(candidate["prices"][(region, "2040")] for region in REGIONS) for candidate in selected
        ),
        "sum_objectives": mean(sum(candidate["objectives"].values()) for candidate in selected),
    }


def fmt(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}"


def build_markdown(candidates: list[dict]) -> str:
    groups = ("old_ch_first", "new_af_first", "new_eu_first")
    labels = {
        "old_ch_first": "Old CH-first (6)",
        "new_af_first": "New AF-first (4)",
        "new_eu_first": "New EU-first (4)",
    }
    summaries = {group: group_summary(candidates, group) for group in groups}
    old = summaries["old_ch_first"]

    lines = [
        "# New versus old accepted equilibrium outcomes",
        "",
        "All entries are simple averages across accepted profiles in the named group. Damping is treated as an algorithmic parameter, not an economic input. Objectives are the regional reference objectives recorded by the common frozen-profile zero-proximal audit; their sum is descriptive and is not a social-welfare measure.",
        "",
        "## Group-level comparison",
        "",
        "| Group | Accepted | Total capacity 2040 | Total capacity 2045 | Mean regional price 2040 | Cross-border trade 2040 | Sum of objectives |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        summary = summaries[group]
        lines.append(
            f"| {labels[group]} | {summary['count']} | {fmt(summary['capacity']['2040'])} | "
            f"{fmt(summary['capacity']['2045'])} | "
            f"{fmt(summary['mean_price_2040'], 2)} | {fmt(summary['cross_border_trade_2040'])} | "
            f"{fmt(summary['sum_objectives'])} |"
        )

    lines.extend(
        [
            "",
            "Relative to the old accepted set:",
            "",
            "| New group | Capacity 2040 | Capacity 2045 | Mean price 2040 | Cross-border trade 2040 | Sum of objectives |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for group in ("new_af_first", "new_eu_first"):
        summary = summaries[group]
        lines.append(
            f"| {labels[group]} | {percent_change(summary['capacity']['2040'], old['capacity']['2040']):+.2f}% | "
            f"{percent_change(summary['capacity']['2045'], old['capacity']['2045']):+.2f}% | "
            f"{percent_change(summary['mean_price_2040'], old['mean_price_2040']):+.2f}% | "
            f"{percent_change(summary['cross_border_trade_2040'], old['cross_border_trade_2040']):+.2f}% | "
            f"{percent_change(summary['sum_objectives'], old['sum_objectives']):+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Regional endpoint comparison (capacity 2045; market outcomes 2040)",
            "",
            "| Region | Old capacity | AF-first capacity | EU-first capacity | Old output | AF-first output | EU-first output |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for region in REGIONS:
        lines.append(
            f"| {REGION_LABELS[region]} | {fmt(old['capacity_by_region_2045'][region])} | "
            f"{fmt(summaries['new_af_first']['capacity_by_region_2045'][region])} | "
            f"{fmt(summaries['new_eu_first']['capacity_by_region_2045'][region])} | "
            f"{fmt(old['output_by_region_2040'][region])} | "
            f"{fmt(summaries['new_af_first']['output_by_region_2040'][region])} | "
            f"{fmt(summaries['new_eu_first']['output_by_region_2040'][region])} |"
        )

    lines.extend(
        [
            "",
            "## Within-group spread",
            "",
            "| Group | Capacity 2045 range | Mean price 2040 range | Trade 2040 range |",
            "|---|---:|---:|---:|",
        ]
    )
    for group in groups:
        selected = [candidate for candidate in candidates if candidate["group"] == group]
        capacities = [sum(candidate["capacities"][(region, "2045")] for region in REGIONS) for candidate in selected]
        prices = [mean(candidate["prices"][(region, "2040")] for region in REGIONS) for candidate in selected]
        trades = [sum(candidate["exports"][(region, "2040")] for region in REGIONS) for candidate in selected]
        lines.append(
            f"| {labels[group]} | {fmt(min(capacities))} to {fmt(max(capacities))} | "
            f"{fmt(min(prices), 2)} to {fmt(max(prices), 2)} | {fmt(min(trades))} to {fmt(max(trades))} |"
        )

    lines.extend(
        [
            "",
            "## Matched-branch update-order comparison",
            "",
            "Each row holds the price factor, capacity-path weight, and damping fixed and changes only the update order. Percentages are the new outcome relative to its old CH-first counterpart.",
            "",
            "| New sequence | Branch | Capacity 2045 | Mean price 2040 | Trade 2040 | Sum of objectives |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    old_sequence = "ch-af-apac-eu-row-us"
    by_key = {(candidate["sequence"], candidate["branch"]): candidate for candidate in candidates}
    for sequence in ("af-eu-us-apac-row-ch", "eu-us-af-row-apac-ch"):
        common_branches = sorted(
            {candidate["branch"] for candidate in candidates if candidate["sequence"] == old_sequence}
            & {candidate["branch"] for candidate in candidates if candidate["sequence"] == sequence}
        )
        for branch in common_branches:
            old_candidate = by_key[(old_sequence, branch)]
            new_candidate = by_key[(sequence, branch)]
            old_capacity = sum(old_candidate["capacities"][(region, "2045")] for region in REGIONS)
            new_capacity = sum(new_candidate["capacities"][(region, "2045")] for region in REGIONS)
            old_price = mean(old_candidate["prices"][(region, "2040")] for region in REGIONS)
            new_price = mean(new_candidate["prices"][(region, "2040")] for region in REGIONS)
            old_trade = sum(old_candidate["exports"][(region, "2040")] for region in REGIONS)
            new_trade = sum(new_candidate["exports"][(region, "2040")] for region in REGIONS)
            old_objective = sum(old_candidate["objectives"].values())
            new_objective = sum(new_candidate["objectives"].values())
            lines.append(
                f"| {sequence} | {branch} | {percent_change(new_capacity, old_capacity):+.2f}% | "
                f"{percent_change(new_price, old_price):+.2f}% | "
                f"{percent_change(new_trade, old_trade):+.2f}% | "
                f"{percent_change(new_objective, old_objective):+.3f}% |"
            )

    lines.extend(
        [
            "",
            "## Regional prices and import reliance (2040)",
            "",
            "| Region | Old price | AF-first price | EU-first price | Old import share | AF-first import share | EU-first import share |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for region in REGIONS:
        lines.append(
            f"| {REGION_LABELS[region]} | {fmt(old['price_by_region_2040'][region], 2)} | "
            f"{fmt(summaries['new_af_first']['price_by_region_2040'][region], 2)} | "
            f"{fmt(summaries['new_eu_first']['price_by_region_2040'][region], 2)} | "
            f"{100.0 * old['import_share_by_region_2040'][region]:.1f}% | "
            f"{100.0 * summaries['new_af_first']['import_share_by_region_2040'][region]:.1f}% | "
            f"{100.0 * summaries['new_eu_first']['import_share_by_region_2040'][region]:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Damping-pair outcome distances",
            "",
            "Relative L1 distance is sum(abs(A-B)) divided by the average absolute mass of the two outcome vectors. Zero means identical. Only accepted 0.30/0.40 pairs with the same sequence, price factor, and capacity-path weight are shown.",
            "",
            "| Sequence/group | Economic branch | Capacity distance | Price distance | Flow distance |",
            "|---|---|---:|---:|---:|",
        ]
    )
    pair_rows = []
    sequences = sorted({candidate["sequence"] for candidate in candidates})
    for sequence in sequences:
        for price_factor in ("pf100", "pf120"):
            for capacity_weight in ("k050", "k100"):
                base = f"{price_factor}_{capacity_weight}"
                first = by_key.get((sequence, f"{base}_a030"))
                second = by_key.get((sequence, f"{base}_a040"))
                if first is None or second is None:
                    continue
                pair_rows.append(
                    (
                        sequence,
                        base,
                        relative_l1(first, second, "capacities"),
                        relative_l1(first, second, "prices"),
                        relative_l1(first, second, "flows"),
                    )
                )
    for sequence, base, capacity_distance, price_distance, flow_distance in pair_rows:
        lines.append(
            f"| {sequence} | {base} | {100.0 * capacity_distance:.3f}% | "
            f"{100.0 * price_distance:.3f}% | {100.0 * flow_distance:.3f}% |"
        )

    lines.extend(
        [
            "",
            "## Audit quality",
            "",
            "| Group | Max gain range | Limiting players |",
            "|---|---:|---|",
        ]
    )
    for group in groups:
        selected = [candidate for candidate in candidates if candidate["group"] == group]
        gains = [100.0 * candidate["max_gain"] for candidate in selected]
        players = ", ".join(sorted({REGION_LABELS[candidate["max_gain_player"]] for candidate in selected}))
        lines.append(f"| {labels[group]} | {min(gains):.3f}% to {max(gains):.3f}% | {players} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates()
    expected = {"old_ch_first": 6, "new_af_first": 4, "new_eu_first": 4}
    actual = {group: sum(candidate["group"] == group for candidate in candidates) for group in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected candidate counts: {actual}")
    write_candidate_csv(candidates)
    report = build_markdown(candidates)
    (OUT_DIR / "comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
