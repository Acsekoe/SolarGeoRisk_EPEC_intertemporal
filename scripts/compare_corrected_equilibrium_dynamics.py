from __future__ import annotations

"""Compare accepted corrected-demand equilibria with the historical O6 profile.

The historical O6 package used inconsistent explicit demand slopes and is kept
only as a qualitative reference.  The corrected candidates are read directly
from their accepted cost-staged checkpoints, so this report does not rerun any
optimization or interfere with live searches.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_PACKAGE = (
    ROOT
    / "outputs/equilibria/ch-row-apac-us-eu-af/o6_sweep_015_20260915"
)
DEFAULT_CORRECTED_ROOT = (
    ROOT
    / "outputs/demand_calibration/diversified_search_corrected_20260915_155249"
)
PERIODS = ["2025", "2030", "2035", "2040"]
REGIONS = ["ch", "row", "apac", "us", "eu", "af"]
MATCHING_SEQUENCE = "ch-row-apac-us-eu-af"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def resolve_recorded_path(value: str, *, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_path = ROOT / path
    return root_path if root_path.exists() else base / path


def keyed(
    rows: list[dict[str, Any]], entity_key: str
) -> dict[tuple[str, str], float]:
    return {
        (str(row[entity_key]), str(row["time"])): float(row["value"])
        for row in rows
        if str(row["time"]) in PERIODS
    }


def load_old(package: Path) -> dict[str, Any]:
    comparison_path = package / "comparison_to_paper.json"
    comparison = read_json(comparison_path)
    candidates = [
        row
        for row in comparison["profiles"]
        if row.get("name") == "best_overnight_candidate"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one best_overnight_candidate in {comparison_path}; "
            f"found {len(candidates)}"
        )
    candidate = candidates[0]
    return {
        "label": "Historical O6 sweep 15",
        "short_label": "Old O6 s15",
        "sequence": MATCHING_SEQUENCE,
        "calibration": "historical_inconsistent_demand_slopes",
        "status": "historical_reference_only",
        "one_start_gain": read_json(package / "certification.json")[
            "maximum_frozen_profile_relative_gain"
        ],
        "three_start_gain": candidate["max_relative_gain"],
        "source": relative(comparison_path),
        "capacities": keyed(candidate["capacities"], "player"),
        "prices": keyed(candidate["clearing_prices"], "region"),
    }


def load_corrected(corrected_root: Path) -> list[dict[str, Any]]:
    profiles = []
    for status_path in sorted(corrected_root.glob("*/cost_staged/status.json")):
        status = read_json(status_path)
        if status.get("status") != "accepted":
            continue
        checkpoint_path = resolve_recorded_path(
            str(status["selected_profile"]), base=corrected_root
        )
        checkpoint = read_json(checkpoint_path)
        ending = checkpoint["ending_profile"]
        sequence = str(status["sequence"])
        profiles.append(
            {
                "label": f"Corrected {sequence}",
                "short_label": sequence.upper(),
                "sequence": sequence,
                "calibration": "corrected_demand_slopes",
                "status": "accepted_one_start_local_1pct_equilibrium",
                "one_start_gain": float(status["one_start_max_relative_gain"]),
                "three_start_gain": float(status["three_start_max_relative_gain"]),
                "source": relative(checkpoint_path),
                "selected_sweep": int(status["selected_sweep"]),
                "capacities": keyed(ending["capacities"], "player"),
                "prices": keyed(ending["market"]["clearing_prices"], "region"),
            }
        )
    if not profiles:
        raise RuntimeError(f"No accepted cost-staged profiles found in {corrected_root}")
    return profiles


def total_capacity(profile: dict[str, Any], period: str) -> float:
    return sum(profile["capacities"][(region, period)] for region in REGIONS)


def mean_absolute_difference(
    candidate: dict[str, Any], reference: dict[str, Any], field: str
) -> float:
    return mean(
        abs(candidate[field][(region, period)] - reference[field][(region, period)])
        for region in REGIONS
        for period in PERIODS
    )


def write_csvs(output: Path, profiles: list[dict[str, Any]]) -> None:
    for filename, field, value_name in (
        ("capacity_paths.csv", "capacities", "capacity_gw"),
        ("price_paths.csv", "prices", "price_usd_per_kw"),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "profile",
                    "calibration",
                    "sequence",
                    "one_start_gain_pct",
                    "region",
                    "time",
                    value_name,
                ],
            )
            writer.writeheader()
            for profile in profiles:
                for period in PERIODS:
                    for region in REGIONS:
                        writer.writerow(
                            {
                                "profile": profile["label"],
                                "calibration": profile["calibration"],
                                "sequence": profile["sequence"],
                                "one_start_gain_pct": 100
                                * float(profile["one_start_gain"]),
                                "region": region,
                                "time": period,
                                value_name: profile[field][(region, period)],
                            }
                        )


def plot_capacity(
    output: Path,
    historical: dict[str, Any],
    corrected: list[dict[str, Any]],
) -> None:
    years = [int(period) for period in PERIODS]
    matching = next(row for row in corrected if row["sequence"] == MATCHING_SEQUENCE)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].plot(
        years,
        [total_capacity(historical, period) for period in PERIODS],
        color="#111111",
        linewidth=2.8,
        marker="o",
        label="Historical O6 s15",
    )
    colors = ["#2E6F40", "#A83232", "#4C72B0"]
    for color, profile in zip(colors, corrected):
        axes[0].plot(
            years,
            [total_capacity(profile, period) for period in PERIODS],
            linewidth=2.0,
            marker="o",
            color=color,
            label=f"Corrected {profile['sequence'].upper()}",
        )
    axes[0].set_title("Total capacity")
    axes[0].set_ylabel("GW")
    axes[0].legend(fontsize=8)

    region_colors = {"ch": "#CA6180", "row": "#3A9DAD", "apac": "#8064A2"}
    for region in ("ch", "row", "apac"):
        axes[1].plot(
            years,
            [historical["capacities"][(region, period)] for period in PERIODS],
            color=region_colors[region],
            linewidth=2.4,
            marker="o",
            label=f"Old {region.upper()}",
        )
        axes[1].plot(
            years,
            [matching["capacities"][(region, period)] for period in PERIODS],
            color=region_colors[region],
            linewidth=2.0,
            marker="s",
            linestyle="--",
            label=f"Corrected {region.upper()}",
        )
    axes[1].set_title("Matching order: regional allocation")
    axes[1].set_ylabel("GW")
    axes[1].legend(fontsize=8, ncol=2)

    for axis in axes:
        axis.set_xticks(years)
        axis.grid(True, linestyle=":", alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "capacity_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "capacity_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prices(
    output: Path,
    historical: dict[str, Any],
    corrected: list[dict[str, Any]],
) -> None:
    years = [int(period) for period in PERIODS]
    matching = next(row for row in corrected if row["sequence"] == MATCHING_SEQUENCE)
    names = {
        "ch": "China",
        "row": "Rest of World",
        "apac": "Asia-Pacific",
        "us": "United States",
        "eu": "Europe",
        "af": "Africa",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex=True, sharey=True)
    for axis, region in zip(axes.flat, REGIONS):
        axis.plot(
            years,
            [historical["prices"][(region, period)] for period in PERIODS],
            color="#111111",
            linewidth=2.3,
            marker="o",
            label="Historical O6 s15",
        )
        axis.plot(
            years,
            [matching["prices"][(region, period)] for period in PERIODS],
            color="#A83232",
            linewidth=2.1,
            marker="s",
            linestyle="--",
            label="Corrected matching order",
        )
        axis.set_title(names[region])
        axis.set_xticks(years)
        axis.grid(True, linestyle=":", alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Price [$/kW]")
    axes[1, 0].set_ylabel("Price [$/kW]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.subplots_adjust(bottom=0.12, wspace=0.18, hspace=0.30)
    fig.savefig(output / "price_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "price_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def write_assessment(
    output: Path,
    historical: dict[str, Any],
    corrected: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    matching = next(row for row in corrected if row["sequence"] == MATCHING_SEQUENCE)

    def cap(profile: dict[str, Any], region: str, period: str) -> float:
        return float(profile["capacities"][(region, period)])

    lines = [
        "# Corrected-equilibrium dynamics compared with historical O6",
        "",
        "The historical O6 profile is retained only as a qualitative reference because it used the inconsistent demand slopes. All corrected profiles below pass the one-start local 1% criterion but fail their stronger three-start diagnostic.",
        "",
        "## Quantitative comparison",
        "",
        "| Corrected player order | Selected sweep | One-start max gain | Three-start max gain | Capacity MAE vs old | Price MAE vs old | Total capacity 2030 / 2035 / 2040 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['sequence'].upper()} | {row['selected_sweep']} | "
            f"{row['one_start_gain_pct']:.3f}% | {row['three_start_gain_pct']:.3f}% | "
            f"{row['capacity_mae_gw']:.1f} GW | ${row['price_mae_usd_per_kw']:.1f}/kW | "
            f"{row['total_capacity_gw']['2030']:.0f} / "
            f"{row['total_capacity_gw']['2035']:.0f} / "
            f"{row['total_capacity_gw']['2040']:.0f} GW |"
        )
    old_totals = " / ".join(
        f"{total_capacity(historical, period):.0f}" for period in PERIODS[1:]
    )
    lines.extend(
        [
            "",
            f"Historical O6 total capacity in 2030 / 2035 / 2040 was {old_totals} GW.",
            "",
            "## Assessment",
            "",
            "The corrected candidates preserve the large post-2025 contraction, near-complete EU and US exit, and Chinese dominance. Two of the three also show a 2030-2035 capacity rebound. The matching-order candidate is closest on market prices and has almost the same Chinese capacity after 2030.",
            "",
            f"The important mismatch is regional allocation. In 2040, historical O6 had CH/ROW/APAC capacity of {cap(historical, 'ch', '2040'):.0f}/{cap(historical, 'row', '2040'):.0f}/{cap(historical, 'apac', '2040'):.0f} GW. The corrected matching-order candidate has {cap(matching, 'ch', '2040'):.0f}/{cap(matching, 'row', '2040'):.0f}/{cap(matching, 'apac', '2040'):.0f} GW. ROW therefore remains a large producer, while APAC contracts sharply after 2035 instead of continuing to expand.",
            "",
            "China's clearing-price path is effectively unchanged. The corrected matching-order profile does not reproduce the historical 2035 price spikes in Europe, the United States, and Rest of World, and instead has a much higher APAC price in 2040.",
            "",
            "These are similar high-level strategic dynamics, but not a like-for-like reproduction of the old O6 storyline. A search explicitly targeting the old ROW-to-APAC capacity reallocation would need to be documented as a separate basin or calibration-continuation experiment.",
        ]
    )
    (output / "ASSESSMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-package", type=Path, default=DEFAULT_OLD_PACKAGE)
    parser.add_argument("--corrected-root", type=Path, default=DEFAULT_CORRECTED_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    old_package = args.old_package.resolve()
    corrected_root = args.corrected_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else corrected_root / "comparison_to_historical_o6"
    )
    output.mkdir(parents=True, exist_ok=True)

    historical = load_old(old_package)
    corrected = load_corrected(corrected_root)
    profiles = [historical, *corrected]
    summaries = []
    for candidate in corrected:
        summaries.append(
            {
                "sequence": candidate["sequence"],
                "source": candidate["source"],
                "selected_sweep": candidate["selected_sweep"],
                "one_start_gain_pct": 100 * candidate["one_start_gain"],
                "three_start_gain_pct": 100 * candidate["three_start_gain"],
                "capacity_mae_gw": mean_absolute_difference(
                    candidate, historical, "capacities"
                ),
                "price_mae_usd_per_kw": mean_absolute_difference(
                    candidate, historical, "prices"
                ),
                "total_capacity_gw": {
                    period: total_capacity(candidate, period) for period in PERIODS
                },
            }
        )

    write_csvs(output, profiles)
    plot_capacity(output, historical, corrected)
    plot_prices(output, historical, corrected)
    write_assessment(output, historical, corrected, summaries)
    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "warning": "Historical O6 used inconsistent demand slopes and is a qualitative reference only.",
        "historical_reference": {
            key: value
            for key, value in historical.items()
            if key not in {"capacities", "prices"}
        },
        "corrected_root": relative(corrected_root),
        "periods": PERIODS,
        "regions": REGIONS,
        "summaries": summaries,
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote comparison to {output}")


if __name__ == "__main__":
    main()
