from __future__ import annotations

"""Create statistical and paper-ready figures for a clean Stage-2 run."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN_ROOT = ROOT / "outputs" / "clean_stage2_factorial_20260923_123037"
PREVIOUS_ANALYSIS = (
    ROOT / "outputs" / "new_equilibria" / "statistical_analysis_20260921"
)

from scripts import analyze_equilibrium_ranges as base


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_recorded_path(value: str, run_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    rooted = (ROOT / path).resolve()
    if rooted.exists():
        return rooted
    return (run_root / path.name).resolve()


def profile_payload(document: dict[str, Any]) -> dict[str, Any]:
    if "ending_profile" in document:
        return document["ending_profile"]
    if "profile" in document:
        return document["profile"]
    raise ValueError("Saved clean Stage-2 document has no profile payload")


def collect_candidates(
    run_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(run_root / "manifest.json")
    candidates: list[dict[str, Any]] = []
    accepted = [row for row in manifest["results"] if row["status"] == "accepted"]
    for result in accepted:
        sequence = str(result["sequence"])
        branch = str(result["branch"])
        order = base.ORDER_LABELS[sequence]
        price_factor, capacity_weight, damping = base.parse_branch(branch)
        profile_path = resolve_recorded_path(result["selected_profile"], run_root)
        audit_path = resolve_recorded_path(result["one_start_audit"], run_root)
        document = load_json(profile_path)
        audit = load_json(audit_path)
        profile = profile_payload(document)
        market = profile["market"]

        if document.get("objective_mode") != "without-mu-and-penalties":
            raise RuntimeError(f"Non-clean profile included: {profile_path}")
        if audit.get("objective_mode") != "without-mu-and-penalties":
            raise RuntimeError(f"Non-clean audit included: {audit_path}")
        if not bool(audit.get("equilibrium_verified")):
            raise RuntimeError(f"Unverified profile included: {profile_path}")

        candidates.append(
            {
                "candidate": f"{sequence}/{branch}",
                "candidate_code": base.short_candidate_code(order, branch),
                "sequence": sequence,
                "order": order,
                "branch": branch,
                "origin": "clean_objective_stage2",
                "price_factor": price_factor,
                "capacity_weight": capacity_weight,
                "damping": damping,
                "selected_sweep": int(result["selected_sweep"]),
                "max_gain_percent": 100.0 * float(audit["max_relative_gain"]),
                "limiting_player": str(audit["max_gain_player"]),
                "multistart_status": "one_start_only",
                "capacities": base.rows_to_map(
                    profile["capacities"], ("player", "time")
                ),
                "prices": base.rows_to_map(
                    market["clearing_prices"], ("region", "time")
                ),
                "demand": base.rows_to_map(
                    market["demand"], ("region", "time")
                ),
                "flows": base.rows_to_map(
                    market["trade_flows"], ("exporter", "importer", "time")
                ),
                "offers": base.rows_to_map(
                    profile["strategy"]["p_offer"],
                    ("exporter", "importer", "time"),
                ),
            }
        )

    if len(candidates) != len(accepted):
        raise RuntimeError("Not all accepted clean profiles were loaded")
    if not candidates:
        raise RuntimeError("No accepted clean profiles found")
    return candidates, manifest


def saved_audit_records(branch_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((branch_root / "audits").glob("audit_*_one_start.json")):
        audit = load_json(path)
        match = re.search(r"audit_sweep_(\d{3})_one_start", path.stem)
        sweep = int(match.group(1)) if match else 0
        records.append(
            {
                "sweep": sweep,
                "path": path,
                "max_gain_percent": 100.0 * float(audit["max_relative_gain"]),
                "limiting_player": str(audit["max_gain_player"]),
                "audit_successful": bool(audit["all_attempts_successful"]),
                "accepted": bool(audit["equilibrium_verified"]),
            }
        )
    return records


def collect_branch_results(
    run_root: Path, manifest: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in manifest["results"]:
        sequence = str(result["sequence"])
        branch = str(result["branch"])
        price_factor, capacity_weight, damping = base.parse_branch(branch)
        audits = saved_audit_records(run_root / sequence / branch)
        if not audits:
            raise RuntimeError(f"No saved audits for {sequence}/{branch}")
        best = min(audits, key=lambda row: float(row["max_gain_percent"]))
        selected_sweep = int(result.get("selected_sweep", best["sweep"]))
        max_gain_percent = 100.0 * float(
            result.get(
                "one_start_max_relative_gain", best["max_gain_percent"] / 100.0
            )
        )
        limiting_player = str(
            result.get("one_start_max_gain_player", best["limiting_player"])
        )
        status = str(result["status"])
        rows.append(
            {
                "source_experiment": "clean objective Stage-2 factorial",
                "sequence": sequence,
                "order": base.ORDER_LABELS[sequence],
                "branch": branch,
                "status": status,
                "selected_sweep": selected_sweep,
                "max_gain_percent": max_gain_percent,
                "limiting_player": limiting_player,
                "selected_audit_successful": bool(best["audit_successful"]),
                "run_completed_without_failure": status != "failed",
                "accepted": status == "accepted",
                "price_factor": price_factor,
                "capacity_weight": capacity_weight,
                "damping": damping,
                "error": result.get("error", ""),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["sequence", "branch"])
    if len(frame) != 36:
        raise RuntimeError(f"Expected 36 clean Stage-2 branches, found {len(frame)}")
    if frame.duplicated(["sequence", "branch"]).any():
        raise RuntimeError("Duplicate clean Stage-2 branch records")
    return frame


def collect_initial_audits(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence in base.ORDER_LABELS:
        for branch_root in sorted((run_root / sequence).glob("pf*_k*_a*")):
            path = branch_root / "audits" / "audit_initial_one_start.json"
            audit = load_json(path)
            price_factor, capacity_weight, damping = base.parse_branch(
                branch_root.name
            )
            rows.append(
                {
                    "sequence": sequence,
                    "order": base.ORDER_LABELS[sequence],
                    "branch": branch_root.name,
                    "price_factor": price_factor,
                    "capacity_weight": capacity_weight,
                    "damping": damping,
                    "max_gain_percent": 100.0 * float(audit["max_relative_gain"]),
                    "limiting_player": str(audit["max_gain_player"]),
                    "all_attempts_successful": bool(
                        audit["all_attempts_successful"]
                    ),
                    "accepted": bool(audit["equilibrium_verified"]),
                }
            )
    return pd.DataFrame(rows)


def plot_clean_search_diagnostics(
    branch_results: pd.DataFrame, output_dir: Path
) -> None:
    price_colors = {0.8: "#7030A0", 1.0: "#4472C4", 1.2: "#ED7D31"}
    orders = ("CH-first", "AF-first", "EU-first")
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.8), sharey=True)
    for axis, order in zip(axes, orders):
        subset = branch_results[branch_results["order"] == order]
        for _, row in subset.iterrows():
            offset = (-0.10 if row["capacity_weight"] == 0.5 else 0.10) + (
                -0.035 if row["damping"] == 0.3 else 0.035
            )
            color = price_colors[float(row["price_factor"])]
            if row["status"] == "failed":
                axis.scatter(
                    row["selected_sweep"] + offset,
                    row["max_gain_percent"],
                    marker="x",
                    s=62,
                    color="#C00000",
                    linewidth=1.8,
                    zorder=5,
                )
            else:
                axis.scatter(
                    row["selected_sweep"] + offset,
                    row["max_gain_percent"],
                    s=52,
                    facecolor=color if row["accepted"] else "white",
                    edgecolor=color,
                    linewidth=1.35,
                    zorder=4,
                )
        accepted = int(subset["accepted"].sum())
        failed = int((subset["status"] == "failed").sum())
        axis.axhspan(0.35, 1.0, color="#E2F0D9", alpha=0.55, zorder=0)
        axis.axhline(1.0, color="#C00000", linestyle="--", linewidth=1.15)
        axis.set_yscale("log")
        axis.set_xlim(-0.5, 15.6)
        axis.set_ylim(0.35, max(6.5, 1.15 * subset["max_gain_percent"].max()))
        axis.set_xticks([0, 5, 10, 15])
        axis.set_xlabel("Selected or best-audited sweep")
        suffix = f"; {failed} failed" if failed else ""
        axis.set_title(f"{order}: {accepted}/12 accepted{suffix}", fontweight="bold")
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    axes[0].set_ylabel("Frozen-profile maximum gain (%)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"PF {factor:.2f}",
        )
        for factor, color in price_colors.items()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#595959",
                markeredgecolor="#595959",
                label="Accepted",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#595959",
                label="No pass",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                color="#C00000",
                linestyle="none",
                label="Run failed",
            ),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "Clean-objective Stage 2: search convergence and equilibrium verification",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Filled markers pass the common frozen-profile one-start 1% criterion; hollow markers show the best audited non-pass.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.90))
    base.save_figure(figure, output_dir, "algorithm_convergence_diagnostics")


def plot_regional_bands(
    frame: pd.DataFrame,
    value_column: str,
    years: tuple[int, ...],
    ylabel: str,
    band_color: str,
    output_dir: Path,
    stem: str,
) -> None:
    """Paper band plot with extra horizontal room for wide clean-price axes."""
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    ):
        figure, axes = plt.subplots(3, 2, figsize=(8.4, 8.5))
        rng = np.random.default_rng(20260923)
        x = np.arange(len(years), dtype=float)
        for index, (axis, region) in enumerate(
            zip(axes.flat, base.PAPER_REGION_ORDER)
        ):
            data = [
                frame[(frame["region"] == region) & (frame["year"] == year)][
                    value_column
                ].to_numpy()
                for year in years
            ]
            matrix = np.column_stack(data)
            observed_minimum = matrix.min(axis=0)
            observed_maximum = matrix.max(axis=0)
            q10, q25, median, q75, q90 = np.quantile(
                matrix, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0
            )
            axis.fill_between(
                x,
                observed_minimum,
                observed_maximum,
                color=band_color,
                alpha=0.20,
                linewidth=0,
            )
            axis.fill_between(
                x, q10, q90, color=band_color, alpha=0.38, linewidth=0
            )
            axis.fill_between(
                x, q25, q75, color=band_color, alpha=0.90, linewidth=0
            )
            for boundary in (
                observed_minimum,
                observed_maximum,
                q10,
                q90,
                q25,
                q75,
            ):
                axis.plot(x, boundary, color="#C7C7C7", linewidth=0.65)
            axis.plot(
                x,
                median,
                color="#222222",
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                zorder=4,
            )
            for position, values in enumerate(data):
                jitter = rng.uniform(-0.055, 0.055, size=len(values))
                axis.scatter(
                    np.full(len(values), x[position]) + jitter,
                    values,
                    s=10,
                    color="#222222",
                    alpha=0.45,
                    linewidths=0,
                    zorder=3,
                )
            axis.set_title(base.PAPER_REGION_NAMES[region], fontsize=15)
            axis.set_xticks(x, [str(year) for year in years])
            if index % 2 == 0:
                axis.set_ylabel(ylabel, fontsize=15)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)
            axis.grid(True, linestyle=":", alpha=0.5)
            axis.tick_params(axis="both", labelsize=12.5)

        legend_handles = [
            Patch(
                facecolor=band_color,
                alpha=0.20,
                edgecolor="#C7C7C7",
                label="Total range",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=1.8,
                marker="o",
                markersize=4.5,
                label="Median",
            ),
            Patch(
                facecolor=band_color,
                alpha=0.38,
                edgecolor="#C7C7C7",
                label="10th–90th percentiles",
            ),
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle="none",
                marker="o",
                markersize=4.0,
                alpha=0.45,
                label="Individual equilibria",
            ),
            Patch(
                facecolor=band_color,
                alpha=0.90,
                edgecolor="#C7C7C7",
                label="25th–75th percentiles",
            ),
        ]
        figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=3,
            fontsize=9.5,
            frameon=True,
            framealpha=0.9,
            handlelength=1.4,
            handletextpad=0.45,
            columnspacing=0.9,
            borderpad=0.45,
            labelspacing=0.35,
            bbox_to_anchor=(0.5, 0.065),
        )
        figure.subplots_adjust(
            left=0.11,
            right=0.98,
            top=0.96,
            bottom=0.20,
            wspace=0.43,
            hspace=0.50,
        )
        base.save_figure(figure, output_dir, stem)


def collect_previous_candidates() -> list[dict[str, Any]]:
    candidates = base.collect_candidates()
    for candidate in candidates:
        profile_path = (
            base.CANDIDATE_ROOT
            / candidate["sequence"]
            / candidate["branch"]
            / "profile.json"
        )
        profile = load_json(profile_path)["ending_profile"]
        candidate["offers"] = base.rows_to_map(
            profile["strategy"]["p_offer"],
            ("exporter", "importer", "time"),
        )
    return candidates


def trade_offer_metrics(
    candidates: list[dict[str, Any]], formulation: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for year in base.MARKET_YEARS:
            period = str(year)
            cross_border_flows = {
                (exporter, importer): candidate["flows"].get(
                    (exporter, importer, period), 0.0
                )
                for exporter in base.REGIONS
                for importer in base.REGIONS
                if exporter != importer
            }
            total_flow = float(sum(cross_border_flows.values()))
            weighted_offer = (
                sum(
                    flow
                    * candidate["offers"][(exporter, importer, period)]
                    for (exporter, importer), flow in cross_border_flows.items()
                )
                / total_flow
                if total_flow > 1e-12
                else np.nan
            )
            active_routes = sum(flow > 1e-6 for flow in cross_border_flows.values())
            rows.append(
                {
                    "candidate": candidate["candidate"],
                    "formulation": formulation,
                    "year": year,
                    "cross_border_trade_gw": total_flow,
                    "flow_weighted_offer_usd_per_kw": float(weighted_offer),
                    "active_cross_border_routes": active_routes,
                }
            )
    return pd.DataFrame(rows)


def plot_regional_formulation_comparison(
    clean_rows: pd.DataFrame,
    previous_rows: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(9.0, 8.4), sharex=True)
    years = np.array(base.MARKET_YEARS)
    styles = (
        ("Previous objective", previous_rows, "#6F7C80"),
        ("Clean objective", clean_rows, "#A83232"),
    )
    for index, (axis, region) in enumerate(zip(axes.flat, base.PAPER_REGION_ORDER)):
        for label, frame, color in styles:
            subset = frame[frame["region"] == region]
            grouped = subset.groupby("year")[value_column]
            median = grouped.median().reindex(years).to_numpy()
            q25 = grouped.quantile(0.25).reindex(years).to_numpy()
            q75 = grouped.quantile(0.75).reindex(years).to_numpy()
            axis.fill_between(years, q25, q75, color=color, alpha=0.16)
            axis.plot(
                years,
                median,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4.2,
                label=label,
            )
        axis.set_title(base.PAPER_REGION_NAMES[region], fontsize=13)
        axis.set_xticks(years)
        if index % 2 == 0:
            axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], color=color, linewidth=2.0, marker="o", label=label)
        for label, _, color in styles
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    figure.text(
        0.5,
        0.075,
        "Lines show medians; shaded bands show interquartile ranges across accepted branch outcomes.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.6, w_pad=2.0)
    base.save_figure(figure, output_dir, stem)


def plot_trade_offer_comparison(
    trade_offer_rows: pd.DataFrame, output_dir: Path
) -> None:
    panels = (
        ("cross_border_trade_gw", "Cross-border trade [GW]"),
        (
            "flow_weighted_offer_usd_per_kw",
            "Flow-weighted bilateral offer [$/kW]",
        ),
    )
    styles = (
        ("previous_objective", "Previous objective", "#6F7C80"),
        ("clean_objective", "Clean objective", "#A83232"),
    )
    years = np.array(base.MARKET_YEARS)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    for axis, (column, ylabel) in zip(axes, panels):
        for formulation, label, color in styles:
            subset = trade_offer_rows[
                trade_offer_rows["formulation"] == formulation
            ]
            grouped = subset.groupby("year")[column]
            median = grouped.median().reindex(years).to_numpy()
            q25 = grouped.quantile(0.25).reindex(years).to_numpy()
            q75 = grouped.quantile(0.75).reindex(years).to_numpy()
            axis.fill_between(years, q25, q75, color=color, alpha=0.18)
            axis.plot(
                years,
                median,
                color=color,
                linewidth=2.1,
                marker="o",
                label=label,
            )
        axis.set_xticks(years)
        axis.set_xlabel("Market year")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=True, loc="best")
    figure.suptitle(
        "Cross-border trade and offers on realized trade flows",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Lines show medians and bands show interquartile ranges across accepted branch outcomes; offers are weighted by realized cross-border flow.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    base.save_figure(figure, output_dir, "comparison_trade_flows_and_offers")


def comparison_table(
    clean_metrics: pd.DataFrame, previous_metrics: pd.DataFrame
) -> pd.DataFrame:
    metrics = {
        "total_capacity_2040_gw": "Total capacity 2040 (GW)",
        "demand_weighted_price_2040_usd_per_kw": (
            "Demand-weighted price 2040 (USD/kW)"
        ),
        "cross_border_trade_2040_gw": "Cross-border trade 2040 (GW)",
        "average_total_capacity_2025_2040_gw": (
            "Horizon-average total capacity (GW)"
        ),
        "average_demand_weighted_price_2025_2040_usd_per_kw": (
            "Horizon-average demand-weighted price (USD/kW)"
        ),
    }
    rows: list[dict[str, Any]] = []
    for column, label in metrics.items():
        for formulation, frame in (
            ("previous_objective", previous_metrics),
            ("clean_objective", clean_metrics),
        ):
            values = frame[column].to_numpy(dtype=float)
            rows.append(
                {
                    "metric": column,
                    "metric_label": label,
                    "formulation": formulation,
                    "n_accepted_branch_outcomes": len(values),
                    "min": float(np.min(values)),
                    "p10": float(np.percentile(values, 10)),
                    "p25": float(np.percentile(values, 25)),
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "p75": float(np.percentile(values, 75)),
                    "p90": float(np.percentile(values, 90)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def plot_previous_clean_comparison(
    clean_metrics: pd.DataFrame,
    previous_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    panels = (
        (
            "average_total_capacity_2025_2040_gw",
            "Average total manufacturing capacity [GW]",
        ),
        (
            "average_demand_weighted_price_2025_2040_usd_per_kw",
            "Demand-weighted price [$/kW]",
        ),
    )
    labels = ("Previous objective", "Clean objective")
    colors = ("#7F8C8D", "#A83232")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
    rng = np.random.default_rng(20260923)
    for axis, (column, ylabel) in zip(axes, panels):
        values = [
            previous_metrics[column].to_numpy(dtype=float),
            clean_metrics[column].to_numpy(dtype=float),
        ]
        boxes = axis.boxplot(
            values,
            tick_labels=labels,
            widths=0.55,
            patch_artist=True,
            whis=(0, 100),
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.7},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.38)
            patch.set_edgecolor(color)
        for position, (series, color) in enumerate(zip(values, colors), start=1):
            jitter = rng.uniform(-0.08, 0.08, len(series))
            axis.scatter(
                np.full(len(series), position) + jitter,
                series,
                s=19,
                color=color,
                alpha=0.66,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            axis.text(
                position,
                axis.get_ylim()[0],
                f"n={len(series)}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DDDDDD", linestyle=":", linewidth=0.7)
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle(
        "Accepted branch outcomes under the previous and clean objectives",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Descriptive comparison only: accepted sets are selection-conditioned and differ in size.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    base.save_figure(figure, output_dir, "comparison_previous_vs_clean")


def build_report(
    candidate_metrics: pd.DataFrame,
    capacity_rows: pd.DataFrame,
    price_summary: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    associations: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    pass_rates: pd.DataFrame,
    branch_results: pd.DataFrame,
    pca_info: dict[str, Any],
    previous_metrics: pd.DataFrame,
    previous_capacity_rows: pd.DataFrame,
    trade_offer_rows: pd.DataFrame,
) -> str:
    count = len(candidate_metrics)
    cap = candidate_metrics["total_capacity_2040_gw"]
    price = candidate_metrics["demand_weighted_price_2040_usd_per_kw"]
    corr = associations[
        (associations["x"] == "total_capacity_2040_gw")
        & (associations["y"] == "demand_weighted_price_2040_usd_per_kw")
    ].iloc[0]
    widest_price = price_summary[price_summary["year"] == 2040].sort_values(
        "iqr", ascending=False
    ).iloc[0]
    widest_capacity = capacity_summary[
        capacity_summary["year"] == 2040
    ].sort_values("iqr", ascending=False).iloc[0]
    accepted = int(branch_results["accepted"].sum())
    no_pass = int((branch_results["status"] == "no_pass_within_schedule").sum())
    failed = int((branch_results["status"] == "failed").sum())
    old_horizon_capacity = previous_metrics[
        "average_total_capacity_2025_2040_gw"
    ].median()
    clean_horizon_capacity = candidate_metrics[
        "average_total_capacity_2025_2040_gw"
    ].median()
    old_horizon_price = previous_metrics[
        "average_demand_weighted_price_2025_2040_usd_per_kw"
    ].median()
    clean_horizon_price = candidate_metrics[
        "average_demand_weighted_price_2025_2040_usd_per_kw"
    ].median()

    def regional_capacity_median(frame: pd.DataFrame, region: str) -> float:
        return float(
            frame[(frame["region"] == region) & (frame["year"] == 2040)][
                "capacity_gw"
            ].median()
        )

    trade_2040 = trade_offer_rows[trade_offer_rows["year"] == 2040].groupby(
        "formulation"
    )
    old_trade = float(
        trade_2040["cross_border_trade_gw"].median().loc["previous_objective"]
    )
    clean_trade = float(
        trade_2040["cross_border_trade_gw"].median().loc["clean_objective"]
    )
    old_offer = float(
        trade_2040["flow_weighted_offer_usd_per_kw"]
        .median()
        .loc["previous_objective"]
    )
    clean_offer = float(
        trade_2040["flow_weighted_offer_usd_per_kw"]
        .median()
        .loc["clean_objective"]
    )

    lines = [
        "# Clean-objective Stage 2 statistical analysis",
        "",
        "## Scope and selection",
        "",
        f"This analysis uses all **{count} accepted branch outcomes** from the 36-run clean-objective Stage 2 factorial. The objective retains the full market-price producer margin (no `-mu_offer` subtraction) and removes both economic-quadratic and algorithmic-proximal penalties. There were **{accepted} accepted**, **{no_pass} no-pass**, and **{failed} failed** branches.",
        "",
        "Accepted branch outcomes are deterministic, selection-conditioned computational results. They are not independent observations and may represent nearby points in the same equilibrium basin. Consequently, percentiles, correlations, clusters, and matched contrasts are descriptive rather than inferential or causal.",
        "",
        "## Main clean-objective ranges",
        "",
        f"- Total installed capacity in 2040 ranges from **{cap.min():,.1f} to {cap.max():,.1f} GW** (median {cap.median():,.1f} GW).",
        f"- The demand-weighted 2040 clearing price ranges from **{price.min():,.1f} to {price.max():,.1f} USD/kW** (median {price.median():,.1f} USD/kW).",
        f"- Across accepted branch outcomes, capacity and price have **Spearman rho = {corr['spearman_rho']:.2f}** and **Pearson r = {corr['pearson_r']:.2f}**.",
        f"- The widest regional 2040 price IQR is in **{widest_price['region_label']}** ({widest_price['iqr']:.1f} USD/kW); the widest capacity IQR is in **{widest_capacity['region_label']}** ({widest_capacity['iqr']:.1f} GW).",
        "",
        "## Comparison with the previous objective",
        "",
        f"- Median horizon-average total capacity increases from **{old_horizon_capacity:,.1f} to {clean_horizon_capacity:,.1f} GW**, while the median horizon-average demand-weighted price increases from **{old_horizon_price:,.1f} to {clean_horizon_price:,.1f} USD/kW**.",
        f"- The former EU/US exit is not robust to the clean objective. Median 2040 EU capacity changes from **{regional_capacity_median(previous_capacity_rows, 'eu'):,.1f} to {regional_capacity_median(capacity_rows, 'eu'):,.1f} GW** and US capacity from **{regional_capacity_median(previous_capacity_rows, 'us'):,.1f} to {regional_capacity_median(capacity_rows, 'us'):,.1f} GW**.",
        f"- Median 2040 cross-border trade changes from **{old_trade:,.1f} to {clean_trade:,.1f} GW**. The median flow-weighted bilateral offer on realized cross-border trade changes from **{old_offer:,.1f} to {clean_offer:,.1f} USD/kW**.",
        "- These shifts are descriptive: the objective and the accepted branch set both changed.",
        "",
        "## Search success",
        "",
        "| Dimension | Level | Accepted | Branches | Pass rate |",
        "|---|---|---:|---:|---:|",
    ]
    display_rates = pass_rates[
        pass_rates["dimension"].isin(["price_factor", "update_order"])
    ]
    for _, row in display_rates.iterrows():
        lines.append(
            f"| {row['dimension']} | {row['level']} | {int(row['n_accepted'])} | {int(row['n_branches'])} | {100.0 * row['pass_rate']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Matched branch contrasts",
            "",
            "Contrasts retain only accepted pairs that match on the other encoded search settings. They diagnose sensitivity to initialization and damping; they do not identify economic treatment effects.",
            "",
            "| Contrast | Outcome | Pairs | Median difference | Range |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in contrast_summary[
        contrast_summary["metric"].isin(
            {
                "total_capacity_2040_gw",
                "demand_weighted_price_2040_usd_per_kw",
            }
        )
    ].iterrows():
        lines.append(
            f"| {row['contrast']} | {base.readable_metric(row['metric'])} | {int(row['n_pairs'])} | {row['median_percent_difference']:+.2f}% | {row['min_percent_difference']:+.2f}% to {row['max_percent_difference']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Figure assessment",
            "",
            "- **Main-text candidates:** `equilibrium_bands_capacity_by_region`, `equilibrium_bands_prices_by_region`, and `capacity_price_pathway_equilibria`. These communicate the pathways and dispersion without implying a sampling distribution.",
            "- **Direct formulation comparison:** `comparison_previous_vs_clean` uses identical horizon metrics for the prior and clean accepted sets. It remains descriptive because acceptance changes with the objective.",
            "- **Regional mechanism comparison:** `comparison_regional_capacity_paths` makes the disappearance of systematic EU/US exit visible; `comparison_regional_price_paths` shows the associated price dispersion.",
            "- **Trade mechanism comparison:** `comparison_trade_flows_and_offers` compares total cross-border trade and the flow-weighted bilateral offer price on realized trade routes.",
            "- **Algorithm evidence:** `algorithm_convergence_diagnostics` uses only clean-objective Stage 2 audits. The historical mixed-objective convergence figure was deliberately not copied unchanged.",
            "- **Supplementary figures:** boxplots, pass-rate heatmap, matched contrasts, and PCA. PCA families are exploratory and should not be presented as structurally distinct equilibria.",
            "",
            "## Outcome families",
            "",
            f"The first two standardized PCA components explain **{pca_info['pc1_pc2_explained_percent']:.1f}%** of pathway variation. Ward clustering selects **{pca_info['best_k']} descriptive families** with silhouette score {pca_info['best_silhouette']:.3f}.",
            "",
            "## Interpretation limits",
            "",
            "1. The unit of analysis is an accepted algorithm branch, not a random draw and not necessarily a unique equilibrium.",
            "2. Damping, price factor, capacity weight, and update order are search settings rather than economic primitives.",
            "3. Failed and no-pass branches enter pass-rate diagnostics but are excluded from economic-outcome ranges.",
            "4. All acceptance audits are one-start local best-response checks; no global or multistart equilibrium claim is made.",
            "5. Differences from `new_equilibria` combine a changed objective with a changed accepted set, so they should not be interpreted as a clean causal decomposition.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--previous-analysis", type=Path, default=PREVIOUS_ANALYSIS)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_root / "statistical_analysis"
    )
    previous_analysis = args.previous_analysis.resolve()
    if not (run_root / "manifest.json").is_file():
        raise FileNotFoundError(run_root / "manifest.json")
    if not (previous_analysis / "candidate_metrics.csv").is_file():
        raise FileNotFoundError(previous_analysis / "candidate_metrics.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    base.apply_plot_style()

    candidates, manifest = collect_candidates(run_root)
    (
        candidate_metrics,
        capacity_rows,
        price_rows,
        regional_metrics,
        system_metrics,
    ) = base.candidate_and_observation_tables(candidates)
    price_summary = base.describe_groups(
        price_rows, ["year", "region", "region_label"], "price_usd_per_kw"
    )
    capacity_summary = base.describe_groups(
        capacity_rows, ["year", "region", "region_label"], "capacity_gw"
    )
    system_summary = base.metric_summary(system_metrics)
    associations, spearman_matrix = base.build_associations(
        candidate_metrics, regional_metrics
    )
    raw_contrasts, contrast_summary = base.matched_contrasts(candidate_metrics)
    branch_results = collect_branch_results(run_root, manifest)
    initial_audits = collect_initial_audits(run_root)
    pass_rates = base.pass_rate_summary(branch_results)
    clusters, pca_loadings, silhouette, pca_info = base.pca_and_clusters(
        candidates, candidate_metrics
    )
    candidate_metrics = candidate_metrics.merge(
        clusters[["candidate", "figure_code", "family", "pc1_score", "pc2_score"]],
        on="candidate",
        how="left",
        validate="one_to_one",
    )
    candidate_metrics = base.add_horizon_capacity_price_indicators(candidate_metrics)

    previous_metrics = pd.read_csv(previous_analysis / "candidate_metrics.csv")
    if "average_total_capacity_2025_2040_gw" not in previous_metrics.columns:
        previous_metrics = base.add_horizon_capacity_price_indicators(
            previous_metrics
        )
    previous_capacity_rows = pd.read_csv(
        previous_analysis / "capacity_observations.csv"
    )
    previous_price_rows = pd.read_csv(previous_analysis / "price_observations.csv")
    previous_candidates = collect_previous_candidates()
    trade_offer_rows = pd.concat(
        (
            trade_offer_metrics(previous_candidates, "previous_objective"),
            trade_offer_metrics(candidates, "clean_objective"),
        ),
        ignore_index=True,
    )
    comparison = comparison_table(candidate_metrics, previous_metrics)
    tables = {
        "candidate_metrics.csv": candidate_metrics,
        "capacity_observations.csv": capacity_rows,
        "price_observations.csv": price_rows,
        "regional_market_metrics.csv": regional_metrics,
        "system_metrics.csv": system_metrics,
        "price_summary.csv": price_summary,
        "capacity_summary.csv": capacity_summary,
        "system_summary.csv": system_summary,
        "associations.csv": associations,
        "spearman_correlation_matrix.csv": spearman_matrix,
        "matched_contrasts.csv": raw_contrasts,
        "matched_contrast_summary.csv": contrast_summary,
        "branch_results.csv": branch_results,
        "initial_audits.csv": initial_audits,
        "pass_rate_summary.csv": pass_rates,
        "cluster_assignments.csv": clusters,
        "pca_loadings.csv": pca_loadings,
        "cluster_silhouette_scores.csv": silhouette,
        "comparison_previous_vs_clean.csv": comparison,
        "trade_flow_offer_metrics.csv": trade_offer_rows,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")

    count = len(candidates)
    base.draw_boxplots(
        price_rows,
        "price_usd_per_kw",
        base.MARKET_YEARS,
        "Clearing price (USD/kW)",
        f"Regional clearing-price distributions across {count} clean-objective outcomes",
        output_dir,
        "boxplots_market_prices",
    )
    base.draw_boxplots(
        capacity_rows,
        "capacity_gw",
        base.CAPACITY_YEARS,
        "Installed manufacturing capacity (GW)",
        f"Regional capacity distributions across {count} clean-objective outcomes",
        output_dir,
        "boxplots_manufacturing_capacity",
    )
    base.plot_capacity_by_region(capacity_rows, output_dir)
    base.plot_prices_by_region(price_rows, output_dir)
    plot_regional_bands(
        capacity_rows,
        "capacity_gw",
        base.CAPACITY_YEARS,
        "Capacity [GW]",
        "#7570B3",
        output_dir,
        "equilibrium_bands_capacity_by_region",
    )
    plot_regional_bands(
        price_rows,
        "price_usd_per_kw",
        base.MARKET_YEARS,
        "Price [$/kW]",
        "#A83232",
        output_dir,
        "equilibrium_bands_prices_by_region",
    )
    base.plot_capacity_price_scatter(candidate_metrics, associations, output_dir)
    base.plot_horizon_capacity_price_equilibria(candidate_metrics, output_dir)
    base.plot_pass_heatmap(branch_results, output_dir)
    plot_clean_search_diagnostics(branch_results, output_dir)
    base.plot_matched_contrasts(raw_contrasts, output_dir)
    base.plot_pca(clusters, pca_info, output_dir)
    plot_previous_clean_comparison(candidate_metrics, previous_metrics, output_dir)
    plot_regional_formulation_comparison(
        capacity_rows,
        previous_capacity_rows,
        value_column="capacity_gw",
        ylabel="Capacity [GW]",
        output_dir=output_dir,
        stem="comparison_regional_capacity_paths",
    )
    plot_regional_formulation_comparison(
        price_rows,
        previous_price_rows,
        value_column="price_usd_per_kw",
        ylabel="Price [$/kW]",
        output_dir=output_dir,
        stem="comparison_regional_price_paths",
    )
    plot_trade_offer_comparison(trade_offer_rows, output_dir)

    report = build_report(
        candidate_metrics,
        capacity_rows,
        price_summary,
        capacity_summary,
        associations,
        contrast_summary,
        pass_rates,
        branch_results,
        pca_info,
        previous_metrics,
        previous_capacity_rows,
        trade_offer_rows,
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    base.build_workbook_bundle(
        output_dir,
        output_dir / "analysis_bundle.json",
        report,
        candidate_metrics,
        price_summary,
        capacity_summary,
        associations,
        contrast_summary,
        pass_rates,
        clusters,
        price_rows,
        capacity_rows,
        branch_results,
        system_summary,
        pca_info,
    )
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "output_dir": str(output_dir),
                "accepted_branch_outcomes": count,
                "figures_png": len(list(output_dir.glob("*.png"))),
                "figures_pdf": len(list(output_dir.glob("*.pdf"))),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
