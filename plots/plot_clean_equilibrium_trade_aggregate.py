"""Summarize bilateral trade across the reported Stage-2 equilibrium profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_clean_equilibrium_capacity_chords import DEFAULT_RUN, PERIODS
from plot_iter21_capacity_chords import (
    CHORD_COLORS,
    REGION_LABEL,
    REGION_ORDER,
    UNUSED_EDGE_COLOR,
    UNUSED_EDGE_LS,
    UNUSED_EDGE_LW,
    _draw_chord_panel,
)


ACTIVE_ROUTE_GW = 1.0


def _load_profiles(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    selected = pd.read_csv(run_dir / "statistical_analysis" / "csv" / "candidate_metrics.csv")
    if selected.candidate_code.duplicated().any():
        raise ValueError("Candidate codes are not unique")

    capacities = []
    routes = []
    for candidate in selected.itertuples():
        path = (
            run_dir / candidate.candidate /
            f"sweep_{int(candidate.selected_sweep):03d}.json"
        )
        profile = json.loads(path.read_text(encoding="utf-8"))["ending_profile"]
        cap_map = {
            (item["time"], item["player"]): float(item["value"])
            for item in profile["capacities"]
        }
        flow_map = {
            (item["time"], item["exporter"], item["importer"]): float(item["value"])
            for item in profile["market"]["trade_flows"]
        }
        demand_map = {
            (item["time"], item["region"]): float(item["value"])
            for item in profile["market"]["demand"]
        }
        for year in PERIODS:
            for exporter in REGION_ORDER:
                capacity = cap_map[(year, exporter)]
                outbound = sum(
                    flow_map.get((year, exporter, importer), 0.0)
                    for importer in REGION_ORDER
                )
                if outbound > capacity + 0.1:
                    raise ValueError(f"Output exceeds capacity: {candidate.candidate_code} {year} {exporter}")
                capacities.append({
                    "candidate_code": candidate.candidate_code,
                    "year": year,
                    "region": exporter,
                    "capacity_gw": capacity,
                    "unused_gw": max(capacity - outbound, 0.0),
                })
                for importer in REGION_ORDER:
                    routes.append({
                        "candidate_code": candidate.candidate_code,
                        "year": year,
                        "exporter": exporter,
                        "importer": importer,
                        "flow_gw": max(flow_map.get((year, exporter, importer), 0.0), 0.0),
                    })
            for importer in REGION_ORDER:
                inbound = sum(
                    flow_map.get((year, exporter, importer), 0.0)
                    for exporter in REGION_ORDER
                )
                if abs(inbound - demand_map[(year, importer)]) > 0.1:
                    raise ValueError(f"Trade imbalance: {candidate.candidate_code} {year} {importer}")

    return pd.DataFrame(capacities), pd.DataFrame(routes), len(selected)


def _summaries(capacities: pd.DataFrame, routes: pd.DataFrame, n: int):
    route_summary = routes.groupby(["year", "exporter", "importer"], as_index=False).agg(
        mean_gw=("flow_gw", "mean"),
        median_gw=("flow_gw", "median"),
        q1_gw=("flow_gw", lambda x: x.quantile(0.25)),
        q3_gw=("flow_gw", lambda x: x.quantile(0.75)),
        active_profiles=("flow_gw", lambda x: int((x > ACTIVE_ROUTE_GW).sum())),
    )
    route_summary["active_share"] = route_summary.active_profiles / n
    capacity_summary = capacities.groupby(["year", "region"], as_index=False).agg(
        mean_capacity_gw=("capacity_gw", "mean"),
        median_capacity_gw=("capacity_gw", "median"),
        mean_unused_gw=("unused_gw", "mean"),
        median_unused_gw=("unused_gw", "median"),
    )
    return route_summary, capacity_summary


def _mean_chords(route_summary: pd.DataFrame, capacity_summary: pd.DataFrame,
                 n: int, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.2))
    fig.suptitle(f"Mean manufacturing capacity and trade flows across {n} reported profiles",
                 fontsize=15, y=0.975)
    for ax, year in zip(axes.flat, PERIODS):
        cap = capacity_summary[capacity_summary.year == year].set_index("region")["mean_capacity_gw"]
        data = route_summary[route_summary.year == year]
        flows = data.rename(columns={"exporter": "exp", "importer": "imp", "mean_gw": "x"})[
            ["exp", "imp", "x"]
        ]
        _draw_chord_panel(ax, cap, flows, min_share_of_exporter=0.006)
        mean_output = float(flows.x.sum())
        total_cap = float(cap.sum())
        ax.text(0.5, 1.075, f"Supply  |  {year}  |  Demand",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=13, clip_on=False)
        ax.text(0.5, -0.055,
                f"Mean capacity {total_cap:.0f}  ·  "
                f"Mean production {mean_output:.0f}  ·  "
                f"Mean unused {total_cap - mean_output:.0f} GW",
                transform=ax.transAxes, ha="center", va="top", fontsize=10.5, clip_on=False)

    handles = [Patch(facecolor=CHORD_COLORS[r], edgecolor="white") for r in REGION_ORDER]
    labels = [REGION_LABEL[r] for r in REGION_ORDER]
    handles.append(Patch(facecolor=CHORD_COLORS["unused"],
                         edgecolor=UNUSED_EDGE_COLOR, linewidth=UNUSED_EDGE_LW,
                         linestyle=UNUSED_EDGE_LS))
    labels.append("UNUSED CAPACITY")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               bbox_to_anchor=(0.5, 0.055), fontsize=10.5)
    fig.text(0.5, 0.025,
             "Descriptive mean, not an equilibrium. Arc widths show within-period shares; small routes are omitted.",
             ha="center", va="bottom", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.91, bottom=0.14,
                        wspace=-0.13, hspace=0.32)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _route_frequency(route_summary: pd.DataFrame, n: int, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 9.4))
    fig.suptitle(f"How often each cross-border trade route is used ({n} reported profiles)",
                 fontsize=15, y=0.98)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#EEEEEE")
    last_image = None
    for ax, year in zip(axes.flat, PERIODS):
        data = route_summary[route_summary.year == year].set_index(["exporter", "importer"])
        frequency = np.full((len(REGION_ORDER), len(REGION_ORDER)), np.nan)
        for i, exporter in enumerate(REGION_ORDER):
            for j, importer in enumerate(REGION_ORDER):
                if exporter == importer:
                    ax.text(j, i, "domestic", ha="center", va="center",
                            fontsize=7, color="0.45", rotation=45)
                    continue
                row = data.loc[(exporter, importer)]
                frequency[i, j] = row.active_share
                if row.active_profiles:
                    color = "white" if row.active_share >= 0.62 else "#202020"
                    mean_label = (
                        f"{row.mean_gw:.1f}" if row.mean_gw < 1.0
                        else f"{row.mean_gw:.0f}"
                    )
                    ax.text(j, i,
                            f"{int(row.active_profiles)}/{n}\n{mean_label} GW",
                            ha="center", va="center", fontsize=7.5, color=color)
        last_image = ax.imshow(frequency, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(REGION_ORDER)), [REGION_LABEL[r] for r in REGION_ORDER])
        ax.set_yticks(range(len(REGION_ORDER)), [REGION_LABEL[r] for r in REGION_ORDER])
        ax.set_xlabel("Destination")
        ax.set_ylabel("Origin")
        ax.set_title(year, fontsize=13)
        ax.set_xticks(np.arange(-0.5, len(REGION_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(REGION_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar_ax = fig.add_axes([0.89, 0.19, 0.022, 0.63])
    fig.colorbar(last_image, cax=colorbar_ax,
                 label="Share of profiles with flow > 1 GW")
    fig.text(0.5, 0.025,
             "Cell labels: active profiles / total; mean flow across all profiles. Grey diagonal is domestic supply.",
             ha="center", va="bottom", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.08, right=0.84, top=0.92, bottom=0.09,
                        wspace=0.23, hspace=0.3)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "statistical_analysis" / "capacity_trade_chords").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    capacities, routes, n = _load_profiles(run_dir)
    route_summary, capacity_summary = _summaries(capacities, routes, n)
    route_summary.to_csv(out_dir / "aggregate_route_summary.csv", index=False)
    capacity_summary.to_csv(out_dir / "aggregate_capacity_summary.csv", index=False)
    _mean_chords(route_summary, capacity_summary, n, out_dir / "aggregate_mean_capacity_trade.png")
    _route_frequency(route_summary, n, out_dir / "aggregate_route_frequency.png")
    print(f"Saved mean chords and route-frequency heatmap for {n} profiles in {out_dir}")


if __name__ == "__main__":
    main()
