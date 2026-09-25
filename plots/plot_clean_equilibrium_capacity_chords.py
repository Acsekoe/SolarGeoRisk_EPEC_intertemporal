"""Capacity, trade, and unused-capacity chords for accepted Stage-2 profiles.

Each image uses the four-period layout of plot_iter21_capacity_chords.py. The
underlying data are the selected sweep JSON files, not intermediate iterations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from plot_iter21_capacity_chords import (
    CHORD_COLORS,
    REGION_LABEL,
    REGION_ORDER,
    UNUSED_EDGE_COLOR,
    UNUSED_EDGE_LS,
    UNUSED_EDGE_LW,
    _draw_chord_panel,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "clean_stage2_factorial_20260923_123037"
PERIODS = ("2025", "2030", "2035", "2040")
FLOW_FLOOR_GW = 0.01  # Hide only numerical dust; retain all material routes.
BALANCE_TOL_GW = 0.1


def _period_data(profile: dict, period: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    caps = pd.Series(
        {
            item["player"]: float(item["value"])
            for item in profile["capacities"]
            if str(item["time"]) == period
        },
        dtype=float,
    ).reindex(REGION_ORDER)
    if caps.isna().any():
        raise ValueError(f"Missing capacity for {period}: {caps[caps.isna()].index.tolist()}")

    market = profile["market"]
    flows = pd.DataFrame(
        [
            {"exp": item["exporter"], "imp": item["importer"], "x": float(item["value"])}
            for item in market["trade_flows"]
            if str(item["time"]) == period
        ]
    )
    if flows.empty:
        raise ValueError(f"No bilateral flows for {period}")
    flows = flows.groupby(["exp", "imp"], as_index=False)["x"].sum()

    demand = pd.Series(
        {
            item["region"]: float(item["value"])
            for item in market["demand"]
            if str(item["time"]) == period
        },
        dtype=float,
    ).reindex(REGION_ORDER)
    if demand.isna().any():
        raise ValueError(f"Missing demand for {period}: {demand[demand.isna()].index.tolist()}")

    outbound = flows.groupby("exp")["x"].sum().reindex(REGION_ORDER).fillna(0.0)
    inbound = flows.groupby("imp")["x"].sum().reindex(REGION_ORDER).fillna(0.0)
    if (outbound - caps).max() > BALANCE_TOL_GW:
        raise ValueError(f"Production exceeds capacity in {period}")
    if (inbound - demand).abs().max() > BALANCE_TOL_GW:
        raise ValueError(f"Trade does not balance demand in {period}")

    # Tiny solver flows need not appear as hairline ribbons. Account for them as
    # output in the annotations, while retaining all material trade routes.
    visible_flows = flows[flows["x"] >= FLOW_FLOOR_GW].copy()
    stats = {
        "capacity_gw": float(caps.sum()),
        "production_gw": float(outbound.sum()),
        "demand_gw": float(demand.sum()),
        "unused_gw": float((caps - outbound).clip(lower=0).sum()),
        "cross_border_gw": float(flows.loc[flows.exp != flows.imp, "x"].sum()),
    }
    return caps, visible_flows, stats


def _figure(candidate: pd.Series, profile: dict) -> tuple[plt.Figure, list[dict]]:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.2))
    fig.suptitle(
        f"{candidate['figure_code']}  |  {candidate['candidate_code']}"
        f"  |  best-response gain {candidate['max_gain_percent']:.2f}%",
        fontsize=14,
        y=0.975,
    )

    records = []
    for ax, period in zip(axes.flat, PERIODS):
        caps, flows, stats = _period_data(profile, period)
        _draw_chord_panel(ax, caps, flows, min_share_of_exporter=0.0)
        ax.text(
            0.5, 1.075, f"Supply  |  {period}  |  Demand",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=13,
            clip_on=False,
        )
        ax.text(
            0.5, -0.055,
            f"Capacity {stats['capacity_gw']:.0f}  ·  "
            f"Production {stats['production_gw']:.0f}  ·  "
            f"Unused {stats['unused_gw']:.0f} GW",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
            clip_on=False,
        )
        records.append({
            "figure_code": candidate["figure_code"],
            "candidate_code": candidate["candidate_code"],
            "candidate": candidate["candidate"],
            "year": period,
            **stats,
        })

    handles = [Patch(facecolor=CHORD_COLORS[r], edgecolor="white") for r in REGION_ORDER]
    labels = [REGION_LABEL[r] for r in REGION_ORDER]
    handles.append(Patch(
        facecolor=CHORD_COLORS["unused"], edgecolor=UNUSED_EDGE_COLOR,
        linewidth=UNUSED_EDGE_LW, linestyle=UNUSED_EDGE_LS,
    ))
    labels.append("UNUSED CAPACITY")
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=True,
        bbox_to_anchor=(0.5, 0.055), fontsize=10.5,
    )
    fig.text(
        0.5, 0.025,
        "Arc widths show shares within each period; GW totals beneath panels give absolute scale.",
        ha="center", va="bottom", fontsize=9, color="0.35",
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.91, bottom=0.14,
                        wspace=-0.13, hspace=0.32)
    return fig, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--candidate-code", default=None,
                        help="Render one accepted profile; default renders all.")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or run_dir / "statistical_analysis" / "capacity_trade_chords").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(run_dir / "statistical_analysis" / "csv" / "candidate_metrics.csv")
    if args.candidate_code:
        candidates = candidates[candidates.candidate_code == args.candidate_code]
        if candidates.empty:
            raise ValueError(f"Accepted candidate not found: {args.candidate_code}")
    candidates = candidates.sort_values("figure_code")

    if not args.candidate_code:
        current_names = {
            f"{row['figure_code']}_{row['candidate_code']}_capacity_trade.png"
            for _, row in candidates.iterrows()
        }
        for old_plot in out_dir.glob("*_capacity_trade.png"):
            if old_plot.name not in current_names:
                old_plot.unlink()

    all_records = []
    for _, candidate in candidates.iterrows():
        sweep_path = (
            run_dir / candidate["candidate"] /
            f"sweep_{int(candidate['selected_sweep']):03d}.json"
        )
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        fig, records = _figure(candidate, payload["ending_profile"])
        stem = f"{candidate['figure_code']}_{candidate['candidate_code']}_capacity_trade"
        fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        all_records.extend(records)

    index_name = (
        f"figure_index_{args.candidate_code}.csv"
        if args.candidate_code else "figure_index.csv"
    )
    pd.DataFrame(all_records).to_csv(out_dir / index_name, index=False)
    print(f"Saved {len(candidates)} four-period figures in {out_dir}")


if __name__ == "__main__":
    main()
