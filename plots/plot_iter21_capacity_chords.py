from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Patch, PathPatch, Wedge
from matplotlib.path import Path as MplPath

# Match IEEEtran serif rendering and text sizes used by plots/plot_prices.py.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


@dataclass(frozen=True)
class PlotConfig:
    excel_path: str = "outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx"
    out_dir: str = "outputs/figures"
    iteration: int = 21
    periods: Tuple[str, ...] = ("2025", "2030", "2035", "2040")
    out_name: str = "capacity_trade_iter21_2x2"


REGION_ORDER = ["ch", "eu", "us", "apac", "af", "row"]
REGION_LABEL = {"ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "af": "AF", "row": "ROW"}
DEST_ORDER = ["unused", *REGION_ORDER]

CHORD_COLORS: Dict[str, str] = {
    # Exact regional colors from plots/plot_capacity_epec_demand.py,
    # used for Fig. capacity_epec_stacked_with_global_demand in the paper.
    "ch": "#CA6180",
    "eu": "#FEFD99",
    "us": "#FCB7C7",
    "apac": "#B7A6D8",
    "af": "#B8D99E",
    "row": "#9ED3DC",
    "unused": "#F2F2F2",
}

EXPORT_ALPHA = 0.45
UNUSED_ALPHA = 0.70
MIN_SHARE_OF_EXPORTER = 0.006
MIN_CAPACITY = 1e-6
R_OUT = 1.0
RING_WIDTH = 0.16
R_IN = R_OUT - RING_WIDTH
EXP_START, EXP_END = 90.0, 270.0
DEST_START, DEST_END = 270.0, 450.0
GAP_DEG = 3.0
UNUSED_EXTRA_GAP = 6.0
UNUSED_EDGE_COLOR = "0.45"
UNUSED_EDGE_LW = 1.4
UNUSED_EDGE_LS = (0, (4, 3))


def _assert_excel_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")


def _polar_xy(angle_deg: float, radius: float) -> Tuple[float, float]:
    rad = np.deg2rad(angle_deg)
    return radius * float(np.cos(rad)), radius * float(np.sin(rad))


def _arc_points(a0: float, a1: float, radius: float, n: int = 18) -> list[Tuple[float, float]]:
    return [_polar_xy(a, radius) for a in np.linspace(a0, a1, max(2, n))]


def _build_arc_spans(
    names: list[str],
    values: pd.Series,
    start_deg: float,
    end_deg: float,
    gap_deg: float,
    extra_gap_after: Dict[str, float] | None = None,
) -> Dict[str, Tuple[float, float]]:
    extra_gap_after = extra_gap_after or {}
    total_span = end_deg - start_deg
    base_gaps = gap_deg * (len(names) - 1) if len(names) > 1 else 0.0
    extra_gaps = sum(extra_gap_after.get(name, 0.0) for name in names[:-1])
    avail = max(0.0, total_span - base_gaps - extra_gaps)
    vals = np.maximum(np.array([values.get(name, 0.0) for name in names], float), 0.0)
    total = float(vals.sum())

    spans: Dict[str, Tuple[float, float]] = {}
    cursor = start_deg
    for idx, (name, value) in enumerate(zip(names, vals)):
        width = 0.0 if total <= 0.0 else avail * (value / total)
        spans[name] = (cursor, cursor + width)
        cursor += width
        if idx < len(names) - 1:
            cursor += gap_deg + extra_gap_after.get(name, 0.0)
    return spans


def _add_ribbon(ax: plt.Axes, a0: float, a1: float, b0: float, b1: float, color, alpha: float) -> None:
    r_attach = R_IN - 0.010
    chord_len = abs((a0 + a1) / 2.0 - (b0 + b1) / 2.0)
    r_ctrl = 0.18 + 0.10 * (1.0 - np.exp(-chord_len / 70.0))
    p0, p1 = _polar_xy(a0, r_attach), _polar_xy(b0, r_attach)
    p2, p3 = _polar_xy(b1, r_attach), _polar_xy(a1, r_attach)
    c0, c1 = _polar_xy(a0, r_ctrl), _polar_xy(b0, r_ctrl)
    c2, c3 = _polar_xy(b1, r_ctrl), _polar_xy(a1, r_ctrl)

    verts = [p0]
    codes = [MplPath.MOVETO]
    verts += [c0, c1, p1]
    codes += [MplPath.CURVE4] * 3
    for point in _arc_points(b0, b1, r_attach)[1:]:
        verts.append(point)
        codes.append(MplPath.LINETO)
    verts += [c2, c3, p3]
    codes += [MplPath.CURVE4] * 3
    for point in _arc_points(a1, a0, r_attach)[1:]:
        verts.append(point)
        codes.append(MplPath.LINETO)
    verts.append((0.0, 0.0))
    codes.append(MplPath.CLOSEPOLY)

    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor=(0, 0, 0, 0.10),
            lw=0.25,
            alpha=alpha,
            zorder=2,
        )
    )


def _mid_angle(span: Tuple[float, float]) -> float:
    return 0.5 * (span[0] + span[1])


def _load_iteration_data(excel_path: str, iteration: int, period: str) -> tuple[pd.Series, pd.DataFrame]:
    detail = pd.read_excel(excel_path, sheet_name="detailed_iters")
    detail["t"] = detail["t"].astype(str)
    detail["r"] = detail["r"].astype(str).str.strip().str.lower()
    selected = detail[(detail["iter"] == iteration) & (detail["t"] == str(period))].copy()
    if selected.empty:
        raise ValueError(f"No detailed_iters rows found for iter={iteration}, period={period}")

    cap = (
        selected.set_index("r")["Kcap"]
        .reindex(REGION_ORDER)
        .apply(lambda x: max(0.0, float(pd.to_numeric(x, errors="coerce") or 0.0)))
    )

    flow_rows = []
    for _, row in selected.iterrows():
        exp = str(row["r"]).strip().lower()
        if exp not in REGION_ORDER:
            continue
        for imp in REGION_ORDER:
            col = f"x_exp_to_{imp}"
            if col not in selected.columns:
                raise ValueError(f"Missing route column in detailed_iters: {col}")
            value = float(pd.to_numeric(row[col], errors="coerce") or 0.0)
            if value > 0.0:
                flow_rows.append({"exp": exp, "imp": imp, "x": value})

    flows = pd.DataFrame(flow_rows, columns=["exp", "imp", "x"])
    flows = flows.groupby(["exp", "imp"], as_index=False)["x"].sum()
    return cap, flows


def _draw_chord_panel(ax: plt.Axes, cap: pd.Series, flows_all: pd.DataFrame) -> None:
    cap = cap.reindex(REGION_ORDER).fillna(0.0)
    left_exporters = [region for region in REGION_ORDER if float(cap.get(region, 0.0)) > MIN_CAPACITY]
    cap_left = cap.reindex(left_exporters).fillna(0.0)

    flows_all = flows_all.copy()
    flows_all["x"] = pd.to_numeric(flows_all["x"], errors="coerce").fillna(0.0)
    flows_all = flows_all[flows_all["x"] > 0.0].copy()

    used_by_exp = flows_all.groupby("exp")["x"].sum().reindex(left_exporters).fillna(0.0)
    flows_plot = flows_all.copy()
    unused_rows = []
    for region in left_exporters:
        unused = max(0.0, float(cap_left[region]) - float(used_by_exp.get(region, 0.0)))
        if unused > 0.0:
            unused_rows.append({"exp": region, "imp": "unused", "x": unused})
    if unused_rows:
        flows_plot = pd.concat([flows_plot, pd.DataFrame(unused_rows)], ignore_index=True)

    flows_plot["share_exp"] = flows_plot.apply(
        lambda row: (
            float(row["x"]) / float(cap.get(row["exp"], 0.0))
            if float(cap.get(row["exp"], 0.0)) > 0.0
            else 0.0
        ),
        axis=1,
    )
    keep = (flows_plot["imp"] == "unused") | (flows_plot["share_exp"] >= MIN_SHARE_OF_EXPORTER)
    flows_plot = flows_plot[keep].copy()

    recv = flows_plot.groupby("imp")["x"].sum().reindex(DEST_ORDER).fillna(0.0)
    exp_spans = _build_arc_spans(left_exporters, cap_left, EXP_START, EXP_END, GAP_DEG)
    dest_spans = _build_arc_spans(DEST_ORDER, recv, DEST_START, DEST_END, GAP_DEG, {"unused": UNUSED_EXTRA_GAP})

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_patch(Circle((0, 0), R_OUT, facecolor="none", edgecolor="0.35", lw=0.9))
    ax.plot([0, 0], [-1.06, 1.06], linestyle=(0, (6, 6)), lw=1.0, color="0.35")

    for region in left_exporters:
        a0, a1 = exp_spans[region]
        if a1 > a0 + 1e-9:
            ax.add_patch(
                Wedge(
                    (0, 0),
                    R_OUT,
                    a0,
                    a1,
                    width=RING_WIDTH,
                    facecolor=CHORD_COLORS[region],
                    edgecolor="white",
                    lw=0.9,
                    zorder=3,
                )
            )

    for dest in DEST_ORDER:
        b0, b1 = dest_spans[dest]
        if b1 <= b0 + 1e-9:
            continue
        if dest == "unused":
            ax.add_patch(
                Wedge(
                    (0, 0),
                    R_OUT,
                    b0,
                    b1,
                    width=RING_WIDTH,
                    facecolor=CHORD_COLORS[dest],
                    edgecolor=UNUSED_EDGE_COLOR,
                    lw=UNUSED_EDGE_LW,
                    linestyle=UNUSED_EDGE_LS,
                    zorder=3,
                )
            )
        else:
            ax.add_patch(
                Wedge(
                    (0, 0),
                    R_OUT,
                    b0,
                    b1,
                    width=RING_WIDTH,
                    facecolor=CHORD_COLORS[dest],
                    edgecolor="white",
                    lw=0.9,
                    zorder=3,
                )
            )

    exp_cursor = {region: exp_spans[region][0] for region in left_exporters}
    dest_cursor = {dest: dest_spans[dest][0] for dest in DEST_ORDER}
    recv_safe = recv.reindex(DEST_ORDER).fillna(0.0)

    flows_non_unused = flows_plot[flows_plot["imp"] != "unused"].sort_values("x", ascending=False).copy()
    flows_unused = flows_plot[flows_plot["imp"] == "unused"].copy()
    flows_unused["exp_mid"] = flows_unused["exp"].map({r: _mid_angle(exp_spans[r]) for r in left_exporters}).fillna(0.0)
    flows_unused = flows_unused.sort_values("exp_mid", ascending=True)

    def draw_flow(exp: str, dest: str, value: float) -> None:
        if exp not in exp_spans or dest not in DEST_ORDER:
            return
        if float(cap.get(exp, 0.0)) <= 0.0 or float(recv_safe.get(dest, 0.0)) <= 0.0 or value <= 0.0:
            return
        exp_span = exp_spans[exp][1] - exp_spans[exp][0]
        dest_span = dest_spans[dest][1] - dest_spans[dest][0]
        if exp_span <= 1e-9 or dest_span <= 1e-9:
            return
        da = exp_span * (value / float(cap.get(exp, 0.0)))
        db = dest_span * (value / float(recv_safe.get(dest, 0.0)))

        a0 = exp_cursor[exp]
        b0 = dest_cursor[dest]
        a1 = min(exp_spans[exp][1], a0 + da)
        b1 = min(dest_spans[dest][1], b0 + db)
        exp_cursor[exp] = a1
        dest_cursor[dest] = b1
        if a1 <= a0 + 1e-6 or b1 <= b0 + 1e-6:
            return
        _add_ribbon(ax, a0, a1, b0, b1, CHORD_COLORS[exp], UNUSED_ALPHA if dest == "unused" else EXPORT_ALPHA)

    for _, row in flows_non_unused.iterrows():
        draw_flow(str(row["exp"]), str(row["imp"]), float(row["x"]))
    for _, row in flows_unused.iterrows():
        draw_flow(str(row["exp"]), "unused", float(row["x"]))

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.08)


def plot_capacity_trade_2x2(cfg: PlotConfig) -> tuple[str, str]:
    _assert_excel_exists(cfg.excel_path)
    os.makedirs(cfg.out_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 15, "legend.fontsize": 13})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0))
    axes_flat = axes.ravel()

    for ax, period in zip(axes_flat, cfg.periods):
        cap, flows = _load_iteration_data(cfg.excel_path, cfg.iteration, period)
        _draw_chord_panel(ax, cap, flows)
        ax.text(
            0.15,
            1.085,
            "Supply",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="normal",
            color="0.15",
            clip_on=False,
        )
        ax.text(
            0.34,
            1.085,
            "|",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="normal",
            color="0.15",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.085,
            f"{period}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="normal",
            color="0.15",
            clip_on=False,
        )
        ax.text(
            0.66,
            1.085,
            "|",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="normal",
            color="0.15",
            clip_on=False,
        )
        ax.text(
            0.85,
            1.085,
            "Demand",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="normal",
            color="0.15",
            clip_on=False,
        )

    handles = [Patch(facecolor=CHORD_COLORS[r], edgecolor="white", linewidth=0.8) for r in REGION_ORDER]
    labels = [REGION_LABEL[r] for r in REGION_ORDER]
    handles.append(
        Patch(
            facecolor=CHORD_COLORS["unused"],
            edgecolor=UNUSED_EDGE_COLOR,
            linewidth=UNUSED_EDGE_LW,
            linestyle=UNUSED_EDGE_LS,
        )
    )
    labels.append("UNUSED CAPACITY")
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=True,
        fancybox=True,
        edgecolor="0.55",
        bbox_to_anchor=(0.5, 0.018),
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.45,
        labelspacing=0.35,
        fontsize=13,
    )
    legend.get_frame().set_linewidth(0.9)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.12, wspace=-0.28, hspace=0.34)

    png_path = os.path.join(cfg.out_dir, f"{cfg.out_name}.png")
    pdf_path = os.path.join(cfg.out_dir, f"{cfg.out_name}.pdf")
    fig.savefig(png_path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png_path, pdf_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2x2 iteration-specific trade/capacity chord diagrams.")
    parser.add_argument("--excel-path", default=PlotConfig.excel_path)
    parser.add_argument("--out-dir", default=PlotConfig.out_dir)
    parser.add_argument("--iteration", type=int, default=PlotConfig.iteration)
    parser.add_argument("--periods", nargs=4, default=list(PlotConfig.periods), help="Exactly four periods for the 2x2 panels.")
    parser.add_argument("--out-name", default=PlotConfig.out_name)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PlotConfig(
        excel_path=args.excel_path,
        out_dir=args.out_dir,
        iteration=args.iteration,
        periods=tuple(str(p) for p in args.periods),
        out_name=args.out_name,
    )
    png_path, pdf_path = plot_capacity_trade_2x2(cfg)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
