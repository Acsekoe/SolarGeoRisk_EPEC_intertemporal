"""Regenerate the four paper-result figures for every packaged candidate.

This batch generator adapts the visual design and accounting used by:

* ``plots/plot_prices.py``
* ``plots/plot_welfare.py``
* ``plots/plot_capacity_epec_demand.py``

Two welfare figures are generated.  The first reports each region's
equilibrium-minus-LLP welfare change relative to that region's LLP welfare and
decomposes the change into consumer-surplus (CS) and producer-surplus (PS)
contributions.  The second exactly follows the paper's period-stacked absolute
gain/loss design in billion USD/year.  PS is net of manufacturing, shipping,
capacity-holding, and investment costs.  The terminal salvage credit is
deliberately excluded from the plotted welfare accounting.

The 2045 value remains active when solving the planner's terminal capacity
state, but all plotted market and operating-welfare results end in 2040.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from scripts.stage2_results_selection import excluded_candidate_ids
DEFAULT_PACKAGE_DIR = ROOT_DIR / "outputs" / "15_equilibria"
DEFAULT_PLOTS_DIR = DEFAULT_PACKAGE_DIR / "plots"
DEFAULT_PLANNER_PATH = DEFAULT_PACKAGE_DIR / "planner_benchmark_corrected.xlsx"

PERIODS = ("2025", "2030", "2035", "2040")
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
REGION_NAMES = {
    "ch": "China",
    "eu": "Europe",
    "us": "United States",
    "apac": "Asia-Pacific",
    "af": "Africa",
    "row": "Rest of World",
}
REGION_LABELS = {
    "ch": "CH",
    "eu": "EU",
    "us": "US",
    "apac": "APAC",
    "af": "AF",
    "row": "ROW",
}
REGION_COLORS = {
    "ch": "#CA6180",
    "eu": "#FEFD99",
    "us": "#FCB7C7",
    "apac": "#B7A6D8",
    "af": "#B8D99E",
    "row": "#9ED3DC",
}

COLOR_PLAN = "#2E6F40"
COLOR_CANDIDATE = "#A83232"
COLOR_COST = "#6E6E6E"
COLOR_DEMAND = "#222222"
COLOR_CS_GAIN = "#A8D5A2"
COLOR_CS_LOSS = "#F1948A"
COLOR_PS_GAIN = "#2E6F40"
COLOR_PS_LOSS = "#A83232"
COLOR_GRID = "#D0D0D0"

# USD/kW multiplied by GW is million USD.
MUSD_TO_TUSD = 1e6


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


@dataclass(frozen=True)
class Candidate:
    sequence: str
    branch: str
    sweep: int
    source_path: Path
    payload: dict[str, Any]

    @property
    def output_dir_parts(self) -> tuple[str, str]:
        return self.sequence, self.branch

    @property
    def label(self) -> str:
        return f"{self.sequence}/{self.branch}/sweep_{self.sweep:03d}"


class SplitLegendKey:
    """Legend key pairing the gain and loss colors of one welfare component."""

    def __init__(
        self, gain_color: str, loss_color: str, *, loss_first: bool = True
    ):
        self.gain_color = gain_color
        self.loss_color = loss_color
        self.loss_first = loss_first


class SplitLegendHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        half_width = width / 2.0
        first_color = (
            orig_handle.loss_color
            if orig_handle.loss_first
            else orig_handle.gain_color
        )
        second_color = (
            orig_handle.gain_color
            if orig_handle.loss_first
            else orig_handle.loss_color
        )
        first = Rectangle(
            (xdescent, ydescent),
            half_width,
            height,
            facecolor=first_color,
            edgecolor="white",
            linewidth=0.8,
            transform=trans,
        )
        second = Rectangle(
            (xdescent + half_width, ydescent),
            half_width,
            height,
            facecolor=second_color,
            edgecolor="white",
            linewidth=0.8,
            transform=trans,
        )
        border = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor="none",
            edgecolor="#BBBBBB",
            linewidth=0.6,
            transform=trans,
        )
        return [first, second, border]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_recorded_path(path_text: str) -> Path:
    path = Path(str(path_text).replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()
    for root in (ROOT_DIR, ROOT_DIR.parent):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (ROOT_DIR / path).resolve()


def load_candidates(
    package_dir: Path,
    workflow_manifest: dict[str, Any] | None = None,
) -> list[Candidate]:
    if workflow_manifest is None:
        profile_root = package_dir / "profiles"
        paths = sorted(profile_root.glob("*/*/sweep_*.json"))
        source_description = f"below {profile_root}"
    else:
        workflow_results = workflow_manifest.get("grid_results")
        if workflow_results is None:
            workflow_results = workflow_manifest.get("results", [])
        accepted = [
            result
            for result in workflow_results
            if bool(result.get("local_one_percent_equilibrium", False))
        ]
        paths = sorted(
            (resolve_recorded_path(str(result["selected_profile"])) for result in accepted),
            key=lambda path: path.as_posix(),
        )
        source_description = "listed as accepted in the workflow manifest"

    if not paths:
        raise ValueError(f"No accepted candidate profiles found {source_description}")

    candidates = []
    for path in paths:
        payload = load_json(path)
        candidates.append(
            Candidate(
                sequence=str(payload["sequence"]),
                branch=str(payload["branch"]),
                sweep=int(payload["sweep"]),
                source_path=path,
                payload=payload,
            )
        )
    return candidates


def configure_model_data(input_path: Path, terminal_salvage_fraction: float):
    """Load the exact corrected calibration used for the 15 profiles."""
    from model.data_prep import load_data_from_excel
    from model import run_gs

    cfg = run_gs.RunConfig(
        excel_path=str(input_path),
        params_region_sheet="params_region_new",
        discount_rate=0.02,
        base_year=2025,
        terminal_salvage_fraction=float(terminal_salvage_fraction),
        terminal_capacity_state_only=True,
        fix_q_offer_to_kcap=True,
        fix_a_bid_to_true_dem=True,
        force_mu_offer_zero=False,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_data_from_excel(
            str(input_path), params_region_sheet=cfg.params_region_sheet
        )
        run_gs._apply_data_overrides(data, cfg)
    return data


def planner_meta(planner_path: Path) -> dict[str, str]:
    if not planner_path.exists():
        return {}
    meta = pd.read_excel(planner_path, sheet_name="meta")
    return {str(row["key"]): str(row["value"]) for _, row in meta.iterrows()}


def planner_is_current(
    planner_path: Path,
    input_path: Path,
    terminal_salvage_fraction: float,
) -> bool:
    meta = planner_meta(planner_path)
    try:
        recorded_salvage = float(meta.get("terminal_salvage_fraction", "nan"))
    except ValueError:
        return False
    return (
        meta.get("input_sha256", "").upper() == sha256(input_path)
        and np.isclose(recorded_salvage, terminal_salvage_fraction)
        and meta.get("discount_rate") == "0.02"
        and meta.get("terminal_capacity_state_only", "").lower() == "true"
    )


def write_planner_workbook(
    planner_path: Path,
    state: dict[str, Any],
    data,
    input_path: Path,
    terminal_salvage_fraction: float,
) -> None:
    times = list(data.times or [])
    plan_times = times[:-1]
    c_man = data.c_man_t or {
        (r, t): float(data.c_man[r]) for r in data.regions for t in times
    }
    rows_regions = []
    for region in data.regions:
        for period in plan_times:
            investment = float(state["Icap_pos"].get((region, period), 0.0))
            decommissioning = float(
                state["Dcap_neg"].get((region, period), 0.0)
            )
            rows_regions.append(
                {
                    "r": region,
                    "t": period,
                    "Kcap": float(state["Kcap"].get((region, period), 0.0)),
                    "Icap_report": investment,
                    "Dcap_report": decommissioning,
                    "x_dem": float(state["x_dem"].get((region, period), 0.0)),
                    "x_man": float(state["x_man"].get((region, period), 0.0)),
                    "lam": float(state["lam"].get((region, period), 0.0)),
                    "mu_cap": float(state["mu_cap"].get((region, period), 0.0)),
                    "c_man_t": float(c_man[(region, period)]),
                }
            )

    rows_flows = []
    for exporter in data.regions:
        for importer in data.regions:
            for period in plan_times:
                flow = float(state["x"].get((exporter, importer, period), 0.0))
                if flow > 1e-9:
                    rows_flows.append(
                        {
                            "exp": exporter,
                            "imp": importer,
                            "t": period,
                            "x": flow,
                            "c_ship": float(data.c_ship[(exporter, importer)]),
                            "c_man": float(c_man[(exporter, period)]),
                        }
                    )

    rows_meta = [
        {"key": "model", "value": "corrected_llp_planner"},
        {"key": "created", "value": datetime.now().astimezone().isoformat()},
        {"key": "input", "value": str(input_path)},
        {"key": "input_sha256", "value": sha256(input_path)},
        {"key": "params_sheet", "value": "params_region_new"},
        {"key": "discount_rate", "value": "0.02"},
        {"key": "base_year", "value": "2025"},
        {
            "key": "terminal_salvage_fraction",
            "value": str(float(terminal_salvage_fraction)),
        },
        {"key": "terminal_capacity_state_only", "value": "true"},
        {"key": "operating_periods", "value": str(plan_times)},
        {"key": "objective_including_salvage", "value": str(state["obj_total"])},
    ]

    planner_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(planner_path, engine="openpyxl") as writer:
        pd.DataFrame(rows_regions).to_excel(writer, sheet_name="regions", index=False)
        pd.DataFrame(rows_flows).to_excel(writer, sheet_name="flows", index=False)
        pd.DataFrame(rows_meta).to_excel(writer, sheet_name="meta", index=False)


def build_planner_benchmark(
    planner_path: Path,
    data,
    input_path: Path,
    terminal_salvage_fraction: float,
) -> None:
    from model.model_llp_planner import (
        build_llp_planner_model,
        extract_llp_state,
        solve_llp_planner,
        validate_llp_solution,
    )

    workdir = Path(tempfile.mkdtemp(prefix="solargeorisk_planner_"))
    context = None
    try:
        context = build_llp_planner_model(data, working_directory=str(workdir))
        solve_llp_planner(context, solver="ipopt")
        state = extract_llp_state(context, data)
        warnings = validate_llp_solution(state, data)
        if warnings:
            raise RuntimeError("Corrected planner validation failed:\n" + "\n".join(warnings))
        write_planner_workbook(
            planner_path,
            state,
            data,
            input_path,
            terminal_salvage_fraction,
        )
    finally:
        if context is not None:
            context.container.close()
        shutil.rmtree(workdir, ignore_errors=True)


def load_planner(planner_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = pd.read_excel(planner_path, sheet_name="regions")
    flows = pd.read_excel(planner_path, sheet_name="flows")
    regions["r"] = regions["r"].astype(str).str.lower()
    regions["t"] = regions["t"].astype(str)
    flows["exp"] = flows["exp"].astype(str).str.lower()
    flows["imp"] = flows["imp"].astype(str).str.lower()
    flows["t"] = flows["t"].astype(str)
    return regions, flows


def profile_records(
    candidate: Candidate,
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str, str], float],
    dict[tuple[str, str], float],
]:
    ending = candidate.payload["ending_profile"]
    prices = {
        (str(row["region"]).lower(), str(row["time"])): float(row["value"])
        for row in ending["market"]["clearing_prices"]
    }
    demand = {
        (str(row["region"]).lower(), str(row["time"])): float(row["value"])
        for row in ending["market"]["demand"]
    }
    capacities = {
        (str(row["player"]).lower(), str(row["time"])): float(row["value"])
        for row in ending["capacities"]
    }
    flows = {
        (
            str(row["exporter"]).lower(),
            str(row["importer"]).lower(),
            str(row["time"]),
        ): float(row["value"])
        for row in ending["market"]["trade_flows"]
    }
    d_k_net = {
        (str(row["region"]).lower(), str(row["time"])): float(row["value"])
        for row in ending["strategy"]["dK_net"]
    }
    return prices, demand, capacities, flows, d_k_net


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prices(
    candidate: Candidate,
    planner_regions: pd.DataFrame,
    data,
    output_dir: Path,
    dpi: int,
) -> None:
    prices, _, _, _, _ = profile_records(candidate)
    costs = data.c_man_t or {
        (r, t): float(data.c_man[r]) for r in REGIONS for t in PERIODS
    }
    years = [int(period) for period in PERIODS]

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.4), sharey=False)
    for index, (ax, region) in enumerate(zip(axes.flat, REGIONS)):
        planner = (
            planner_regions[planner_regions["r"] == region]
            .set_index("t")
            .reindex(PERIODS)
        )
        ax.plot(
            years,
            planner["lam"].to_numpy(float),
            color=COLOR_PLAN,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            label="Global welfare maximization",
        )
        ax.plot(
            years,
            [prices[(region, period)] for period in PERIODS],
            color=COLOR_CANDIDATE,
            linewidth=2.2,
            marker="s",
            markersize=5.5,
            label="Strategic market-clearing prices",
        )
        ax.plot(
            years,
            [float(costs[(region, period)]) for period in PERIODS],
            color=COLOR_COST,
            linewidth=1.8,
            linestyle="--",
            marker="^",
            markersize=5.0,
            label="Regional manufacturing costs",
        )
        ax.set_title(REGION_NAMES[region], fontsize=18, fontweight="normal")
        ax.set_xticks(years)
        if index % 2 == 0:
            ax.set_ylabel("Price [$/kW]", fontsize=18)
        ax.set_ylim(0, 370)
        ax.set_yticks([0, 100, 200, 300])
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="both", labelsize=15)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=1,
        fontsize=15,
        framealpha=0.9,
        handletextpad=0.5,
        borderpad=0.45,
        labelspacing=0.35,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(
        left=0.13, right=0.98, top=0.95, bottom=0.17, wspace=0.34, hspace=0.50
    )
    save_figure(fig, output_dir, "market_prices", dpi)


def plot_capacity(
    candidate: Candidate,
    planner_regions: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    _, demand, capacities, _, _ = profile_records(candidate)
    x = np.arange(len(PERIODS), dtype=float)
    bar_width = 0.46
    bottom = np.zeros(len(PERIODS), dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for region in REGIONS:
        values = np.array(
            [max(capacities[(region, period)], 0.0) for period in PERIODS],
            dtype=float,
        )
        ax.bar(
            x,
            values,
            width=bar_width,
            bottom=bottom,
            color=REGION_COLORS[region],
            alpha=0.78,
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )
        bottom += values

    candidate_capacity = bottom.copy()
    global_demand = np.array(
        [sum(demand[(region, period)] for region in REGIONS) for period in PERIODS],
        dtype=float,
    )
    planner_capacity = (
        planner_regions.groupby("t")["Kcap"].sum().reindex(PERIODS).to_numpy(float)
    )
    max_total = max(
        float(candidate_capacity.max()),
        float(global_demand.max()),
        float(planner_capacity.max()),
    )

    ax.plot(
        x,
        global_demand,
        color=COLOR_DEMAND,
        linestyle="-.",
        marker="^",
        markerfacecolor=COLOR_DEMAND,
        markeredgecolor=COLOR_DEMAND,
        markeredgewidth=1.6,
        markersize=6.8,
        linewidth=1.8,
        zorder=5,
    )
    ax.plot(
        x,
        planner_capacity,
        color=COLOR_PLAN,
        linestyle="--",
        marker="*",
        markerfacecolor=COLOR_PLAN,
        markeredgecolor=COLOR_PLAN,
        markeredgewidth=1.4,
        markersize=9.0,
        linewidth=1.8,
        zorder=5,
    )
    for xpos, total in zip(x, candidate_capacity):
        ax.text(
            xpos,
            total + max_total * 0.018,
            f"{total:.0f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color=COLOR_DEMAND,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS, fontsize=15)
    ax.set_ylabel("GW", fontsize=15)
    ax.set_ylim(0, max_total * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", length=0)

    handles: list[Any] = [
        Patch(
            facecolor=REGION_COLORS[region],
            edgecolor="white",
            linewidth=0.8,
            label=REGION_LABELS[region],
        )
        for region in REGIONS
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=COLOR_DEMAND,
                linestyle="-.",
                marker="^",
                linewidth=1.8,
                markersize=6.2,
                label="Global demand",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_PLAN,
                linestyle="--",
                marker="*",
                linewidth=1.8,
                markersize=8.0,
                label="Planner capacity",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        frameon=True,
        fontsize=11.5,
        framealpha=0.9,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=0.9,
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.31)
    save_figure(fig, output_dir, "manufacturing_capacity_pathway", dpi)


def planner_welfare_components(
    planner_regions: pd.DataFrame,
    planner_flows: pd.DataFrame,
    data,
) -> dict[str, tuple[float, float]]:
    costs = data.c_man_t or {
        (r, t): float(data.c_man[r]) for r in REGIONS for t in PERIODS
    }
    components: dict[str, tuple[float, float]] = {}
    indexed = planner_regions.set_index(["r", "t"])
    for region in REGIONS:
        cs_total = 0.0
        ps_total = 0.0
        for period in PERIODS:
            row = indexed.loc[(region, period)]
            demand = float(row["x_dem"])
            price = float(row["lam"])
            a_dem = float(data.a_dem_t[(region, period)])
            b_dem = float(data.b_dem_t[(region, period)])
            weight = float(data.beta_t[period]) * float(data.years_to_next[period])
            net_cs = a_dem * demand - 0.5 * b_dem * demand**2 - price * demand

            exported = planner_flows[
                (planner_flows["exp"] == region)
                & (planner_flows["t"] == period)
            ]
            producer_margin = 0.0
            for _, flow_row in exported.iterrows():
                importer = str(flow_row["imp"])
                importer_price = float(indexed.loc[(importer, period), "lam"])
                producer_margin += (
                    importer_price
                    - float(costs[(region, period)])
                    - float(data.c_ship[(region, importer)])
                ) * float(flow_row["x"])

            capacity_cost = (
                float(data.f_hold[region]) * float(row["Kcap"])
                + float(data.c_inv[region]) * float(row["Icap_report"])
            )
            cs_total += weight * net_cs
            ps_total += weight * (producer_margin - capacity_cost)
        components[region] = (cs_total, ps_total)
    return components


def candidate_welfare_components(
    candidate: Candidate,
    data,
) -> dict[str, tuple[float, float]]:
    prices, demand, capacities, flows, d_k_net = profile_records(candidate)
    costs = data.c_man_t or {
        (r, t): float(data.c_man[r]) for r in REGIONS for t in PERIODS
    }
    components: dict[str, tuple[float, float]] = {}
    for region in REGIONS:
        cs_total = 0.0
        ps_total = 0.0
        for period in PERIODS:
            quantity = demand[(region, period)]
            price = prices[(region, period)]
            weight = float(data.beta_t[period]) * float(data.years_to_next[period])
            net_cs = (
                float(data.a_dem_t[(region, period)]) * quantity
                - 0.5 * float(data.b_dem_t[(region, period)]) * quantity**2
                - price * quantity
            )
            producer_margin = sum(
                (
                    prices[(importer, period)]
                    - float(costs[(region, period)])
                    - float(data.c_ship[(region, importer)])
                )
                * flows[(region, importer, period)]
                for importer in REGIONS
            )
            investment = max(float(d_k_net[(region, period)]), 0.0)
            capacity_cost = (
                float(data.f_hold[region]) * capacities[(region, period)]
                + float(data.c_inv[region]) * investment
            )
            cs_total += weight * net_cs
            ps_total += weight * (producer_margin - capacity_cost)
        components[region] = (cs_total, ps_total)
    return components


def planner_annual_welfare(
    planner_regions: pd.DataFrame,
    planner_flows: pd.DataFrame,
    data,
) -> dict[tuple[str, str], float]:
    """Undiscounted annual operating welfare in million USD/year."""
    costs = data.c_man_t or {
        (r, t): float(data.c_man[r]) for r in REGIONS for t in PERIODS
    }
    indexed = planner_regions.set_index(["r", "t"])
    welfare: dict[tuple[str, str], float] = {}
    for region in REGIONS:
        for period in PERIODS:
            row = indexed.loc[(region, period)]
            demand = float(row["x_dem"])
            price = float(row["lam"])
            net_cs = (
                float(data.a_dem_t[(region, period)]) * demand
                - 0.5 * float(data.b_dem_t[(region, period)]) * demand**2
                - price * demand
            )

            exported = planner_flows[
                (planner_flows["exp"] == region)
                & (planner_flows["t"] == period)
            ]
            producer_margin = 0.0
            for _, flow_row in exported.iterrows():
                importer = str(flow_row["imp"])
                importer_price = float(indexed.loc[(importer, period), "lam"])
                producer_margin += (
                    importer_price
                    - float(costs[(region, period)])
                    - float(data.c_ship[(region, importer)])
                ) * float(flow_row["x"])

            capacity_cost = (
                float(data.f_hold[region]) * float(row["Kcap"])
                + float(data.c_inv[region]) * float(row["Icap_report"])
            )
            welfare[(region, period)] = net_cs + producer_margin - capacity_cost
    return welfare


def candidate_annual_welfare(
    candidate: Candidate,
    data,
) -> dict[tuple[str, str], float]:
    """Undiscounted annual operating welfare in million USD/year."""
    prices, demand, capacities, flows, d_k_net = profile_records(candidate)
    costs = data.c_man_t or {
        (r, t): float(data.c_man[r]) for r in REGIONS for t in PERIODS
    }
    welfare: dict[tuple[str, str], float] = {}
    for region in REGIONS:
        for period in PERIODS:
            quantity = demand[(region, period)]
            price = prices[(region, period)]
            net_cs = (
                float(data.a_dem_t[(region, period)]) * quantity
                - 0.5 * float(data.b_dem_t[(region, period)]) * quantity**2
                - price * quantity
            )
            producer_margin = sum(
                (
                    prices[(importer, period)]
                    - float(costs[(region, period)])
                    - float(data.c_ship[(region, importer)])
                )
                * flows[(region, importer, period)]
                for importer in REGIONS
            )
            investment = max(float(d_k_net[(region, period)]), 0.0)
            capacity_cost = (
                float(data.f_hold[region]) * capacities[(region, period)]
                + float(data.c_inv[region]) * investment
            )
            welfare[(region, period)] = net_cs + producer_margin - capacity_cost
    return welfare


def plot_welfare_absolute_annual(
    candidate: Candidate,
    planner_regions: pd.DataFrame,
    planner_flows: pd.DataFrame,
    planner_components: dict[str, tuple[float, float]],
    data,
    output_dir: Path,
    dpi: int,
) -> None:
    """Reproduce the paper's period-stacked absolute welfare-difference plot."""
    candidate_annual = candidate_annual_welfare(candidate, data)
    planner_annual = planner_annual_welfare(planner_regions, planner_flows, data)
    candidate_components = candidate_welfare_components(candidate, data)
    regional_pv_delta = {
        region: (
            sum(candidate_components[region]) - sum(planner_components[region])
        )
        / MUSD_TO_TUSD
        for region in REGIONS
    }
    others = sorted(
        [region for region in REGIONS if region != "ch"],
        key=lambda region: abs(regional_pv_delta[region]),
        reverse=True,
    )
    order = ["ch", *others]
    y = np.arange(len(order), dtype=float)
    delta_annual = {
        (region, period): (
            candidate_annual[(region, period)]
            - planner_annual[(region, period)]
        )
        / 1e3
        for region in order
        for period in PERIODS
    }

    gain_colors = {
        "2025": "#DCEFD9",
        "2030": "#A8D5A2",
        "2035": "#5EA267",
        "2040": "#2E6F40",
    }
    loss_colors = {
        "2025": "#FADBD8",
        "2030": "#F1948A",
        "2035": "#D95F5F",
        "2040": "#A83232",
    }
    positive_base = np.zeros(len(order), dtype=float)
    negative_base = np.zeros(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for period in PERIODS:
        values = np.array(
            [delta_annual[(region, period)] for region in order], dtype=float
        )
        positive = np.where(values > 0.0, values, 0.0)
        negative = np.where(values < 0.0, values, 0.0)
        ax.barh(
            y,
            positive,
            left=positive_base,
            height=0.62,
            color=gain_colors[period],
            edgecolor="white",
            linewidth=1.2,
        )
        ax.barh(
            y,
            np.abs(negative),
            left=negative_base + negative,
            height=0.62,
            color=loss_colors[period],
            edgecolor="white",
            linewidth=1.2,
        )
        positive_base += positive
        negative_base += negative

    ax.axvline(0.0, color="#1A1A1A", linewidth=1.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([REGION_NAMES[region] for region in order], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("[billion $/year]", fontsize=15)
    ax.grid(True, axis="x", linestyle=":", color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", length=0, labelsize=15)
    legend_handles = [
        SplitLegendKey(
            gain_colors[period], loss_colors[period], loss_first=False
        )
        for period in PERIODS
    ]
    ax.legend(
        handles=legend_handles,
        labels=PERIODS,
        handler_map={SplitLegendKey: SplitLegendHandler()},
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.34),
        frameon=True,
        fontsize=13,
        framealpha=0.9,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.45,
        labelspacing=0.35,
    )
    stack_limit = max(
        abs(float(negative_base.min())), float(positive_base.max()), 1e-6
    ) * 1.12
    ax.set_xlim(-stack_limit, stack_limit)
    fig.subplots_adjust(left=0.23, right=0.96, top=0.96, bottom=0.36)
    save_figure(
        fig,
        output_dir,
        "welfare_epec_vs_planner_stacked_annual_v2",
        dpi,
    )


def plot_welfare_cs_ps(
    candidate: Candidate,
    planner_components: dict[str, tuple[float, float]],
    data,
    output_dir: Path,
    dpi: int,
) -> None:
    candidate_components = candidate_welfare_components(candidate, data)
    relative_components: dict[str, np.ndarray] = {}
    relative_total: dict[str, float] = {}
    for region in REGIONS:
        candidate_values = np.asarray(candidate_components[region], dtype=float)
        planner_values = np.asarray(planner_components[region], dtype=float)
        planner_welfare = float(planner_values.sum())
        if np.isclose(planner_welfare, 0.0):
            raise ValueError(
                f"Cannot normalize regional welfare change by a zero LLP value: {region}"
            )
        contributions = 100.0 * (candidate_values - planner_values) / planner_welfare
        relative_components[region] = contributions
        relative_total[region] = float(contributions.sum())

    loss_order = sorted(
        [region for region in REGIONS if relative_total[region] < 0.0],
        key=lambda region: abs(relative_total[region]),
        reverse=True,
    )
    other_gain_order = sorted(
        [
            region
            for region in REGIONS
            if region != "ch" and relative_total[region] >= 0.0
        ],
        key=lambda region: abs(relative_total[region]),
        reverse=True,
    )
    order = ["ch", *loss_order, *other_gain_order]
    y = np.arange(len(order), dtype=float)
    component_values = np.array(
        [relative_components[region] for region in order], dtype=float
    )
    totals = np.array([relative_total[region] for region in order], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    positive_base = np.zeros(len(order), dtype=float)
    negative_base = np.zeros(len(order), dtype=float)
    component_styles = (
        (component_values[:, 0], COLOR_CS_GAIN, COLOR_CS_LOSS),
        (component_values[:, 1], COLOR_PS_GAIN, COLOR_PS_LOSS),
    )
    for values, gain_color, loss_color in component_styles:
        positive = np.where(values > 0.0, values, 0.0)
        negative = np.where(values < 0.0, values, 0.0)
        ax.barh(
            y,
            positive,
            left=positive_base,
            height=0.62,
            color=gain_color,
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        ax.barh(
            y,
            negative,
            left=negative_base,
            height=0.62,
            color=loss_color,
            edgecolor="white",
            linewidth=1.2,
            zorder=2,
        )
        positive_base += positive
        negative_base += negative

    bar_extent = max(
        float(positive_base.max()),
        abs(float(negative_base.min())),
        1e-6,
    )
    axis_limit = 1.30 * bar_extent
    label_offset = 0.025 * axis_limit
    for ypos, total, positive_end, negative_end in zip(
        y, totals, positive_base, negative_base
    ):
        if total >= 0.0:
            xpos = positive_end + label_offset
            ha = "left"
            color = COLOR_PS_GAIN
        else:
            xpos = negative_end - label_offset
            ha = "right"
            color = COLOR_PS_LOSS
        ax.text(
            xpos,
            ypos,
            f"{total:+.1f}%",
            ha=ha,
            va="center",
            fontsize=12.5,
            color=color,
        )

    ax.axvline(0.0, color="#1A1A1A", linewidth=1.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([REGION_NAMES[region] for region in order], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Contribution to regional welfare change [% of LLP regional welfare]",
        fontsize=13,
    )
    ax.set_xlim(-axis_limit, axis_limit)
    ax.grid(True, axis="x", linestyle=":", color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", length=0)
    ax.text(
        0.015,
        0.985,
        r"Loss $\leftarrow$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color=COLOR_PS_LOSS,
    )
    ax.text(
        0.985,
        0.985,
        r"$\rightarrow$ Gain",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color=COLOR_PS_GAIN,
    )

    legend_handles = [
        SplitLegendKey(COLOR_CS_GAIN, COLOR_CS_LOSS),
        SplitLegendKey(COLOR_PS_GAIN, COLOR_PS_LOSS),
    ]
    ax.legend(
        handles=legend_handles,
        labels=("Consumer surplus (CS)", "Producer surplus (PS)"),
        handler_map={SplitLegendKey: SplitLegendHandler()},
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.36),
        frameon=True,
        fontsize=12.5,
        framealpha=0.9,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.0,
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.25, right=0.97, top=0.96, bottom=0.31)
    save_figure(fig, output_dir, "welfare_cs_ps_decomposition", dpi)


def candidate_output_dir(
    candidate: Candidate,
    plots_dir: Path,
    plots_in_profile_dirs: bool,
) -> Path:
    if plots_in_profile_dirs:
        return candidate.source_path.parent / "plots"
    return plots_dir.joinpath(*candidate.output_dir_parts)


def write_plot_index(
    candidate_outputs: list[tuple[Candidate, Path]], plots_dir: Path
) -> None:
    rows = []
    for candidate, output_dir in candidate_outputs:
        try:
            relative_dir = output_dir.relative_to(plots_dir)
        except ValueError:
            relative_dir = output_dir.relative_to(ROOT_DIR)
        rows.append(
            {
                "candidate": candidate.label,
                "source_profile": str(candidate.source_path.relative_to(ROOT_DIR)),
                "plot_directory": str(relative_dir),
                "market_prices": str(relative_dir / "market_prices.png"),
                "welfare_cs_ps": str(
                    relative_dir / "welfare_cs_ps_decomposition.png"
                ),
                "welfare_absolute_annual": str(
                    relative_dir
                    / "welfare_epec_vs_planner_stacked_annual_v2.png"
                ),
                "manufacturing_capacity": str(
                    relative_dir / "manufacturing_capacity_pathway.png"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(plots_dir / "plot_index.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the four paper-result figures for accepted candidates."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--planner-path", type=Path, default=DEFAULT_PLANNER_PATH)
    parser.add_argument(
        "--workflow-manifest",
        type=Path,
        default=None,
        help=(
            "Optional workflow manifest. When supplied, plot every grid result marked "
            "as a local one-percent equilibrium instead of scanning package-dir/profiles."
        ),
    )
    parser.add_argument(
        "--plots-in-profile-dirs",
        action="store_true",
        help=(
            "Write each accepted candidate's figures to a plots/ directory next "
            "to its selected profile. The plot index is still written to plots-dir."
        ),
    )
    parser.add_argument(
        "--rebuild-planner",
        action="store_true",
        help="Re-solve the corrected global-planner benchmark even if the cache matches.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    plots_dir = args.plots_dir.resolve()
    planner_path = args.planner_path.resolve()
    workflow_manifest = None
    if args.workflow_manifest is not None:
        workflow_manifest = load_json(args.workflow_manifest.resolve())
        manifest = workflow_manifest
    else:
        manifest = load_json(package_dir / "source_metadata" / "factorial" / "manifest.json")
    protocol = manifest.get("protocol", manifest)
    input_path = resolve_recorded_path(str(protocol["input"]))
    recorded_hash = str(protocol["input_sha256"]).upper()
    if sha256(input_path) != recorded_hash:
        raise RuntimeError(f"Corrected input hash no longer matches the manifest: {input_path}")
    terminal_salvage_fraction = float(protocol["terminal_salvage_fraction"])
    data = configure_model_data(input_path, terminal_salvage_fraction)

    if args.rebuild_planner or not planner_is_current(
        planner_path, input_path, terminal_salvage_fraction
    ):
        print("Building corrected, like-for-like planner benchmark...")
        build_planner_benchmark(
            planner_path,
            data,
            input_path,
            terminal_salvage_fraction,
        )
        print(f"Saved corrected planner benchmark to {planner_path}")

    planner_regions, planner_flows = load_planner(planner_path)
    planner_components = planner_welfare_components(planner_regions, planner_flows, data)
    candidates = load_candidates(package_dir, workflow_manifest)
    selection_root = args.workflow_manifest.resolve().parent if args.workflow_manifest else package_dir
    excluded = excluded_candidate_ids(selection_root)
    loaded_ids = {f"{candidate.sequence}/{candidate.branch}" for candidate in candidates}
    if not excluded <= loaded_ids:
        raise ValueError(f"Reporting exclusions are not loaded candidates: {excluded - loaded_ids}")
    candidates = [
        candidate for candidate in candidates
        if f"{candidate.sequence}/{candidate.branch}" not in excluded
    ]
    plots_dir.mkdir(parents=True, exist_ok=True)

    candidate_outputs = [
        (
            candidate,
            candidate_output_dir(
                candidate,
                plots_dir,
                args.plots_in_profile_dirs,
            ),
        )
        for candidate in candidates
    ]
    for index, (candidate, output_dir) in enumerate(candidate_outputs, start=1):
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_prices(candidate, planner_regions, data, output_dir, args.dpi)
        plot_welfare_cs_ps(
            candidate,
            planner_components,
            data,
            output_dir,
            args.dpi,
        )
        plot_welfare_absolute_annual(
            candidate,
            planner_regions,
            planner_flows,
            planner_components,
            data,
            output_dir,
            args.dpi,
        )
        plot_capacity(candidate, planner_regions, output_dir, args.dpi)
        print(f"[{index:02d}/{len(candidates):02d}] {candidate.label}")

    write_plot_index(candidate_outputs, plots_dir)
    print(f"Generated {len(candidates) * 4} figures in PNG and PDF: {plots_dir}")


if __name__ == "__main__":
    main()
