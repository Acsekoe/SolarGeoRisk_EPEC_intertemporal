"""Statistical summaries of prices and capacities across the 15 candidates.

Creates publication-ready PNG and PDF figures plus the underlying descriptive
statistics. Box-and-whisker summaries use all 15 candidate profiles; overlaid
points distinguish the ``pf100`` and ``pf120`` initialization families.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_equilibrium_paper_figures import (  # noqa: E402
    DEFAULT_PACKAGE_DIR,
    PERIODS,
    REGIONS,
    REGION_NAMES,
    load_candidates,
    profile_records,
)


DEFAULT_OUTPUT_DIR = DEFAULT_PACKAGE_DIR / "plots" / "statistics"

PF100_COLOR = "#0072B2"
PF120_COLOR = "#D55E00"
BOX_FACE = "#ECECEC"
BOX_EDGE = "#555555"
MEDIAN_COLOR = "#111111"
GRID_COLOR = "#D3D3D3"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "savefig.facecolor": "white",
    }
)


def candidate_offer_factor(candidate) -> float:
    return float(
        candidate.payload["branch_specification"][
            "price_offer_factor_to_manufacturing_cost"
        ]
    )


def candidate_arrays(candidates):
    prices = np.empty((len(candidates), len(REGIONS), len(PERIODS)), dtype=float)
    capacities = np.empty_like(prices)
    factors = np.empty(len(candidates), dtype=float)
    for candidate_index, candidate in enumerate(candidates):
        price_map, _, capacity_map, _, _ = profile_records(candidate)
        factors[candidate_index] = candidate_offer_factor(candidate)
        for region_index, region in enumerate(REGIONS):
            for period_index, period in enumerate(PERIODS):
                prices[candidate_index, region_index, period_index] = price_map[
                    (region, period)
                ]
                capacities[candidate_index, region_index, period_index] = capacity_map[
                    (region, period)
                ]
    return prices, capacities, factors


def draw_distribution_panel(
    ax: plt.Axes,
    values: np.ndarray,
    factors: np.ndarray,
) -> None:
    positions = np.arange(1, len(PERIODS) + 1, dtype=float)
    boxplot = ax.boxplot(
        [values[:, index] for index in range(len(PERIODS))],
        positions=positions,
        widths=0.52,
        whis=(0, 100),
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": BOX_FACE, "edgecolor": BOX_EDGE, "linewidth": 1.1},
        medianprops={"color": MEDIAN_COLOR, "linewidth": 1.8},
        whiskerprops={"color": BOX_EDGE, "linewidth": 1.0},
        capprops={"color": BOX_EDGE, "linewidth": 1.0},
    )
    for patch in boxplot["boxes"]:
        patch.set_alpha(0.82)

    for factor, color, marker, left, right in (
        (1.0, PF100_COLOR, "o", -0.19, -0.035),
        (1.2, PF120_COLOR, "s", 0.035, 0.19),
    ):
        indices = np.flatnonzero(np.isclose(factors, factor))
        offsets = np.linspace(left, right, len(indices))
        for offset, candidate_index in zip(offsets, indices):
            ax.scatter(
                positions + offset,
                values[candidate_index, :],
                s=24,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.82,
                zorder=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(PERIODS)
    ax.grid(axis="y", color=GRID_COLOR, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_distributions(
    values: np.ndarray,
    factors: np.ndarray,
    output_dir: Path,
    stem: str,
    ylabel: str,
    title: str,
    share_y: bool,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8.0, 8.6),
        sharex=True,
        sharey=share_y,
    )
    for region_index, (ax, region) in enumerate(zip(axes.flat, REGIONS)):
        region_values = values[:, region_index, :]
        draw_distribution_panel(ax, region_values, factors)
        ax.set_title(REGION_NAMES[region])
        if not share_y:
            lower = float(region_values.min())
            upper = float(region_values.max())
            spread = upper - lower
            padding = max(spread * 0.10, upper * 0.035, 0.2)
            ax.set_ylim(max(0.0, lower - padding), upper + padding)

    if share_y:
        lower = float(values.min())
        upper = float(values.max())
        padding = max((upper - lower) * 0.06, 1.0)
        axes.flat[0].set_ylim(max(0.0, lower - padding), upper + padding)

    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PF100_COLOR,
            markeredgecolor="white",
            markersize=7,
            label=f"pf100 candidates (n={sum(np.isclose(factors, 1.0))})",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=PF120_COLOR,
            markeredgecolor="white",
            markersize=7,
            label=f"pf120 candidates (n={sum(np.isclose(factors, 1.2))})",
        ),
        Line2D(
            [0],
            [0],
            color=BOX_EDGE,
            linewidth=7,
            alpha=0.4,
            label="All-candidate IQR; whiskers show range",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.suptitle(title, fontsize=18, fontweight="normal")
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.925,
        bottom=0.17,
        hspace=0.34,
        wspace=0.25,
    )
    save_figure(fig, output_dir, stem, dpi)


def dispersion_matrices(prices: np.ndarray, capacities: np.ndarray):
    price_median = np.median(prices, axis=0)
    price_range_pct = (
        100.0 * (prices.max(axis=0) - prices.min(axis=0)) / price_median
    )
    initial_capacity = capacities[:, :, 0].mean(axis=0)[:, None]
    capacity_range_pct = (
        100.0
        * (capacities.max(axis=0) - capacities.min(axis=0))
        / initial_capacity
    )
    return price_range_pct, capacity_range_pct


def annotate_heatmap(ax: plt.Axes, image, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            color = "white" if image.norm(value) > 0.55 else "#222222"
            ax.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )


def plot_dispersion_heatmap(
    prices: np.ndarray,
    capacities: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> tuple[np.ndarray, np.ndarray]:
    price_range_pct, capacity_range_pct = dispersion_matrices(prices, capacities)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9))

    specs = (
        (
            axes[0],
            price_range_pct,
            "Market-price range",
            "Range as % of cell median",
            "Blues",
        ),
        (
            axes[1],
            capacity_range_pct,
            "Capacity range",
            "Range as % of 2025 regional capacity",
            "Oranges",
        ),
    )
    for ax, values, title, colorbar_label, cmap in specs:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0)
        annotate_heatmap(ax, image, values)
        ax.set_xticks(np.arange(len(PERIODS)))
        ax.set_xticklabels(PERIODS)
        ax.set_yticks(np.arange(len(REGIONS)))
        ax.set_yticklabels([REGION_NAMES[region] for region in REGIONS])
        ax.set_xlabel("Year")
        ax.set_title(title)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.15)
        colorbar.set_label(colorbar_label, fontsize=11)
        colorbar.ax.tick_params(labelsize=10)

    fig.suptitle("Full range across the 15 one-start local candidates", fontsize=17)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.18, wspace=0.30)
    save_figure(fig, output_dir, "price_capacity_range_heatmap", dpi)
    return price_range_pct, capacity_range_pct


def write_statistics(
    prices: np.ndarray,
    capacities: np.ndarray,
    price_range_pct: np.ndarray,
    capacity_range_pct: np.ndarray,
    output_dir: Path,
) -> None:
    rows = []
    for region_index, region in enumerate(REGIONS):
        for period_index, period in enumerate(PERIODS):
            price_values = prices[:, region_index, period_index]
            capacity_values = capacities[:, region_index, period_index]
            rows.append(
                {
                    "region": region,
                    "period": period,
                    "n_candidates": len(price_values),
                    "price_min_usd_per_kw": price_values.min(),
                    "price_q1_usd_per_kw": np.quantile(price_values, 0.25),
                    "price_median_usd_per_kw": np.median(price_values),
                    "price_q3_usd_per_kw": np.quantile(price_values, 0.75),
                    "price_max_usd_per_kw": price_values.max(),
                    "price_range_pct_of_median": price_range_pct[
                        region_index, period_index
                    ],
                    "capacity_min_gw": capacity_values.min(),
                    "capacity_q1_gw": np.quantile(capacity_values, 0.25),
                    "capacity_median_gw": np.median(capacity_values),
                    "capacity_q3_gw": np.quantile(capacity_values, 0.75),
                    "capacity_max_gw": capacity_values.max(),
                    "capacity_range_pct_of_2025_capacity": capacity_range_pct[
                        region_index, period_index
                    ],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "distribution_statistics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot statistical price and capacity distributions for 15 candidates."
    )
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(package_dir)
    prices, capacities, factors = candidate_arrays(candidates)
    plot_distributions(
        prices,
        factors,
        output_dir,
        stem="market_price_distributions",
        ylabel="Price ($/kW)",
        title="Market-price distributions across 15 candidates",
        share_y=True,
        dpi=args.dpi,
    )
    plot_distributions(
        capacities,
        factors,
        output_dir,
        stem="capacity_distributions",
        ylabel="Capacity (GW)",
        title="Manufacturing-capacity distributions across 15 candidates",
        share_y=False,
        dpi=args.dpi,
    )
    price_range_pct, capacity_range_pct = plot_dispersion_heatmap(
        prices, capacities, output_dir, args.dpi
    )
    write_statistics(
        prices,
        capacities,
        price_range_pct,
        capacity_range_pct,
        output_dir,
    )
    print(f"Saved statistical figures and data to {output_dir}")


if __name__ == "__main__":
    main()
