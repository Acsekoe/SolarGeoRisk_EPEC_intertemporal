"""Create publication-ready figures for the packaged 15 candidate profiles.

The source package deliberately distinguishes profiles that pass the declared
one-start local 1% audit from equilibria that would be robust to a broader
multistart best-response search.  Figure labels retain that distinction.

Outputs
-------
candidate_audit_robustness.{png,pdf}
    Dumbbell comparison of one-start and three-start maximum relative gains.
candidate_price_paths.{png,pdf}
    Regional market-price paths, grouped by the Stage-2 offer-price seed.
candidate_capacity_paths.{png,pdf}
    Regional capacity paths with the median and full range across candidates.
candidate_summary.csv
    Compact candidate metadata and audit results used by the figures.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = ROOT_DIR / "outputs" / "15_equilibria"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plots"

PERIODS = (2025, 2030, 2035, 2040)
REGIONS = ("ch", "apac", "eu", "us", "af", "row")
REGION_NAMES = {
    "ch": "China",
    "apac": "Asia-Pacific",
    "eu": "Europe",
    "us": "United States",
    "af": "Africa",
    "row": "Rest of World",
}

ORDER_CODES = {
    "ch-af-apac-eu-row-us": "A",
    "ch-af-eu-us-row-apac": "B",
    "ch-row-apac-us-eu-af": "C",
}
ORDER_DESCRIPTIONS = {
    "A": "CH-AF-APAC-EU-ROW-US",
    "B": "CH-AF-EU-US-ROW-APAC",
    "C": "CH-ROW-APAC-US-EU-AF",
}

# Colorblind-safe colors (Okabe-Ito family).
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#007A4D"
LIGHT_GREEN = "#B8D8C8"
GREY = "#777777"
DARK_GREY = "#333333"
GRID_GREY = "#D6D6D6"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "savefig.facecolor": "white",
    }
)


@dataclass(frozen=True)
class Candidate:
    sequence: str
    branch: str
    sweep: int
    offer_factor: float
    capacity_weight: float
    alpha: float
    one_start_gain_pct: float
    one_start_player: str
    three_start_gain_pct: float
    three_start_player: str
    prices: dict[tuple[str, int], float]
    capacities: dict[tuple[str, int], float]

    @property
    def order_code(self) -> str:
        return ORDER_CODES[self.sequence]

    @property
    def identifier(self) -> str:
        return f"{self.sequence}/{self.branch}/sweep_{self.sweep:03d}"

    @property
    def short_label(self) -> str:
        return f"{self.order_code}   {self.branch.replace('_', ' / ')}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keyed_values(
    records: list[dict[str, Any]], entity_key: str
) -> dict[tuple[str, int], float]:
    return {
        (str(record[entity_key]).lower(), int(record["time"])): float(record["value"])
        for record in records
    }


def load_candidates(input_dir: Path) -> list[Candidate]:
    profiles_dir = input_dir / "profiles"
    profile_paths = sorted(profiles_dir.glob("*/*/sweep_*.json"))
    if len(profile_paths) != 15:
        raise ValueError(
            f"Expected 15 packaged profiles in {profiles_dir}, found {len(profile_paths)}."
        )

    candidates: list[Candidate] = []
    for profile_path in profile_paths:
        profile = read_json(profile_path)
        sequence = str(profile["sequence"])
        branch = str(profile["branch"])
        sweep = int(profile["sweep"])
        if sequence not in ORDER_CODES:
            raise ValueError(f"Unrecognized player order in {profile_path}: {sequence}")

        relative_parent = profile_path.relative_to(profiles_dir).parent
        one_path = (
            input_dir
            / "one_start_audits"
            / relative_parent
            / f"audit_sweep_{sweep:03d}_one_start.json"
        )
        three_path = (
            input_dir
            / "three_start_audits"
            / relative_parent
            / "audit_three_start.json"
        )
        if not one_path.exists() or not three_path.exists():
            raise FileNotFoundError(
                f"Missing matched audit for {profile_path}: {one_path} or {three_path}"
            )

        one_audit = read_json(one_path)
        three_audit = read_json(three_path)
        ending_profile = profile["ending_profile"]
        specification = profile["branch_specification"]

        prices = keyed_values(
            ending_profile["market"]["clearing_prices"], "region"
        )
        capacities = keyed_values(ending_profile["capacities"], "player")

        missing_prices = set((r, t) for r in REGIONS for t in PERIODS) - set(prices)
        missing_capacities = set((r, t) for r in REGIONS for t in PERIODS) - set(
            capacities
        )
        if missing_prices or missing_capacities:
            raise ValueError(
                f"Incomplete economic-horizon data in {profile_path}: "
                f"prices={sorted(missing_prices)}, capacities={sorted(missing_capacities)}"
            )

        candidates.append(
            Candidate(
                sequence=sequence,
                branch=branch,
                sweep=sweep,
                offer_factor=float(
                    specification["price_offer_factor_to_manufacturing_cost"]
                ),
                capacity_weight=float(specification["capacity_change_path_weight"]),
                alpha=float(profile["alpha"]),
                one_start_gain_pct=100.0 * float(one_audit["max_relative_gain"]),
                one_start_player=str(one_audit["max_gain_player"]).upper(),
                three_start_gain_pct=100.0 * float(three_audit["max_relative_gain"]),
                three_start_player=str(three_audit["max_gain_player"]).upper(),
                prices=prices,
                capacities=capacities,
            )
        )

    return candidates


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(candidates: list[Candidate], output_dir: Path) -> None:
    fieldnames = [
        "candidate",
        "order_code",
        "player_order",
        "branch",
        "selected_sweep",
        "offer_price_factor",
        "capacity_path_weight",
        "damping_alpha",
        "one_start_max_gain_pct",
        "one_start_max_gain_player",
        "three_start_max_gain_pct",
        "three_start_max_gain_player",
    ]
    with (output_dir / "candidate_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda c: (c.order_code, c.branch)):
            writer.writerow(
                {
                    "candidate": candidate.identifier,
                    "order_code": candidate.order_code,
                    "player_order": candidate.sequence,
                    "branch": candidate.branch,
                    "selected_sweep": candidate.sweep,
                    "offer_price_factor": f"{candidate.offer_factor:.2f}",
                    "capacity_path_weight": f"{candidate.capacity_weight:.2f}",
                    "damping_alpha": f"{candidate.alpha:.2f}",
                    "one_start_max_gain_pct": f"{candidate.one_start_gain_pct:.9f}",
                    "one_start_max_gain_player": candidate.one_start_player,
                    "three_start_max_gain_pct": f"{candidate.three_start_gain_pct:.9f}",
                    "three_start_max_gain_player": candidate.three_start_player,
                }
            )


def plot_audit_robustness(
    candidates: list[Candidate], output_dir: Path, dpi: int
) -> None:
    ordered = sorted(candidates, key=lambda c: c.three_start_gain_pct)
    y = np.arange(len(ordered), dtype=float)
    one = np.array([c.one_start_gain_pct for c in ordered])
    three = np.array([c.three_start_gain_pct for c in ordered])

    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    for ypos, x0, x1 in zip(y, one, three):
        ax.plot([x0, x1], [ypos, ypos], color="#A6A6A6", linewidth=1.2, zorder=1)

    ax.scatter(
        one,
        y,
        s=34,
        marker="o",
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.5,
        label="One-start audit",
        zorder=3,
    )
    ax.scatter(
        three,
        y,
        s=36,
        marker="s",
        facecolor=VERMILLION,
        edgecolor=VERMILLION,
        linewidth=1.0,
        label="Three-start audit",
        zorder=3,
    )

    for ypos, x0, x1 in zip(y, one, three):
        ax.annotate(
            f"{x0:.2f}",
            (x0, ypos),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=7.5,
            color=BLUE,
        )
        ax.annotate(
            f"{x1:.2f}",
            (x1, ypos),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=VERMILLION,
        )

    ax.axvline(1.0, color=DARK_GREY, linestyle="--", linewidth=1.2, zorder=0)
    ax.text(
        1.0,
        len(ordered) - 0.15,
        "1% criterion",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=DARK_GREY,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([c.short_label for c in ordered])
    ax.set_ylim(-0.7, len(ordered) - 0.05)
    ax.set_xlim(0.0, max(three) * 1.13)
    ax.set_xlabel("Maximum unilateral welfare gain (%)")
    ax.set_title("Audit sensitivity of the 15 one-start local candidates")
    ax.grid(axis="x", color=GRID_GREY, linestyle=":", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False)

    order_note = "   |   ".join(
        f"{code}: {ORDER_DESCRIPTIONS[code]}" for code in ("A", "B", "C")
    )
    fig.text(0.5, 0.018, order_note, ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.92, bottom=0.12)
    save_figure(fig, output_dir, "candidate_audit_robustness", dpi)


def plot_price_paths(candidates: list[Candidate], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.7), sharex=True, sharey=True)
    x = np.array(PERIODS, dtype=float)
    groups = (
        (1.0, BLUE, "pf100: offers seeded at cost"),
        (1.2, VERMILLION, "pf120: offers seeded at 1.2x cost"),
    )

    for ax, region in zip(axes.flat, REGIONS):
        for offer_factor, color, _ in groups:
            group = [c for c in candidates if np.isclose(c.offer_factor, offer_factor)]
            values = np.array(
                [[c.prices[(region, period)] for period in PERIODS] for c in group],
                dtype=float,
            )
            for path in values:
                ax.plot(x, path, color=color, linewidth=0.8, alpha=0.25, zorder=1)
            ax.plot(
                x,
                np.median(values, axis=0),
                color=color,
                linewidth=2.2,
                marker="o" if offer_factor == 1.0 else "s",
                markersize=3.5,
                zorder=3,
            )

        ax.set_title(REGION_NAMES[region])
        ax.set_xticks(PERIODS)
        ax.set_ylim(0.0, 330.0)
        ax.set_yticks([0, 100, 200, 300])
        ax.grid(color=GRID_GREY, linestyle=":", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Price ($/kW)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o" if offer_factor == 1.0 else "s",
            linewidth=2.2,
            markersize=4,
            label=f"{label} (n={sum(np.isclose(c.offer_factor, offer_factor) for c in candidates)})",
        )
        for offer_factor, color, label in groups
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle("Market-price paths across the 15 one-start local candidates")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.13, hspace=0.36)
    save_figure(fig, output_dir, "candidate_price_paths", dpi)


def plot_capacity_paths(
    candidates: list[Candidate], output_dir: Path, dpi: int
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.7), sharex=True)
    x = np.array(PERIODS, dtype=float)

    for ax, region in zip(axes.flat, REGIONS):
        values = np.array(
            [[c.capacities[(region, period)] for period in PERIODS] for c in candidates],
            dtype=float,
        )
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        median = np.median(values, axis=0)

        ax.fill_between(
            x,
            lower,
            upper,
            color=LIGHT_GREEN,
            alpha=0.75,
            linewidth=0,
            zorder=1,
        )
        for path in values:
            ax.plot(x, path, color=GREY, linewidth=0.65, alpha=0.25, zorder=2)
        ax.plot(
            x,
            median,
            color=GREEN,
            linewidth=2.2,
            marker="o",
            markersize=3.6,
            zorder=3,
        )

        ax.set_title(REGION_NAMES[region])
        ax.set_xticks(PERIODS)
        ymax = max(float(upper.max()) * 1.12, 1.0)
        ax.set_ylim(0.0, ymax)
        ax.grid(color=GRID_GREY, linestyle=":", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Capacity (GW)")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=GREEN,
            marker="o",
            linewidth=2.2,
            markersize=4,
            label="Median",
        ),
        Line2D([0], [0], color=LIGHT_GREEN, linewidth=8, label="Full range"),
        Line2D(
            [0],
            [0],
            color=GREY,
            linewidth=0.9,
            alpha=0.55,
            label="Individual candidate",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.suptitle("Manufacturing-capacity paths across the 15 candidates")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.12, hspace=0.36)
    save_figure(fig, output_dir, "candidate_capacity_paths", dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot results from the packaged 15 one-start local candidates."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing profiles/ and the matched audit folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG, PDF, and CSV outputs.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(input_dir)
    write_summary_csv(candidates, output_dir)
    plot_audit_robustness(candidates, output_dir, args.dpi)
    plot_price_paths(candidates, output_dir, args.dpi)
    plot_capacity_paths(candidates, output_dir, args.dpi)

    print(f"Loaded {len(candidates)} candidates from {input_dir}")
    print(f"Saved figures and candidate_summary.csv to {output_dir}")


if __name__ == "__main__":
    main()
