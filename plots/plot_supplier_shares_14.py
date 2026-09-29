"""Plot mean supplier shares in the four importing regions across the 14 retained equilibria.

Run ``python analysis/china_apac_competition.py`` first so that the route
observations exist. Figures are written to the 14-equilibria paper folder and
copied into the Overleaf repository.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUTES = ROOT / "outputs/china_apac_competition/csv/route_observations.csv"
FOLDER = ROOT / "outputs/paper_plots/14_equilibria"
OVERLEAF_FIGURES = ROOT / "IEEE Paper/images/results"
STEM = "supplier_shares_by_region"
REGIONS = ("ch", "eu", "us", "apac", "af", "row")
# China and APAC supply their own markets almost entirely, so only importers are shown.
DESTINATIONS = ("eu", "us", "af", "row")
YEARS = (2025, 2030, 2035, 2040)
LABELS = {"ch": "CH", "eu": "EU", "us": "US", "apac": "APAC", "af": "AF", "row": "ROW"}
TITLES = {"ch": "China", "eu": "Europe", "us": "United States",
          "apac": "Asia-Pacific", "af": "Africa", "row": "Rest of World"}
# Same colors as the median regional capacity pathway.
COLORS = {
    "ch": "#CA6180", "eu": "#FEFD99", "us": "#FCB7C7",
    "apac": "#B7A6D8", "af": "#B8D99E", "row": "#9ED3DC",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def retained() -> list[str]:
    lines = (FOLDER / "retained_profiles.txt").read_text(encoding="utf-8").splitlines()
    ids = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(ids) != 14 or len(set(ids)) != 14:
        raise ValueError(f"Expected 14 unique retained profiles, got {len(ids)}")
    return ids


def mean_shares(routes: pd.DataFrame) -> pd.DataFrame:
    shares = (routes.groupby(["importer", "year", "exporter"], as_index=False)
              .agg(mean_share=("share_of_market", "mean"),
                   mean_flow_gw=("flow_gw", "mean")))
    if not np.allclose(shares.groupby(["importer", "year"]).mean_share.sum(), 1, atol=1e-6):
        raise ValueError("Mean supplier shares do not add up to one")
    return shares


def plot(shares: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True, sharey=True)
    x = np.arange(len(YEARS), dtype=float)
    for ax, importer in zip(axes.flat, DESTINATIONS):
        bottom = np.zeros(len(YEARS))
        for exporter in REGIONS:
            values = np.array([
                shares.loc[shares.importer.eq(importer) & shares.year.eq(year)
                           & shares.exporter.eq(exporter), "mean_share"].sum() * 100
                for year in YEARS
            ])
            ax.bar(x, values, width=0.58, bottom=bottom, color=COLORS[exporter],
                   alpha=0.78, edgecolor="white", linewidth=1.0, zorder=2)
            bottom += values
        ax.set_title(TITLES[importer], fontsize=16, pad=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x, YEARS)
        ax.tick_params(axis="both", labelsize=13)
        ax.tick_params(axis="x", length=0, labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of demand [%]", fontsize=14)
    handles = [Patch(facecolor=COLORS[r], edgecolor="#888888", linewidth=0.5, label=LABELS[r])
               for r in REGIONS]
    fig.legend(handles=handles, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=True, framealpha=0.95, fontsize=13, columnspacing=1.2,
               handlelength=1.35)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.16,
                        wspace=0.2, hspace=0.4)
    OVERLEAF_FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        target = FOLDER / f"{STEM}.{extension}"
        fig.savefig(target, dpi=300, bbox_inches="tight", pad_inches=0.04)
        shutil.copyfile(target, OVERLEAF_FIGURES / target.name)
    plt.close(fig)


def main() -> None:
    ids = retained()
    routes = pd.read_csv(ROUTES)
    routes = routes[routes.candidate.isin(ids)]
    if routes.candidate.nunique() != 14:
        raise ValueError("Route observations do not contain all 14 retained profiles")
    shares = mean_shares(routes)
    shares.to_csv(FOLDER / f"{STEM}.csv", index=False)
    plot(shares)
    print(f"Wrote {STEM} for {routes.candidate.nunique()} profiles")


if __name__ == "__main__":
    main()
