"""Plot China's and APAC's CS, PS and welfare changes against China's exports.

Changes are differences from the centralized benchmark, with PS net of capacity
costs as in the paper's CS/PS boxplot, so CS + PS equals the regional welfare
change. China's exports are the horizon-weighted averages of
price_supply_drivers.csv (run plots/plot_price_drivers_14.py first). The shaded
area marks the equilibria in which China's welfare gain exceeds APAC's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs/paper_plots/14_equilibria"
STEM = "welfare_china_apac_vs_exports"
sys.path.insert(0, str(FOLDER))

import welfare_cs_ps_net  # noqa: E402

REGIONS = (("ch", "(a) China"), ("apac", "(b) APAC"))
# Component, label, color, marker. CS and PS colors match the CS/PS boxplot.
SERIES = (
    ("cs", "CS", "#B43C38", "s"),
    ("ps", "PS", "#2E6F40", "o"),
    ("w", "Welfare (CS + PS)", "#303030", "x"),
)
# Label offsets in points per panel, chosen to keep labels clear of nearby markers.
OFFSETS = {
    "ch": {"Eq 1": (0, 26), "Eq 2": (0, 12)},
    "apac": {"Eq 1": (18, 16), "Eq 2": (0, 12)},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def load() -> pd.DataFrame:
    obs = welfare_cs_ps_net.observations()
    obs = obs[obs.region.isin(["ch", "apac"])]
    obs = obs.assign(component=obs.component.map({"Consumer surplus": "cs",
                                                  "Producer surplus": "ps"}))
    data = obs.pivot_table(index="candidate", columns=["region", "component"],
                           values="absolute")
    data.columns = [f"{region}_{component}" for region, component in data.columns]
    for region, _ in REGIONS:
        data[f"{region}_w"] = data[f"{region}_cs"] + data[f"{region}_ps"]
    drivers = pd.read_csv(FOLDER / "price_supply_drivers.csv").set_index("candidate")
    data = data.join(drivers[["ch_exports", "label"]], how="inner")
    if len(data) != 14:
        raise ValueError(f"Expected 14 profiles, got {len(data)}")
    return data


def plot(data: pd.DataFrame) -> float:
    china_more = data.ch_w > data.apac_w
    low, high = data.ch_exports[~china_more].max(), data.ch_exports[china_more].min()
    if low >= high:
        raise ValueError("China's exports do not separate the two groups")
    threshold = (low + high) / 2

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharey=True)
    x = data.ch_exports
    for ax, (region, title) in zip(axes, REGIONS):
        ax.axvspan(threshold, x.max() + 8, color="#9E9E9E", alpha=0.15, linewidth=0,
                   zorder=0)
        ax.axhline(0, color="#333333", linewidth=1.1, zorder=1)
        for component, label, color, marker in SERIES:
            y = data[f"{region}_{component}"]
            style = {"linewidth": 1.8} if marker == "x" else {
                "edgecolor": "white", "linewidth": 0.8}
            ax.scatter(x, y, s=46, marker=marker, color=color, label=label, zorder=3,
                       **style)
        for candidate, name in data.label.dropna().items():
            point = (x[candidate], data.at[candidate, f"{region}_w"])
            ax.annotate(name, point, xytext=OFFSETS[region][name], textcoords="offset points",
                        ha="center", va="center", fontsize=12, color="#303030", zorder=4,
                        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.7,
                                    "shrinkA": 6, "shrinkB": 5})
        ax.set_title(title, fontsize=14, pad=6)
        ax.set_xlabel("China exports [GW]", fontsize=14)
        ax.set_xlim(x.min() - 8, x.max() + 8)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Difference [billion USD]", fontsize=14)
    axes[0].text(threshold + 3, 0.97, "China gains\nmore than APAC",
                 transform=axes[0].get_xaxis_transform(), ha="left", va="top",
                 fontsize=11, color="#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.25, wspace=0.08)
    for extension in ("pdf", "png"):
        fig.savefig(FOLDER / f"{STEM}.{extension}", dpi=300, bbox_inches="tight",
                    pad_inches=0.04)
    plt.close(fig)
    return threshold


def main() -> None:
    data = load()
    threshold = plot(data)
    columns = ["ch_exports", "ch_cs", "ch_ps", "ch_w", "apac_cs", "apac_ps", "apac_w",
               "label"]
    print(data[columns].sort_values("ch_exports").round(1).to_string())
    print(f"Threshold between groups: {threshold:.1f} GW")


if __name__ == "__main__":
    main()
