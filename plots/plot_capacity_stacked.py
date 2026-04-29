"""
plot_capacity_stacked.py
========================
Grouped stacked installed-capacity figure for the paper results section.

For each planning period, the figure compares aggregate installed production
capacity under the global planner benchmark and the converged EPEC equilibrium
(Iteration 21), with each bar stacked by regional contribution.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Match IEEEtran serif rendering used by the newer paper figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PLANNER_PATH = os.path.join(ROOT_DIR, "outputs", "llp_planner_results.xlsx")
EPEC_PATH = os.path.join(
    ROOT_DIR,
    "outputs",
    "sens",
    "converged",
    "sens_ch-row-apac-us-eu-af.xlsx",
)
OUT_DIR = os.path.join(ROOT_DIR, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PERIODS = ["2025", "2030", "2035", "2040"]
REGIONS = ["ch", "eu", "us", "apac", "af", "row"]
REGION_NAMES = {
    "ch": "CH",
    "eu": "EU",
    "us": "US",
    "apac": "APAC",
    "af": "AF",
    "row": "ROW",
}

# Colorblind-friendly qualitative palette. The same mapping is used for
# planner and EPEC bars so regional composition can be compared directly.
REGION_COLORS = {
    "ch": "#2F3A67",
    "eu": "#3A9BC1",
    "us": "#C75D4A",
    "apac": "#5FAE8B",
    "af": "#D9A441",
    "row": "#9B6FB5",
}

COLOR_GRID = "#D0D0D0"
BAR_ALPHA = 0.78


def load_capacity(path):
    df = pd.read_excel(path, sheet_name="regions")
    df["r"] = df["r"].astype(str).str.lower()
    df["t"] = df["t"].astype(str)
    df = df[df["t"].isin(PERIODS) & df["r"].isin(REGIONS)].copy()

    # Solver tolerances can leave decommissioned capacity as tiny negatives.
    df["Kcap"] = df["Kcap"].clip(lower=0.0)

    cap = (
        df.pivot_table(index="t", columns="r", values="Kcap", aggfunc="sum")
        .reindex(index=PERIODS, columns=REGIONS)
        .fillna(0.0)
    )
    return cap


planner = load_capacity(PLANNER_PATH)
epec = load_capacity(EPEC_PATH)


fig, ax = plt.subplots(figsize=(7.4, 4.8))

x = np.arange(len(PERIODS), dtype=float)
bar_width = 0.32
offset = bar_width / 2 + 0.025
x_planner = x - offset
x_epec = x + offset

bottom_planner = np.zeros(len(PERIODS))
bottom_epec = np.zeros(len(PERIODS))

for region in REGIONS:
    planner_values = planner[region].to_numpy(dtype=float)
    epec_values = epec[region].to_numpy(dtype=float)

    ax.bar(
        x_planner,
        planner_values,
        width=bar_width,
        bottom=bottom_planner,
        color=REGION_COLORS[region],
        alpha=BAR_ALPHA,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.bar(
        x_epec,
        epec_values,
        width=bar_width,
        bottom=bottom_epec,
        color=REGION_COLORS[region],
        alpha=BAR_ALPHA,
        edgecolor="white",
        linewidth=0.9,
    )

    bottom_planner += planner_values
    bottom_epec += epec_values

totals_planner = planner.sum(axis=1).to_numpy(dtype=float)
totals_epec = epec.sum(axis=1).to_numpy(dtype=float)
max_total = max(totals_planner.max(), totals_epec.max())

for xpos, total in zip(x_planner, totals_planner):
    ax.text(
        xpos,
        total + max_total * 0.018,
        f"{total:.0f}",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#222222",
    )

for xpos, total in zip(x_epec, totals_epec):
    ax.text(
        xpos,
        total + max_total * 0.018,
        f"{total:.0f}",
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#222222",
    )

ax.set_xticks(x)
ax.set_xticklabels(PERIODS, fontsize=15)
ax.set_xlabel("")
ax.set_ylabel("Installed capacity (GW)", fontsize=15)
ax.set_ylim(0, max_total * 1.15)
ax.grid(True, axis="y", linestyle=":", color=COLOR_GRID)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", length=0)

# Compact model labels below each paired bar. They identify the adjacent bars
# without adding a second legend that competes with the regional color legend.
for xpos in x_planner:
    ax.text(
        xpos,
        -0.07,
        "Planner",
        ha="center",
        va="top",
        fontsize=10.5,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
for xpos in x_epec:
    ax.text(
        xpos,
        -0.07,
        "EPEC",
        ha="center",
        va="top",
        fontsize=10.5,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

legend_handles = [
    Patch(
        facecolor=REGION_COLORS[region],
        edgecolor="white",
        linewidth=0.8,
        label=REGION_NAMES[region],
    )
    for region in REGIONS
]
ax.legend(
    handles=legend_handles,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.30),
    frameon=True,
    fontsize=12,
    framealpha=0.9,
    handlelength=1.4,
    handletextpad=0.45,
    columnspacing=1.0,
    borderpad=0.45,
    labelspacing=0.35,
)

fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.30)

out_png = os.path.join(OUT_DIR, "capacity_stacked_planner_vs_epec.png")
out_pdf = os.path.join(OUT_DIR, "capacity_stacked_planner_vs_epec.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved to {out_png}")
print(f"Saved to {out_pdf}")
