"""
plot_capacity_epec_demand.py
============================
Installed production capacity in the converged EPEC equilibrium, with global
demand overlaid.

The stacked bars show regional installed capacity (Kcap). Black markers show
aggregate global demand, computed as the sum of regional x_dem in each period.
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

REGION_COLORS = {
    "ch": "#2F3A67",
    "eu": "#3A9BC1",
    "us": "#C75D4A",
    "apac": "#5FAE8B",
    "af": "#D9A441",
    "row": "#9B6FB5",
}

BAR_ALPHA = 0.78
COLOR_DEMAND = "#D9822B"
COLOR_PLANNER_CAPACITY = "#4A4A4A"


df = pd.read_excel(EPEC_PATH, sheet_name="regions")
df["r"] = df["r"].astype(str).str.lower()
df["t"] = df["t"].astype(str)
df = df[df["t"].isin(PERIODS) & df["r"].isin(REGIONS)].copy()

# Solver tolerances can leave decommissioned capacity as tiny negatives.
df["Kcap"] = df["Kcap"].clip(lower=0.0)

capacity = (
    df.pivot_table(index="t", columns="r", values="Kcap", aggfunc="sum")
    .reindex(index=PERIODS, columns=REGIONS)
    .fillna(0.0)
)
demand = df.groupby("t")["x_dem"].sum().reindex(PERIODS).fillna(0.0)

df_plan = pd.read_excel(PLANNER_PATH, sheet_name="regions")
df_plan["r"] = df_plan["r"].astype(str).str.lower()
df_plan["t"] = df_plan["t"].astype(str)
df_plan = df_plan[df_plan["t"].isin(PERIODS) & df_plan["r"].isin(REGIONS)].copy()
df_plan["Kcap"] = df_plan["Kcap"].clip(lower=0.0)
planner_capacity = (
    df_plan.pivot_table(index="t", columns="r", values="Kcap", aggfunc="sum")
    .reindex(index=PERIODS, columns=REGIONS)
    .fillna(0.0)
)


fig, ax = plt.subplots(figsize=(7.4, 4.8))

x = np.arange(len(PERIODS), dtype=float)
bar_width = 0.46
bottom = np.zeros(len(PERIODS))

for region in REGIONS:
    values = capacity[region].to_numpy(dtype=float)
    ax.bar(
        x,
        values,
        width=bar_width,
        bottom=bottom,
        color=REGION_COLORS[region],
        alpha=BAR_ALPHA,
        edgecolor="white",
        linewidth=1.0,
        zorder=2,
    )
    bottom += values

capacity_totals = capacity.sum(axis=1).to_numpy(dtype=float)
demand_values = demand.to_numpy(dtype=float)
planner_capacity_totals = planner_capacity.sum(axis=1).to_numpy(dtype=float)
max_total = max(capacity_totals.max(), demand_values.max(), planner_capacity_totals.max())

ax.plot(
    x,
    demand_values,
    color=COLOR_DEMAND,
    linestyle="-.",
    marker="^",
    markerfacecolor=COLOR_DEMAND,
    markeredgecolor=COLOR_DEMAND,
    markeredgewidth=1.6,
    markersize=6.8,
    linewidth=1.8,
    zorder=5,
    label="Global demand",
)

ax.plot(
    x,
    planner_capacity_totals,
    color=COLOR_PLANNER_CAPACITY,
    linestyle="--",
    marker="*",
    markerfacecolor=COLOR_PLANNER_CAPACITY,
    markeredgecolor=COLOR_PLANNER_CAPACITY,
    markeredgewidth=1.4,
    markersize=9.0,
    linewidth=1.8,
    zorder=5,
    label="Planner capacity",
)

for xpos, total in zip(x, capacity_totals):
    ax.text(
        xpos,
        total + max_total * 0.018,
        f"{total:.0f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#222222",
    )

ax.set_xticks(x)
ax.set_xticklabels(PERIODS, fontsize=15)
ax.set_ylabel("GW", fontsize=15)
ax.set_ylim(0, max_total * 1.15)
ax.grid(False)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", length=0)

legend_handles = [
    Patch(
        facecolor=REGION_COLORS[region],
        edgecolor="white",
        linewidth=0.8,
        label=REGION_NAMES[region],
    )
    for region in REGIONS
]
legend_handles.append(
    Line2D(
        [0],
        [0],
        color=COLOR_DEMAND,
        linestyle="-.",
        marker="^",
        markerfacecolor=COLOR_DEMAND,
        markeredgecolor=COLOR_DEMAND,
        markeredgewidth=1.4,
        linewidth=1.8,
        markersize=6.2,
        label="Global demand",
    )
)
legend_handles.append(
    Line2D(
        [0],
        [0],
        color=COLOR_PLANNER_CAPACITY,
        linestyle="--",
        marker="*",
        markerfacecolor=COLOR_PLANNER_CAPACITY,
        markeredgecolor=COLOR_PLANNER_CAPACITY,
        markeredgewidth=1.2,
        linewidth=1.8,
        markersize=8.0,
        label="Planner capacity",
    )
)

ax.legend(
    handles=legend_handles,
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

out_png = os.path.join(OUT_DIR, "capacity_epec_stacked_with_global_demand.png")
out_pdf = os.path.join(OUT_DIR, "capacity_epec_stacked_with_global_demand.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved to {out_png}")
print(f"Saved to {out_pdf}")
