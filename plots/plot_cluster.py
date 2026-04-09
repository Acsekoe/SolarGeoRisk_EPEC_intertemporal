"""
plot_cluster.py
===============
Cluster (grouped bar) plots showing Kcap and lam per region per year
across all EPEC player-order runs, with the planner benchmark overlaid.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SENS_DIR     = os.path.join(SCRIPT_DIR, "..", "outputs", "sens")
PLANNER_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "llp_planner_results.xlsx")
OUT_DIR      = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PERIODS      = ["2025", "2030", "2035", "2040"]
REGIONS      = ["ch", "eu", "us", "apac", "af", "row"]
REGION_NAMES = {"ch": "China", "eu": "Europe", "us": "United States",
                "apac": "Asia-Pacific", "af": "Africa", "row": "Rest of World"}

# exclude the duplicate early run
EXCLUDE_RUNS = {"sens_us-apac-af-row-eu-ch_early"}

# ---------------------------------------------------------------------------
# Load all EPEC runs
# ---------------------------------------------------------------------------
runs = {}
for fpath in sorted(glob.glob(os.path.join(SENS_DIR, "sens_*.xlsx"))):
    name = os.path.splitext(os.path.basename(fpath))[0]
    if name in EXCLUDE_RUNS:
        continue
    label = name.replace("sens_", "")
    df = pd.read_excel(fpath, sheet_name="regions")
    df["t"] = df["t"].astype(str)
    runs[label] = df

run_names = list(runs.keys())
n_runs    = len(run_names)

# short labels for legend readability
def short_label(label):
    parts = label.split("-")
    return "→".join(p.upper() for p in parts)

short_names = [short_label(r) for r in run_names]

# colormap for runs
cmap   = plt.cm.get_cmap("tab10", n_runs)
colors = [cmap(i) for i in range(n_runs)]

# planner
df_plan = pd.read_excel(PLANNER_PATH, sheet_name="regions")
df_plan["t"] = df_plan["t"].astype(str)

# ---------------------------------------------------------------------------
# Helper: build cluster plot for one variable
# ---------------------------------------------------------------------------
def cluster_plot(var, ylabel, title_suffix, out_name, ylim_bottom=0):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes = axes.flatten()

    n_periods = len(PERIODS)
    bar_width = 0.8 / (n_runs + 1)   # +1 slot for planner
    x = np.arange(n_periods)

    for ax, r in zip(axes, REGIONS):
        # planner reference line per period
        plan_vals = []
        for t in PERIODS:
            row = df_plan[(df_plan["r"] == r) & (df_plan["t"] == t)]
            plan_vals.append(float(row[var].values[0]) if len(row) else np.nan)

        # EPEC bars
        for i, (run_label, df_run) in enumerate(runs.items()):
            vals = []
            for t in PERIODS:
                row = df_run[(df_run["r"] == r) & (df_run["t"] == t)]
                v = float(row[var].values[0]) if len(row) else np.nan
                vals.append(max(v, 0))   # clip negatives (near-zero exits) to 0
            offset = (i - n_runs / 2) * bar_width + bar_width / 2
            ax.bar(x + offset, vals, width=bar_width, color=colors[i],
                   alpha=0.85, label=short_names[i] if r == "ch" else "_nolegend_")

        # planner as horizontal tick marks
        for xi, pv in zip(x, plan_vals):
            if not np.isnan(pv):
                ax.plot([xi - 0.38, xi + 0.38], [pv, pv],
                        color="black", linewidth=2.0, zorder=5,
                        label="Planner" if xi == x[0] and r == "ch" else "_nolegend_")

        ax.set_title(REGION_NAMES[r], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(PERIODS, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim_bottom is not None:
            ax.set_ylim(bottom=ylim_bottom)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

    # legend
    run_handles = [mpatches.Patch(color=colors[i], label=short_names[i])
                   for i in range(n_runs)]
    plan_handle = plt.Line2D([0], [0], color="black", linewidth=2, label="Planner")
    fig.legend(handles=run_handles + [plan_handle],
               loc="lower center", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(f"{title_suffix} by region and period — all player orderings",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name), dpi=150, bbox_inches="tight")
    print(f"Saved {out_name}")

# ---------------------------------------------------------------------------
# Plot 1: Installed capacity (Kcap)
# ---------------------------------------------------------------------------
cluster_plot("Kcap",    "Capacity (GW)",   "Installed capacity", "cluster_kcap.png", ylim_bottom=0)

# ---------------------------------------------------------------------------
# Plot 2: Market price (lam)
# ---------------------------------------------------------------------------
cluster_plot("lam",     "Price ($/kW)",    "Market price",       "cluster_lam.png",  ylim_bottom=None)
