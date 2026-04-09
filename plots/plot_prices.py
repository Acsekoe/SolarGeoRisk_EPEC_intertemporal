"""
plot_prices.py
==============
Market prices (lam) and marginal cost (c_man_var) per region over 2025-2040,
comparing the LLP planner benchmark vs the selected EPEC run (ch-row-apac-us-eu-af).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PLANNER_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "llp_planner_results.xlsx")
EPEC_PATH    = os.path.join(SCRIPT_DIR, "..", "outputs", "sens", "sens_ch-row-apac-us-eu-af.xlsx")
OUT_DIR      = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PERIODS      = ["2025", "2030", "2035", "2040"]
REGIONS      = ["ch", "eu", "us", "apac", "af", "row"]
REGION_NAMES = {"ch": "China", "eu": "Europe", "us": "United States",
                "apac": "Asia-Pacific", "af": "Africa", "row": "Rest of World"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_plan = pd.read_excel(PLANNER_PATH, sheet_name="regions")
df_plan["t"] = df_plan["t"].astype(str)

df_epec = pd.read_excel(EPEC_PATH, sheet_name="regions")
df_epec["t"] = df_epec["t"].astype(str)

# ---------------------------------------------------------------------------
# Plot — 2x3 grid, one panel per region
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
axes = axes.flatten()

COLOR_PLAN  = "#2196F3"   # blue
COLOR_EPEC  = "#E53935"   # red
COLOR_COST  = "#757575"   # grey for marginal cost

for ax, r in zip(axes, REGIONS):
    plan_r = df_plan[df_plan["r"] == r].set_index("t").reindex(PERIODS)
    epec_r = df_epec[df_epec["r"] == r].set_index("t").reindex(PERIODS)

    periods_int = [int(p) for p in PERIODS]

    ax.plot(periods_int, plan_r["lam"],     color=COLOR_PLAN, linewidth=2,
            marker="o", markersize=5, label="Planner")
    ax.plot(periods_int, epec_r["lam"],     color=COLOR_EPEC, linewidth=2,
            marker="s", markersize=5, label="EPEC")
    ax.plot(periods_int, epec_r["c_man_var"], color=COLOR_COST, linewidth=1.4,
            linestyle="--", marker="^", markersize=4, label="Marginal cost")

    ax.set_title(REGION_NAMES[r], fontsize=11, fontweight="bold")
    ax.set_xticks(periods_int)
    ax.set_xlabel("Period", fontsize=9)
    ax.set_ylabel("Price ($/kW)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=8)

# shared legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Market prices by region — Planner vs EPEC (ch→row→apac→us→eu→af)",
             fontsize=12, fontweight="bold", y=1.01)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "prices_planner_vs_epec.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
