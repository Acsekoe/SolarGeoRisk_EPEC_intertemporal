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

# Match IEEEtran serif rendering used by the other paper figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PLANNER_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "llp_planner_results.xlsx")
EPEC_PATH    = os.path.join(SCRIPT_DIR, "..", "outputs", "sens", "converged", "sens_ch-row-apac-us-eu-af.xlsx")
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
fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.4), sharey=False)
axes = axes.flatten()

COLOR_PLAN  = "#2E6F40"   # green
COLOR_EPEC  = "#A83232"   # red
COLOR_COST  = "#6E6E6E"   # grey

for i, (ax, r) in enumerate(zip(axes, REGIONS)):
    plan_r = df_plan[df_plan["r"] == r].set_index("t").reindex(PERIODS)
    epec_r = df_epec[df_epec["r"] == r].set_index("t").reindex(PERIODS)
    cost_col = "c_man_var" if "c_man_var" in epec_r.columns else "c_man_t"
    cost_r = epec_r[cost_col] if cost_col in epec_r.columns else plan_r["c_man_t"]

    periods_int = [int(p) for p in PERIODS]

    ax.plot(periods_int, plan_r["lam"],     color=COLOR_PLAN, linewidth=2.2,
            marker="o", markersize=5.5, label="Planner")
    ax.plot(periods_int, epec_r["lam"],     color=COLOR_EPEC, linewidth=2.2,
            marker="s", markersize=5.5, label="EPEC")
    ax.plot(periods_int, cost_r, color=COLOR_COST, linewidth=1.8,
            linestyle="--", marker="^", markersize=5.0, label="Regional manufacturing costs")
    ax.set_title(REGION_NAMES[r], fontsize=18, fontweight="normal")
    ax.set_xticks(periods_int)
    ax.set_xlabel("")
    if i % 2 == 0:
        ax.set_ylabel("Price [$/kW]", fontsize=20)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(axis="both", labelsize=15)

# shared legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=13,
           framealpha=0.9, handletextpad=0.5, columnspacing=1.0,
           borderpad=0.45, labelspacing=0.35, bbox_to_anchor=(0.5, 0.015))

fig.text(
    0.5,
    0.083,
    "EU and US: 2025 offers marginally underbid regional manufacturing costs; subsequent EPEC prices remain above costs.",
    ha="center",
    va="center",
    fontsize=10.5,
)

fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.17,
                    wspace=0.34, hspace=0.50)
out_path = os.path.join(OUT_DIR, "prices_planner_vs_epec.png")
out_pdf = os.path.join(OUT_DIR, "prices_planner_vs_epec.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved to {out_path}")
print(f"Saved to {out_pdf}")
