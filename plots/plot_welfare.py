"""
plot_welfare.py
===============
Three welfare comparison plots — Planner vs EPEC (ch->row->apac->us->eu->af):

  Plot 2: Stacked bar — welfare decomposition (net_CS, PS, CapCost) per region
  Plot 3: Waterfall   — CS transfer, PS transfer, deadweight loss per region
  Plot 4: Time series — W per region over periods
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
WELFARE_PATH  = os.path.join(SCRIPT_DIR, "..", "outputs", "sens", "welfare_comparison.xlsx")
OUT_DIR       = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

SELECTED_RUN  = "sens_ch-row-apac-us-eu-af"
PERIODS       = ["2025", "2030", "2035", "2040"]
REGIONS       = ["ch", "eu", "us", "apac", "af", "row"]
REGION_NAMES  = {"ch": "China", "eu": "Europe", "us": "United States",
                 "apac": "Asia-Pacific", "af": "Africa", "row": "Rest of World"}

COLOR_PLAN    = "#2196F3"
COLOR_EPEC    = "#E53935"
COLOR_CS      = "#4CAF50"
COLOR_PS      = "#FF9800"
COLOR_CAP     = "#9E9E9E"
COLOR_DW      = "#7B1FA2"
COLOR_POS     = "#43A047"
COLOR_NEG     = "#E53935"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_welfare  = pd.read_excel(WELFARE_PATH, sheet_name="welfare")
df_summary  = pd.read_excel(WELFARE_PATH, sheet_name="summary")
df_dw       = pd.read_excel(WELFARE_PATH, sheet_name="deadweight")

df_welfare["t"] = df_welfare["t"].astype(str)
# normalise column name
W_COL = [c for c in df_welfare.columns if c.startswith("W ")][0]
df_welfare = df_welfare.rename(columns={W_COL: "W"})

# filter to selected run + planner
df_sum_plan = df_summary[df_summary["run"] == "planner"].set_index("r")
df_sum_epec = df_summary[df_summary["run"] == SELECTED_RUN].set_index("r")
df_dw_sel   = df_dw[df_dw["run"] == SELECTED_RUN].set_index("r")

df_w_plan   = df_welfare[df_welfare["run"] == "planner"]
df_w_epec   = df_welfare[df_welfare["run"] == SELECTED_RUN]

# scale to billions for readability
SCALE       = 1e6       # values are in M$, divide by 1e6 → trillions; or keep M$
UNIT_LABEL  = "M USD"

# ===========================================================================
# PLOT 2 — Stacked bar: welfare decomposition per region
# ===========================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes2 = axes2.flatten()

for ax, r in zip(axes2, REGIONS):
    x      = np.array([0, 1])
    labels = ["Planner", "EPEC"]

    for idx, df_s in enumerate([df_sum_plan, df_sum_epec]):
        net_cs   =  df_s.loc[r, "net_CS"]
        ps       =  df_s.loc[r, "PS"]
        cap_cost = -df_s.loc[r, "CapCost"]   # negative (cost)

        # stack positive components first, then negative
        pos_vals = [v for v in [net_cs, ps] if v >= 0]
        neg_vals = [v for v in [net_cs, ps, cap_cost] if v < 0]

        bottom_pos, bottom_neg = 0.0, 0.0
        for val, color, label in [(net_cs, COLOR_CS, "Net CS"),
                                   (ps,     COLOR_PS, "PS"),
                                   (cap_cost, COLOR_CAP, "−CapCost")]:
            ax.bar(x[idx], val, bottom=bottom_pos if val >= 0 else bottom_neg,
                   color=color, width=0.5,
                   label=label if idx == 0 else "_nolegend_")
            if val >= 0:
                bottom_pos += val
            else:
                bottom_neg += val

    ax.set_title(REGION_NAMES[r], fontsize=11, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"W_npv  ({UNIT_LABEL})", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=8)

handles = [mpatches.Patch(color=COLOR_CS,  label="Net Consumer Surplus"),
           mpatches.Patch(color=COLOR_PS,  label="Producer Surplus"),
           mpatches.Patch(color=COLOR_CAP, label="−Capacity Cost")]
fig2.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig2.suptitle("Welfare decomposition by region — Planner vs EPEC",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "welfare_decomposition.png"), dpi=150, bbox_inches="tight")
print("Saved welfare_decomposition.png")

# ===========================================================================
# PLOT 3 — Waterfall: CS transfer, PS transfer, deadweight loss per region
# ===========================================================================
fig3, ax3 = plt.subplots(figsize=(11, 6))

r_labels   = [REGION_NAMES[r] for r in REGIONS]
n          = len(REGIONS)
x          = np.arange(n)
width      = 0.25

cs_vals  = [df_dw_sel.loc[r, "CS_transfer"]   for r in REGIONS]
ps_vals  = [df_dw_sel.loc[r, "PS_transfer"]   for r in REGIONS]
dw_vals  = [df_dw_sel.loc[r, "deadweight_loss"] for r in REGIONS]

bars_cs = ax3.bar(x - width, cs_vals, width, label="CS transfer (consumers)",
                  color=[COLOR_NEG if v < 0 else COLOR_POS for v in cs_vals], alpha=0.85)
bars_ps = ax3.bar(x,         ps_vals, width, label="PS transfer (producers)",
                  color=[COLOR_POS if v > 0 else COLOR_NEG for v in ps_vals], alpha=0.85)
bars_dw = ax3.bar(x + width, dw_vals, width, label="Deadweight loss (total W)",
                  color=COLOR_DW, alpha=0.85)

ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(r_labels, fontsize=10)
ax3.set_ylabel(f"EPEC − Planner  ({UNIT_LABEL})", fontsize=10)
ax3.set_title("Welfare redistribution and deadweight loss — EPEC vs Planner",
              fontsize=12, fontweight="bold")
ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

handles = [mpatches.Patch(color=COLOR_NEG, label="CS transfer (consumers lose)"),
           mpatches.Patch(color=COLOR_POS, label="PS transfer (producers gain)"),
           mpatches.Patch(color=COLOR_DW,  label="Net deadweight loss")]
ax3.legend(handles=handles, fontsize=9, loc="lower right")
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "welfare_waterfall.png"), dpi=150, bbox_inches="tight")
print("Saved welfare_waterfall.png")

# ===========================================================================
# PLOT 4 — Time series: W per region over periods
# ===========================================================================
fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes4 = axes4.flatten()

for ax, r in zip(axes4, REGIONS):
    plan_r = df_w_plan[df_w_plan["r"] == r].set_index("t").reindex(PERIODS)
    epec_r = df_w_epec[df_w_epec["r"] == r].set_index("t").reindex(PERIODS)

    periods_int = [int(p) for p in PERIODS]

    ax.plot(periods_int, plan_r["W"], color=COLOR_PLAN, linewidth=2,
            marker="o", markersize=5, label="Planner")
    ax.plot(periods_int, epec_r["W"], color=COLOR_EPEC, linewidth=2,
            marker="s", markersize=5, label="EPEC")

    ax.fill_between(periods_int, plan_r["W"], epec_r["W"],
                    where=[True]*len(PERIODS),
                    alpha=0.12, color=COLOR_DW, label="Difference")

    ax.set_title(REGION_NAMES[r], fontsize=11, fontweight="bold")
    ax.set_xticks(periods_int)
    ax.set_xlabel("Period", fontsize=9)
    ax.set_ylabel(f"W  ({UNIT_LABEL}/yr)", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=8)

handles = [mpatches.Patch(color=COLOR_PLAN, label="Planner"),
           mpatches.Patch(color=COLOR_EPEC, label="EPEC"),
           mpatches.Patch(color=COLOR_DW,   alpha=0.3, label="Difference")]
fig4.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
            framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig4.suptitle("Regional welfare over time — Planner vs EPEC",
              fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "welfare_timeseries.png"), dpi=150, bbox_inches="tight")
print("Saved welfare_timeseries.png")
