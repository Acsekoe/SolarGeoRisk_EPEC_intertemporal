"""
plot_welfare_iters.py
=====================
Regional welfare (W, NPV-weighted sum over periods) over Gauss-Seidel iterations
for the selected EPEC run (ch->row->apac->us->eu->af), with the planner benchmark
shown as a horizontal reference line.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "model"))
from data_prep import load_data_from_excel

EPEC_PATH    = os.path.join(SCRIPT_DIR, "..", "outputs", "sens", "sens_ch-row-apac-us-eu-af.xlsx")
PLANNER_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "llp_planner_results.xlsx")
INPUT_PATH   = os.path.join(SCRIPT_DIR, "..", "inputs", "input_data_intertemporal.xlsx")
OUT_DIR      = os.path.join(SCRIPT_DIR, "..", "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

REGIONS      = ["ch", "eu", "us", "apac", "af", "row"]
REGION_NAMES = {"ch": "China", "eu": "Europe", "us": "United States",
                "apac": "Asia-Pacific", "af": "Africa", "row": "Rest of World"}
EXCLUDE_YEARS = {"2045"}

# ---------------------------------------------------------------------------
# Load structural parameters
# ---------------------------------------------------------------------------
print("Loading input data...")
data = load_data_from_excel(INPUT_PATH, params_region_sheet="params_region_new")

times      = [t for t in (data.times or ["2025","2030","2035","2040","2045"]) if t not in EXCLUDE_YEARS]
beta_d     = dict(data.beta_t)        if data.beta_t        else {t: 1.0 for t in times}
ytn_d      = dict(data.years_to_next) if data.years_to_next else {t: 5.0 for t in times}
f_hold_d   = dict(data.f_hold)        if data.f_hold        else {r: 0.0 for r in REGIONS}
c_inv_d    = dict(data.c_inv)         if data.c_inv         else {r: 0.0 for r in REGIONS}
c_ship_d   = data.c_ship              or {}
b_dem_t_d  = dict(data.b_dem_t)       if data.b_dem_t       else {(r,t): float(data.b_dem.get(r,1.0)) for r in REGIONS for t in times}
c_man_t_d  = data.c_man_t             or {}

# ---------------------------------------------------------------------------
# Load detailed_iters
# ---------------------------------------------------------------------------
print("Loading detailed_iters...")
df = pd.read_excel(EPEC_PATH, sheet_name="detailed_iters")
df["t"] = df["t"].astype(str)
df = df[~df["t"].isin(EXCLUDE_YEARS)]

# derive x_dem = sum of all imports
imp_cols = [c for c in df.columns if c.startswith("x_imp_from_")]
df["x_dem"] = df[imp_cols].sum(axis=1)

# build lam lookup: (iter, r, t) -> lam
lam_lkp = df.set_index(["iter", "r", "t"])["lam"].to_dict()

# ---------------------------------------------------------------------------
# Compute welfare per (iter, r) — NPV sum over periods
# ---------------------------------------------------------------------------
records = []
for (it, r, t), grp in df.groupby(["iter", "r", "t"]):
    row      = grp.iloc[0]
    x_dem    = float(row["x_dem"])
    lam_r    = float(row["lam"])
    a        = float(row["a_bid"])           # = a_dem (fix_a_bid_to_true_dem=True)
    b        = b_dem_t_d.get((r, t), 0.0)
    kcap     = float(row["Kcap"])
    icap     = float(row["Icap_report"])

    gross_cs = a * x_dem - 0.5 * b * x_dem ** 2
    net_cs   = gross_cs - lam_r * x_dem

    # producer surplus from exports
    ps = 0.0
    for j in REGIONS:
        x_flow = float(row.get(f"x_exp_to_{j}", 0.0))
        if x_flow > 1e-6:
            lam_j  = lam_lkp.get((it, j, t), 0.0)
            c_man  = c_man_t_d.get((r, t), float(data.c_man.get(r, 0.0)))
            c_ship = float(c_ship_d.get((r, j), 0.0))
            ps    += (lam_j - c_man - c_ship) * x_flow

    cap_cost = f_hold_d.get(r, 0.0) * kcap + c_inv_d.get(r, 0.0) * icap
    w        = net_cs + ps - cap_cost
    beta     = beta_d.get(t, 1.0)
    ytn      = ytn_d.get(t, 5.0)

    records.append({"iter": it, "r": r, "t": t, "W_npv": beta * ytn * w})

df_w = pd.DataFrame(records)
df_iter_welfare = df_w.groupby(["iter", "r"])["W_npv"].sum().reset_index()

# ---------------------------------------------------------------------------
# Load planner benchmark (NPV sum over periods per region)
# ---------------------------------------------------------------------------
df_plan = pd.read_excel(PLANNER_PATH, sheet_name="regions")
df_plan["t"] = df_plan["t"].astype(str)
df_plan = df_plan[~df_plan["t"].isin(EXCLUDE_YEARS)]

plan_welfare = {}
for r in REGIONS:
    pr = df_plan[df_plan["r"] == r]
    total = 0.0
    for _, row in pr.iterrows():
        t     = row["t"]
        x_dem = float(row["x_dem"])
        lam_r = float(row["lam"])
        a     = float(row.get("a_dem_used", 0.0)) if "a_dem_used" in row else 0.0
        b     = b_dem_t_d.get((r, t), 0.0)
        kcap  = float(row["Kcap"])
        icap  = float(row["Icap_report"])

        if a == 0.0:
            a = b_dem_t_d.get((r, t), 0.0)
            from data_prep import load_data_from_excel as _ld
            a_dem_t_d = dict(data.a_dem_t) if data.a_dem_t else {(rr, tt): float(data.a_dem.get(rr, 0.0)) for rr in REGIONS for tt in times}
            a = a_dem_t_d.get((r, t), 0.0)

        gross_cs = a * x_dem - 0.5 * b * x_dem ** 2
        net_cs   = gross_cs - lam_r * x_dem
        cap_cost = f_hold_d.get(r, 0.0) * kcap + c_inv_d.get(r, 0.0) * icap

        # PS from flows sheet
        df_pf = pd.read_excel(PLANNER_PATH, sheet_name="flows")
        df_pf["t"] = df_pf["t"].astype(str)
        lam_plan = {(rr, tt): float(df_plan[(df_plan["r"]==rr) & (df_plan["t"]==tt)]["lam"].values[0])
                    for rr in REGIONS for tt in times
                    if len(df_plan[(df_plan["r"]==rr) & (df_plan["t"]==tt)]) > 0}
        exports = df_pf[(df_pf["exp"] == r) & (df_pf["t"] == t)]
        ps = sum((lam_plan.get((fl["imp"], t), 0.0) - float(fl["c_man"]) - float(fl["c_ship"])) * float(fl["x"])
                 for _, fl in exports.iterrows())

        w    = net_cs + ps - cap_cost
        beta = beta_d.get(t, 1.0)
        ytn  = ytn_d.get(t, 5.0)
        total += beta * ytn * w
    plan_welfare[r] = total

# ---------------------------------------------------------------------------
# Plot — 2x3 grid, one panel per region
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()

COLOR_EPEC = "#E53935"
COLOR_PLAN = "#2196F3"

for ax, r in zip(axes, REGIONS):
    sub = df_iter_welfare[df_iter_welfare["r"] == r].sort_values("iter")
    ax.plot(sub["iter"], sub["W_npv"], color=COLOR_EPEC, linewidth=2,
            marker="o", markersize=3, label="EPEC (per iter)")
    ax.axhline(plan_welfare[r], color=COLOR_PLAN, linewidth=1.8,
               linestyle="--", label="Planner benchmark")

    ax.set_title(REGION_NAMES[r], fontsize=11, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=9)
    ax.set_ylabel("W_npv  (M USD)", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.tick_params(labelsize=8)

handles = [mpatches.Patch(color=COLOR_EPEC, label="EPEC welfare per iteration"),
           mpatches.Patch(color=COLOR_PLAN, label="Planner benchmark")]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Regional welfare over iterations — EPEC vs Planner benchmark\n(ch→row→apac→us→eu→af)",
             fontsize=12, fontweight="bold", y=1.01)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "welfare_over_iterations.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
