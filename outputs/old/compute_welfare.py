"""
compute_welfare.py
==================
Post-processing script that computes comparable regional welfare contributions
for both the LLP planner benchmark and one or more EPEC equilibrium runs.

Welfare decomposition per region r, period t:
    CS_r_t      = a_dem * x_dem  -  0.5 * b_dem * x_dem²
    PS_r_t      = Σ_j [ (lam_j_t - c_man_r_t - c_ship_rj) * x_rjt ]
    CapCost_r_t = f_hold_r * Kcap_r_t  +  c_inv_r * Icap_r_t
    W_r_t       = CS_r_t + PS_r_t - CapCost_r_t

NPV-weighted contribution:
    W_r_t_npv   = beta_t * ytn_t * W_r_t

Output: outputs/welfare_comparison.xlsx
    Sheet "welfare"   — long-format table (model, run, r, t, CS, PS, CapCost, W, W_npv)
    Sheet "summary"   — W_npv summed over periods, by model × region
    Sheet "deadweight"— planner W_npv minus EPEC W_npv (welfare loss per region)
"""

import os
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH   = os.path.join(SCRIPT_DIR, "inputs", "input_data_intertemporal.xlsx")
PLANNER_PATH = os.path.join(SCRIPT_DIR, "outputs", "llp_planner_results.xlsx")
SENS_DIR     = os.path.join(SCRIPT_DIR, "outputs", "sens")
OUT_PATH     = os.path.join(SCRIPT_DIR, "outputs", "welfare_comparison.xlsx")

EXCLUDE_YEARS = {"2045"}

# ---------------------------------------------------------------------------
# Load structural parameters from input data
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(SCRIPT_DIR, "model"))
from data_prep import load_data_from_excel

print("Loading input data …")
data = load_data_from_excel(INPUT_PATH, params_region_sheet="params_region_new")

regions    = data.regions
times      = [t for t in (data.times or ["2025","2030","2035","2040","2045"]) if t not in EXCLUDE_YEARS]
beta_d     = dict(data.beta_t)   if data.beta_t      else {t: 1.0 for t in times}
ytn_d      = dict(data.years_to_next) if data.years_to_next else {t: 5.0 for t in times}
f_hold_d   = dict(data.f_hold)   if data.f_hold      else {r: 0.0 for r in regions}
c_inv_d    = dict(data.c_inv)    if data.c_inv       else {r: 0.0 for r in regions}

# ---------------------------------------------------------------------------
# Helper: compute welfare rows from a regions + flows DataFrame pair
# ---------------------------------------------------------------------------
def compute_welfare_rows(df_reg, df_flows, model_label, run_label):
    """
    df_reg   must have columns: r, t, x_dem, Kcap, Icap_report, a_dem_used, b_dem_used, lam
    df_flows must have columns: exp, imp, t, x, c_ship, c_man

    Welfare decomposition:
        gross_CS  = a*x_dem - 0.5*b*x_dem²          (area under demand curve)
        net_CS    = gross_CS - lam_r * x_dem          (what consumers keep after paying)
        PS        = Σ_j (lam_j - c_man - c_ship)*x    (producer rent net of var costs)
        CapCost   = f_hold*Kcap + c_inv*Icap
        W         = net_CS + PS - CapCost
                  = gross_CS - total_var_costs - CapCost   (lam cancels by market clearing)

    net_CS and PS are lam-dependent, so they capture redistribution between
    consumers and producers. W (total) only shows the deadweight loss.
    """
    rows = []
    lam_lkp = {(row["r"], str(row["t"])): float(row["lam"]) for _, row in df_reg.iterrows()}
    cost_col = "c_man_t" if "c_man_t" in df_reg.columns else "c_man_var"
    c_man_lkp = (
        {
            (row["r"], str(row["t"])): float(row[cost_col])
            for _, row in df_reg.iterrows()
        }
        if cost_col in df_reg.columns
        else {}
    )

    for _, reg in df_reg.iterrows():
        r, t = reg["r"], str(reg["t"])
        if t in EXCLUDE_YEARS:
            continue

        x_dem    = float(reg["x_dem"])
        a        = float(reg["a_dem_used"])
        b        = float(reg["b_dem_used"])
        kcap     = float(reg["Kcap"])
        icap     = float(reg["Icap_report"])
        lam_r    = lam_lkp.get((r, t), 0.0)

        gross_cs = a * x_dem - 0.5 * b * x_dem ** 2
        net_cs   = gross_cs - lam_r * x_dem

        # Producer surplus: revenue minus variable costs for all exports from r
        exports_r = df_flows[(df_flows["exp"] == r) & (df_flows["t"].astype(str) == t)]
        ps = 0.0
        for _, fl in exports_r.iterrows():
            j      = fl["imp"]
            x_flow = float(fl["x"])
            c_ship = float(fl["c_ship"])
            c_man  = c_man_lkp.get((r, t), float(fl["c_man"]))
            lam_j  = lam_lkp.get((j, t), 0.0)
            ps    += (lam_j - c_man - c_ship) * x_flow

        cap_cost = f_hold_d.get(r, 0.0) * kcap + c_inv_d.get(r, 0.0) * icap
        w        = net_cs + ps - cap_cost
        beta     = beta_d.get(t, 1.0)
        ytn      = ytn_d.get(t, 5.0)

        rows.append({
            "model":    model_label,
            "run":      run_label,
            "r":        r,
            "t":        t,
            "lam":      lam_r,
            "gross_CS": gross_cs,
            "net_CS":   net_cs,
            "PS":       ps,
            "CapCost":  cap_cost,
            "W":        w,
            "W_npv":    beta * ytn * w,
        })

    return rows

# ---------------------------------------------------------------------------
# 1. Planner welfare
# ---------------------------------------------------------------------------
print("Reading planner output …")
df_plan_reg   = pd.read_excel(PLANNER_PATH, sheet_name="regions")
df_plan_flows = pd.read_excel(PLANNER_PATH, sheet_name="flows")

# planner regions sheet uses column names without a_dem_used — rename to match helper
df_plan_reg = df_plan_reg.rename(columns={"region": "r"}) if "region" in df_plan_reg.columns else df_plan_reg

# planner sheet already has cs/prod_cost but let's recompute consistently from raw cols
# we need a_dem_used and b_dem_used — fall back to data dicts if not present
a_dem_t_dict = dict(data.a_dem_t) if data.a_dem_t else {(r, t): float(data.a_dem.get(r, 0.0)) for r in regions for t in times}
b_dem_t_dict = dict(data.b_dem_t) if data.b_dem_t else {(r, t): float(data.b_dem.get(r, 1.0)) for r in regions for t in times}

if "a_dem_used" not in df_plan_reg.columns:
    df_plan_reg["a_dem_used"] = df_plan_reg.apply(lambda row: a_dem_t_dict.get((row["r"], str(row["t"])), 0.0), axis=1)
if "b_dem_used" not in df_plan_reg.columns:
    df_plan_reg["b_dem_used"] = df_plan_reg.apply(lambda row: b_dem_t_dict.get((row["r"], str(row["t"])), 0.0), axis=1)

plan_rows = compute_welfare_rows(df_plan_reg, df_plan_flows, "planner", "planner")
print(f"  -> {len(plan_rows)} region-period rows")

# ---------------------------------------------------------------------------
# 2. EPEC welfare — all sens files
# ---------------------------------------------------------------------------
all_rows = list(plan_rows)

sens_files = sorted(glob.glob(os.path.join(SENS_DIR, "**", "sens_*.xlsx"), recursive=True))
print(f"\nFound {len(sens_files)} EPEC sens files")

for fpath in sens_files:
    run_label = os.path.splitext(os.path.basename(fpath))[0]  # e.g. sens_eu-us-af-row-apac-ch
    print(f"  Reading {run_label} …")

    df_reg   = pd.read_excel(fpath, sheet_name="regions")
    df_flows = pd.read_excel(fpath, sheet_name="flows")

    # EPEC regions sheet has a_dem_used and b_dem_used already
    rows = compute_welfare_rows(df_reg, df_flows, "epec", run_label)
    all_rows.extend(rows)

# ---------------------------------------------------------------------------
# 3. Build output
# ---------------------------------------------------------------------------
df_all = pd.DataFrame(all_rows)

# summary: key welfare components summed over periods
df_summary = (
    df_all.groupby(["model", "run", "r"])[["gross_CS", "net_CS", "PS", "CapCost", "W_npv"]]
    .sum()
    .reset_index()
    .rename(columns={"W_npv": "W_npv_total"})
)

# deadweight: planner W_npv_total minus EPEC W_npv_total per region
df_plan_sum = (
    df_summary[df_summary["model"] == "planner"]
    [["r", "gross_CS", "net_CS", "PS", "CapCost", "W_npv_total"]]
    .rename(columns={"gross_CS": "gross_CS_plan", "net_CS": "net_CS_plan",
                     "PS": "PS_plan", "CapCost": "CapCost_plan", "W_npv_total": "W_planner"})
)

df_dw_rows = []
for _, epec_row in df_summary[df_summary["model"] == "epec"].iterrows():
    r     = epec_row["r"]
    plan  = df_plan_sum[df_plan_sum["r"] == r]
    if plan.empty:
        continue
    plan  = plan.iloc[0]
    df_dw_rows.append({
        "run":              epec_row["run"],
        "r":                r,
        # total welfare
        "W_planner":        plan["W_planner"],
        "W_epec":           epec_row["W_npv_total"],
        "deadweight_loss":  plan["W_planner"] - epec_row["W_npv_total"],
        "deadweight_pct":   (plan["W_planner"] - epec_row["W_npv_total"]) / abs(plan["W_planner"]) * 100 if plan["W_planner"] else float("nan"),
        # redistribution: net_CS shift (negative = consumers lose)
        "net_CS_plan":      plan["net_CS_plan"],
        "net_CS_epec":      epec_row["net_CS"],
        "CS_transfer":      epec_row["net_CS"] - plan["net_CS_plan"],
        # redistribution: PS shift (positive = producers gain)
        "PS_plan":          plan["PS_plan"],
        "PS_epec":          epec_row["PS"],
        "PS_transfer":      epec_row["PS"] - plan["PS_plan"],
    })

df_dw = pd.DataFrame(df_dw_rows)
df_totals = (
    df_summary.groupby(["model", "run"], as_index=False)["W_npv_total"]
    .sum()
    .rename(columns={"W_npv_total": "W_npv_total_all_regions"})
)

# ---------------------------------------------------------------------------
# 4. Write Excel
# ---------------------------------------------------------------------------
print(f"\nWriting {OUT_PATH} …")
with pd.ExcelWriter(OUT_PATH) as writer:
    df_all.to_excel(writer, sheet_name="welfare", index=False)
    df_summary.to_excel(writer, sheet_name="summary", index=False)
    df_dw.to_excel(writer, sheet_name="deadweight", index=False)
    df_totals.to_excel(writer, sheet_name="totals", index=False)

print("Done.")
print("\n=== Planner total W_npv by region ===")
print(df_plan_sum.set_index("r").to_string())
print("\n=== Mean welfare loss vs planner (across EPEC runs) ===")
print(df_dw.groupby("r")[["deadweight_loss", "deadweight_pct", "CS_transfer", "PS_transfer"]].mean().round(2).to_string())
