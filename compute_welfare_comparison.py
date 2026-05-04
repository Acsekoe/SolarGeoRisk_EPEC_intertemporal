"""
Compute welfare comparison from raw planner and selected EPEC outputs.

Inputs:
    outputs/llp_planner_results.xlsx
    outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx

Output:
    outputs/welfare_comparison.xlsx
"""

import ast
import contextlib
import io
import os
import sys

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "inputs", "input_data_intertemporal.xlsx")
PLANNER_PATH = os.path.join(SCRIPT_DIR, "outputs", "llp_planner_results.xlsx")
EPEC_PATH = os.path.join(
    SCRIPT_DIR,
    "outputs",
    "sens",
    "converged",
    "sens_ch-row-apac-us-eu-af.xlsx",
)
OUT_PATH = os.path.join(SCRIPT_DIR, "outputs", "welfare_comparison.xlsx")

SELECTED_RUN = "sens_ch-row-apac-us-eu-af"
PERIODS = ["2025", "2030", "2035", "2040"]
REGIONS = ["ch", "eu", "us", "apac", "af", "row"]


def load_model_data():
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "model"))
    from data_prep import load_data_from_excel

    with contextlib.redirect_stdout(io.StringIO()):
        return load_data_from_excel(INPUT_PATH, params_region_sheet="params_region_new")


def read_meta_dict(path):
    meta = pd.read_excel(path, sheet_name="meta")
    return dict(zip(meta["key"], meta["value"]))


def parse_mapping(value, fallback):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return fallback
    return parsed if isinstance(parsed, dict) else fallback


def add_demand_parameters(df_reg, data):
    df_reg = df_reg.copy()
    df_reg["t"] = df_reg["t"].astype(str)

    a_dem = dict(data.a_dem_t) if data.a_dem_t else {
        (r, t): float(data.a_dem.get(r, 0.0))
        for r in data.regions
        for t in data.times
    }
    b_dem = dict(data.b_dem_t) if data.b_dem_t else {
        (r, t): float(data.b_dem.get(r, 1.0))
        for r in data.regions
        for t in data.times
    }

    if "a_dem_used" not in df_reg.columns:
        df_reg["a_dem_used"] = df_reg.apply(
            lambda row: a_dem.get((row["r"], row["t"]), 0.0), axis=1
        )
    if "b_dem_used" not in df_reg.columns:
        df_reg["b_dem_used"] = df_reg.apply(
            lambda row: b_dem.get((row["r"], row["t"]), 0.0), axis=1
        )
    return df_reg


def compute_welfare(df_reg, df_flows, data, model, run):
    df_reg = add_demand_parameters(df_reg, data)
    df_flows = df_flows.copy()
    df_flows["t"] = df_flows["t"].astype(str)

    beta_t = dict(data.beta_t) if data.beta_t else {t: 1.0 for t in PERIODS}
    ytn_t = dict(data.years_to_next) if data.years_to_next else {t: 5.0 for t in PERIODS}
    f_hold = dict(data.f_hold) if data.f_hold else {r: 0.0 for r in REGIONS}
    c_inv = dict(data.c_inv) if data.c_inv else {r: 0.0 for r in REGIONS}

    lam = {(row["r"], row["t"]): float(row["lam"]) for _, row in df_reg.iterrows()}
    cost_col = "c_man_t" if "c_man_t" in df_reg.columns else "c_man_var"
    c_man = (
        {
            (row["r"], row["t"]): float(row[cost_col])
            for _, row in df_reg.iterrows()
        }
        if cost_col in df_reg.columns
        else None
    )

    rows = []
    for _, reg in df_reg[df_reg["t"].isin(PERIODS)].iterrows():
        r = reg["r"]
        t = reg["t"]
        x_dem = float(reg["x_dem"])
        gross_cs = float(reg["a_dem_used"]) * x_dem - 0.5 * float(reg["b_dem_used"]) * x_dem**2
        net_cs = gross_cs - float(reg["lam"]) * x_dem

        ps = 0.0
        flows_r = df_flows[(df_flows["exp"] == r) & (df_flows["t"] == t)]
        for _, flow in flows_r.iterrows():
            imp = flow["imp"]
            cost = c_man[(r, t)] if c_man is not None else float(flow["c_man"])
            ps += (
                lam.get((imp, t), 0.0)
                - cost
                - float(flow["c_ship"])
            ) * float(flow["x"])

        cap_cost = (
            f_hold.get(r, 0.0) * float(reg["Kcap"])
            + c_inv.get(r, 0.0) * float(reg["Icap_report"])
        )
        welfare = net_cs + ps - cap_cost
        rows.append(
            {
                "model": model,
                "run": run,
                "r": r,
                "t": t,
                "lam": float(reg["lam"]),
                "gross_CS": gross_cs,
                "net_CS": net_cs,
                "PS": ps,
                "CapCost": cap_cost,
                "W": welfare,
                "W_npv": beta_t.get(t, 1.0) * ytn_t.get(t, 5.0) * welfare,
            }
        )
    return pd.DataFrame(rows)


def main():
    data = load_model_data()
    epec_meta = read_meta_dict(EPEC_PATH)
    data.beta_t = parse_mapping(epec_meta.get("beta_t"), data.beta_t or {t: 1.0 for t in PERIODS})
    data.years_to_next = parse_mapping(
        epec_meta.get("ytn"), data.years_to_next or {t: 5.0 for t in PERIODS}
    )

    df_plan = compute_welfare(
        pd.read_excel(PLANNER_PATH, sheet_name="regions"),
        pd.read_excel(PLANNER_PATH, sheet_name="flows"),
        data,
        "planner",
        "planner",
    )
    df_epec = compute_welfare(
        pd.read_excel(EPEC_PATH, sheet_name="regions"),
        pd.read_excel(EPEC_PATH, sheet_name="flows"),
        data,
        "epec",
        SELECTED_RUN,
    )
    df_all = pd.concat([df_plan, df_epec], ignore_index=True)

    df_summary = (
        df_all.groupby(["model", "run", "r"], as_index=False)
        [["gross_CS", "net_CS", "PS", "CapCost", "W_npv"]]
        .sum()
        .rename(columns={"W_npv": "W_npv_total"})
    )

    planner = (
        df_summary[df_summary["model"] == "planner"]
        .set_index("r")
        .rename(columns={"W_npv_total": "W_planner"})
    )
    epec = (
        df_summary[df_summary["model"] == "epec"]
        .set_index("r")
        .rename(columns={"W_npv_total": "W_epec"})
    )
    df_regional = pd.DataFrame(index=REGIONS)
    df_regional["W_planner"] = planner["W_planner"]
    df_regional["W_epec"] = epec["W_epec"]
    df_regional["delta"] = df_regional["W_epec"] - df_regional["W_planner"]
    df_regional["delta_pct"] = 100 * df_regional["delta"] / df_regional["W_planner"]
    df_regional = df_regional.reset_index().rename(columns={"index": "r"})

    df_totals = (
        df_summary.groupby(["model", "run"], as_index=False)["W_npv_total"]
        .sum()
        .rename(columns={"W_npv_total": "W_npv_total_all_regions"})
    )
    planner_total = float(
        df_totals.loc[df_totals["model"] == "planner", "W_npv_total_all_regions"].iloc[0]
    )
    epec_total = float(
        df_totals.loc[df_totals["model"] == "epec", "W_npv_total_all_regions"].iloc[0]
    )
    df_aggregate = pd.DataFrame(
        [
            {
                "planner_W_npv_total": planner_total,
                "epec_W_npv_total": epec_total,
                "delta": epec_total - planner_total,
                "delta_pct": 100 * (epec_total - planner_total) / planner_total,
                "planner_trillion_usd": planner_total / 1e6,
                "epec_trillion_usd": epec_total / 1e6,
                "delta_trillion_usd": (epec_total - planner_total) / 1e6,
            }
        ]
    )

    with pd.ExcelWriter(OUT_PATH) as writer:
        df_all.to_excel(writer, sheet_name="welfare", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_regional.to_excel(writer, sheet_name="regional_comparison", index=False)
        df_totals.to_excel(writer, sheet_name="totals", index=False)
        df_aggregate.to_excel(writer, sheet_name="aggregate_comparison", index=False)

    print(f"Wrote {OUT_PATH}")
    print(df_aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
