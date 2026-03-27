"""
model_llp_planner.py — Global-Planner LLP Benchmark
====================================================
Welfare-maximizing global market-clearing benchmark for PV module trade.

A central planner allocates production and trade to minimise total cost
minus aggregate consumer utility, subject to capacity limits and demand
bounds.  No tariffs, no strategic behaviour, no upper-level logic.

The model is one integrated NLP over all regions and periods.  Each
period is separable (no intertemporal coupling).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gamspy as gp
from gamspy import (
    Container,
    Equation,
    Model,
    Parameter,
    Problem,
    Sense,
    Set,
    Sum,
    Variable,
    VariableType,
)

# Re-use existing data structures
try:
    from .model_main import ModelContext, ModelData
except ImportError:
    from model_main import ModelContext, ModelData


# =====================================================================
# Default time periods
# =====================================================================
_DEFAULT_TIMES = ("2025", "2030", "2035", "2040", "2045", "2050", "2055")


# =====================================================================
#  build_llp_planner_model
# =====================================================================
def build_llp_planner_model(
    data: ModelData,
    *,
    working_directory: str | None = None,
) -> ModelContext:
    """Build the global-planner LLP benchmark model.

    Returns a ModelContext whose model key "planner" can be solved
    directly with ``ctx.models["planner"].solve(...)``.
    """
    times: List[str] = data.times or list(_DEFAULT_TIMES)
    regions = data.regions

    # ---- Container ----
    kw = {}
    if working_directory:
        kw["working_directory"] = working_directory
    m = Container(**kw)

    # ---- Sets ----
    R   = Set(m, "R",   records=regions)
    exp = Set(m, "exp", domain=R, records=regions)
    imp = Set(m, "imp", domain=R, records=regions)
    T   = Set(m, "T",   records=times)

    z = gp.Number(0)

    # ---- Parameters ----
    # a_dem, b_dem, Dmax  (region, time)
    a_dem_recs = [(r, t, float(data.a_dem_t[(r, t)])) for r in regions for t in times]
    b_dem_recs = [(r, t, float(data.b_dem_t[(r, t)])) for r in regions for t in times]
    Dmax_recs  = [(r, t, float(data.Dmax_t[(r, t)]))  for r in regions for t in times]

    a_dem_p = Parameter(m, "a_dem", domain=[R, T], records=a_dem_recs)
    b_dem_p = Parameter(m, "b_dem", domain=[R, T], records=b_dem_recs)
    Dmax_p  = Parameter(m, "Dmax",  domain=[R, T], records=Dmax_recs)

    # c_man  (region) — time-invariant in current data
    c_man_recs = [(r, float(data.c_man[r])) for r in regions]
    c_man_p = Parameter(m, "c_man", domain=[R], records=c_man_recs)

    # c_ship  (exp, imp) — time-invariant
    c_ship_recs = [(e, i, float(data.c_ship[(e, i)])) for e in regions for i in regions]
    c_ship_p = Parameter(m, "c_ship", domain=[exp, imp], records=c_ship_recs)

    # Kcap  (region, time)
    # Use Qcap (full installed capacity) as the primary source.
    # Kcap_2025 is often a placeholder; Qcap has realistic data.
    kcap_dict = {r: float(data.Qcap.get(r, 0.0)) for r in regions}
    # Fall back to Kcap_2025 only if Qcap is missing/zero
    if all(v == 0.0 for v in kcap_dict.values()) and data.Kcap_2025:
        kcap_dict = {r: float(data.Kcap_2025[r]) for r in regions}
    Kcap_recs = [(r, t, float(kcap_dict[r])) for r in regions for t in times]
    Kcap_p = Parameter(m, "Kcap", domain=[R, T], records=Kcap_recs)

    print(f"[LLP] Kcap by region: {kcap_dict}")
    print(f"[LLP] Total Kcap = {sum(kcap_dict.values()):.1f}  |  Total Dmax(2025) = {sum(float(data.Dmax_t.get((r, times[0]), 0)) for r in regions):.1f}")

    # Discount & period length  (purely cosmetic weighting here)
    beta_dict = data.beta_t or {t: 1.0 for t in times}
    ytn_dict  = data.years_to_next or {"2025": 5.0, "2030": 5.0, "2035": 5.0, "2040": 5.0, "2045": 5.0, "2050": 5.0, "2055": 5.0}
    beta_p = Parameter(m, "beta", domain=[T], records=[(t, float(beta_dict[t])) for t in times])
    ytn_p  = Parameter(m, "ytn",  domain=[T], records=[(t, float(ytn_dict[t]))  for t in times])

    # ---- Variables ----
    x     = Variable(m, "x",     domain=[exp, imp, T], type=VariableType.POSITIVE)
    x_dem = Variable(m, "x_dem", domain=[R, T],        type=VariableType.POSITIVE)

    # Demand ceiling
    x_dem.up[R, T] = Dmax_p[R, T]

    # ---- Constraints ----
    # (1)  Demand balance:  sum_e x[e,i,t] = x_dem[i,t]
    eq_dem_bal = Equation(m, "eq_dem_bal", domain=[imp, T])
    eq_dem_bal[imp, T] = Sum(exp, x[exp, imp, T]) - x_dem[imp, T] == z

    # (2)  Exporter capacity:  sum_i x[e,i,t] <= Kcap[e,t]
    eq_cap = Equation(m, "eq_cap", domain=[exp, T])
    eq_cap[exp, T] = Sum(imp, x[exp, imp, T]) - Kcap_p[exp, T] <= z

    # ---- Objective ----
    # min  sum_t beta_t * y_t * [ sum_{e,i} (c_man_e + c_ship_{e,i}) x_{ei}
    #                              - sum_i ( a_i x_dem_i - 0.5 b_i x_dem_i^2 ) ]
    obj = Sum(
        [T],
        beta_p[T] * ytn_p[T] * (
            # Production + shipping cost
            Sum([exp, imp], (c_man_p[exp] + c_ship_p[exp, imp]) * x[exp, imp, T])
            # Minus consumer utility
            - Sum(R, a_dem_p[R, T] * x_dem[R, T]
                     - (b_dem_p[R, T] / gp.Number(2.0)) * x_dem[R, T] * x_dem[R, T])
        ),
    )

    planner = Model(
        m,
        name="planner",
        equations=[eq_dem_bal, eq_cap],
        problem=Problem.QCP,
        sense=Sense.MIN,
        objective=obj,
    )

    return ModelContext(
        container=m,
        sets={"R": R, "exp": exp, "imp": imp, "T": T},
        params={
            "a_dem": a_dem_p,
            "b_dem": b_dem_p,
            "Dmax":  Dmax_p,
            "c_man": c_man_p,
            "c_ship": c_ship_p,
            "Kcap":  Kcap_p,
            "beta":  beta_p,
            "ytn":   ytn_p,
        },
        vars={
            "x":     x,
            "x_dem": x_dem,
        },
        equations={
            "eq_dem_bal": eq_dem_bal,
            "eq_cap":     eq_cap,
        },
        models={"planner": planner},
    )


# =====================================================================
#  solve
# =====================================================================
def solve_llp_planner(
    ctx: ModelContext,
    *,
    solver: str = "conopt",
    solver_options: Dict[str, float] | None = None,
) -> None:
    """Solve the planner model in-place."""
    kw: dict = {"solver": solver}
    if solver_options:
        kw["solver_options"] = solver_options
    ctx.models["planner"].solve(**kw)


# =====================================================================
#  extract state
# =====================================================================
def extract_llp_state(ctx: ModelContext, data: ModelData) -> Dict[str, object]:
    """Pull results from the solved planner model.

    Returns dict with keys:
        x           Dict[(exp,imp,t), float]  trade flows
        x_dem       Dict[(r,t), float]         consumption
        x_man       Dict[(r,t), float]         total production per exporter
        lam         Dict[(r,t), float]         demand-balance dual (market price)
        mu_cap      Dict[(r,t), float]         physical capacity scarcity rent
                                               (dual of sum_imp x[exp,imp,t] <= Kcap[exp,t])
        obj_total   float                      total objective
    """
    m = ctx.container

    x_rec     = m["x"].records
    x_dem_rec = m["x_dem"].records
    eq_dem_rec = m["eq_dem_bal"].records
    eq_cap_rec = m["eq_cap"].records

    # --- Trade flows ---
    x_dict: Dict[Tuple[str, str, str], float] = {}
    if x_rec is not None:
        for _, row in x_rec.iterrows():
            x_dict[(row["exp"], row["imp"], row["T"])] = float(row["level"])

    # --- Demand ---
    x_dem_dict: Dict[Tuple[str, str], float] = {}
    if x_dem_rec is not None:
        for _, row in x_dem_rec.iterrows():
            x_dem_dict[(row["R"], row["T"])] = float(row["level"])

    # --- Production by exporter (derived) ---
    x_man_dict: Dict[Tuple[str, str], float] = {}
    times = data.times or list(_DEFAULT_TIMES)
    for e in data.regions:
        for t in times:
            x_man_dict[(e, t)] = sum(
                x_dict.get((e, i, t), 0.0) for i in data.regions
            )

    # --- Duals: lambda (market clearing price) ---
    # Raw GAMS marginal includes beta*ytn weighting; divide out.
    beta_dict = data.beta_t or {t: 1.0 for t in times}
    ytn_dict  = data.years_to_next or {"2025": 5.0, "2030": 5.0, "2035": 5.0, "2040": 5.0, "2045": 5.0, "2050": 5.0, "2055": 5.0}

    lam_dict: Dict[Tuple[str, str], float] = {}
    if eq_dem_rec is not None:
        for _, row in eq_dem_rec.iterrows():
            t = row["T"]
            weight = float(beta_dict.get(t, 1.0)) * float(ytn_dict.get(t, 5.0))
            lam_dict[(row["imp"], t)] = float(row["marginal"]) / weight

    # --- Duals: mu_cap (physical capacity scarcity rent) ---
    mu_cap_dict: Dict[Tuple[str, str], float] = {}
    if eq_cap_rec is not None:
        for _, row in eq_cap_rec.iterrows():
            t = row["T"]
            weight = float(beta_dict.get(t, 1.0)) * float(ytn_dict.get(t, 5.0))
            mu_cap_dict[(row["exp"], t)] = -float(row["marginal"]) / weight

    obj_total = float(ctx.models["planner"].objective_value)

    return {
        "x":       x_dict,
        "x_dem":   x_dem_dict,
        "x_man":   x_man_dict,
        "lam":     lam_dict,
        "mu_cap":  mu_cap_dict,
        "obj_total": obj_total,
    }


# =====================================================================
#  warm-start mapping
# =====================================================================
def build_epec_warmstart(
    state: Dict[str, object],
    data: ModelData,
) -> Dict[str, Dict]:
    """Map planner solution into an initial-state dict for the EPEC GS solver."""
    times = data.times or list(_DEFAULT_TIMES)

    x_man_d  = state["x_man"]
    lam_d    = state["lam"]

    Q_offer: Dict[Tuple[str, str], float] = {}
    for r in data.players:
        for t in times:
            Q_offer[(r, t)] = x_man_d.get((r, t), 0.0)

    p_offer: Dict[Tuple[str, str, str], float] = {}
    for e in data.regions:
        for i in data.regions:
            for t in times:
                lam_val = lam_d.get((e, t), 0.0)
                p_offer[(e, i, t)] = max(0.0, lam_val)

    # a_bid  ← a_dem (no demand withholding in warm start)
    a_bid: Dict[Tuple[str, str], float] = {}
    for r in data.regions:
        for t in times:
            a_bid[(r, t)] = float(
                data.a_dem_t[(r, t)] if data.a_dem_t else data.a_dem.get(r, 0.0)
            )

    return {
        "Q_offer": Q_offer,
        "p_offer": p_offer,
        "a_bid":   a_bid,
    }


# =====================================================================
#  validation checks
# =====================================================================
def validate_llp_solution(state: Dict[str, object], data: ModelData) -> List[str]:
    """Run the 6 validation checks from the spec.  Returns list of warnings."""
    times = data.times or list(_DEFAULT_TIMES)
    msgs: List[str] = []

    x_dict  = state["x"]
    x_dem_d = state["x_dem"]
    x_man_d = state["x_man"]
    lam_d   = state["lam"]

    kcap = data.Kcap_2025 if data.Kcap_2025 is not None else {
        r: float(data.Qcap.get(r, 0.0)) for r in data.regions
    }

    for i in data.regions:
        for t in times:
            # (1) demand balance residual
            total_in = sum(x_dict.get((e, i, t), 0.0) for e in data.regions)
            dem      = x_dem_d.get((i, t), 0.0)
            resid    = abs(total_in - dem)
            if resid > 1e-3:
                msgs.append(f"[CHECK 1] dem-balance resid {resid:.4e} for {i},{t}")

            # (3) x_dem <= Dmax
            dmax = float(data.Dmax_t[(i, t)])
            if dem > dmax + 1e-3:
                msgs.append(f"[CHECK 3] x_dem={dem:.2f} > Dmax={dmax:.2f} for {i},{t}")

    for e in data.regions:
        for t in times:
            # (2) no exporter exceeds Kcap
            prod = x_man_d.get((e, t), 0.0)
            cap  = float(kcap[e])
            if prod > cap + 1e-3:
                msgs.append(f"[CHECK 2] prod={prod:.2f} > Kcap={cap:.2f} for {e},{t}")

    # (4) lambda plausibility
    for i in data.regions:
        for t in times:
            lam_val = lam_d.get((i, t), 0.0)
            if lam_val < -1e-3:
                msgs.append(f"[CHECK 4] negative lambda={lam_val:.4f} for {i},{t}")

    # (5/6) cheapest exporter serves each market
    for i in data.regions:
        for t in times:
            delivered = []
            for e in data.regions:
                cost = float(data.c_man[e]) + float(data.c_ship[(e, i)])
                flow = x_dict.get((e, i, t), 0.0)
                delivered.append((cost, flow, e))
            delivered.sort(key=lambda x: x[0])
            # cheapest with flow > 0 should come first
            active = [(c, f, e) for c, f, e in delivered if f > 1e-3]
            if active:
                cheapest_active_cost = active[0][0]
                for c, f, e in active:
                    if c > cheapest_active_cost + 50:  # allow some slack
                        msgs.append(
                            f"[CHECK 5] {e}->{i} t={t}: active at cost={c:.1f} but cheapest={cheapest_active_cost:.1f}"
                        )

    return msgs


# =====================================================================
#  pretty-print
# =====================================================================
def print_llp_summary(state: Dict[str, object], data: ModelData) -> None:
    """Console summary of planner results."""
    times = data.times or list(_DEFAULT_TIMES)
    x_dict  = state["x"]
    x_dem_d = state["x_dem"]
    x_man_d = state["x_man"]
    lam_d      = state["lam"]
    mu_cap_d   = state["mu_cap"]

    kcap = data.Kcap_2025 if data.Kcap_2025 is not None else {
        r: float(data.Qcap.get(r, 0.0)) for r in data.regions
    }

    print(f"\nObjective value: {state['obj_total']:.2f}")
    print(f"\n{'':4} {'t':4} {'x_dem':>8} {'x_man':>8} {'Kcap':>8} {'lam':>8} {'mu_cap':>8} {'c_man':>8}")
    print("-" * 64)
    for r in data.regions:
        for t in times:
            dem    = x_dem_d.get((r, t), 0.0)
            man    = x_man_d.get((r, t), 0.0)
            cap    = float(kcap[r])
            lam    = lam_d.get((r, t), 0.0)
            mu_cap = mu_cap_d.get((r, t), 0.0)
            cm     = float(data.c_man[r])
            print(f"{r:4} {t:4} {dem:8.1f} {man:8.1f} {cap:8.1f} {lam:8.2f} {mu_cap:8.2f} {cm:8.2f}")

    # Trade flow matrix for first period
    t0 = times[0]
    print(f"\nTrade matrix t={t0}:")
    header = f"{'':6}" + "".join(f"{i:>8}" for i in data.regions)
    print(header)
    for e in data.regions:
        vals = "".join(f"{x_dict.get((e, i, t0), 0.0):8.1f}" for i in data.regions)
        print(f"{e:6}{vals}")


# =====================================================================
#  CLI entry point
# =====================================================================
if __name__ == "__main__":
    try:
        from . import data_prep
    except ImportError:
        import data_prep

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(
        os.path.join(script_dir, "..", "inputs", "input_data_intertemporal.xlsx")
    )
    data = data_prep.load_data_from_excel(input_path)

    print("=== Building LLP Planner ===")
    ctx = build_llp_planner_model(data)

    print("=== Solving ===")
    solve_llp_planner(ctx, solver="conopt")
    print(f"Solve status: {ctx.models['planner'].status}")

    state = extract_llp_state(ctx, data)
    print_llp_summary(state, data)

    print("\n=== Validation ===")
    warnings = validate_llp_solution(state, data)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("All checks passed.")

    # Generate warm-start dict
    ws = build_epec_warmstart(state, data)
    print(f"\nWarm-start keys: {list(ws.keys())}")
    print(f"Q_offer entries: {len(ws['Q_offer'])}")
    print(f"p_offer entries: {len(ws['p_offer'])}")

    # Save to Excel
    import pandas as pd
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llp_planner_results.xlsx")

    times_list = data.times or list(_DEFAULT_TIMES)
    try:
        with pd.ExcelWriter(out_path) as writer:
            # Trade flows
            rows_x = [
                {"exp": e, "imp": i, "t": t, "flow": state["x"].get((e, i, t), 0.0)}
                for e in data.regions for i in data.regions for t in times_list
            ]
            df_x = pd.DataFrame(rows_x)
            df_x = df_x[df_x["flow"] > 0.001]
            df_x.to_excel(writer, sheet_name="Trade_Flows", index=False)

            # Demand & production
            rows_rp = []
            for r in data.regions:
                for t in times_list:
                    rows_rp.append({
                        "region": r, "t": t,
                        "x_dem":  state["x_dem"].get((r, t), 0.0),
                        "x_man":  state["x_man"].get((r, t), 0.0),
                        "lam":    state["lam"].get((r, t), 0.0),
                        "mu_cap": state["mu_cap"].get((r, t), 0.0),
                        "Dmax":  float(data.Dmax_t[(r, t)]),
                        "Kcap":  float((data.Kcap_2025 or data.Qcap)[r]),
                        "c_man": float(data.c_man[r]),
                    })
            pd.DataFrame(rows_rp).to_excel(writer, sheet_name="Regional_Summary", index=False)

        print(f"\nResults saved to {out_path}")
    except PermissionError:
        print(f"\n[WARN] Could not write {out_path} — file may be open in Excel.")
