"""
model_llp_planner.py — Global-Planner LLP Benchmark (intertemporal)
====================================================================
Welfare-maximising global market-clearing benchmark for PV module trade.

A central planner chooses production, trade, and capacity investment to
maximise aggregate social welfare (consumer surplus + producer surplus minus
costs) over periods 2025–2040, subject to:
  - Demand balance per region and period
  - Physical capacity limits (endogenous Kcap path)
  - Endogenous investment / decommissioning (Icap_pos, Dcap_neg)
  - Investment rate bounds (g_exp_ub, g_dec_ub)
  - Exogenous Swanson's Law cost decline schedule (c_man_t)
  - Holding costs (f_hold) and annualised capital costs (c_inv)

Period 2045 is a terminal buffer: the capacity path is solved through 2045
(to avoid boundary effects on the 2040 investment decision) but excluded
from the welfare objective.

No tariffs, no strategic behaviour, no upper-level logic.
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

try:
    from .model_main import ModelContext, ModelData
    from .data_prep import load_data_from_excel
except ImportError:
    from model_main import ModelContext, ModelData
    from data_prep import load_data_from_excel


_DEFAULT_TIMES = ["2025", "2030", "2035", "2040", "2045"]


# =====================================================================
#  build_llp_planner_model
# =====================================================================
def build_llp_planner_model(
    data: ModelData,
    *,
    working_directory: str | None = None,
    discount_rate: float = 0.02,
    base_year: int = 2025,
) -> ModelContext:
    """Build the intertemporal global-planner LLP benchmark.

    Returns a ModelContext whose model key "planner" can be solved
    directly with ``ctx.models["planner"].solve(...)``.

    The objective sums over the *planning* horizon only (all periods except
    the terminal buffer 2045).  The capacity path is still modelled through
    2045 to avoid end-of-horizon distortions on the 2040 investment decision.
    """
    times: List[str] = data.times or list(_DEFAULT_TIMES)
    regions = data.regions

    # Planning horizon excludes terminal buffer (last period = 2045)
    plan_times: List[str] = times[:-1]   # ["2025","2030","2035","2040"]
    # Investment periods: same as plan_times (there is no 2045→next transition)
    move_times: List[str] = plan_times

    # Transition pairs for capacity evolution (includes 2040→2045)
    transition_pairs: List[Tuple[str, str]] = list(zip(times[:-1], times[1:]))

    # ---- GAMS container ----
    kw: dict = {}
    if working_directory:
        kw["working_directory"] = working_directory
    m = Container(**kw)

    # ---- Sets ----
    R      = Set(m, "R",      records=regions)
    exp    = Set(m, "exp",    domain=R, records=regions)
    imp    = Set(m, "imp",    domain=R, records=regions)
    T      = Set(m, "T",      records=times)
    T_plan = Set(m, "T_plan", domain=[T], records=plan_times)   # objective domain (no 2045)
    T_move = Set(m, "T_move", domain=[T], records=move_times)   # investment domain

    z = gp.Number(0)

    # ---- Helper dicts (mirrors model_main.py fallback logic) ----
    a_dem_t_dict: Dict[Tuple[str, str], float] = dict(data.a_dem_t) if data.a_dem_t else {
        (r, tp): float(data.a_dem.get(r, 0.0)) for r in regions for tp in times
    }
    b_dem_t_dict: Dict[Tuple[str, str], float] = dict(data.b_dem_t) if data.b_dem_t else {
        (r, tp): float(data.b_dem.get(r, 1.0)) for r in regions for tp in times
    }
    dmax_t_dict: Dict[Tuple[str, str], float] = dict(data.Dmax_t) if data.Dmax_t else {
        (r, tp): float(data.Dmax.get(r, 1.0)) for r in regions for tp in times
    }

    kcap_init_dict: Dict[str, float] = dict(data.Kcap_2025) if data.Kcap_2025 else {
        r: float(data.Qcap.get(r, 0.0)) for r in regions
    }

    g_exp_dict: Dict[str, float] = dict(data.g_exp_ub) if data.g_exp_ub else {r: 0.1 for r in regions}
    g_dec_dict: Dict[str, float] = dict(data.g_dec_ub) if data.g_dec_ub else {r: 0.1 for r in regions}
    g_exp_is_abs: bool = bool(getattr(data, "g_exp_ub_is_absolute", False))

    f_hold_dict: Dict[str, float] = dict(data.f_hold) if data.f_hold else {r: 0.0 for r in regions}
    c_inv_dict:  Dict[str, float] = dict(data.c_inv)  if data.c_inv  else {r: 0.0 for r in regions}

    # Swanson's Law time-indexed manufacturing cost; fall back to static c_man
    c_man_t_dict: Dict[Tuple[str, str], float] = dict(data.c_man_t) if data.c_man_t else {
        (r, tp): float(data.c_man.get(r, 0.0)) for r in regions for tp in times
    }

    # Discounting — use RunConfig-style override if non-default, else use data.beta_t
    if data.beta_t:
        beta_t_dict: Dict[str, float] = dict(data.beta_t)
    else:
        r_disc = float(discount_rate)
        beta_t_dict = {
            tp: (1.0 if r_disc == 0.0 else 1.0 / ((1.0 + r_disc) ** (int(tp) - base_year)))
            for tp in times
        }

    ytn_dict: Dict[str, float] = dict(data.years_to_next) if data.years_to_next else {
        tp: 5.0 for tp in times
    }

    # ---- Parameters ----
    a_dem_p = Parameter(m, "a_dem", domain=[R, T],
                        records=[(r, t, a_dem_t_dict[(r, t)]) for r in regions for t in times])
    b_dem_p = Parameter(m, "b_dem", domain=[R, T],
                        records=[(r, t, b_dem_t_dict[(r, t)]) for r in regions for t in times])
    Dmax_p  = Parameter(m, "Dmax",  domain=[R, T],
                        records=[(r, t, dmax_t_dict[(r, t)]) for r in regions for t in times])

    # Swanson's Law cost schedule (region × period)
    c_man_t_p = Parameter(m, "c_man_t", domain=[R, T],
                          records=[(r, t, c_man_t_dict[(r, t)]) for r in regions for t in times])

    c_ship_p = Parameter(m, "c_ship", domain=[exp, imp],
                         records=[(e, i, float(data.c_ship[(e, i)])) for e in regions for i in regions])

    Kcap_init_p = Parameter(m, "Kcap_init", domain=[R],
                            records=[(r, kcap_init_dict[r]) for r in regions])

    g_exp_p = Parameter(m, "g_exp", domain=[R],
                        records=[(r, g_exp_dict[r]) for r in regions])
    g_dec_p = Parameter(m, "g_dec", domain=[R],
                        records=[(r, g_dec_dict[r]) for r in regions])

    f_hold_p = Parameter(m, "f_hold", domain=[R],
                         records=[(r, f_hold_dict[r]) for r in regions])
    c_inv_p  = Parameter(m, "c_inv",  domain=[R],
                         records=[(r, c_inv_dict[r])  for r in regions])

    beta_p = Parameter(m, "beta_t", domain=[T],
                       records=[(t, beta_t_dict[t]) for t in times])
    ytn_p  = Parameter(m, "ytn",   domain=[T],
                       records=[(t, ytn_dict[t])   for t in times])

    # ---- Variables ----
    x      = Variable(m, "x",        domain=[exp, imp, T], type=VariableType.POSITIVE)
    x_dem  = Variable(m, "x_dem",    domain=[R, T],        type=VariableType.POSITIVE)
    Kcap   = Variable(m, "Kcap",     domain=[R, T],        type=VariableType.POSITIVE)
    Icap   = Variable(m, "Icap_pos", domain=[R, T],        type=VariableType.POSITIVE)
    Dcap   = Variable(m, "Dcap_neg", domain=[R, T],        type=VariableType.POSITIVE)

    x_dem.up[R, T] = Dmax_p[R, T]

    # Fix investment/decommissioning to zero in the terminal buffer period (2045)
    Icap.fx[R, times[-1]] = z
    Dcap.fx[R, times[-1]] = z

    # ---- Constraints ----

    # (1) Demand balance
    eq_dem_bal = Equation(m, "eq_dem_bal", domain=[imp, T])
    eq_dem_bal[imp, T] = Sum(exp, x[exp, imp, T]) == x_dem[imp, T]

    # (2) Physical capacity: total exports ≤ Kcap
    eq_cap = Equation(m, "eq_cap", domain=[exp, T])
    eq_cap[exp, T] = Sum(imp, x[exp, imp, T]) <= Kcap[exp, T]

    # (3) Initial capacity
    eq_kcap_init = Equation(m, "eq_kcap_init", domain=[R])
    eq_kcap_init[R] = Kcap[R, times[0]] == Kcap_init_p[R]

    # (4) Capacity transitions (covers all consecutive pairs including 2040→2045)
    eq_kcap_trans: Dict[str, Equation] = {}
    for tp, tp_next in transition_pairs:
        eq = Equation(m, f"eq_kcap_trans_{tp_next}", domain=[R])
        eq[R] = Kcap[R, tp_next] == Kcap[R, tp] + ytn_p[tp] * (Icap[R, tp] - Dcap[R, tp])
        eq_kcap_trans[tp_next] = eq

    # (5) Investment rate upper bounds (only for move_times)
    eq_icap_ub: Dict[str, Equation] = {}
    eq_dcap_ub: Dict[str, Equation] = {}
    for tp in move_times:
        eq_i = Equation(m, f"eq_icap_ub_{tp}", domain=[R])
        if g_exp_is_abs:
            eq_i[R] = Icap[R, tp] <= g_exp_p[R]
        else:
            eq_i[R] = Icap[R, tp] <= g_exp_p[R] * Kcap[R, tp]
        eq_icap_ub[tp] = eq_i

        eq_d = Equation(m, f"eq_dcap_ub_{tp}", domain=[R])
        eq_d[R] = Dcap[R, tp] <= g_dec_p[R] * Kcap[R, tp]
        eq_dcap_ub[tp] = eq_d

    # ---- Objective (planning horizon only: T_plan, excludes 2045) ----
    # Maximise: consumer surplus - manufacturing cost - shipping - holding cost - investment cost
    obj = Sum(
        [T_plan],
        beta_p[T_plan] * ytn_p[T_plan] * (
            # Consumer surplus
            Sum(R, a_dem_p[R, T_plan] * x_dem[R, T_plan]
                   - (b_dem_p[R, T_plan] / gp.Number(2.0))
                   * x_dem[R, T_plan] * x_dem[R, T_plan])
            # Production + shipping cost
            - Sum([exp, imp], (c_man_t_p[exp, T_plan] + c_ship_p[exp, imp]) * x[exp, imp, T_plan])
            # Capacity holding cost
            - Sum(R, f_hold_p[R] * Kcap[R, T_plan])
            # Investment cost (annualised capital charge)
            - Sum(R, c_inv_p[R] * Icap[R, T_plan])
        ),
    )

    # Collect all equations
    all_equations = [eq_dem_bal, eq_cap, eq_kcap_init]
    all_equations += list(eq_kcap_trans.values())
    all_equations += list(eq_icap_ub.values())
    all_equations += list(eq_dcap_ub.values())

    planner = Model(
        m,
        name="planner",
        equations=all_equations,
        problem=Problem.QCP,
        sense=Sense.MAX,
        objective=obj,
    )

    # Print capacity initialisation summary
    print(f"[LLP] Kcap_init by region:")
    for r in regions:
        print(f"  {r}: {kcap_init_dict[r]:.1f} GW  |  c_man_2025={c_man_t_dict.get((r, times[0]), 0):.1f}  c_man_2040={c_man_t_dict.get((r, plan_times[-1]), 0):.1f}")
    print(f"[LLP] Planning horizon: {plan_times}  (terminal buffer {times[-1]} in capacity path but excluded from objective)")

    return ModelContext(
        container=m,
        sets={"R": R, "exp": exp, "imp": imp, "T": T, "T_plan": T_plan, "T_move": T_move},
        params={
            "a_dem":    a_dem_p,
            "b_dem":    b_dem_p,
            "Dmax":     Dmax_p,
            "c_man_t":  c_man_t_p,
            "c_ship":   c_ship_p,
            "Kcap_init": Kcap_init_p,
            "g_exp":    g_exp_p,
            "g_dec":    g_dec_p,
            "f_hold":   f_hold_p,
            "c_inv":    c_inv_p,
            "beta_t":   beta_p,
            "ytn":      ytn_p,
        },
        vars={
            "x":        x,
            "x_dem":    x_dem,
            "Kcap":     Kcap,
            "Icap_pos": Icap,
            "Dcap_neg": Dcap,
        },
        equations={"eq_dem_bal": eq_dem_bal, "eq_cap": eq_cap, **eq_kcap_trans},
        models={"planner": planner},
    )


# =====================================================================
#  solve
# =====================================================================
def solve_llp_planner(
    ctx: ModelContext,
    *,
    solver: str = "ipopt",
    solver_options: Dict[str, float] | None = None,
) -> None:
    """Solve the planner model in-place."""
    kw: dict = {"solver": solver}
    if solver_options:
        kw["solver_options"] = solver_options
    ctx.models["planner"].solve(**kw)
    print(f"[LLP] Solve status: {ctx.models['planner'].status}  obj={ctx.models['planner'].objective_value:.4g}")


# =====================================================================
#  extract state
# =====================================================================
def extract_llp_state(ctx: ModelContext, data: ModelData) -> Dict[str, object]:
    """Pull results from the solved planner model.

    Returns dict with keys:
        x           Dict[(exp,imp,t), float]   trade flows
        x_dem       Dict[(r,t), float]          consumption
        x_man       Dict[(r,t), float]          total production per exporter
        Kcap        Dict[(r,t), float]          installed capacity path
        Icap_pos    Dict[(r,t), float]          investment rate
        Dcap_neg    Dict[(r,t), float]          decommissioning rate
        lam         Dict[(r,t), float]          demand-balance dual (market price)
        mu_cap      Dict[(r,t), float]          physical capacity scarcity rent
        obj_total   float                       total objective value
    """
    m = ctx.container
    times = data.times or list(_DEFAULT_TIMES)

    def _records_to_dict_3(name: str, cols: Tuple[str, str, str]) -> Dict[Tuple[str, str, str], float]:
        rec = m[name].records
        d: Dict[Tuple[str, str, str], float] = {}
        if rec is not None:
            for _, row in rec.iterrows():
                d[(row[cols[0]], row[cols[1]], row[cols[2]])] = float(row["level"])
        return d

    def _records_to_dict_2(name: str, cols: Tuple[str, str], value_col: str = "level") -> Dict[Tuple[str, str], float]:
        rec = m[name].records
        d: Dict[Tuple[str, str], float] = {}
        if rec is not None:
            for _, row in rec.iterrows():
                d[(row[cols[0]], row[cols[1]])] = float(row[value_col])
        return d

    x_dict    = _records_to_dict_3("x",        ("exp", "imp", "T"))
    x_dem_dict = _records_to_dict_2("x_dem",   ("R", "T"))
    kcap_dict  = _records_to_dict_2("Kcap",    ("R", "T"))
    icap_dict  = _records_to_dict_2("Icap_pos", ("R", "T"))
    dcap_dict  = _records_to_dict_2("Dcap_neg", ("R", "T"))

    # Derived: total production per exporter
    x_man_dict: Dict[Tuple[str, str], float] = {
        (e, t): sum(x_dict.get((e, i, t), 0.0) for i in data.regions)
        for e in data.regions for t in times
    }

    # Duals — GAMS marginal includes beta*ytn weighting; divide out
    beta_dict = data.beta_t or {t: 1.0 for t in times}
    ytn_dict  = data.years_to_next or {t: 5.0 for t in times}

    lam_dict: Dict[Tuple[str, str], float] = {}
    rec = m["eq_dem_bal"].records
    if rec is not None:
        for _, row in rec.iterrows():
            t = row["T"]
            w = float(beta_dict.get(t, 1.0)) * float(ytn_dict.get(t, 5.0))
            lam_dict[(row["imp"], t)] = -float(row["marginal"]) / w if w else 0.0

    mu_cap_dict: Dict[Tuple[str, str], float] = {}
    rec = m["eq_cap"].records
    if rec is not None:
        for _, row in rec.iterrows():
            t = row["T"]
            w = float(beta_dict.get(t, 1.0)) * float(ytn_dict.get(t, 5.0))
            mu_cap_dict[(row["exp"], t)] = -float(row["marginal"]) / w if w else 0.0

    obj_total = float(ctx.models["planner"].objective_value)

    return {
        "x":        x_dict,
        "x_dem":    x_dem_dict,
        "x_man":    x_man_dict,
        "Kcap":     kcap_dict,
        "Icap_pos": icap_dict,
        "Dcap_neg": dcap_dict,
        "lam":      lam_dict,
        "mu_cap":   mu_cap_dict,
        "obj_total": obj_total,
    }


# =====================================================================
#  pretty-print
# =====================================================================
def print_llp_summary(state: Dict[str, object], data: ModelData) -> None:
    """Console summary of planner results (planning horizon only)."""
    times     = data.times or list(_DEFAULT_TIMES)
    plan_times = times[:-1]

    x_dict    = state["x"]
    x_dem_d   = state["x_dem"]
    x_man_d   = state["x_man"]
    kcap_d    = state["Kcap"]
    icap_d    = state["Icap_pos"]
    lam_d     = state["lam"]
    mu_cap_d  = state["mu_cap"]

    print(f"\nObjective value (planning horizon): {state['obj_total']:.2f}")
    print(f"\n{'r':4} {'t':4} {'x_dem':>8} {'x_man':>8} {'Kcap':>8} {'Icap':>7} {'lam':>8} {'mu_cap':>8} {'c_man_t':>8}")
    print("-" * 78)

    c_man_t_dict = data.c_man_t or {(r, tp): float(data.c_man.get(r, 0.0)) for r in data.regions for tp in times}

    for r in data.regions:
        for t in plan_times:
            print(
                f"{r:4} {t:4}"
                f" {x_dem_d.get((r, t), 0.0):8.1f}"
                f" {x_man_d.get((r, t), 0.0):8.1f}"
                f" {kcap_d.get((r, t), 0.0):8.1f}"
                f" {icap_d.get((r, t), 0.0):7.2f}"
                f" {lam_d.get((r, t), 0.0):8.2f}"
                f" {mu_cap_d.get((r, t), 0.0):8.2f}"
                f" {c_man_t_dict.get((r, t), 0.0):8.2f}"
            )

    print(f"\nTrade matrices (planning horizon):")
    header = f"{'':6}" + "".join(f"{i:>8}" for i in data.regions)
    for t in plan_times:
        print(f"\n  t={t}:")
        print(f"  {header}")
        for e in data.regions:
            vals = "".join(f"{x_dict.get((e, i, t), 0.0):8.1f}" for i in data.regions)
            print(f"  {e:6}{vals}")


# =====================================================================
#  validation
# =====================================================================
def validate_llp_solution(state: Dict[str, object], data: ModelData) -> List[str]:
    """Basic feasibility checks. Returns list of warning strings."""
    times     = data.times or list(_DEFAULT_TIMES)
    plan_times = times[:-1]
    msgs: List[str] = []

    x_dict   = state["x"]
    x_dem_d  = state["x_dem"]
    x_man_d  = state["x_man"]
    kcap_d   = state["Kcap"]

    for t in plan_times:
        for i in data.regions:
            total_in = sum(x_dict.get((e, i, t), 0.0) for e in data.regions)
            dem      = x_dem_d.get((i, t), 0.0)
            if abs(total_in - dem) > 1e-3:
                msgs.append(f"[CHECK 1] dem-balance resid {abs(total_in-dem):.4e} {i},{t}")

            dmax = float(data.Dmax_t[(i, t)]) if data.Dmax_t else float(data.Dmax.get(i, 0.0))
            if dem > dmax + 1e-3:
                msgs.append(f"[CHECK 2] x_dem={dem:.2f} > Dmax={dmax:.2f} for {i},{t}")

        for e in data.regions:
            prod = x_man_d.get((e, t), 0.0)
            cap  = kcap_d.get((e, t), 0.0)
            if prod > cap + 1e-3:
                msgs.append(f"[CHECK 3] prod={prod:.2f} > Kcap={cap:.2f} for {e},{t}")

        for i in data.regions:
            lam_val = state["lam"].get((i, t), 0.0)
            if lam_val < -1e-3:
                msgs.append(f"[CHECK 4] negative lambda={lam_val:.4f} for {i},{t}")

    return msgs


# =====================================================================
#  CLI entry point
# =====================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.normpath(
        os.path.join(script_dir, "..", "inputs", "input_data_intertemporal.xlsx")
    )

    print("=== Loading data (params_region_new) ===")
    data = load_data_from_excel(input_path, params_region_sheet="params_region_new")

    print("=== Building LLP Planner ===")
    ctx = build_llp_planner_model(data)

    print("=== Solving ===")
    solve_llp_planner(ctx, solver="ipopt")

    state = extract_llp_state(ctx, data)
    print_llp_summary(state, data)

    print("\n=== Validation ===")
    warnings = validate_llp_solution(state, data)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("All checks passed.")

    import pandas as pd
    out_dir = os.path.normpath(os.path.join(script_dir, "..", "outputs"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llp_planner_results.xlsx")

    times_list = data.times or list(_DEFAULT_TIMES)
    plan_times = times_list[:-1]
    c_man_t_dict   = data.c_man_t or {(r, t): float(data.c_man.get(r, 0.0)) for r in data.regions for t in times_list}
    c_ship_dict    = data.c_ship or {}
    kcap_init_dict = data.Kcap_2025 or data.Qcap or {}
    ytn_d          = dict(data.years_to_next) if data.years_to_next else {t: 5.0 for t in times_list}
    beta_d         = dict(data.beta_t) if data.beta_t else {t: 1.0 for t in times_list}
    a_dem_t_dict   = dict(data.a_dem_t) if data.a_dem_t else {(r, t): float(data.a_dem.get(r, 0.0)) for r in data.regions for t in times_list}
    b_dem_t_dict   = dict(data.b_dem_t) if data.b_dem_t else {(r, t): float(data.b_dem.get(r, 1.0)) for r in data.regions for t in times_list}
    f_hold_dict    = dict(data.f_hold) if data.f_hold else {r: 0.0 for r in data.regions}
    c_inv_dict     = dict(data.c_inv)  if data.c_inv  else {r: 0.0 for r in data.regions}

    try:
        with pd.ExcelWriter(out_path) as writer:
            # --- regions sheet (mirrors EPEC format) ---
            rows_rp = []
            for r in data.regions:
                for t in plan_times:
                    icap = state["Icap_pos"].get((r, t), 0.0)
                    dcap = state["Dcap_neg"].get((r, t), 0.0)
                    ytn  = ytn_d.get(t, 5.0)
                    exports = sum(state["x"].get((r, i, t), 0.0) for i in data.regions)
                    imports = sum(state["x"].get((e, r, t), 0.0) for e in data.regions)
                    kcap    = state["Kcap"].get((r, t), 0.0)
                    x_dem_val = state["x_dem"].get((r, t), 0.0)
                    beta = beta_d.get(t, 1.0)
                    a    = a_dem_t_dict.get((r, t), 0.0)
                    b    = b_dem_t_dict.get((r, t), 0.0)
                    cs   = a * x_dem_val - 0.5 * b * x_dem_val ** 2
                    prod_cost  = sum((c_man_t_dict.get((r, t), 0.0) + float(c_ship_dict.get((r, i), 0.0)))
                                     * state["x"].get((r, i, t), 0.0) for i in data.regions)
                    hold_cost  = f_hold_dict.get(r, 0.0) * kcap
                    inv_cost   = c_inv_dict.get(r, 0.0) * icap
                    obj_r_t    = beta * ytn * (cs - prod_cost - hold_cost - inv_cost)
                    rows_rp.append({
                        "r":              r,
                        "t":              t,
                        "Kcap":           kcap,
                        "net_cap_change": ytn * (icap - dcap),
                        "Icap_report":    icap,
                        "Dcap_report":    dcap,
                        "x_dem":          x_dem_val,
                        "x_man":          state["x_man"].get((r, t), 0.0),
                        "lam":            state["lam"].get((r, t), 0.0),
                        "mu_cap":         state["mu_cap"].get((r, t), 0.0),
                        "imports":        imports,
                        "exports":        exports,
                        "Kcap_init":      kcap_init_dict.get(r, 0.0),
                        "c_man_t":        c_man_t_dict.get((r, t), 0.0),
                        "cs":             cs,
                        "prod_cost":      prod_cost,
                        "hold_cost":      hold_cost,
                        "inv_cost":       inv_cost,
                        "obj":            obj_r_t,
                    })
            pd.DataFrame(rows_rp).to_excel(writer, sheet_name="regions", index=False)

            # --- flows sheet (mirrors EPEC format) ---
            rows_x = []
            for e in data.regions:
                for i in data.regions:
                    for t in plan_times:
                        flow = state["x"].get((e, i, t), 0.0)
                        if flow > 0.001:
                            rows_x.append({
                                "exp":    e,
                                "imp":    i,
                                "t":      t,
                                "x":      flow,
                                "c_ship": c_ship_dict.get((e, i), 0.0),
                                "c_man":  c_man_t_dict.get((e, t), 0.0),
                            })
            pd.DataFrame(rows_x).to_excel(writer, sheet_name="flows", index=False)

            # --- meta sheet ---
            pd.DataFrame([
                {"key": "model",           "value": "llp_planner"},
                {"key": "obj_total",       "value": round(state["obj_total"], 2)},
                {"key": "plan_times",      "value": str(plan_times)},
                {"key": "excel_path",      "value": input_path},
                {"key": "params_sheet",    "value": "params_region_new"},
            ]).to_excel(writer, sheet_name="meta", index=False)

        print(f"\nResults saved to {out_path}")
    except PermissionError:
        print(f"\n[WARN] Could not write {out_path} — file may be open in Excel.")
