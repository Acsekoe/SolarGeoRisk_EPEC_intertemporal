"""
Intertemporal (4-period) perfect-foresight EPEC model using Offer-Based Uniform-Price Settlement.

Each strategic player maximises the discounted sum of welfare across
T = {"2025", "2030", "2035", "2040"}, subject to per-period LLP KKT
conditions, dynamic capacity transitions, and offer price decisions.
"""
from __future__ import annotations

INTERTEMPORAL_IMPLEMENTED = True

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set as PySet, Tuple

import gamspy as gp
from gamspy import (
    Alias,
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

z = gp.Number(0)

_DEFAULT_TIMES = ["2025", "2030", "2035", "2040"]
_DEFAULT_YTN = {"2025": 5.0, "2030": 5.0, "2035": 5.0, "2040": 5.0}


# =============================================================================
# Step 1 — ModelData
# =============================================================================
@dataclass
class ModelData:
    regions: List[str]
    players: List[str]
    non_strategic: PySet[str]

    D: Dict[str, float]

    a_dem: Dict[str, float]
    b_dem: Dict[str, float]
    Dmax: Dict[str, float]

    Qcap: Dict[str, float]
    c_man: Dict[str, float]
    c_ship: Dict[Tuple[str, str], float]

    p_offer_ub: Dict[Tuple[str, str], float]

    rho_p: Dict[str, float]

    eps_x: float
    eps_comp: float

    kappa_Q: Dict[str, float] | None = None
    settings: Dict[str, object] | None = None

    # --- Intertemporal extensions ---
    times: List[str] | None = None

    # Time-indexed demand (region, year) → value
    a_dem_t: Dict[Tuple[str, str], float] | None = None
    b_dem_t: Dict[Tuple[str, str], float] | None = None
    Dmax_t: Dict[Tuple[str, str], float] | None = None

    # Capacity
    Kcap_2025: Dict[str, float] | None = None  # fallback to Qcap

    # Max capacity expansion/decommission rates (g_exp, g_dec)
    g_exp_ub: Dict[str, float] | None = None
    g_dec_ub: Dict[str, float] | None = None

    # Capacity costs
    f_hold: Dict[str, float] | None = None  # holding cost
    c_inv: Dict[str, float] | None = None   # investment cost

    # Discounting
    beta_t: Dict[str, float] | None = None  # discount per period (default 1)
    years_to_next: Dict[str, float] | None = None  # interval lengths


@dataclass
class ModelContext:
    container: Container
    sets: Dict[str, Set]
    params: Dict[str, Parameter]
    vars: Dict[str, Variable]
    equations: Dict[str, Equation]
    models: Dict[str, Model]


# =============================================================================
# Sanity checks
# =============================================================================
def _sanity_check_data(data: ModelData) -> None:
    times = data.times or _DEFAULT_TIMES

    # -- per-period demand checks --
    if data.Dmax_t is not None:
        bad = sorted(
            [k for k in data.Dmax_t if float(data.Dmax_t[k]) <= 0.0]
        )
        if bad:
            raise ValueError(f"All Dmax_t must be > 0. Invalid keys: {bad}")
    else:
        bad_dmax = sorted([r for r in data.regions if float(data.Dmax[r]) <= 0.0])
        if bad_dmax:
            raise ValueError(f"All Dmax must be > 0. Invalid regions: {bad_dmax}")

    if data.b_dem_t is not None:
        bad = sorted(
            [k for k in data.b_dem_t if float(data.b_dem_t[k]) <= 0.0]
        )
        if bad:
            raise ValueError(f"All b_dem_t must be > 0. Invalid keys: {bad}")
    else:
        bad_b = sorted([r for r in data.regions if float(data.b_dem.get(r, 1.0)) <= 0.0])
        if bad_b:
            raise ValueError(f"All b_dem must be > 0. Invalid regions: {bad_b}")

    # -- capacity init --
    kcap_init = data.Kcap_2025 or data.Qcap
    bad_k = sorted([r for r in data.regions if float(kcap_init.get(r, 0.0)) < 0.0])
    if bad_k:
        raise ValueError(f"All Kcap_2025 must be >= 0. Invalid regions: {bad_k}")

    # -- rates bounds --
    g_exp_map = data.g_exp_ub or {}
    bad_gexp = sorted([r for r in g_exp_map if float(g_exp_map[r]) < 0.0])
    if bad_gexp:
        raise ValueError(f"All g_exp_ub must be >= 0. Invalid regions: {bad_gexp}")

    g_dec_map = data.g_dec_ub or {}
    bad_gdec = sorted([r for r in g_dec_map if float(g_dec_map[r]) < 0.0])
    if bad_gdec:
        raise ValueError(f"All g_dec_ub must be >= 0. Invalid regions: {bad_gdec}")

    # -- cost params --
    f_hold_map = data.f_hold or {}
    bad_fh = sorted([r for r in f_hold_map if float(f_hold_map[r]) < 0.0])
    if bad_fh:
        raise ValueError(f"All f_hold must be >= 0. Invalid regions: {bad_fh}")

    c_inv_map = data.c_inv or {}
    bad_ci = sorted([r for r in c_inv_map if float(c_inv_map[r]) < 0.0])
    if bad_ci:
        raise ValueError(f"All c_inv must be >= 0. Invalid regions: {bad_ci}")

    kappa_map = getattr(data, "kappa_Q", None) or {}
    bad_kappa = sorted([r for r in kappa_map if float(kappa_map[r]) < 0.0])
    if bad_kappa:
        raise ValueError(f"All kappa_Q must be >= 0. Invalid regions: {bad_kappa}")


# =============================================================================
# Step 2–8: build_model
# =============================================================================
def build_model(data: ModelData, working_directory: str | None = None) -> ModelContext:
    if working_directory and " " in str(working_directory):
        raise ValueError(
            f"GAMS working directory must be space-free. Got: {working_directory}"
        )

    _sanity_check_data(data)

    unknown_players = sorted(set(data.players) - set(data.regions))
    if unknown_players:
        raise ValueError(
            f"All players must be in regions. Unknown players: {unknown_players}"
        )

    settings = data.settings or {}
    use_quad = bool(settings.get("use_quad", False))

    # =====================================================================
    # Step 2 — Container, sets, time set
    # =====================================================================
    m = Container(working_directory=working_directory, debugging_level="keep")

    R = Set(m, "R", records=data.regions)
    exp = Alias(m, "exp", R)
    imp = Alias(m, "imp", R)
    j = Alias(m, "j", R)

    times = data.times or list(_DEFAULT_TIMES)
    T = Set(m, "T", records=times)
    # Note: do NOT create Alias(m, "t", T) — 't' is a GAMS built-in symbol.

    # =====================================================================
    # Step 8 — Compatibility fallback (before building params)
    # =====================================================================
    # Demand
    a_dem_t_dict: Dict[Tuple[str, str], float] = {}
    b_dem_t_dict: Dict[Tuple[str, str], float] = {}
    dmax_t_dict: Dict[Tuple[str, str], float] = {}

    if data.a_dem_t is not None:
        a_dem_t_dict = dict(data.a_dem_t)
    else:
        for r in data.regions:
            for tp in times:
                a_dem_t_dict[(r, tp)] = float(data.a_dem.get(r, 0.0))

    if data.b_dem_t is not None:
        b_dem_t_dict = dict(data.b_dem_t)
    else:
        for r in data.regions:
            for tp in times:
                b_dem_t_dict[(r, tp)] = float(data.b_dem.get(r, 1.0))

    if data.Dmax_t is not None:
        dmax_t_dict = dict(data.Dmax_t)
    else:
        for r in data.regions:
            for tp in times:
                dmax_t_dict[(r, tp)] = float(data.Dmax.get(r, 1.0))

    # Capacity init
    kcap_2025_dict: Dict[str, float] = {}
    if data.Kcap_2025 is not None:
        kcap_2025_dict = dict(data.Kcap_2025)
    else:
        kcap_2025_dict = {r: float(data.Qcap.get(r, 0.0)) for r in data.regions}

    # Growth bounds
    g_exp_dict: Dict[str, float] = {}
    if data.g_exp_ub is not None:
        g_exp_dict = dict(data.g_exp_ub)
    else:
        g_exp_dict = {r: 0.1 for r in data.regions} # fallback value

    g_dec_dict: Dict[str, float] = {}
    if data.g_dec_ub is not None:
        g_dec_dict = dict(data.g_dec_ub)
    else:
        g_dec_dict = {r: 0.1 for r in data.regions} # fallback value

    # Capacity costs
    f_hold_dict: Dict[str, float] = {}
    if data.f_hold is not None:
        f_hold_dict = dict(data.f_hold)
    else:
        f_hold_dict = {r: 0.0 for r in data.regions}

    c_inv_dict: Dict[str, float] = {}
    if data.c_inv is not None:
        c_inv_dict = dict(data.c_inv)
    else:
        c_inv_dict = {r: 0.0 for r in data.regions}

    # Discount and interval
    beta_t_dict: Dict[str, float] = {}
    if data.beta_t is not None:
        beta_t_dict = dict(data.beta_t)
    else:
        beta_t_dict = {tp: 1.0 for tp in times}

    ytn_dict: Dict[str, float] = {}
    if data.years_to_next is not None:
        ytn_dict = dict(data.years_to_next)
    else:
        ytn_dict = dict(_DEFAULT_YTN)

    # =====================================================================
    # Step 3 — Parameters
    # =====================================================================
    # Time-invariant params (region only)
    c_man_base_p = Parameter(
        m, "c_man_base", domain=[R],
        records=[(r, data.c_man[r]) for r in data.regions],
    )
    theta_lbd_p = Parameter(
        m, "theta_lbd", domain=[R],
        records=[(r, 0.022) for r in data.regions], # Default realistic learning rate
    )
    c_man_floor_p = Parameter(
        m, "c_man_floor", domain=[R],
        records=[(r, max(50.0, data.c_man[r] * 0.5)) for r in data.regions], # Absolute minimum cost floor
    )
    
    c_ship = Parameter(
        m, "c_ship", domain=[exp, imp],
        records=[
            (r, i, data.c_ship[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )
    rho_p_p = Parameter(
        m, "rho_p", domain=[R],
        records=[(r, data.rho_p[r]) for r in data.regions],
    )

    # Time-indexed demand params [R, T]
    a_dem_t_p = Parameter(
        m, "a_dem_t", domain=[R, T],
        records=[(r, tp, a_dem_t_dict[(r, tp)]) for r in data.regions for tp in times],
    )
    b_dem_t_p = Parameter(
        m, "b_dem_t", domain=[R, T],
        records=[(r, tp, b_dem_t_dict[(r, tp)]) for r in data.regions for tp in times],
    )
    Dmax_t_p = Parameter(
        m, "Dmax_t", domain=[R, T],
        records=[(r, tp, dmax_t_dict[(r, tp)]) for r in data.regions for tp in times],
    )

    p_offer_ub_p = Parameter(
        m, "p_offer_ub", domain=[exp, imp],
        records=[
            (r, i, data.p_offer_ub[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )

    # Capacity constraints & costs params 
    Kcap_init_p = Parameter(
        m, "Kcap_init", domain=[R],
        records=[(r, kcap_2025_dict[r]) for r in data.regions],
    )
    g_exp_p = Parameter(
        m, "g_exp", domain=[R],
        records=[(r, g_exp_dict[r]) for r in data.regions],
    )
    g_dec_p = Parameter(
        m, "g_dec", domain=[R],
        records=[(r, g_dec_dict[r]) for r in data.regions],
    )
    f_hold_p = Parameter(
        m, "f_hold", domain=[R],
        records=[(r, f_hold_dict[r]) for r in data.regions],
    )
    c_inv_p = Parameter(
        m, "c_inv", domain=[R],
        records=[(r, c_inv_dict[r]) for r in data.regions],
    )

    # Discount & interval
    beta_p = Parameter(
        m, "beta_t", domain=[T],
        records=[(tp, beta_t_dict[tp]) for tp in times],
    )
    ytn_p = Parameter(
        m, "ytn", domain=[T],
        records=[(tp, ytn_dict[tp]) for tp in times],
    )

    # Regularization scalars
    eps_x = gp.Number(float(data.eps_x))
    eps_comp = float(data.eps_comp)
    eps_value = gp.Number(eps_comp)

    rho_prox_val = float(settings.get("rho_prox", 0.0))
    rho_prox = gp.Number(rho_prox_val)

    kappa_map = getattr(data, "kappa_Q", None) or {}
    kappa_by_r = {k: float(kappa_map.get(k, 0.0)) for k in data.regions}
    kappa_Q = Parameter(
        m, "kappa_Q", domain=[R],
        records=[(k, kappa_by_r[k]) for k in data.regions],
    )

    # Proximal reference params [R, T] and [imp, exp, T]
    Q_offer_last = Parameter(m, "Q_offer_last", domain=[R, T])
    p_offer_last = Parameter(m, "p_offer_last", domain=[exp, imp, T])

    # Initialize proximal references
    Q_offer_last[R, T] = Kcap_init_p[R]
    p_offer_last[exp, imp, T] = z

    # lam upper bound for variable bounding (use max p_offer_ub)
    max_p_offer = max(float(v) for v in data.p_offer_ub.values()) if data.p_offer_ub else 1000.0
    lam_ub_values: Dict[str, float] = {}
    for i in data.regions:
        lam_ub_values[i] = max_p_offer * 10.0 # safe upper bound
    lam_ub = Parameter(
        m, "lam_ub", domain=[R],
        records=[(i, lam_ub_values[i]) for i in data.regions],
    )

    # mu upper bound
    mu_ub_values: Dict[str, float] = {}
    for r in data.regions:
        mu_ub_values[r] = max(
            0.0,
            max(
                float(lam_ub_values[i])
                for i in data.regions
            ),
        )
    mu_ub = Parameter(
        m, "mu_ub", domain=[R],
        records=[(r, mu_ub_values[r]) for r in data.regions],
    )

    # gamma upper bound - based on reasonable max bounds for p_offer
    gamma_ub_values: Dict[Tuple[str, str], float] = {}
    for r in data.regions:
        for i in data.regions:
            gamma_ub_values[(r, i)] = (
                float(data.c_ship[(r, i)])
                + float(data.p_offer_ub[(r, i)])
                + float(data.eps_x) * float(kcap_2025_dict.get(r, 0.0) * 10) # arbitrary scalar for upper bound
                + float(mu_ub_values.get(r, 0.0))
            )
    gamma_ub = Parameter(
        m, "gamma_ub", domain=[exp, imp],
        records=[
            (r, i, gamma_ub_values[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )

    # =====================================================================
    # Step 3 — Variables (all time-indexed)
    # =====================================================================

    # -- ULP strategic variables --
    Q_offer = Variable(m, "Q_offer", domain=[R, T], type=VariableType.POSITIVE)
    p_offer = Variable(m, "p_offer", domain=[exp, imp, T], type=VariableType.POSITIVE)
    a_bid = Variable(m, "a_bid", domain=[R, T], type=VariableType.POSITIVE)
    
    # Capacity variables
    Kcap = Variable(m, "Kcap", domain=[R, T], type=VariableType.POSITIVE)
    k_exp = Variable(m, "k_exp", domain=[R, T], type=VariableType.POSITIVE)
    k_dec = Variable(m, "k_dec", domain=[R, T], type=VariableType.POSITIVE)

    # Link p_offer up bound
    p_offer.up[exp, imp, T] = p_offer_ub_p[exp, imp]
    
    # Restrict a_bid to not exceed true demand intercept
    a_bid.up[R, T] = a_dem_t_p[R, T]

    # -- Learning-By-Doing Variables --
    W_cum = Variable(m, "W_cum", domain=[R, T], type=VariableType.POSITIVE)
    c_man_var = Variable(m, "c_man_var", domain=[R, T], type=VariableType.POSITIVE)
    
    # Establish strict variable lower bound for cost floor
    c_man_var.lo[R, T] = c_man_floor_p[R]

    # Pre-fix Kcap initially and lock end-period expansion
    for r in data.regions:
        Kcap.fx[r, times[0]] = kcap_2025_dict[r]
        # Initialize cumulative volume at 0 for the first period
        W_cum.fx[r, times[0]] = 0.0
        
        # No investment or decommission in 2040 (terminal)
        k_exp.fx[r, times[-1]] = 0.0
        k_dec.fx[r, times[-1]] = 0.0

    # -- LLP market variables --
    z_llp = Variable(m, "z_llp", type=VariableType.FREE)

    x = Variable(m, "x", domain=[exp, imp, T], type=VariableType.POSITIVE)
    x_dem = Variable(m, "x_dem", domain=[R, T], type=VariableType.POSITIVE)
    x_dem.up[R, T] = Dmax_t_p[R, T]

    lam_var = Variable(m, "lam", domain=[R, T], type=VariableType.FREE)
    mu = Variable(m, "mu", domain=[R, T], type=VariableType.POSITIVE)
    gamma = Variable(m, "gamma", domain=[exp, imp, T], type=VariableType.POSITIVE)
    beta_dem = Variable(m, "beta_dem", domain=[R, T], type=VariableType.POSITIVE)
    psi_dem = Variable(m, "psi_dem", domain=[R, T], type=VariableType.POSITIVE)

    #lam_var.lo[R, T] = 0.0
    lam_var.up[R, T] = lam_ub[R]
    beta_dem.up[R, T] = lam_ub[R]
    psi_dem.up[R, T] = lam_ub[R]
    mu.up[R, T] = mu_ub[R]
    gamma.up[exp, imp, T] = gamma_ub[exp, imp]

    # =====================================================================
    # Step 4 — LLP equations (time-indexed)
    # =====================================================================

    # --- Primal LLP Objective (not directly used in MPEC solve but kept for reporting) ---
    llp_gross_surplus = Sum(
        [R, T],
        a_dem_t_p[R, T] * x_dem[R, T]
        - (b_dem_t_p[R, T] / gp.Number(2.0)) * x_dem[R, T] * x_dem[R, T],
    )

    llp_total_cost = (
        Sum(
            [exp, imp, T],
            (p_offer[exp, imp, T] + c_ship[exp, imp]) * x[exp, imp, T],
        )
        + Sum(
            [exp, imp, T],
            (eps_x / gp.Number(2.0)) * x[exp, imp, T] * x[exp, imp, T],
        )
    )

    eq_obj_llp = Equation(m, "eq_obj_llp")
    eq_obj_llp[...] = z_llp == llp_total_cost - llp_gross_surplus

    # --- Primal Constraints ---
    eq_bal = Equation(m, "eq_bal", domain=[imp, T])
    eq_bal[imp, T] = Sum(exp, x[exp, imp, T]) - x_dem[imp, T] == z

    eq_cap = Equation(m, "eq_cap", domain=[exp, T])
    eq_cap[exp, T] = Q_offer[exp, T] - Sum(imp, x[exp, imp, T]) >= z

    # --- Stationarity (KKT) ---
    eq_stat_x = Equation(m, "eq_stat_x", domain=[exp, imp, T])
    eq_stat_x[exp, imp, T] = (
        (p_offer[exp, imp, T] + c_ship[exp, imp])
        + eps_x * x[exp, imp, T]
        - lam_var[imp, T]
        + mu[exp, T]
        - gamma[exp, imp, T]
        == z
    )

    eq_stat_dem = Equation(m, "eq_stat_dem", domain=[imp, T])
    eq_stat_dem[imp, T] = (
        -(a_bid[imp, T] - b_dem_t_p[imp, T] * x_dem[imp, T])
        + lam_var[imp, T]
        + beta_dem[imp, T]
        - psi_dem[imp, T]
        == z
    )

    # --- Complementarity (KKT) ---
    eq_comp_mu = Equation(m, "eq_comp_mu", domain=[exp, T])
    if eps_comp == 0.0:
        eq_comp_mu[exp, T] = (
            mu[exp, T] * (Q_offer[exp, T] - Sum(imp, x[exp, imp, T])) == z
        )
    else:
        eq_comp_mu[exp, T] = (
            mu[exp, T] * (Q_offer[exp, T] - Sum(imp, x[exp, imp, T])) <= eps_value
        )

    eq_comp_gamma = Equation(m, "eq_comp_gamma", domain=[exp, imp, T])
    if eps_comp == 0.0:
        eq_comp_gamma[exp, imp, T] = gamma[exp, imp, T] * x[exp, imp, T] == z
    else:
        eq_comp_gamma[exp, imp, T] = gamma[exp, imp, T] * x[exp, imp, T] <= eps_value

    eq_comp_beta_dem = Equation(m, "eq_comp_beta_dem", domain=[imp, T])
    if eps_comp == 0.0:
        eq_comp_beta_dem[imp, T] = (
            beta_dem[imp, T] * (Dmax_t_p[imp, T] - x_dem[imp, T]) == z
        )
    else:
        eq_comp_beta_dem[imp, T] = (
            beta_dem[imp, T] * (Dmax_t_p[imp, T] - x_dem[imp, T]) <= eps_value
        )

    eq_comp_psi_dem = Equation(m, "eq_comp_psi_dem", domain=[imp, T])
    if eps_comp == 0.0:
        eq_comp_psi_dem[imp, T] = psi_dem[imp, T] * x_dem[imp, T] == z
    else:
        eq_comp_psi_dem[imp, T] = psi_dem[imp, T] * x_dem[imp, T] <= eps_value

    # =====================================================================
    # Step 5 — Capacity transitions + offer linkage + rate limits
    # =====================================================================

    # 3 hard-coded transition equations
    eq_cap_trans_30 = Equation(m, "eq_cap_trans_30", domain=[R])
    eq_cap_trans_30[R] = (
        Kcap[R, "2030"] == Kcap[R, "2025"] + k_exp[R, "2025"] - k_dec[R, "2025"]
    )

    eq_cap_trans_35 = Equation(m, "eq_cap_trans_35", domain=[R])
    eq_cap_trans_35[R] = (
        Kcap[R, "2035"] == Kcap[R, "2030"] + k_exp[R, "2030"] - k_dec[R, "2030"]
    )

    eq_cap_trans_40 = Equation(m, "eq_cap_trans_40", domain=[R])
    eq_cap_trans_40[R] = (
        Kcap[R, "2040"] == Kcap[R, "2035"] + k_exp[R, "2035"] - k_dec[R, "2035"]
    )

    # Offer linkage: Q_offer <= Kcap (explicit constraint)
    eq_offer_cap = Equation(m, "eq_offer_cap", domain=[R, T])
    eq_offer_cap[R, T] = Q_offer[R, T] <= Kcap[R, T]

    # Pin domestic self-offer to domestic manufacturing cost
    eq_self_offer = Equation(m, "eq_self_offer", domain=[R, T])
    eq_self_offer[R, T] = p_offer[R, R, T] == c_man_var[R, T]

    # Expansion rate limit: k_exp <= g_exp * y_t * K_cap
    eq_exp_limit = Equation(m, "eq_exp_limit", domain=[R, T])
    eq_exp_limit[R, T] = k_exp[R, T] <= g_exp_p[R] * ytn_p[T] * Kcap[R, T]

    # Decommissioning rate limit: k_dec <= g_dec * y_t * K_cap
    eq_dec_limit = Equation(m, "eq_dec_limit", domain=[R, T])
    eq_dec_limit[R, T] = k_dec[R, T] <= g_dec_p[R] * ytn_p[T] * Kcap[R, T]

    # Prevent simultaneous expansion and decommissioning (k_exp * k_dec = 0)
    eq_comp_exp_dec = Equation(m, "eq_comp_exp_dec", domain=[R, T])
    if eps_comp == 0.0:
        eq_comp_exp_dec[R, T] = k_exp[R, T] * k_dec[R, T] == z
    else:
        eq_comp_exp_dec[R, T] = k_exp[R, T] * k_dec[R, T] <= eps_value

    # --- Learning-By-Doing (LBD) Equations ---
    
    # Cumulative volume transitions
    eq_w_trans_30 = Equation(m, "eq_w_trans_30", domain=[R])
    eq_w_trans_30[R] = (
        W_cum[R, "2030"] == W_cum[R, "2025"] + ytn_p["2025"] * Sum(imp, x[R, imp, "2025"])
    )

    eq_w_trans_35 = Equation(m, "eq_w_trans_35", domain=[R])
    eq_w_trans_35[R] = (
        W_cum[R, "2035"] == W_cum[R, "2030"] + ytn_p["2030"] * Sum(imp, x[R, imp, "2030"])
    )

    eq_w_trans_40 = Equation(m, "eq_w_trans_40", domain=[R])
    eq_w_trans_40[R] = (
        W_cum[R, "2040"] == W_cum[R, "2035"] + ytn_p["2035"] * Sum(imp, x[R, imp, "2035"])
    )

    # Cost curve constraints (Inequality allows W_cum to grow past the floor bound)
    eq_c_man_curve = Equation(m, "eq_c_man_curve", domain=[R, T])
    eq_c_man_curve[R, T] = (
        c_man_var[R, T] >= c_man_base_p[R] - (theta_lbd_p[R] * W_cum[R, T])
    )

    # Pin domestic offer price to marginal manufacturing cost
    eq_p_offer_self = Equation(m, "eq_p_offer_self", domain=[R, T])
    eq_p_offer_self[R, T] = p_offer[R, R, T] == c_man_var[R, T]
    # =====================================================================
    # Collect equations
    # =====================================================================
    equations = {
        "eq_bal": eq_bal,
        "eq_cap": eq_cap,
        "eq_stat_x": eq_stat_x,
        "eq_stat_dem": eq_stat_dem,
        "eq_comp_mu": eq_comp_mu,
        "eq_comp_gamma": eq_comp_gamma,
        "eq_comp_beta_dem": eq_comp_beta_dem,
        "eq_comp_psi_dem": eq_comp_psi_dem,
        "eq_obj_llp": eq_obj_llp,
        "eq_cap_trans_30": eq_cap_trans_30,
        "eq_cap_trans_35": eq_cap_trans_35,
        "eq_cap_trans_40": eq_cap_trans_40,
        "eq_offer_cap": eq_offer_cap,
        "eq_self_offer": eq_self_offer,
        "eq_exp_limit": eq_exp_limit,
        "eq_dec_limit": eq_dec_limit,
        "eq_comp_exp_dec": eq_comp_exp_dec,
        "eq_w_trans_30": eq_w_trans_30,
        "eq_w_trans_35": eq_w_trans_35,
        "eq_w_trans_40": eq_w_trans_40,
        "eq_c_man_curve": eq_c_man_curve,
        "eq_p_offer_self": eq_p_offer_self,
    }

    # =====================================================================
    # Step 6 — Objective = sum over time
    # =====================================================================
    models: Dict[str, Model] = {}
    for rname in data.players:
        r = rname

        # ---- Per-period welfare components (summed over T) ----

        # U_rt(d) - lam_r * d_r
        d_surplus_t = Sum(
            T,
            beta_p[T] * ytn_p[T] * (
                a_dem_t_p[r, T] * x_dem[r, T]
                - (b_dem_t_p[r, T] / gp.Number(2.0)) * x_dem[r, T] * x_dem[r, T]
                - lam_var[r, T] * x_dem[r, T]
            ),
        )

        # Producer term
        producer_term_t = Sum(
            [j, T],
            beta_p[T] * ytn_p[T] * (
                lam_var[j, T]
                - c_man_var[r, T]
                - c_ship[r, j]
            ) * x[r, j, T],
        )

        # Holding cost
        hold_cost_t = Sum(
            T,
            beta_p[T] * f_hold_p[r] * ytn_p[T] * Kcap[r, T],
        )

        # Investment cost
        inv_cost_t = Sum(
            T,
            beta_p[T] * c_inv_p[r] * ytn_p[T] * k_exp[r, T],
        )

        # ---- Penalties (replicated per period) ----
        pen_p_offer_quad = z
        pen_q_quad = z
        if use_quad:
            pen_p_offer_quad = Sum(
                T,
                -gp.Number(0.5) * ytn_p[T] * rho_p_p[r] * Sum(j, p_offer[r, j, T] * p_offer[r, j, T]),
            )
            pen_q_quad = Sum(
                T,
                -gp.Number(0.5) * ytn_p[T] * kappa_Q[r] * (
                    (Q_offer[r, T] - Sum(j, x[r, j, T]))
                    * (Q_offer[r, T] - Sum(j, x[r, j, T]))
                ),
            )

        # Linear penalties
        pen_p_offer_lin = Sum(
            T,
            -ytn_p[T] * rho_p_p[r] * Sum(j, p_offer[r, j, T]),
        )
        pen_q_lin = Sum(
            T,
            -ytn_p[T] * kappa_Q[r] * Q_offer[r, T],
        )

        # Proximal regularization
        pen_prox_q = Sum(
            T,
            -gp.Number(0.5) * ytn_p[T] * rho_prox * (
                (Q_offer[r, T] - Q_offer_last[r, T])
                * (Q_offer[r, T] - Q_offer_last[r, T])
            ),
        )
        pen_prox_poffer = Sum(
            T,
            -gp.Number(0.5) * ytn_p[T] * rho_prox * Sum(
                j,
                (p_offer[r, j, T] - p_offer_last[r, j, T])
                * (p_offer[r, j, T] - p_offer_last[r, j, T]),
            ),
        )

        # ---- Assemble objective ----
        if use_quad:
            obj_welfare = (
                d_surplus_t
                + producer_term_t
                - hold_cost_t
                - inv_cost_t
                + pen_p_offer_quad
                + pen_q_quad
                + pen_prox_q
                + pen_prox_poffer
            )
        else:
            obj_welfare = (
                d_surplus_t
                + producer_term_t
                - hold_cost_t
                - inv_cost_t
                + pen_p_offer_lin
                + pen_q_lin
                + pen_prox_q
                + pen_prox_poffer
            )

        models[r] = Model(
            m,
            f"mpec_{r}",
            equations=list(equations.values()),
            problem=Problem.NLP,
            sense=Sense.MAX,
            objective=obj_welfare,
        )

    # =====================================================================
    # Return context
    # =====================================================================
    return ModelContext(
        container=m,
        sets={"R": R, "exp": exp, "imp": imp, "j": j, "T": T},
        params={
            "Dmax_t": Dmax_t_p,
            "a_dem_t": a_dem_t_p,
            "b_dem_t": b_dem_t_p,
            "Kcap_init": Kcap_init_p,
            "c_man_base": c_man_base_p,
            "theta_lbd": theta_lbd_p,
            "c_man_floor": c_man_floor_p,
            "c_ship": c_ship,
            "p_offer_ub": p_offer_ub_p,
            "rho_p": rho_p_p,
            "kappa_Q": kappa_Q,
            "g_exp": g_exp_p,
            "g_dec": g_dec_p,
            "f_hold": f_hold_p,
            "c_inv": c_inv_p,
            "beta_t": beta_p,
            "ytn": ytn_p,
            "Q_offer_last": Q_offer_last,
            "p_offer_last": p_offer_last,
        },
        vars={
            "Q_offer": Q_offer,
            "p_offer": p_offer,
            "a_bid": a_bid,
            "Kcap": Kcap,
            "k_exp": k_exp,
            "k_dec": k_dec,
            "x": x,
            "x_dem": x_dem,
            "lam": lam_var,
            "mu": mu,
            "gamma": gamma,
            "W_cum": W_cum,
            "c_man_var": c_man_var,
            "beta_dem": beta_dem,
            "psi_dem": psi_dem,
        },
        equations=equations,
        models=models,
    )


# =============================================================================
# Step 7 — apply_player_fixings (time-indexed)
# =============================================================================
def apply_player_fixings(
    ctx: ModelContext,
    data: ModelData,
    theta_Q: Dict[Tuple[str, str], float],
    theta_p_offer: Dict[Tuple[str, str, str], float],
    theta_k_exp: Dict[Tuple[str, str], float],
    theta_k_dec: Dict[Tuple[str, str], float],
    theta_a_bid: Dict[Tuple[str, str], float],
    *,
    player: str,
) -> None:
    """Fix all other players' strategies; free current player's strategies."""
    times = data.times or list(_DEFAULT_TIMES)
    kcap_2025_dict = data.Kcap_2025 if data.Kcap_2025 is not None else {r: float(data.Qcap.get(r, 0.0)) for r in data.regions}

    Q_offer = ctx.vars["Q_offer"]
    p_offer = ctx.vars["p_offer"]
    a_bid = ctx.vars["a_bid"]
    Kcap = ctx.vars["Kcap"]
    k_exp = ctx.vars["k_exp"]
    k_dec = ctx.vars["k_dec"]

    for r in data.regions:
        for tp in times:
            if r == player:
                # Free current player's strategies
                # Q_offer bounded by Kcap (via eq_offer_cap constraint)
                Q_offer.lo[r, tp] = 0.0
                Q_offer.up[r, tp] = float("inf")  # constraint handles upper bound

                # Free capacity decisions
                if tp == times[0]:
                    # Kcap fixed at initial
                    Kcap.lo[r, tp] = float(kcap_2025_dict[r])
                    Kcap.up[r, tp] = float(kcap_2025_dict[r])
                else:
                    Kcap.lo[r, tp] = 0.0
                    Kcap.up[r, tp] = float("inf")

                if tp == times[-1]:
                    k_exp.lo[r, tp] = 0.0
                    k_exp.up[r, tp] = 0.0
                    k_dec.lo[r, tp] = 0.0
                    k_dec.up[r, tp] = 0.0
                else:
                    k_exp.lo[r, tp] = 0.0
                    k_exp.up[r, tp] = float("inf") # Bound managed by eq_exp_limit
                    k_dec.lo[r, tp] = 0.0
                    k_dec.up[r, tp] = float("inf") # Bound managed by eq_dec_limit
                
                # a_bid freed for current player's own region
                a_bid.lo[r, tp] = 0.0
                a_bid.up[r, tp] = float(data.a_dem_t[(r, tp)] if data.a_dem_t else data.a_dem.get(r, 0.0))

            elif r in data.players:
                # Fix other strategic player
                q_val = float(theta_Q.get((r, tp), float(kcap_2025_dict.get(r, 0.0))))
                Q_offer.lo[r, tp] = q_val
                Q_offer.up[r, tp] = q_val

                i_val = float(theta_k_exp.get((r, tp), 0.0))
                k_exp.lo[r, tp] = i_val
                k_exp.up[r, tp] = i_val

                d_val = float(theta_k_dec.get((r, tp), 0.0))
                k_dec.lo[r, tp] = d_val
                k_dec.up[r, tp] = d_val
                
                a_val = float(theta_a_bid.get((r, tp), data.a_dem_t[(r, tp)] if data.a_dem_t else data.a_dem.get(r, 0.0)))
                a_bid.lo[r, tp] = a_val
                a_bid.up[r, tp] = a_val

                # Kcap is determined by transitions — fix it
                if tp == times[0]:
                    Kcap.lo[r, tp] = float(kcap_2025_dict[r])
                    Kcap.up[r, tp] = float(kcap_2025_dict[r])
                else:
                    # Kcap implied by transitions; keep free so transition eqs work
                    Kcap.lo[r, tp] = 0.0
                    Kcap.up[r, tp] = float("inf")

            else:
                # Non-strategic: no capacity changes, fixed Q_offer
                v = float(kcap_2025_dict.get(r, 0.0))
                Q_offer.lo[r, tp] = v
                Q_offer.up[r, tp] = v

                Kcap.lo[r, tp] = v
                Kcap.up[r, tp] = v

                k_exp.lo[r, tp] = 0.0
                k_exp.up[r, tp] = 0.0
                k_dec.lo[r, tp] = 0.0
                k_dec.up[r, tp] = 0.0
                
                a_val = float(data.a_dem_t[(r, tp)] if data.a_dem_t else data.a_dem.get(r, 0.0))
                a_bid.lo[r, tp] = a_val
                a_bid.up[r, tp] = a_val

    # -- Offer Prices --
    for ex in data.regions:
        for im in data.regions:
            for tp in times:
                ub = data.p_offer_ub[(ex, im)]
                if ex in data.non_strategic:
                    p_offer.lo[ex, im, tp] = 0.0
                    p_offer.up[ex, im, tp] = 0.0
                elif ex == player:
                    p_offer.lo[ex, im, tp] = 0.0
                    p_offer.up[ex, im, tp] = ub
                else:
                    v = float(theta_p_offer.get((ex, im, tp), 0.0))
                    p_offer.lo[ex, im, tp] = v
                    p_offer.up[ex, im, tp] = v


# =============================================================================
# Step 7 — extract_state (time-indexed)
# =============================================================================
def extract_state(
    ctx: ModelContext, variables: List[str] | None = None
) -> Dict[str, Dict]:
    def _maybe_var(name: str):
        if variables is not None and name not in variables:
            return {}
        v = ctx.vars.get(name)
        if v is None:
            return {}
        out = v.toDict()
        return out if isinstance(out, dict) else {}

    # Extract objective values from solved models
    obj_values = {}
    if variables is None or "obj" in variables:
        for r, model in ctx.models.items():
            try:
                obj_values[r] = float(model.objective_value)
            except (AttributeError, TypeError):
                pass

    return {
        "Q_offer": _maybe_var("Q_offer"),
        "p_offer": _maybe_var("p_offer"),
        "a_bid": _maybe_var("a_bid"),
        "Kcap": _maybe_var("Kcap"),
        "k_exp": _maybe_var("k_exp"),
        "k_dec": _maybe_var("k_dec"),
        "x": _maybe_var("x"),
        "x_dem": _maybe_var("x_dem"),
        "lam": _maybe_var("lam"),
        "mu": _maybe_var("mu"),
        "gamma": _maybe_var("gamma"),
        "beta_dem": _maybe_var("beta_dem"),
        "psi_dem": _maybe_var("psi_dem"),
        "W_cum": _maybe_var("W_cum"),
        "c_man_var": _maybe_var("c_man_var"),
        "obj": obj_values,
    }

