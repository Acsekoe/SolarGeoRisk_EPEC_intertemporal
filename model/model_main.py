"""
Intertemporal perfect-foresight EPEC model using Offer-Based Uniform-Price Settlement.

Each strategic player maximises the present-value sum of welfare across the
operating periods {"2025", "2030", "2035", "2040"}.  The additional 2045
label is a capacity-state date only: the 2040 capacity decision determines
Kcap[2045], which receives the terminal value, but no 2045 market is cleared.

Discounting
-----------
beta_t[T] = 1 / (1 + r)^(T - base_year) where r = discount_rate.
When discount_rate = 0 the objective reduces to block-length-weighted
(undiscounted) aggregation: Sum_t ytn[t] * welfare_t.

Investment cost interpretation
-------------------------------
c_inv [mUSD/GW-yr]: annualized capital charge per GW-year of expansion rate.
    Annual cost = c_inv * Icap_pos   (Icap_pos in GW/yr)
    Over period of ytn years: c_inv * Icap_pos * ytn
    Discounted: beta_t * ytn * c_inv * Icap_pos
f_hold [mUSD/GW-yr]: annual O&M holding cost per GW of installed capacity.
    Cost per period: beta_t * ytn * f_hold * Kcap

Shadow-price naming
-------------------
mu_offer  (EPEC model_main.py): dual on the offer-capacity constraint
          Q_offer[exp,t] - sum_imp x[exp,imp,t] >= 0.
          This is NOT a physical capacity scarcity rent; it reflects
          the exporter's residual offer margin above actual shipments.
mu_cap    (model_llp_planner.py): dual on the physical capacity constraint
          sum_imp x[exp,imp,t] <= Kcap[exp,t] in the planner.
          This IS a physical scarcity rent.
"""
from __future__ import annotations

INTERTEMPORAL_IMPLEMENTED = True

import math
import warnings
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

_DEFAULT_TIMES = ["2025", "2030", "2035", "2040", "2045"]
_DEFAULT_YTN = {"2025": 5.0, "2030": 5.0, "2035": 5.0, "2040": 5.0, "2045": 5.0}


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

    eps_x: float
    eps_comp: float
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
    # g_exp_ub is interpreted as absolute GW/year when g_exp_ub_is_absolute=True,
    # otherwise as a per-year rate multiplied by current Kcap.
    g_exp_ub: Dict[str, float] | None = None
    g_exp_ub_is_absolute: bool = False
    g_dec_ub: Dict[str, float] | None = None

    # Capacity costs
    f_hold: Dict[str, float] | None = None  # holding cost
    c_inv: Dict[str, float] | None = None   # investment cost

    # Discounting
    beta_t: Dict[str, float] | None = None  # NPV discount factor: beta_t = 1/(1+r)^(t-base_year)
    years_to_next: Dict[str, float] | None = None  # interval lengths (ytn[t] = years in block t)

    # Exogenous LBD cost schedule: (region, time) -> cost (USD/kW), pre-computed by
    # compute_lbd_schedule() in data_prep.py using Swanson's Law + IEA NZE projections.
    # Falls back to time-invariant c_man[r] when None (backward-compat).
    c_man_t: Dict[Tuple[str, str], float] | None = None


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


def _true_demand_intercept(data: ModelData, region: str, tp: str) -> float:
    if data.a_dem_t is not None:
        return float(data.a_dem_t[(region, tp)])
    return float(data.a_dem.get(region, 0.0))


def _initial_capacity_by_region(data: ModelData) -> Dict[str, float]:
    cap_source = data.Kcap_2025 if data.Kcap_2025 is not None else data.Qcap
    return {
        r: float(cap_source.get(r, data.Qcap.get(r, 0.0)))
        for r in data.regions
    }


def _initial_capacity_path(
    data: ModelData,
    times: List[str],
) -> Dict[Tuple[str, str], float]:
    kcap_init = _initial_capacity_by_region(data)
    return {(r, tp): kcap_init[r] for r in data.regions for tp in times}


def _transition_pairs(times: List[str]) -> List[Tuple[str, str]]:
    return list(zip(times[:-1], times[1:]))


def _move_times(times: List[str]) -> List[str]:
    return times[:-1]


def _non_strategic_regions(data: ModelData) -> PySet[str]:
    return set(data.non_strategic) | (set(data.regions) - set(data.players))


def _fix_a_bid_to_true_dem(data: ModelData) -> bool:
    settings = data.settings or {}
    return bool(settings.get("fix_a_bid_to_true_dem", False))


def _fix_q_offer_to_kcap(data: ModelData) -> bool:
    settings = data.settings or {}
    return bool(settings.get("fix_q_offer_to_kcap", False))


def _terminal_capacity_state_only(data: ModelData) -> bool:
    """Whether the last time label is a capacity state rather than a market."""
    return bool((data.settings or {}).get("terminal_capacity_state_only", False))


def _operating_times(data: ModelData) -> List[str]:
    """Return time labels with economic operations and market clearing."""
    times = list(data.times or _DEFAULT_TIMES)
    if _terminal_capacity_state_only(data):
        if len(times) < 2:
            raise ValueError(
                "terminal_capacity_state_only requires at least one operating "
                "period and one terminal capacity-state date"
            )
        return times[:-1]
    return times


def _terminal_salvage_discount_factor(data: ModelData) -> float:
    """Return the discount factor at the terminal stock valuation date.

    With ``terminal_capacity_state_only=True``, the last time label is already
    the end-of-horizon stock date (2045 in the paper calibration), so its beta
    is used directly.  The legacy full-operating-horizon mode instead values
    stock at the end of the final operating block.
    """
    times = list(data.times or _DEFAULT_TIMES)
    terminal = times[-1]
    beta_at_terminal_start = float((data.beta_t or {}).get(terminal, 1.0))
    if _terminal_capacity_state_only(data):
        return beta_at_terminal_start
    years_to_terminal_end = float(
        (data.years_to_next or _DEFAULT_YTN).get(terminal, 0.0)
    )
    if years_to_terminal_end < 0.0:
        raise ValueError("terminal years_to_next must be non-negative")
    discount_rate = float((data.settings or {}).get("discount_rate", 0.0))
    if discount_rate <= -1.0:
        raise ValueError("discount_rate must be greater than -1")
    return beta_at_terminal_start / ((1.0 + discount_rate) ** years_to_terminal_end)


def _terminal_salvage_credit(
    data: ModelData,
    player: str,
    terminal_capacity: float,
) -> float:
    """Evaluate the terminal stock credit for reporting and nested solves."""
    fraction = float((data.settings or {}).get("terminal_salvage_fraction", 0.0))
    if fraction < 0.0:
        raise ValueError("terminal_salvage_fraction must be non-negative")
    investment_cost = float((data.c_inv or {}).get(player, 0.0))
    return (
        _terminal_salvage_discount_factor(data)
        * fraction
        * investment_cost
        * float(terminal_capacity)
    )


def _implied_capacity_path(
    data: ModelData,
    times: List[str],
    theta_dK_net: Dict[Tuple[str, str], float] | None = None,
) -> Dict[Tuple[str, str], float]:
    """Recover the endogenous Kcap path implied by Kcap_init and net changes."""
    kcap_init = _initial_capacity_by_region(data)
    ytn = data.years_to_next or _DEFAULT_YTN
    move_times = _move_times(times)
    out: Dict[Tuple[str, str], float] = {}
    for r in data.regions:
        out[(r, times[0])] = float(kcap_init[r])
        for tp, tp_next in _transition_pairs(times):
            d_val = 0.0
            if theta_dK_net is not None:
                d_val = float(theta_dK_net.get((r, tp), 0.0))
            out[(r, tp_next)] = out[(r, tp)] + float(ytn.get(tp, _DEFAULT_YTN.get(tp, 1.0))) * d_val
        for tp in move_times:
            out.setdefault((r, tp), float(kcap_init[r]))
    return out


def _clip_value(value: float, lo: float, up: float) -> float:
    return max(lo, min(value, up))


def _warn_model_structure(data: ModelData, times: List[str]) -> None:
    """Surface data/model inconsistencies without changing the intended game."""
    for r in data.regions:
        c_floor = max(50.0, float(data.c_man.get(r, 0.0)) * 0.5)
        domestic_ub = float(data.p_offer_ub.get((r, r), 0.0))
        if domestic_ub + 1e-9 < c_floor:
            warnings.warn(
                (
                    f"Domestic p_offer_ub[{r},{r}]={domestic_ub:.6g} is below "
                    f"c_man_floor={c_floor:.6g}; the self-offer equality may be infeasible."
                ),
                stacklevel=2,
            )

    for ex in _non_strategic_regions(data):
        benchmark = float(data.c_man.get(ex, 0.0))
        for im in data.regions:
            if ex == im:
                continue
            ub = float(data.p_offer_ub.get((ex, im), 0.0))
            if benchmark > ub + 1e-9:
                warnings.warn(
                    (
                        f"Non-strategic offer benchmark {benchmark:.6g} for route "
                        f"({ex},{im}) exceeds p_offer_ub={ub:.6g}; fixings will clip to the upper bound."
                    ),
                    stacklevel=2,
                )

    # Keep a guard here so future edits do not silently revert the follower
    # back to true demand while eq_stat_dem still uses a_bid.
    llp_objective_demand_symbol = "a_bid"
    llp_stationarity_demand_symbol = "a_bid"
    if llp_objective_demand_symbol != llp_stationarity_demand_symbol:
        warnings.warn(
            "LLP objective/stationarity mismatch detected: eq_obj_llp must use a_bid whenever eq_stat_dem uses a_bid.",
            stacklevel=2,
        )


def validate_strategy_inputs(
    data: ModelData,
    theta_Q: Dict[Tuple[str, str], float],
    theta_a_bid: Dict[Tuple[str, str], float] | None = None,
    theta_dK_net: Dict[Tuple[str, str], float] | None = None,
) -> None:
    """Warn when iterates exceed the feasible intertemporal strategy space."""
    times = data.times or list(_DEFAULT_TIMES)
    move_times = _move_times(times)
    transition_pairs = _transition_pairs(times)
    kcap_init = _initial_capacity_by_region(data)
    implied_kcap = _implied_capacity_path(data, times, theta_dK_net)
    for key, q_val in theta_Q.items():
        cap = float(implied_kcap.get(key, 0.0))
        if float(q_val) > cap + 1e-9:
            warnings.warn(
                (
                    f"theta_Q{key}={float(q_val):.6g} exceeds implied Kcap={cap:.6g}; "
                    f"apply_player_fixings will clip it."
                ),
                stacklevel=2,
            )

    if theta_a_bid is None:
        theta_a_bid = {}

    for (r, tp), a_val in theta_a_bid.items():
        a_true = _true_demand_intercept(data, r, tp)
        if float(a_val) < -1e-9 or float(a_val) > a_true + 1e-9:
            warnings.warn(
                (
                    f"theta_a_bid[{r},{tp}]={float(a_val):.6g} lies outside [0, {a_true:.6g}]; "
                    f"apply_player_fixings will clip it."
                ),
                stacklevel=2,
            )

    if theta_dK_net is not None:
        for r in data.players:
            for tp in move_times:
                d_val = float(theta_dK_net.get((r, tp), 0.0))
                k_val = max(float(implied_kcap.get((r, tp), kcap_init[r])), 0.0)
                exp_lim = float((data.g_exp_ub or {}).get(r, 0.1))
                if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
                    exp_lim *= k_val
                dec_lim = float((data.g_dec_ub or {}).get(r, 0.1)) * k_val
                if d_val > exp_lim + 1e-9:
                    warnings.warn(
                        f"theta_dK_net[{r},{tp}]={d_val:.6g} exceeds expansion limit {exp_lim:.6g}.",
                        stacklevel=2,
                    )
                if -d_val > dec_lim + 1e-9:
                    warnings.warn(
                        f"theta_dK_net[{r},{tp}]={d_val:.6g} is below the decommissioning lower bound {-dec_lim:.6g}.",
                        stacklevel=2,
                    )

            for tp in times:
                k_val = float(implied_kcap.get((r, tp), 0.0))
                if k_val < -1e-9:
                    warnings.warn(
                        f"Implied Kcap[{r},{tp}]={k_val:.6g} is negative under theta_dK_net.",
                        stacklevel=2,
                    )


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

    # =====================================================================
    # Step 2 — Container, sets, time set
    # =====================================================================
    m = Container(working_directory=working_directory, debugging_level="keep")

    R = Set(m, "R", records=data.regions)
    exp = Alias(m, "exp", R)
    imp = Alias(m, "imp", R)
    j = Alias(m, "j", R)

    times = data.times or list(_DEFAULT_TIMES)
    operating_times = _operating_times(data)
    T = Set(m, "T", records=times)
    T_op = Set(m, "T_op", domain=[T], records=operating_times)
    _warn_model_structure(data, times)
    transition_pairs = _transition_pairs(times)
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
    # Exogenous LBD cost schedule: pre-computed via Swanson's Law + IEA NZE projections.
    # Falls back to time-invariant c_man[r] when c_man_t is not provided.
    _c_man_t_src = data.c_man_t or {
        (r, tp): data.c_man[r] for r in data.regions for tp in times
    }
    c_man_t_p = Parameter(
        m, "c_man_t", domain=[R, T],
        records=[(r, tp, _c_man_t_src[(r, tp)]) for r in data.regions for tp in times],
    )
    
    c_ship = Parameter(
        m, "c_ship", domain=[exp, imp],
        records=[
            (r, i, data.c_ship[(r, i)])
            for r in data.regions for i in data.regions
        ],
    )
    # Deleted rho_p_p parameter here

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

    # Algorithmic proximal penalties — stored as scalar Parameters so gauss_seidel.py
    # can update them between iterations (penalty annealing) without rebuilding the model.
    c_pen_q  = Parameter(m, "c_pen_q_scalar",  records=float(settings.get("c_pen_q",  0.0)))
    c_pen_p  = Parameter(m, "c_pen_p_scalar",  records=float(settings.get("c_pen_p",  0.0)))
    c_pen_a  = Parameter(m, "c_pen_a_scalar",  records=float(settings.get("c_pen_a",  0.0)))
    c_pen_dk = Parameter(m, "c_pen_dk_scalar", records=float(settings.get("c_pen_dk", 0.0)))
    
    # Economic quadratic penalties (convex costs / disutility)
    c_quad_q = gp.Number(float(settings.get("c_quad_q", 0.0)))
    c_quad_p = gp.Number(float(settings.get("c_quad_p", 0.0)))
    c_quad_a = gp.Number(float(settings.get("c_quad_a", 0.0)))
    subtract_mu_offer_from_producer_margin = bool(
        settings.get("subtract_mu_offer_from_producer_margin", True)
    )

    # Capacity-policy incentives (all optional; default off).
    # Positive cap_keep_reward and capex_subsidy encourage retaining/expanding capacity.
    # Positive terminal_salvage_fraction values the installed stock once, at the
    # end of the terminal block, as a fraction of region-specific investment cost.
    # Positive decommission_penalty discourages dismantling capacity.
    cap_keep_reward = gp.Number(float(settings.get("cap_keep_reward", 0.0)))
    capex_subsidy = gp.Number(float(settings.get("capex_subsidy", 0.0)))
    terminal_salvage_fraction_value = float(
        settings.get("terminal_salvage_fraction", 0.0)
    )
    if terminal_salvage_fraction_value < 0.0:
        raise ValueError("terminal_salvage_fraction must be non-negative")
    terminal_salvage_fraction = gp.Number(terminal_salvage_fraction_value)
    terminal_salvage_discount = gp.Number(_terminal_salvage_discount_factor(data))
    decommission_penalty = gp.Number(float(settings.get("decommission_penalty", 0.0)))

    # Proximal reference parameters — updated by gauss_seidel before each player solve.
    p_offer_last  = Parameter(m, "p_offer_last",  domain=[exp, imp, T])
    Q_offer_last  = Parameter(m, "Q_offer_last",  domain=[R, T])
    Icap_pos_last = Parameter(m, "Icap_pos_last", domain=[R, T])
    Dcap_neg_last = Parameter(m, "Dcap_neg_last", domain=[R, T])
    a_bid_last    = Parameter(m, "a_bid_last",    domain=[R, T])

    # Initialise all reference values to zero
    p_offer_last[exp, imp, T]  = z
    Q_offer_last[R, T]         = z
    Icap_pos_last[R, T]        = z
    Dcap_neg_last[R, T]        = z
    a_bid_last[R, T]           = z

    # lam upper bound for variable bounding.
    # By default, cap each import-region lambda by the largest offer cap that can
    # be sent into that region. Optional settings allow a small additive buffer.
    lam_ub_buffer = float(settings.get("lam_ub_buffer", 0.0))
    lam_ub_floor = float(settings.get("lam_ub_floor", 1.0))
    lam_ub_values: Dict[str, float] = {}
    for i in data.regions:
        inbound_offer_caps = [
            float(data.p_offer_ub[(ex, i)])
            for ex in data.regions
        ]
        inbound_max = max(inbound_offer_caps) if inbound_offer_caps else 0.0
        lam_ub_values[i] = max(lam_ub_floor, inbound_max + lam_ub_buffer)
    lam_ub = Parameter(
        m, "lam_ub", domain=[R],
        records=[(i, lam_ub_values[i]) for i in data.regions],
    )

    # mu_offer upper bound (dual of offer-capacity constraint Q_offer - sum_imp x >= 0)
    mu_offer_ub_values: Dict[str, float] = {}
    for r in data.regions:
        mu_offer_ub_values[r] = max(
            0.0,
            max(
                float(lam_ub_values[i])
                for i in data.regions
            ),
        )
    mu_offer_ub = Parameter(
        m, "mu_offer_ub", domain=[R],
        records=[(r, mu_offer_ub_values[r]) for r in data.regions],
    )

    # gamma upper bound - based on reasonable max bounds for p_offer
    gamma_ub_values: Dict[Tuple[str, str], float] = {}
    for r in data.regions:
        for i in data.regions:
            gamma_ub_values[(r, i)] = (
                float(data.c_ship[(r, i)])
                + float(data.p_offer_ub[(r, i)])
                + float(data.eps_x) * float(kcap_2025_dict.get(r, 0.0) * 10) # arbitrary scalar for upper bound
                + float(mu_offer_ub_values.get(r, 0.0))
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
    # Capacity variables use every state date.  Market/offer variables retain
    # the full T domain for backward-compatible state serialization, but the
    # terminal-only entries are fixed to zero and never enter an equation or
    # objective when terminal_capacity_state_only is enabled.
    Kcap = Variable(m, "Kcap", domain=[R, T], type=VariableType.POSITIVE)
    Icap_pos = Variable(m, "Icap_pos", domain=[R, T], type=VariableType.POSITIVE)
    Dcap_neg = Variable(m, "Dcap_neg", domain=[R, T], type=VariableType.POSITIVE)
    Q_offer = Variable(m, "Q_offer", domain=[R, T], type=VariableType.POSITIVE)
    p_offer = Variable(m, "p_offer", domain=[exp, imp, T], type=VariableType.POSITIVE)
    a_bid = Variable(m, "a_bid", domain=[R, T], type=VariableType.POSITIVE)

    # Link p_offer up bound
    p_offer.up[exp, imp, T] = p_offer_ub_p[exp, imp]

    a_bid.up[R, T] = a_dem_t_p[R, T]

    # -- LLP market variables --
    z_llp = Variable(m, "z_llp", type=VariableType.FREE)

    x = Variable(m, "x", domain=[exp, imp, T], type=VariableType.POSITIVE)
    x_dem = Variable(m, "x_dem", domain=[R, T], type=VariableType.POSITIVE)
    x_dem.up[R, T] = Dmax_t_p[R, T]

    lam_var = Variable(m, "lam", domain=[R, T], type=VariableType.FREE)
    mu_offer = Variable(m, "mu_offer", domain=[R, T], type=VariableType.POSITIVE)
    gamma = Variable(m, "gamma", domain=[exp, imp, T], type=VariableType.POSITIVE)
    beta_dem = Variable(m, "beta_dem", domain=[R, T], type=VariableType.POSITIVE)
    psi_dem = Variable(m, "psi_dem", domain=[R, T], type=VariableType.POSITIVE)

    lam_var.lo[R, T] = 0.0
    lam_var.up[R, T] = lam_ub[R]
    beta_dem.up[R, T] = lam_ub[R]
    psi_dem.up[R, T] = lam_ub[R]
    mu_offer.up[R, T] = mu_offer_ub[R]
    gamma.up[exp, imp, T] = gamma_ub[exp, imp]

    if bool(settings.get("force_mu_offer_zero", False)):
        mu_offer.lo[R, T] = 0.0
        mu_offer.up[R, T] = 0.0

    inactive_market_times = [tp for tp in times if tp not in operating_times]
    for tp in inactive_market_times:
        Q_offer.lo[R, tp] = 0.0
        Q_offer.up[R, tp] = 0.0
        p_offer.lo[exp, imp, tp] = 0.0
        p_offer.up[exp, imp, tp] = 0.0
        a_bid.lo[R, tp] = 0.0
        a_bid.up[R, tp] = 0.0
        x.lo[exp, imp, tp] = 0.0
        x.up[exp, imp, tp] = 0.0
        x_dem.lo[R, tp] = 0.0
        x_dem.up[R, tp] = 0.0
        lam_var.lo[R, tp] = 0.0
        lam_var.up[R, tp] = 0.0
        mu_offer.lo[R, tp] = 0.0
        mu_offer.up[R, tp] = 0.0
        gamma.lo[exp, imp, tp] = 0.0
        gamma.up[exp, imp, tp] = 0.0
        beta_dem.lo[R, tp] = 0.0
        beta_dem.up[R, tp] = 0.0
        psi_dem.lo[R, tp] = 0.0
        psi_dem.up[R, tp] = 0.0

    # =====================================================================
    # Step 4 — LLP equations (time-indexed)
    # =====================================================================

    # --- Primal LLP Objective (kept for reporting) ---
    llp_gross_surplus = Sum(
        [R, T_op],
        a_bid[R, T_op] * x_dem[R, T_op]
        - (b_dem_t_p[R, T_op] / gp.Number(2.0))
        * x_dem[R, T_op]
        * x_dem[R, T_op],
    )

    llp_total_cost = (
        Sum(
            [exp, imp, T_op],
            (p_offer[exp, imp, T_op] + c_ship[exp, imp])
            * x[exp, imp, T_op],
        )
        + Sum(
            [exp, imp, T_op],
            (eps_x / gp.Number(2.0))
            * x[exp, imp, T_op]
            * x[exp, imp, T_op],
        )
    )

    eq_obj_llp = Equation(m, "eq_obj_llp")
    eq_obj_llp[...] = z_llp == llp_total_cost - llp_gross_surplus

    # --- Primal Constraints ---
    eq_bal = Equation(m, "eq_bal", domain=[imp, T_op])
    eq_bal[imp, T_op] = Sum(exp, x[exp, imp, T_op]) - x_dem[imp, T_op] == z

    eq_cap = Equation(m, "eq_cap", domain=[exp, T_op])
    eq_cap[exp, T_op] = Q_offer[exp, T_op] - Sum(imp, x[exp, imp, T_op]) >= z

    eq_q_offer_cap = Equation(m, "eq_q_offer_cap", domain=[R, T_op])
    if bool(settings.get("fix_q_offer_to_kcap", False)):
        eq_q_offer_cap[R, T_op] = Q_offer[R, T_op] == Kcap[R, T_op]
    else:
        eq_q_offer_cap[R, T_op] = Q_offer[R, T_op] <= Kcap[R, T_op]

    # --- Stationarity (KKT) ---
    eq_stat_x = Equation(m, "eq_stat_x", domain=[exp, imp, T_op])
    eq_stat_x[exp, imp, T_op] = (
        (p_offer[exp, imp, T_op] + c_ship[exp, imp])
        + eps_x * x[exp, imp, T_op]
        - lam_var[imp, T_op]
        + mu_offer[exp, T_op]
        - gamma[exp, imp, T_op]
        == z
    )

    eq_stat_dem = Equation(m, "eq_stat_dem", domain=[imp, T_op])
    eq_stat_dem[imp, T_op] = (
        -(a_bid[imp, T_op] - b_dem_t_p[imp, T_op] * x_dem[imp, T_op])
        + lam_var[imp, T_op]
        + beta_dem[imp, T_op]
        - psi_dem[imp, T_op]
        == z
    )

    # --- Complementarity (KKT) ---
    # Complementarity for offer-capacity constraint: mu_offer * (Q_offer - sum_imp x) = 0
    eq_comp_mu_offer = Equation(m, "eq_comp_mu_offer", domain=[exp, T_op])
    if eps_comp == 0.0:
        eq_comp_mu_offer[exp, T_op] = (
            mu_offer[exp, T_op]
            * (Q_offer[exp, T_op] - Sum(imp, x[exp, imp, T_op]))
            == z
        )
    else:
        eq_comp_mu_offer[exp, T_op] = (
            mu_offer[exp, T_op]
            * (Q_offer[exp, T_op] - Sum(imp, x[exp, imp, T_op]))
            <= eps_value
        )

    eq_comp_gamma = Equation(m, "eq_comp_gamma", domain=[exp, imp, T_op])
    if eps_comp == 0.0:
        eq_comp_gamma[exp, imp, T_op] = (
            gamma[exp, imp, T_op] * x[exp, imp, T_op] == z
        )
    else:
        eq_comp_gamma[exp, imp, T_op] = (
            gamma[exp, imp, T_op] * x[exp, imp, T_op] <= eps_value
        )

    eq_comp_beta_dem = Equation(m, "eq_comp_beta_dem", domain=[imp, T_op])
    if eps_comp == 0.0:
        eq_comp_beta_dem[imp, T_op] = (
            beta_dem[imp, T_op]
            * (Dmax_t_p[imp, T_op] - x_dem[imp, T_op])
            == z
        )
    else:
        eq_comp_beta_dem[imp, T_op] = (
            beta_dem[imp, T_op]
            * (Dmax_t_p[imp, T_op] - x_dem[imp, T_op])
            <= eps_value
        )

    eq_comp_psi_dem = Equation(m, "eq_comp_psi_dem", domain=[imp, T_op])
    if eps_comp == 0.0:
        eq_comp_psi_dem[imp, T_op] = (
            psi_dem[imp, T_op] * x_dem[imp, T_op] == z
        )
    else:
        eq_comp_psi_dem[imp, T_op] = (
            psi_dem[imp, T_op] * x_dem[imp, T_op] <= eps_value
        )

    # =====================================================================
    # Step 5 — Capacity transitions + offer linkage + rate limits
    # =====================================================================

    eq_kcap_init = Equation(m, "eq_kcap_init", domain=[R])
    eq_kcap_init[R] = Kcap[R, times[0]] == Kcap_init_p[R]

    eq_kcap_transitions: Dict[str, Equation] = {}
    eq_icap_ub: Dict[str, Equation] = {}
    eq_dcap_ub: Dict[str, Equation] = {}
    for tp, tp_next in transition_pairs:
        eq_kcap_trans = Equation(m, f"eq_kcap_trans_{tp_next}", domain=[R])
        eq_kcap_trans[R] = (
            Kcap[R, tp_next] == Kcap[R, tp] + ytn_p[tp] * (Icap_pos[R, tp] - Dcap_neg[R, tp])
        )
        eq_kcap_transitions[tp_next] = eq_kcap_trans

        eq_iub = Equation(m, f"eq_icap_ub_{tp}", domain=[R])
        if bool(getattr(data, "g_exp_ub_is_absolute", False)):
            eq_iub[R] = Icap_pos[R, tp] <= g_exp_p[R]
        else:
            eq_iub[R] = Icap_pos[R, tp] <= g_exp_p[R] * Kcap[R, tp]
        eq_icap_ub[tp] = eq_iub

        eq_dub = Equation(m, f"eq_dcap_ub_{tp}", domain=[R])
        eq_dub[R] = Dcap_neg[R, tp] <= g_dec_p[R] * Kcap[R, tp]
        eq_dcap_ub[tp] = eq_dub

    # Pin domestic self-offer to the exogenous LBD cost schedule
    eq_self_offer = Equation(m, "eq_self_offer", domain=[R, T_op])
    eq_self_offer[R, T_op] = (
        p_offer[R, R, T_op] == c_man_t_p[R, T_op]
    )

    # =====================================================================
    # Collect equations
    # =====================================================================
    equations = {
        "eq_bal": eq_bal,
        "eq_cap": eq_cap,
        "eq_q_offer_cap": eq_q_offer_cap,
        "eq_stat_x": eq_stat_x,
        "eq_stat_dem": eq_stat_dem,
        "eq_comp_mu_offer": eq_comp_mu_offer,
        "eq_comp_gamma": eq_comp_gamma,
        "eq_comp_beta_dem": eq_comp_beta_dem,
        "eq_comp_psi_dem": eq_comp_psi_dem,
        "eq_obj_llp": eq_obj_llp,
        "eq_kcap_init": eq_kcap_init,
        "eq_self_offer": eq_self_offer,
    }
    equations.update({f"eq_kcap_trans_{tp_next}": eq for tp_next, eq in eq_kcap_transitions.items()})
    equations.update({f"eq_icap_ub_{tp}": eq for tp, eq in eq_icap_ub.items()})
    equations.update({f"eq_dcap_ub_{tp}": eq for tp, eq in eq_dcap_ub.items()})

    # =====================================================================
    # Step 6 — Objective = sum over time
    # =====================================================================
    models: Dict[str, Model] = {}
    for rname in data.players:
        r = rname

        # ---- Per-period welfare components (operating periods only) ----

        d_surplus_t = Sum(
            T_op,
            beta_p[T_op] * ytn_p[T_op] * (
                a_dem_t_p[r, T_op] * x_dem[r, T_op]
                - (b_dem_t_p[r, T_op] / gp.Number(2.0))
                * x_dem[r, T_op]
                * x_dem[r, T_op]
                - lam_var[r, T_op] * x_dem[r, T_op]
            ),
        )

        # Producer term. The historical mode subtracts mu_offer; the comparison
        # mode omits that subtraction and retains the full market-price margin.
        producer_term_t = Sum(
            [j, T_op],
            beta_p[T_op] * ytn_p[T_op] * (
                lam_var[j, T_op]
                - (
                    mu_offer[r, T_op]
                    if subtract_mu_offer_from_producer_margin
                    else z
                )
                - c_man_t_p[r, T_op]
                - c_ship[r, j]
            ) * x[r, j, T_op],
        )

        capacity_cost_t = Sum(
            T_op,
            -beta_p[T_op] * ytn_p[T_op] * f_hold_p[r] * Kcap[r, T_op]
            - beta_p[T_op]
            * ytn_p[T_op]
            * c_inv_p[r]
            * Icap_pos[r, T_op],
        )

        capacity_policy_t = Sum(
            T_op,
            beta_p[T_op] * ytn_p[T_op] * cap_keep_reward * Kcap[r, T_op]
            + beta_p[T_op]
            * ytn_p[T_op]
            * capex_subsidy
            * Icap_pos[r, T_op]
            - beta_p[T_op]
            * ytn_p[T_op]
            * decommission_penalty
            * Dcap_neg[r, T_op],
        )

        terminal_salvage_credit = (
            terminal_salvage_discount
            * terminal_salvage_fraction
            * c_inv_p[r]
            * Kcap[r, times[-1]]
        )

        # ---- Penalties (replicated per period) ----
        # Economic quadratic penalty: penalise deviation from full capacity
        pen_quad_q = Sum(
            T_op,
            -gp.Number(0.5)
            * beta_p[T_op]
            * ytn_p[T_op]
            * c_quad_q
            * (Q_offer[r, T_op] - Kcap[r, T_op])
            * (Q_offer[r, T_op] - Kcap[r, T_op]),
        )

        # Penalise markup above manufacturing cost (overpricing penalty)
        pen_quad_p = Sum(
            T_op,
            -gp.Number(0.5) * beta_p[T_op] * ytn_p[T_op] * c_quad_p * Sum(
                j,
                (p_offer[r, j, T_op] - c_man_t_p[r, T_op])
                * (p_offer[r, j, T_op] - c_man_t_p[r, T_op]),
            ),
        )

        pen_quad_a = Sum(
            T_op,
            -gp.Number(0.5)
            * beta_p[T_op]
            * ytn_p[T_op]
            * c_quad_a
            * (a_dem_t_p[r, T_op] - a_bid[r, T_op])
            * (a_dem_t_p[r, T_op] - a_bid[r, T_op]),
        )

        # Algorithmic proximal penalties (solver stabilization)
        pen_prox_poffer = Sum(
            T_op,
            -gp.Number(0.5) * beta_p[T_op] * ytn_p[T_op] * c_pen_p * Sum(
                j,
                (p_offer[r, j, T_op] - p_offer_last[r, j, T_op])
                * (p_offer[r, j, T_op] - p_offer_last[r, j, T_op]),
            ),
        )

        pen_prox_q = Sum(
            T_op,
            -gp.Number(0.5) * beta_p[T_op] * ytn_p[T_op] * c_pen_q
            * (Q_offer[r, T_op] - Q_offer_last[r, T_op])
            * (Q_offer[r, T_op] - Q_offer_last[r, T_op]),
        )

        pen_prox_dk = Sum(
            T_op,
            -gp.Number(0.5) * beta_p[T_op] * ytn_p[T_op] * c_pen_dk
            * (
                (Icap_pos[r, T_op] - Icap_pos_last[r, T_op])
                * (Icap_pos[r, T_op] - Icap_pos_last[r, T_op])
                + (Dcap_neg[r, T_op] - Dcap_neg_last[r, T_op])
                * (Dcap_neg[r, T_op] - Dcap_neg_last[r, T_op])
            ),
        )

        pen_prox_a = Sum(
            T_op,
            -gp.Number(0.5) * beta_p[T_op] * ytn_p[T_op] * c_pen_a
            * (a_bid[r, T_op] - a_bid_last[r, T_op])
            * (a_bid[r, T_op] - a_bid_last[r, T_op]),
        )

        # ---- Assemble objective ----
        obj_welfare = (
            d_surplus_t
            + producer_term_t
            + capacity_cost_t
            + capacity_policy_t
            + terminal_salvage_credit
            + pen_quad_q
            + pen_quad_p
            + pen_quad_a
            + pen_prox_poffer
            + pen_prox_q
            + pen_prox_dk
            + pen_prox_a
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
        sets={"R": R, "exp": exp, "imp": imp, "j": j, "T": T, "T_op": T_op},
        params={
            "Dmax_t": Dmax_t_p,
            "a_dem_t": a_dem_t_p,
            "b_dem_t": b_dem_t_p,
            "Kcap_init": Kcap_init_p,
            "c_man_t": c_man_t_p,
            "c_ship": c_ship,
            "p_offer_ub": p_offer_ub_p,
            "g_exp": g_exp_p,
            "g_dec": g_dec_p,
            "f_hold": f_hold_p,
            "c_inv": c_inv_p,
            "beta_t": beta_p,
            "ytn": ytn_p,
            "p_offer_last":  p_offer_last,
            "Q_offer_last":  Q_offer_last,
            "Icap_pos_last": Icap_pos_last,
            "Dcap_neg_last": Dcap_neg_last,
            "a_bid_last":    a_bid_last,
            "c_pen_q_scalar":  c_pen_q,
            "c_pen_p_scalar":  c_pen_p,
            "c_pen_a_scalar":  c_pen_a,
            "c_pen_dk_scalar": c_pen_dk,
        },
        vars={
            "Kcap": Kcap,
            "Icap_pos": Icap_pos,
            "Dcap_neg": Dcap_neg,
            "Q_offer": Q_offer,
            "p_offer": p_offer,
            "a_bid": a_bid,
            "x": x,
            "x_dem": x_dem,
            "lam": lam_var,
            "mu_offer": mu_offer,
            "gamma": gamma,
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
    theta_dK_net: Dict[Tuple[str, str], float],
    theta_p_offer: Dict[Tuple[str, str, str], float],
    theta_a_bid: Dict[Tuple[str, str], float],
    *,
    player: str,
    theta_Kcap: Dict[Tuple[str, str], float] | None = None,
) -> None:
    """Fix all other players' strategies; free current player's strategies."""
    times = data.times or list(_DEFAULT_TIMES)
    operating_times = _operating_times(data)
    operating_time_set = set(operating_times)
    move_times = _move_times(times)
    # Use actual solved Kcap if provided, otherwise fall back to reconstructing from dK_net
    if theta_Kcap is not None:
        implied_kcap = theta_Kcap
    else:
        implied_kcap = _implied_capacity_path(data, times, theta_dK_net)
    non_strategic_regions = _non_strategic_regions(data)
    fix_a_bid = _fix_a_bid_to_true_dem(data)

    Kcap = ctx.vars["Kcap"]
    Icap_pos = ctx.vars["Icap_pos"]
    Dcap_neg = ctx.vars["Dcap_neg"]
    Q_offer = ctx.vars["Q_offer"]
    p_offer = ctx.vars["p_offer"]
    a_bid = ctx.vars["a_bid"]

    for r in data.regions:
        for tp in times:
            a_true = _true_demand_intercept(data, r, tp)
            Kcap.lo[r, tp] = 0.0
            Kcap.up[r, tp] = float("inf")
            if tp not in operating_time_set:
                # The terminal label is a capacity-state date only.  Its Kcap
                # remains endogenous through the final transition, while every
                # market/offer coordinate is inactive and fixed to zero.
                Icap_pos.lo[r, tp] = 0.0
                Icap_pos.up[r, tp] = 0.0
                Dcap_neg.lo[r, tp] = 0.0
                Dcap_neg.up[r, tp] = 0.0
                Q_offer.lo[r, tp] = 0.0
                Q_offer.up[r, tp] = 0.0
                a_bid.lo[r, tp] = 0.0
                a_bid.up[r, tp] = 0.0
                continue
            if r == player:
                # Active player: Icap_pos and Dcap_neg are free decision variables.
                # IMPORTANT: set explicit .up from data bounds so that ipopt
                # respects the capacity-rate limits even on early termination
                # (equation constraints alone are insufficient for MPECs).
                if tp not in move_times:
                    icap_ub = 0.0
                    dcap_ub = 0.0
                else:
                    g_exp = float((data.g_exp_ub or {}).get(r, 0.0))
                    if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
                        g_exp *= max(float(implied_kcap.get((r, tp), 0.0)), 0.0)
                    icap_ub = g_exp
                    g_dec = float((data.g_dec_ub or {}).get(r, 1.0))
                    dcap_ub = g_dec * max(float(implied_kcap.get((r, tp), 0.0)), 0.0)
                Icap_pos.lo[r, tp] = 0.0
                Icap_pos.up[r, tp] = icap_ub
                Dcap_neg.lo[r, tp] = 0.0
                Dcap_neg.up[r, tp] = dcap_ub
                # Active player: eq_q_offer_cap enforces Q_offer <= Kcap structurally.
                # Do NOT set Q_offer.lo to implied_kcap here — Kcap is endogenous
                # for the active player, so a fixed lower bound can conflict with
                # the optimised Kcap and cause infeasibility.
                Q_offer.lo[r, tp] = 0.0
                Q_offer.up[r, tp] = float("inf")

                if fix_a_bid:
                    a_bid.lo[r, tp] = a_true
                    a_bid.up[r, tp] = a_true
                else:
                    a_bid.lo[r, tp] = 0.0
                    a_bid.up[r, tp] = a_true

            elif r in data.players:
                # Other strategic players: fix gross invest/decommission from GS iterate.
                d_val = float(theta_dK_net.get((r, tp), 0.0)) if tp in move_times else 0.0
                i_fix = max(d_val, 0.0)
                d_fix = max(-d_val, 0.0)
                Icap_pos.lo[r, tp] = i_fix
                Icap_pos.up[r, tp] = i_fix
                Dcap_neg.lo[r, tp] = d_fix
                Dcap_neg.up[r, tp] = d_fix

                k_implied = max(float(implied_kcap.get((r, tp), 0.0)), 0.0)
                if _fix_q_offer_to_kcap(data):
                    q_val = k_implied
                else:
                    q_val = _clip_value(float(theta_Q.get((r, tp), 0.0)), 0.0, k_implied)
                Q_offer.lo[r, tp] = q_val
                Q_offer.up[r, tp] = q_val

                a_val = a_true if fix_a_bid else _clip_value(float(theta_a_bid.get((r, tp), a_true)), 0.0, a_true)
                a_bid.lo[r, tp] = a_val
                a_bid.up[r, tp] = a_val
            else:
                # Non-strategic regions: no capacity changes.
                Icap_pos.lo[r, tp] = 0.0
                Icap_pos.up[r, tp] = 0.0
                Dcap_neg.lo[r, tp] = 0.0
                Dcap_neg.up[r, tp] = 0.0
                v = max(float(implied_kcap.get((r, tp), 0.0)), 0.0)
                Q_offer.lo[r, tp] = v
                Q_offer.up[r, tp] = v

                a_bid.lo[r, tp] = a_true
                a_bid.up[r, tp] = a_true

    # -- Offer Prices --
    # Active player: export routes p_offer[player, j!=player, T] are free;
    #   domestic route p_offer[player, player, T] is pinned to c_man_t_p[player,T]
    #   by eq_self_offer, so we set bounds wide enough to not conflict.
    # Other players: ALL p_offer[other, j, T] are fixed to GS iterate,
    #   with domestic route fixed to the exogenous schedule c_man_t to guarantee
    #   self-offer == manufacturing cost exactly.
    c_man_t_src = data.c_man_t or {}
    for ex in data.regions:
        for im in data.regions:
            for tp in operating_times:
                ub = float(data.p_offer_ub[(ex, im)])
                if ex == player:
                    if im == player:
                        # Domestic: eq_self_offer pins this to c_man_t_p[player,T].
                        # Set bounds wide enough so the equality is feasible.
                        p_offer.lo[ex, im, tp] = 0.0
                        p_offer.up[ex, im, tp] = max(ub, float(data.c_man.get(ex, 0.0)) * 2.0)
                    else:
                        # Export routes: free strategic choice
                        p_offer.lo[ex, im, tp] = 0.0
                        p_offer.up[ex, im, tp] = ub
                else:
                    if im == ex:
                        # Other player's domestic route: fix to their exogenous LBD cost.
                        cv = float(c_man_t_src.get((ex, tp), data.c_man.get(ex, 0.0)))
                        p_offer.lo[ex, im, tp] = cv
                        p_offer.up[ex, im, tp] = cv
                    else:
                        # Other player's export routes: fix to iterate
                        v = _clip_value(float(theta_p_offer.get((ex, im, tp), 0.0)), 0.0, ub)
                        p_offer.lo[ex, im, tp] = v
                        p_offer.up[ex, im, tp] = v

    # -- Route-level sanity bounds on trade flows x --
    # Bound each bilateral flow by the importer's demand ceiling only. Do not
    # use per-route variable bounds as a surrogate for the exporter aggregate
    # limit; eq_cap continues to enforce that structurally.
    x = ctx.vars["x"]
    if data.Dmax_t is not None:
        importer_caps = {
            tp: {im: max(float(data.Dmax_t.get((im, tp), 0.0)), 0.0) for im in data.regions}
            for tp in times
        }
    else:
        importer_caps = {
            tp: {im: max(float(data.Dmax.get(im, 0.0)), 0.0) for im in data.regions}
            for tp in times
        }
    for ex in data.regions:
        for tp in operating_times:
            for im in data.regions:
                x.up[ex, im, tp] = importer_caps[tp][im]



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

    # Derive dK_net = Icap_pos - Dcap_neg (no longer a GAMS variable)
    icap_d = _maybe_var("Icap_pos")
    dcap_d = _maybe_var("Dcap_neg")
    dK_net_d = {k: float(icap_d.get(k, 0.0)) - float(dcap_d.get(k, 0.0)) for k in set(icap_d) | set(dcap_d)}

    return {
        "Kcap": _maybe_var("Kcap"),
        "dK_net": dK_net_d,
        "Icap_pos": icap_d,
        "Dcap_neg": dcap_d,
        "Q_offer": _maybe_var("Q_offer"),
        "p_offer": _maybe_var("p_offer"),
        "a_bid": _maybe_var("a_bid"),
        "x": _maybe_var("x"),
        "x_dem": _maybe_var("x_dem"),
        "lam": _maybe_var("lam"),
        "mu_offer": _maybe_var("mu_offer"),
        "gamma": _maybe_var("gamma"),
        "beta_dem": _maybe_var("beta_dem"),
        "psi_dem": _maybe_var("psi_dem"),
        "obj": obj_values,
    }

