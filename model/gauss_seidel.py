from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

try:
    from . import model_main as _it
except ImportError:
    import model_main as _it


def solve_gs_intertemporal(
    data: "_it.ModelData",
    *,
    iters: int = 50,
    omega: float = 0.8,
    tol_rel: float = 1e-4,
    stable_iters: int = 3,
    solver: str = "conopt",
    solver_options: Dict[str, float] | None = None,
    working_directory: str | None = None,
    iter_callback: Callable[[int, Dict[str, Dict], float, int], None] | None = None,
    initial_state: Dict[str, Dict] | None = None,
    convergence_mode: str = "strategy",
    tol_obj: float = 1e-6,
    shuffle_players: bool = False,
    player_order: List[str] | None = None,
    force_ch_last: bool = True,
    exclude_terminal_from_convergence: bool = False,
) -> tuple[Dict[str, Dict], List[Dict[str, object]]]:
    """Gauss-Seidel solver for the 4-period intertemporal EPEC Offer model.

    Theta dicts are keyed by (region, time) for scalar strategies and by
    (exp, imp, time) for offer prices.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1].")
    if tol_rel <= 0.0:
        raise ValueError("tol_rel must be > 0")
    if stable_iters < 1:
        raise ValueError("stable_iters must be >= 1")

    if player_order is not None:
        normalized = [str(p).strip().lower() for p in player_order if str(p).strip()]
        unknown = sorted(set(normalized) - set(data.players))
        missing = [p for p in data.players if p not in normalized]
        duplicates = sorted({p for p in normalized if normalized.count(p) > 1})
        if unknown:
            raise ValueError(f"player_order contains unknown players: {unknown}. Valid players: {data.players}")
        if missing:
            raise ValueError(f"player_order missing players: {missing}. Valid players: {data.players}")
        if duplicates:
            raise ValueError(f"player_order contains duplicates: {duplicates}")
        base_order = normalized
    else:
        base_order = list(data.players)

    times: List[str] = data.times or ["2025", "2030", "2035", "2040", "2045"]
    move_times = _it._move_times(times)
    init_kcap = _it._initial_capacity_by_region(data)
    fix_a_bid = _it._fix_a_bid_to_true_dem(data)

    ctx = _it.build_model(data, working_directory=working_directory)

    # ---- Initialize theta dicts ----
    if initial_state:
        raw_theta_dK_net: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("dK_net", {}).get((r, tp), 0.0))
            for r in data.players for tp in move_times
        }
        _it.validate_strategy_inputs(
            data,
            {(r, tp): float(initial_state.get("Q_offer", {}).get((r, tp), 0.8 * float(init_kcap[r]))) for r in data.players for tp in times},
            None if fix_a_bid else {(r, tp): float(initial_state.get("a_bid", {}).get((r, tp), _it._true_demand_intercept(data, r, tp))) for r in data.players for tp in times},
            raw_theta_dK_net,
        )
        theta_dK_net: Dict[Tuple[str, str], float] = {
            (r, tp): float(raw_theta_dK_net[(r, tp)])
            for r in data.players for tp in move_times
        }
        implied_kcap = _it._implied_capacity_path(data, times, theta_dK_net)
        theta_Kcap: Dict[Tuple[str, str], float] = dict(implied_kcap)
        raw_theta_Q: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("Q_offer", {}).get((r, tp), 0.8 * float(init_kcap[r])))
            for r in data.players for tp in times
        }
        theta_Q: Dict[Tuple[str, str], float] = {
            (r, tp): _it._clip_value(raw_theta_Q[(r, tp)], 0.0, max(float(implied_kcap[(r, tp)]), 0.0))
            for r in data.players for tp in times
        }
        theta_p_offer: Dict[Tuple[str, str, str], float] = {
            (ex, im, tp): float(initial_state.get("p_offer", {}).get((ex, im, tp), 0.0))
            for ex in data.regions for im in data.regions for tp in times
        }
        raw_theta_a_bid: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("a_bid", {}).get((r, tp), _it._true_demand_intercept(data, r, tp)))
            for r in data.players for tp in times
        }
        theta_a_bid: Dict[Tuple[str, str], float] = {
            (r, tp): _it._true_demand_intercept(data, r, tp) if fix_a_bid
            else _it._clip_value(raw_theta_a_bid[(r, tp)], 0.0, _it._true_demand_intercept(data, r, tp))
            for r in data.players for tp in times
        }
    else:
        theta_dK_net = {(r, tp): 0.0 for r in data.players for tp in move_times}
        implied_kcap = _it._implied_capacity_path(data, times, theta_dK_net)
        theta_Kcap = dict(implied_kcap)
        theta_Q = {(r, tp): 0.8 * float(implied_kcap[(r, tp)]) for r in data.players for tp in times}
        theta_p_offer = {(ex, im, tp): 0.5 * float(data.p_offer_ub[(ex, im)]) for ex in data.regions for im in data.regions for tp in times}
        theta_a_bid = {(r, tp): _it._true_demand_intercept(data, r, tp) for r in data.players for tp in times}

    theta_obj: Dict[str, float] = {r: 0.0 for r in data.players}

    # ---- Warm-start GAMS variable levels from theta dicts ----
    # Without this, ipopt starts all POSITIVE variables at level 0, which
    # traps the MPEC solver in a no-investment local optimum.
    _Kcap_var = ctx.vars.get("Kcap")
    _Icap_var = ctx.vars.get("Icap_pos")
    _Dcap_var = ctx.vars.get("Dcap_neg")
    _Q_var    = ctx.vars.get("Q_offer")
    _p_var    = ctx.vars.get("p_offer")
    _a_var    = ctx.vars.get("a_bid")

    if _Kcap_var is not None:
        for (r, tp), v in theta_Kcap.items():
            _Kcap_var.l[r, tp] = max(v, 0.0)
    if _Icap_var is not None and _Dcap_var is not None:
        for r in data.players:
            for tp in move_times:
                d_val = float(theta_dK_net.get((r, tp), 0.0))
                _Icap_var.l[r, tp] = max(d_val, 0.0)
                _Dcap_var.l[r, tp] = max(-d_val, 0.0)
    if _Q_var is not None:
        for (r, tp), v in theta_Q.items():
            _Q_var.l[r, tp] = v
    if _p_var is not None:
        for (ex, im, tp), v in theta_p_offer.items():
            _p_var.l[ex, im, tp] = v
    if _a_var is not None:
        for (r, tp), v in theta_a_bid.items():
            _a_var.l[r, tp] = v

    def _scaled_change(new: float, old: float, scale: float) -> float:
        return abs(new - old) / max(scale, 1e-12)

    def _q_scale(r: str) -> float:
        return max(float(init_kcap.get(r, 0.0)), 1.0)

    def _dk_scale(r: str) -> float:
        exp_scale = float((data.g_exp_ub or {}).get(r, 0.0))
        if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
            exp_scale *= float(init_kcap.get(r, 0.0))
        dec_scale = float((data.g_dec_ub or {}).get(r, 0.0)) * float(init_kcap.get(r, 0.0))
        return max(exp_scale, dec_scale, 1.0)

    def _p_scale(ex: str, im: str) -> float:
        return max(float(data.p_offer_ub[(ex, im)]), 1e-3)

    iter_rows: List[Dict[str, object]] = []
    stable_count = 0
    last_state: Dict[str, Dict] = {}

    solve_kwargs: Dict[str, object] = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options

    def _update_prox_reference() -> None:
        # p_offer_last
        poffer_last = ctx.params.get("p_offer_last")
        if poffer_last is not None:
            for ex in data.regions:
                for im in data.regions:
                    for tp in times:
                        poffer_last[ex, im, tp] = float(theta_p_offer.get((ex, im, tp), 0.0))

        # Q_offer_last
        q_last = ctx.params.get("Q_offer_last")
        if q_last is not None:
            for r in data.regions:
                for tp in times:
                    q_last[r, tp] = float(theta_Q.get((r, tp), 0.0))

        # Icap_pos_last / Dcap_neg_last — derived from theta_dK_net
        icap_last = ctx.params.get("Icap_pos_last")
        dcap_last = ctx.params.get("Dcap_neg_last")
        if icap_last is not None and dcap_last is not None:
            for r in data.players:
                for tp in move_times:
                    d_val = float(theta_dK_net.get((r, tp), 0.0))
                    icap_last[r, tp] = max(d_val, 0.0)
                    dcap_last[r, tp] = max(-d_val, 0.0)

        # a_bid_last
        a_last = ctx.params.get("a_bid_last")
        if a_last is not None:
            for r in data.regions:
                for tp in times:
                    a_last[r, tp] = float(
                        theta_a_bid.get((r, tp), _it._true_demand_intercept(data, r, tp))
                    )

    for it in range(1, iters + 1):
        r_strat = 0.0

        prev_Q = dict(theta_Q)
        prev_dK_net = dict(theta_dK_net)
        prev_Kcap = dict(theta_Kcap)
        prev_poffer = dict(theta_p_offer)
        prev_a_bid = dict(theta_a_bid)
        prev_obj = dict(theta_obj)

        sweep_order = list(base_order)
        if shuffle_players:
            random.shuffle(sweep_order)

        # Optional legacy behavior to keep China last.
        if force_ch_last and "ch" in sweep_order:
            sweep_order.remove("ch")
            sweep_order.append("ch")

        for p in sweep_order:
            _update_prox_reference()
            _it.apply_player_fixings(
                ctx, data,
                theta_Q, theta_dK_net, theta_p_offer,
                theta_a_bid,
                player=p,
                theta_Kcap=theta_Kcap,
            )
            ctx.models[p].solve(**solve_kwargs)

            state = _it.extract_state(ctx)
            last_state = state

            dK_net_sol = state.get("dK_net", {})
            Q_sol = state.get("Q_offer", {})
            Kcap_sol = state.get("Kcap", {})
            poffer_sol = state.get("p_offer", {})
            a_bid_sol = state.get("a_bid", {})
            obj_sol = state.get("obj", {})

            # Update net capacity changes.
            for tp in move_times:
                key = (p, tp)
                if key in dK_net_sol:
                    br = float(dK_net_sol[key])
                    theta_dK_net[key] = (1.0 - omega) * theta_dK_net[key] + omega * br

            # Recompute theta_Kcap from the damped theta_dK_net so that the
            # capacity path and the net-change rates stay consistent.
            theta_Kcap.update(_it._implied_capacity_path(data, times, theta_dK_net))

            # Update Q_offer — clip against the solved Kcap.
            # When fix_q_offer_to_kcap is active, pin theta_Q == theta_Kcap so
            # that Q_offer_last (used in the proximal penalty) stays consistent
            # with the bound that apply_player_fixings imposes on fixed players.
            # Omega-damping is applied in both branches so all strategies converge
            # at the same rate; without damping, theta_Q could jump by the full
            # step while theta_dK_net (and thus theta_Kcap) are only moved by omega.
            fix_q = _it._fix_q_offer_to_kcap(data)
            for tp in times:
                key = (p, tp)
                solved_kcap = float(theta_Kcap.get(key, 0.0))
                if fix_q:
                    theta_Q[key] = (1.0 - omega) * theta_Q[key] + omega * solved_kcap
                elif key in Q_sol:
                    br = _it._clip_value(float(Q_sol[key]), 0.0, max(solved_kcap, 0.0))
                    theta_Q[key] = (1.0 - omega) * theta_Q[key] + omega * br

            # Update p_offer
            for im in data.regions:
                for tp in times:
                    key = (p, im, tp)
                    if key in poffer_sol:
                        br = float(poffer_sol[key])
                        theta_p_offer[key] = (1.0 - omega) * theta_p_offer[key] + omega * br

            # Update a_bid
            if not fix_a_bid:
                for tp in times:
                    sk = (p, tp)
                    if sk in a_bid_sol:
                        theta_a_bid[sk] = (1.0 - omega) * theta_a_bid[sk] + omega * float(a_bid_sol[sk])

            if isinstance(obj_sol, dict):
                theta_obj[p] = float(obj_sol.get(p, 0.0))

        # ---- Convergence metrics ----
        # theta_Kcap is fully determined by theta_dK_net (derived, not independent),
        # so it is excluded from r_strat to avoid double-counting capacity changes.
        # theta_Q is the independent offer strategy; it is always included.
        # When exclude_terminal_from_convergence=True the terminal buffer period
        # (times[-1], e.g. 2045) is dropped from the times loop, and the last
        # move_time (e.g. the 2040→2045 transition) is dropped from move_times.
        conv_times = times[:-1] if exclude_terminal_from_convergence and len(times) > 1 else times
        conv_move_times = move_times[:-1] if exclude_terminal_from_convergence and len(move_times) > 1 else move_times

        for r in data.players:
            for tp in conv_move_times:
                r_strat = max(r_strat, _scaled_change(theta_dK_net[(r, tp)], prev_dK_net[(r, tp)], _dk_scale(r)))
            for tp in conv_times:
                r_strat = max(r_strat, _scaled_change(theta_Q[(r, tp)], prev_Q[(r, tp)], _q_scale(r)))

                if not fix_a_bid:
                    a_scale = _it._true_demand_intercept(data, r, tp)
                    r_strat = max(r_strat, _scaled_change(theta_a_bid[(r, tp)], prev_a_bid[(r, tp)], a_scale))

        for ex in data.regions:
            for im in data.regions:
                for tp in conv_times:
                    key = (ex, im, tp)
                    r_strat = max(r_strat, _scaled_change(theta_p_offer[key], prev_poffer[key], _p_scale(ex, im)))

        r_obj = 0.0
        for r in data.players:
            r_obj = max(r_obj, _scaled_change(theta_obj.get(r, 0.0), prev_obj.get(r, 0.0), 1000.0))

        metric_met = False
        if convergence_mode == "combined":
            metric_met = (r_strat <= tol_rel) and (r_obj <= tol_obj)
        elif convergence_mode == "objective":
            metric_met = r_obj <= tol_obj
        else:  # "strategy"
            metric_met = r_strat <= tol_rel

        stable_count = stable_count + 1 if metric_met else 0
        row_data: Dict[str, object] = {
            "iter": it,
            "r_strat": float(r_strat),
            "r_obj": float(r_obj),
            "stable_count": int(stable_count),
            "omega": float(omega),
        }
        if shuffle_players:
            row_data["sweep_order"] = list(sweep_order)
        iter_rows.append(row_data)

        if iter_callback is not None:
            if shuffle_players:
                last_state["_sweep_order"] = list(sweep_order)
            iter_callback(it, last_state, float(r_strat), int(stable_count))
            last_state.pop("_sweep_order", None)

        if stable_count >= stable_iters:
            break

    return last_state, iter_rows
