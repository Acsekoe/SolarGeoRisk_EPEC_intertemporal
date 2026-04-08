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
    adaptive_omega: bool = True,
    omega_min: float = 0.2,
    omega_aggressive_sweeps: int = 3,
    omega_ramp_iters: int = 10,
    tol_rel: float = 1e-4,
    stable_iters: int = 3,
    solver: str = "conopt",
    solver_options: Dict[str, float] | None = None,
    working_directory: str | None = None,
    iter_callback: Callable[[int, Dict[str, Dict], float, int], None] | None = None,
    initial_state: Dict[str, Dict] | None = None,
    convergence_mode: str = "absolute",
    tol_obj: float = 1e-6,
    shuffle_players: bool = False,
    player_order: List[str] | None = None,
    force_ch_last: bool = True,
    exclude_terminal_from_convergence: bool = True,
    tol_p_abs: float = 1.0,
    tol_dk_abs: float = 0.1,
    c_pen_q_mid: float | None = None,
    c_pen_p_mid: float | None = None,
    c_pen_a_mid: float | None = None,
    c_pen_dk_mid: float | None = None,
    c_pen_q_final: float | None = None,
    c_pen_p_final: float | None = None,
    c_pen_a_final: float | None = None,
    c_pen_dk_final: float | None = None,
    c_pen_ramp_iters: int = 10,
) -> tuple[Dict[str, Dict], List[Dict[str, object]]]:
    """Gauss-Seidel solver for the 4-period intertemporal EPEC Offer model.

    Theta dicts are keyed by (region, time) for scalar strategies and by
    (exp, imp, time) for offer prices.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if not (0.0 < omega <= 1.0):
        raise ValueError("omega must be in (0, 1].")
    if not (0.0 < omega_min <= 1.0):
        raise ValueError("omega_min must be in (0, 1].")
    if omega_min > omega:
        raise ValueError("omega_min must be <= omega.")
    if omega_aggressive_sweeps < 0:
        raise ValueError("omega_aggressive_sweeps must be >= 0.")
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
    omega_current = float(omega)
    prev_metrics: Dict[str, float] | None = None

    solve_kwargs: Dict[str, object] = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options

    # Penalty annealing setup: read initial values from ctx, compute schedule.
    penalty_schedule: Dict[str, Dict[str, object]] = {}
    for key, param_name, mid_value, final_value in (
        ("q", "c_pen_q_scalar", c_pen_q_mid, c_pen_q_final),
        ("p", "c_pen_p_scalar", c_pen_p_mid, c_pen_p_final),
        ("a", "c_pen_a_scalar", c_pen_a_mid, c_pen_a_final),
        ("dk", "c_pen_dk_scalar", c_pen_dk_mid, c_pen_dk_final),
    ):
        param = ctx.params.get(param_name)
        start = float(param.records.iloc[0, 0]) if param is not None else 0.0
        penalty_schedule[key] = {
            "param": param,
            "start": start,
            "mid": float(mid_value) if mid_value is not None else None,
            "end": float(final_value) if final_value is not None else start,
            "do_mid": (mid_value is not None) and (param is not None),
            "do_ramp": (final_value is not None) and (param is not None),
        }

    def _scheduled_penalty(
        start: float,
        mid: float | None,
        end: float,
        do_mid: bool,
        do_ramp: bool,
        it: int,
    ) -> float:
        """Piecewise-linear schedule from start -> mid -> end across GS sweeps."""
        if not do_mid and not do_ramp:
            return start
        if do_mid and not do_ramp:
            end = float(mid if mid is not None else start)
        if c_pen_ramp_iters <= 1:
            return end
        if do_mid and mid is not None and c_pen_ramp_iters >= 3:
            mid_iter = max(2, (c_pen_ramp_iters + 1) // 2)
            if it <= mid_iter:
                frac = (it - 1) / max(mid_iter - 1, 1)
                return start + frac * (mid - start)
            frac = min((it - mid_iter) / max(c_pen_ramp_iters - mid_iter, 1), 1.0)
            return mid + frac * (end - mid)
        frac = min((it - 1) / (c_pen_ramp_iters - 1), 1.0)
        return start + frac * (end - start)

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

    def _scheduled_omega(it: int) -> float:
        if not adaptive_omega:
            return float(omega)
        if it <= omega_aggressive_sweeps:
            return float(omega)
        if omega_ramp_iters <= 1:
            return float(omega_min)
        ramp_it = it - omega_aggressive_sweeps
        frac = min((ramp_it - 1) / (omega_ramp_iters - 1), 1.0)
        return float(omega + frac * (omega_min - omega))

    for it in range(1, iters + 1):
        r_strat = 0.0
        omega_it = float(omega_current)

        # Update penalty annealing
        c_pen_current: Dict[str, float] = {}
        for key, spec in penalty_schedule.items():
            current = _scheduled_penalty(
                float(spec["start"]),
                None if spec["mid"] is None else float(spec["mid"]),
                float(spec["end"]),
                bool(spec["do_mid"]),
                bool(spec["do_ramp"]),
                it,
            )
            c_pen_current[key] = current
            param = spec["param"]
            if bool(spec["do_ramp"]) and param is not None:
                param.setRecords(current)

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

            _ss = getattr(ctx.models[p], "solve_status", None)
            _ms = getattr(ctx.models[p], "model_status", None)
            _ss_str = str(_ss).lower().replace(" ", "") if _ss is not None else ""
            if _ss is not None and "normal" not in _ss_str and "1" != _ss_str:
                print(f"  WARNING: solver status for {p} at iter {it}: solve_status={_ss}, model_status={_ms}")

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
                    theta_dK_net[key] = (1.0 - omega_it) * theta_dK_net[key] + omega_it * br

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
                    theta_Q[key] = (1.0 - omega_it) * theta_Q[key] + omega_it * solved_kcap
                elif key in Q_sol:
                    br = _it._clip_value(float(Q_sol[key]), 0.0, max(solved_kcap, 0.0))
                    theta_Q[key] = (1.0 - omega_it) * theta_Q[key] + omega_it * br
                # Clip to ensure theta_Q never exceeds theta_Kcap after damping
                theta_Q[key] = min(theta_Q[key], max(solved_kcap, 0.0))

            # Update p_offer
            for im in data.regions:
                for tp in times:
                    key = (p, im, tp)
                    if key in poffer_sol:
                        br = float(poffer_sol[key])
                        theta_p_offer[key] = (1.0 - omega_it) * theta_p_offer[key] + omega_it * br

            # Update a_bid
            if not fix_a_bid:
                for tp in times:
                    sk = (p, tp)
                    if sk in a_bid_sol:
                        theta_a_bid[sk] = (1.0 - omega_it) * theta_a_bid[sk] + omega_it * float(a_bid_sol[sk])

            if isinstance(obj_sol, dict):
                theta_obj[p] = float(obj_sol.get(p, 0.0))

        # ---- Convergence metrics ----
        # When exclude_terminal_from_convergence=True:
        #  - 2045 is dropped from conv_times (excludes Q_offer, p_offer, a_bid at 2045)
        #  - 2040 is dropped from conv_move_times (excludes dK_net at 2040, i.e. the 2040→2045 transition)
        # Result: convergence only checks 2025-2040 for prices/quantities, 2025-2035 for capacity changes.
        conv_times = times[:-1] if exclude_terminal_from_convergence and len(times) > 1 else times
        conv_move_times = move_times[:-1] if exclude_terminal_from_convergence and len(move_times) > 1 else move_times

        # Track absolute changes for "absolute" convergence mode
        max_abs_dp  = 0.0   # max |Δp_offer|  across all arcs/periods  [USD/kW]
        max_abs_ddk = 0.0   # max |ΔdK_net|   across all players/periods [GW/yr]

        # Per-variable diagnostics: collect the worst offenders each sweep
        _diag_dk: List[Tuple[str, str, float, float]]  = []   # (r, tp, old, new)
        _diag_q:  List[Tuple[str, str, float, float]]  = []
        _diag_p:  List[Tuple[str, str, str, float, float]] = []  # (ex, im, tp, old, new)

        for r in data.players:
            for tp in conv_move_times:
                old_dk, new_dk = prev_dK_net[(r, tp)], theta_dK_net[(r, tp)]
                abs_dk = abs(new_dk - old_dk)
                max_abs_ddk = max(max_abs_ddk, abs_dk)
                r_strat = max(r_strat, _scaled_change(new_dk, old_dk, _dk_scale(r)))
                _diag_dk.append((r, tp, old_dk, new_dk))
            for tp in conv_times:
                old_q, new_q = prev_Q[(r, tp)], theta_Q[(r, tp)]
                r_strat = max(r_strat, _scaled_change(new_q, old_q, _q_scale(r)))
                _diag_q.append((r, tp, old_q, new_q))

                if not fix_a_bid:
                    a_scale = _it._true_demand_intercept(data, r, tp)
                    r_strat = max(r_strat, _scaled_change(theta_a_bid[(r, tp)], prev_a_bid[(r, tp)], a_scale))

        for ex in data.regions:
            for im in data.regions:
                for tp in conv_times:
                    key = (ex, im, tp)
                    old_p, new_p = prev_poffer[key], theta_p_offer[key]
                    abs_dp = abs(new_p - old_p)
                    max_abs_dp = max(max_abs_dp, abs_dp)
                    r_strat = max(r_strat, _scaled_change(new_p, old_p, _p_scale(ex, im)))
                    _diag_p.append((ex, im, tp, old_p, new_p))

        # Print top movers for this sweep
        _diag_dk.sort(key=lambda x: abs(x[3] - x[2]), reverse=True)
        _diag_q.sort(key=lambda x: abs(x[3] - x[2]), reverse=True)
        _diag_p.sort(key=lambda x: abs(x[4] - x[3]), reverse=True)
        print(f"  [iter {it}] top dK_net changes:")
        for r, tp, old, new in _diag_dk[:3]:
            print(f"    {r}/{tp}: {old:+.3f} -> {new:+.3f}  (d={new-old:+.3f})")
        print(f"  [iter {it}] top Q_offer changes:")
        for r, tp, old, new in _diag_q[:3]:
            print(f"    {r}/{tp}: {old:.2f} -> {new:.2f}  (d={new-old:+.2f})")
        print(f"  [iter {it}] top p_offer changes:")
        for ex, im, tp, old, new in _diag_p[:5]:
            print(f"    {ex}->{im}/{tp}: {old:.2f} -> {new:.2f}  (d={new-old:+.2f})")

        r_obj = 0.0
        for r in data.players:
            r_obj = max(r_obj, _scaled_change(theta_obj.get(r, 0.0), prev_obj.get(r, 0.0), 1000.0))

        metric_met = False
        if convergence_mode == "combined":
            metric_met = (r_strat <= tol_rel) and (r_obj <= tol_obj)
        elif convergence_mode == "objective":
            metric_met = r_obj <= tol_obj
        elif convergence_mode == "absolute":
            metric_met = (max_abs_dp <= tol_p_abs) and (max_abs_ddk <= tol_dk_abs)
        else:  # "strategy"
            metric_met = r_strat <= tol_rel

        stable_count = stable_count + 1 if metric_met else 0

        omega_next = _scheduled_omega(it + 1)
        omega_reason_parts: List[str] = []
        if adaptive_omega and prev_metrics is not None and it >= omega_aggressive_sweeps:
            if max_abs_ddk > max(1.5 * prev_metrics["max_abs_ddk"], 2.5 * tol_dk_abs):
                omega_next = min(omega_next, max(float(omega_min), 0.75 * omega_it))
                omega_reason_parts.append("dK spike")
            if convergence_mode in {"strategy", "combined"} and r_strat > 1.2 * prev_metrics["r_strat"]:
                omega_next = min(omega_next, max(float(omega_min), 0.8 * omega_it))
                omega_reason_parts.append("strategy residual worsened")
            if convergence_mode in {"strategy", "combined", "absolute"} and max_abs_dp > max(1.5 * prev_metrics["max_abs_dp"], 2.5 * tol_p_abs):
                omega_next = min(omega_next, max(float(omega_min), 0.75 * omega_it))
                omega_reason_parts.append("price spike")
            if (it - omega_aggressive_sweeps) >= omega_ramp_iters and stable_count == 0 and max_abs_ddk > 5.0 * tol_dk_abs:
                omega_next = min(omega_next, max(float(omega_min), 0.85 * omega_it))
                omega_reason_parts.append("late-stage damping")

        omega_next = min(float(omega), max(float(omega_min), omega_next))
        if adaptive_omega:
            omega_next = min(float(omega_it), float(omega_next))
        if not omega_reason_parts:
            if adaptive_omega and it <= omega_aggressive_sweeps:
                omega_reason_parts.append("aggressive warm-up")
            elif adaptive_omega and omega_next < omega_it - 1e-12:
                omega_reason_parts.append("scheduled decay")
            elif adaptive_omega and abs(omega_next - omega_it) <= 1e-12:
                omega_reason_parts.append("held after prior cut")
            else:
                omega_reason_parts.append("fixed")
        omega_reason = ", ".join(omega_reason_parts)
        row_data: Dict[str, object] = {
            "iter": it,
            "r_strat": float(r_strat),
            "r_obj": float(r_obj),
            "max_abs_dp": float(max_abs_dp),
            "max_abs_ddk": float(max_abs_ddk),
            "c_pen_q": float(c_pen_current["q"]),
            "c_pen_p": float(c_pen_current["p"]),
            "c_pen_a": float(c_pen_current["a"]),
            "c_pen_dk": float(c_pen_current["dk"]),
            "stable_count": int(stable_count),
            "omega": float(omega_it),
            "omega_next": float(omega_next),
            "omega_reason": omega_reason,
        }
        if shuffle_players:
            row_data["sweep_order"] = list(sweep_order)
        iter_rows.append(row_data)

        if iter_callback is not None:
            if shuffle_players:
                last_state["_sweep_order"] = list(sweep_order)
            last_state["_max_abs_dp"]      = float(max_abs_dp)
            last_state["_max_abs_ddk"]     = float(max_abs_ddk)
            last_state["_omega_current"]   = float(omega_it)
            last_state["_omega_next"]      = float(omega_next)
            last_state["_omega_reason"]    = omega_reason
            last_state["_c_pen_q_current"]  = float(c_pen_current["q"])
            last_state["_c_pen_p_current"]  = float(c_pen_current["p"])
            last_state["_c_pen_a_current"]  = float(c_pen_current["a"])
            last_state["_c_pen_dk_current"] = float(c_pen_current["dk"])
            iter_callback(it, last_state, float(r_strat), int(stable_count))
            last_state.pop("_sweep_order",      None)
            last_state.pop("_max_abs_dp",       None)
            last_state.pop("_max_abs_ddk",      None)
            last_state.pop("_omega_current",    None)
            last_state.pop("_omega_next",       None)
            last_state.pop("_omega_reason",     None)
            last_state.pop("_c_pen_q_current",  None)
            last_state.pop("_c_pen_p_current",  None)
            last_state.pop("_c_pen_a_current",  None)
            last_state.pop("_c_pen_dk_current", None)

        prev_metrics = {
            "r_strat": float(r_strat),
            "max_abs_dp": float(max_abs_dp),
            "max_abs_ddk": float(max_abs_ddk),
        }
        omega_current = float(omega_next)

        if stable_count >= stable_iters:
            break

    return last_state, iter_rows
