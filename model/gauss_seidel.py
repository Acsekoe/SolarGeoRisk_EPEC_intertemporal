from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

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

    times: List[str] = data.times or ["2025", "2030", "2035", "2040"]
    kcap_2025 = data.Kcap_2025 if data.Kcap_2025 is not None else {r: float(data.Qcap.get(r, 0.0)) for r in data.regions}

    ctx = _it.build_model(data, working_directory=working_directory)

    # ---- Initialize theta dicts ----
    if initial_state:
        theta_Q: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("Q_offer", {}).get((r, tp), 0.8 * float(kcap_2025[r])))
            for r in data.players for tp in times
        }
        theta_p_offer: Dict[Tuple[str, str, str], float] = {
            (ex, im, tp): float(initial_state.get("p_offer", {}).get((ex, im, tp), 0.0))
            for ex in data.regions for im in data.regions for tp in times
        }
        theta_k_exp: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("k_exp", {}).get((r, tp), 0.0))
            for r in data.players for tp in times
        }
        theta_k_dec: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("k_dec", {}).get((r, tp), 0.0))
            for r in data.players for tp in times
        }
        theta_a_bid: Dict[Tuple[str, str], float] = {
            (r, tp): float(initial_state.get("a_bid", {}).get((r, tp), data.a_dem_t[(r, tp)] if data.a_dem_t else data.a_dem.get(r, 0.0)))
            for r in data.players for tp in times
        }
    else:
        theta_Q = {(r, tp): 0.8 * float(kcap_2025[r]) for r in data.players for tp in times}
        theta_p_offer = {(ex, im, tp): 0.5 * float(data.p_offer_ub[(ex, im)]) for ex in data.regions for im in data.regions for tp in times}
        theta_k_exp = {(r, tp): 0.0 for r in data.players for tp in times}
        theta_k_dec = {(r, tp): 0.0 for r in data.players for tp in times}
        theta_a_bid = {(r, tp): float(data.a_dem_t[(r, tp)] if data.a_dem_t else data.a_dem.get(r, 0.0)) for r in data.players for tp in times}

    theta_obj: Dict[str, float] = {r: 0.0 for r in data.players}

    def _scaled_change(new: float, old: float, scale: float) -> float:
        return abs(new - old) / max(scale, 1e-12)

    def _q_scale(r: str) -> float:
        return max(float(kcap_2025.get(r, 0.0)), 1.0)

    def _p_scale(ex: str, im: str) -> float:
        return max(float(data.p_offer_ub[(ex, im)]), 1e-3)

    iter_rows: List[Dict[str, object]] = []
    stable_count = 0
    last_state: Dict[str, Dict] = {}

    solve_kwargs: Dict[str, object] = {"solver": solver}
    if solver_options:
        solve_kwargs["solver_options"] = solver_options

    def _update_prox_reference() -> None:
        q_last = ctx.params.get("Q_offer_last")
        poffer_last = ctx.params.get("p_offer_last")
        if q_last is None or poffer_last is None:
            return
        for r in data.regions:
            for tp in times:
                q_last[r, tp] = float(theta_Q.get((r, tp), float(kcap_2025[r])))
        for ex in data.regions:
            for im in data.regions:
                for tp in times:
                    poffer_last[ex, im, tp] = float(theta_p_offer.get((ex, im, tp), 0.0))

    for it in range(1, iters + 1):
        r_strat = 0.0

        prev_Q = dict(theta_Q)
        prev_poffer = dict(theta_p_offer)
        prev_k_exp = dict(theta_k_exp)
        prev_k_dec = dict(theta_k_dec)
        prev_a_bid = dict(theta_a_bid)
        prev_obj = dict(theta_obj)

        sweep_order = list(data.players)
        if shuffle_players:
            random.shuffle(sweep_order)

        # Ensure China plays last
        if "ch" in sweep_order:
            sweep_order.remove("ch")
            sweep_order.append("ch")

        for p in sweep_order:
            _update_prox_reference()
            _it.apply_player_fixings(
                ctx, data,
                theta_Q, theta_p_offer,
                theta_k_exp, theta_k_dec,
                theta_a_bid,
                player=p,
            )
            ctx.models[p].solve(**solve_kwargs)

            state = _it.extract_state(ctx)
            last_state = state

            Q_sol = state.get("Q_offer", {})
            poffer_sol = state.get("p_offer", {})
            k_exp_sol = state.get("k_exp", {})
            k_dec_sol = state.get("k_dec", {})
            a_bid_sol = state.get("a_bid", {})
            obj_sol = state.get("obj", {})

            # Update Q_offer
            for tp in times:
                key = (p, tp)
                if key in Q_sol:
                    br = float(Q_sol[key])
                    theta_Q[key] = (1.0 - omega) * theta_Q[key] + omega * br

            # Update p_offer
            for im in data.regions:
                for tp in times:
                    key = (p, im, tp)
                    if key in poffer_sol:
                        br = float(poffer_sol[key])
                        theta_p_offer[key] = (1.0 - omega) * theta_p_offer[key] + omega * br

            # Update capacity decisions and a_bid
            for tp in times:
                sk = (p, tp)
                if sk in k_exp_sol:
                    theta_k_exp[sk] = (1.0 - omega) * theta_k_exp[sk] + omega * float(k_exp_sol[sk])
                if sk in k_dec_sol:
                    theta_k_dec[sk] = (1.0 - omega) * theta_k_dec[sk] + omega * float(k_dec_sol[sk])
                if sk in a_bid_sol:
                    theta_a_bid[sk] = (1.0 - omega) * theta_a_bid[sk] + omega * float(a_bid_sol[sk])

            if isinstance(obj_sol, dict):
                theta_obj[p] = float(obj_sol.get(p, 0.0))

        # ---- Convergence metrics ----
        for r in data.players:
            for tp in times:
                r_strat = max(r_strat, _scaled_change(theta_Q[(r, tp)], prev_Q[(r, tp)], _q_scale(r)))
                
                # Compare a_bid change scaled by a_true
                a_scale = float(data.a_dem_t[(r, tp)] if data.a_dem_t else data.a_dem.get(r, 100.0))
                r_strat = max(r_strat, _scaled_change(theta_a_bid[(r, tp)], prev_a_bid[(r, tp)], a_scale))
                # Optionally add k_exp, k_dec to convergence metric

        for ex in data.regions:
            for im in data.regions:
                for tp in times:
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

