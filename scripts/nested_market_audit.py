from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import clarabel
from scipy.optimize import minimize
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import (
    _candidate_state,
    _economic_objective,
    _solve_market_reference,
    _strategy_distance,
    _zero_prox_data,
)


def nested_economic_objective(
    data: mm.ModelData,
    candidate: dict[str, dict],
    market_state: dict[str, dict],
    player: str,
) -> float:
    """Evaluate welfare using the model's KKT substitution for producer revenue."""
    value = _economic_objective(data, candidate, market_state, player)
    if not bool(
        (data.settings or {}).get(
            "subtract_mu_offer_from_producer_margin", True
        )
    ):
        return value
    for tp in mm._operating_times(data):
        weight = float((data.beta_t or {}).get(tp, 1.0)) * float(
            (data.years_to_next or {}).get(tp, 1.0)
        )
        mu = float(market_state["mu_offer"][(player, tp)])
        for importer in data.regions:
            key = (player, importer, tp)
            flow = float(market_state["x"][key])
            original_margin = (
                float(market_state["lam"][(importer, tp)])
                - mu
                - float(data.c_ship[(player, importer)])
            )
            substituted_margin = (
                float(candidate["p_offer"][key]) + float(data.eps_x) * flow
            )
            value += weight * (substituted_margin - original_margin) * flow
    return value


def solve_nested_market(
    data: mm.ModelData,
    candidate: dict[str, dict],
    warm: dict[str, dict] | None = None,
) -> tuple[dict[str, dict], dict[str, float]]:
    regions = list(data.regions)
    times = list(data.times or [])
    operating_times = mm._operating_times(data)
    if bool((data.settings or {}).get("fix_p_offer_to_c_man_t", False)):
        for exporter in regions:
            for importer in regions:
                for period in operating_times:
                    cost = float((data.c_man_t or {}).get((exporter, period), data.c_man[exporter]))
                    offer = float(candidate["p_offer"][(exporter, importer, period)])
                    if abs(offer - cost) > 1e-7:
                        raise ValueError(
                            f"Fixed-cost offer violated at {exporter}/{importer}/{period}: "
                            f"{offer:g} != {cost:g}"
                        )
    routes = [(exporter, importer) for exporter in regions for importer in regions]
    n_flow = len(routes)
    n_dem = len(regions)
    n_var = n_flow + n_dem
    route_index = {route: index for index, route in enumerate(routes)}
    demand_index = {region: n_flow + index for index, region in enumerate(regions)}
    kcap = mm._implied_capacity_path(data, times, candidate["dK_net"])

    market_state = {
        "x": {},
        "x_dem": {},
        "lam": {},
        "mu_offer": {},
        "gamma": {},
        "beta_dem": {},
        "psi_dem": {},
    }
    max_eq = 0.0
    max_cap = 0.0
    max_stationarity = 0.0
    total_iterations = 0

    for tp in operating_times:
        linear_cost = np.array(
            [
                float(candidate["p_offer"][(exporter, importer, tp)])
                + float(data.c_ship[(exporter, importer)])
                for exporter, importer in routes
            ],
            dtype=float,
        )
        a = np.array(
            [float((data.a_dem_t or {})[(importer, tp)]) for importer in regions],
            dtype=float,
        )
        b = np.array(
            [float((data.b_dem_t or {})[(importer, tp)]) for importer in regions],
            dtype=float,
        )
        q = np.array([max(float(kcap[(exporter, tp)]), 0.0) for exporter in regions])
        dmax = np.array(
            [float((data.Dmax_t or {})[(importer, tp)]) for importer in regions],
            dtype=float,
        )

        aeq = np.zeros((n_dem, n_var), dtype=float)
        for importer_pos, importer in enumerate(regions):
            for exporter in regions:
                aeq[importer_pos, route_index[(exporter, importer)]] = 1.0
            aeq[importer_pos, demand_index[importer]] = -1.0

        acap = np.zeros((len(regions), n_var), dtype=float)
        for exporter_pos, exporter in enumerate(regions):
            for importer in regions:
                acap[exporter_pos, route_index[(exporter, importer)]] = 1.0

        p_matrix = sparse.diags(
            np.concatenate((np.full(n_flow, float(data.eps_x)), b)), format="csc"
        )
        q_vector = np.concatenate((linear_cost, -a))
        lower_matrix = -sparse.eye(n_var, format="csc")
        demand_upper = np.zeros((n_dem, n_var), dtype=float)
        for importer_pos, importer in enumerate(regions):
            demand_upper[importer_pos, demand_index[importer]] = 1.0
        constraint_matrix = sparse.vstack(
            (
                sparse.csc_matrix(aeq),
                sparse.csc_matrix(acap),
                lower_matrix,
                sparse.csc_matrix(demand_upper),
            ),
            format="csc",
        )
        rhs = np.concatenate((np.zeros(n_dem), q, np.zeros(n_var), dmax))
        cones = [
            clarabel.ZeroConeT(n_dem),
            clarabel.NonnegativeConeT(len(regions)),
            clarabel.NonnegativeConeT(n_var),
            clarabel.NonnegativeConeT(n_dem),
        ]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.tol_gap_abs = 1e-9
        settings.tol_gap_rel = 1e-9
        settings.tol_feas = 1e-9
        solver = clarabel.DefaultSolver(
            p_matrix, q_vector, constraint_matrix, rhs, cones, settings
        )
        result = solver.solve()
        if str(result.status) not in {"Solved", "AlmostSolved"}:
            raise RuntimeError(f"Nested market failed at {tp}: {result.status}")
        total_iterations += int(result.iterations)
        z = np.asarray(result.x, dtype=float)
        dual = np.asarray(result.z, dtype=float)
        lam = -dual[:n_dem]
        mu = dual[n_dem : n_dem + len(regions)]
        lower_dual = dual[n_dem + len(regions) : n_dem + len(regions) + n_var]
        demand_upper_dual = dual[-n_dem:]

        max_eq = max(max_eq, float(np.max(np.abs(aeq @ z))))
        max_cap = max(max_cap, float(np.max(np.maximum(acap @ z - q, 0.0))))
        for (exporter, importer), index in route_index.items():
            key = (exporter, importer, tp)
            flow = max(float(z[index]), 0.0)
            stationarity = (
                linear_cost[index]
                + float(data.eps_x) * flow
                - float(lam[regions.index(importer)])
                + float(mu[regions.index(exporter)])
            )
            gamma = max(float(lower_dual[index]), 0.0)
            market_state["x"][key] = flow
            market_state["gamma"][key] = gamma
            full_stationarity = stationarity - gamma
            max_stationarity = max(max_stationarity, abs(float(full_stationarity)))
        for importer_pos, importer in enumerate(regions):
            demand = max(float(z[demand_index[importer]]), 0.0)
            lam_value = float(lam[importer_pos])
            market_state["x_dem"][(importer, tp)] = demand
            market_state["lam"][(importer, tp)] = lam_value
            market_state["beta_dem"][(importer, tp)] = max(
                float(demand_upper_dual[importer_pos]), 0.0
            )
            market_state["psi_dem"][(importer, tp)] = max(
                float(lower_dual[demand_index[importer]]), 0.0
            )
        for exporter_pos, exporter in enumerate(regions):
            market_state["mu_offer"][(exporter, tp)] = max(float(mu[exporter_pos]), 0.0)

    # Keep payloads backward-compatible while making clear that the final time
    # label is a capacity state only and has no market outcome.
    for tp in times:
        if tp in operating_times:
            continue
        for exporter in regions:
            market_state["mu_offer"][(exporter, tp)] = 0.0
            market_state["x_dem"][(exporter, tp)] = 0.0
            market_state["lam"][(exporter, tp)] = 0.0
            market_state["beta_dem"][(exporter, tp)] = 0.0
            market_state["psi_dem"][(exporter, tp)] = 0.0
            for importer in regions:
                market_state["x"][(exporter, importer, tp)] = 0.0
                market_state["gamma"][(exporter, importer, tp)] = 0.0

    diagnostics = {
        "max_balance_residual": max_eq,
        "max_capacity_violation": max_cap,
        "max_positive_flow_stationarity": max_stationarity,
        "total_slsqp_iterations": float(total_iterations),
    }
    return market_state, diagnostics


def nested_best_response(
    data: mm.ModelData,
    candidate: dict[str, dict],
    reference_market: dict[str, dict],
    player: str,
    *,
    maxiter: int = 250,
    starts: int = 1,
    proximal_coefficients: dict[str, float] | None = None,
    proximal_reference_state: dict[str, dict] | None = None,
    start_states: list[dict[str, dict]] | None = None,
    start_state_labels: list[str] | None = None,
) -> tuple[float, dict[str, dict], dict[str, object]]:
    """Solve one player's nested best response.

    The return value is always the *unpenalized economic* objective at the
    selected response.  When ``proximal_coefficients`` is supplied, SLSQP
    instead maximizes the economic objective minus the requested algorithmic
    proximal costs; the selected response's penalized objective and proximal
    cost are exposed in the diagnostics.  Existing zero-proximal callers are
    therefore unchanged.

    ``start_states`` separates optimizer initialization from the frozen game
    profile.  This is useful for process-parallel multistart audits: each worker
    can solve from a different active-player initialization while retaining the
    exact same candidate, bounds, and reference objective.
    """
    times = list(data.times or [])
    operating_times = mm._operating_times(data)
    move_times = mm._move_times(times)
    importers = [importer for importer in data.regions if importer != player]
    fixed_cost_offers = bool((data.settings or {}).get("fix_p_offer_to_c_man_t", False))
    price_keys = [] if fixed_cost_offers else [
        (player, importer, tp)
        for importer in importers
        for tp in operating_times
    ]
    dk_keys = [(player, tp) for tp in move_times]
    x0 = np.array(
        [float(candidate["dK_net"][key]) for key in dk_keys]
        + [float(candidate["p_offer"][key]) for key in price_keys],
        dtype=float,
    )
    initial_capacity = float(mm._initial_capacity_by_region(data)[player])
    candidate_kcap = mm._implied_capacity_path(data, times, candidate["dK_net"])
    expansion = float((data.g_exp_ub or {}).get(player, 0.0))
    if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
        expansion *= initial_capacity
    g_dec = float((data.g_dec_ub or {}).get(player, 1.0))
    bounds = [
        (-g_dec * max(float(candidate_kcap[(player, tp)]), 0.0), expansion)
        for _, tp in dk_keys
    ] + [
        (0.0, float(data.p_offer_ub[(player, importer)]))
        for _, importer, _ in price_keys
    ]
    dk_scale = max(expansion, g_dec * initial_capacity, 1.0)
    variable_scale = np.array(
        [dk_scale for _ in dk_keys]
        + [max(float(upper), 1.0) for _, upper in bounds[len(dk_keys) :]],
        dtype=float,
    )
    scaled_bounds = [
        (float(lower) / variable_scale[index], float(upper) / variable_scale[index])
        for index, (lower, upper) in enumerate(bounds)
    ]
    reference_objective = nested_economic_objective(
        data, candidate, reference_market, player
    )
    objective_scale = max(abs(reference_objective), 1.0)
    prox = {name: 0.0 for name in ("q", "p", "a", "dk")}
    proximal_penalties_enabled = bool(
        (data.settings or {}).get("algorithmic_proximal_penalties_enabled", True)
    )
    if proximal_coefficients and proximal_penalties_enabled:
        for name in prox:
            prox[name] = float(proximal_coefficients.get(name, 0.0))
    prox_reference = proximal_reference_state or candidate
    evaluations = 0
    market_failures = 0
    cache: dict[
        bytes,
        tuple[float, float, float, dict[str, dict], dict[str, float]],
    ] = {}

    def state_from_vector(vector: np.ndarray) -> dict[str, dict]:
        state = {
            "dK_net": dict(candidate["dK_net"]),
            "p_offer": dict(candidate["p_offer"]),
            "a_bid": dict(candidate["a_bid"]),
            "Q_offer": dict(candidate["Q_offer"]),
        }
        for index, key in enumerate(dk_keys):
            state["dK_net"][key] = float(vector[index])
        offset = len(dk_keys)
        for index, key in enumerate(price_keys):
            state["p_offer"][key] = float(vector[offset + index])
        kcap = mm._implied_capacity_path(data, times, state["dK_net"])
        state["Q_offer"] = {
            (region, tp): (
                max(float(kcap[(region, tp)]), 0.0)
                if tp in operating_times
                else 0.0
            )
            for region in data.players
            for tp in times
        }
        return state

    def proximal_cost(state: dict[str, dict]) -> float:
        if not any(value != 0.0 for value in prox.values()):
            return 0.0
        value = 0.0
        for tp in operating_times:
            weight = float((data.beta_t or {}).get(tp, 1.0)) * float(
                (data.years_to_next or {}).get(tp, 1.0)
            )
            if prox["q"]:
                q_key = (player, tp)
                delta_q = float(state["Q_offer"][q_key]) - float(
                    prox_reference["Q_offer"][q_key]
                )
                value += 0.5 * weight * prox["q"] * delta_q * delta_q
            if prox["a"]:
                a_key = (player, tp)
                delta_a = float(state["a_bid"][a_key]) - float(
                    prox_reference["a_bid"][a_key]
                )
                value += 0.5 * weight * prox["a"] * delta_a * delta_a
            if prox["p"]:
                for importer in data.regions:
                    p_key = (player, importer, tp)
                    delta_p = float(state["p_offer"][p_key]) - float(
                        prox_reference["p_offer"][p_key]
                    )
                    value += 0.5 * weight * prox["p"] * delta_p * delta_p
            if prox["dk"]:
                dk_key = (player, tp)
                current = float(state["dK_net"].get(dk_key, 0.0))
                reference = float(prox_reference["dK_net"].get(dk_key, 0.0))
                delta_i = max(current, 0.0) - max(reference, 0.0)
                delta_d = max(-current, 0.0) - max(-reference, 0.0)
                value += 0.5 * weight * prox["dk"] * (
                    delta_i * delta_i + delta_d * delta_d
                )
        return float(value)

    def evaluate(vector: np.ndarray):
        nonlocal evaluations, market_failures
        key = np.asarray(vector, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        evaluations += 1
        state = state_from_vector(vector)
        try:
            market, diagnostic = solve_nested_market(data, state, reference_market)
            economic_value = nested_economic_objective(data, state, market, player)
            prox_cost = proximal_cost(state)
            optimization_value = economic_value - prox_cost
        except RuntimeError:
            market_failures += 1
            market = reference_market
            diagnostic = {"failed": 1.0}
            economic_value = reference_objective - 100.0 * objective_scale
            prox_cost = 0.0
            optimization_value = economic_value
        cache[key] = (
            economic_value,
            optimization_value,
            prox_cost,
            market,
            diagnostic,
        )
        return cache[key]

    def scaled_objective(vector: np.ndarray) -> float:
        return -float(evaluate(vector)[1]) / objective_scale

    def unscale(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=float) * variable_scale

    def capacity_feasibility(vector: np.ndarray) -> np.ndarray:
        state = state_from_vector(vector)
        kcap = mm._implied_capacity_path(data, times, state["dK_net"])
        values = []
        for index, (_, tp) in enumerate(dk_keys):
            dk = float(vector[index])
            current = float(kcap[(player, tp)])
            values.append(current)
            values.append(g_dec * current - max(-dk, 0.0))
        values.extend(float(kcap[(player, tp)]) for tp in times)
        return np.asarray(values, dtype=float)

    def scaled_capacity_feasibility(vector: np.ndarray) -> np.ndarray:
        return capacity_feasibility(unscale(vector)) / max(initial_capacity, 1.0)

    if starts < 1 and not start_states:
        raise ValueError("starts must be positive")
    price_offset = len(dk_keys)
    if start_states:
        start_vectors = []
        for start_state in start_states:
            start_vectors.append(
                np.array(
                    [float(start_state["dK_net"][key]) for key in dk_keys]
                    + [float(start_state["p_offer"][key]) for key in price_keys],
                    dtype=float,
                )
            )
        if start_state_labels is None:
            start_labels = [f"explicit_start_{index}" for index in range(len(start_vectors))]
        else:
            if len(start_state_labels) != len(start_vectors):
                raise ValueError("start_state_labels must match start_states")
            start_labels = list(start_state_labels)
    else:
        start_vectors = [x0]
        start_labels = ["candidate_strategy"]
        if starts >= 2:
            # A model-based standard initialization: no net capacity change and
            # exporter-period manufacturing cost for every strategic export price.
            # This is an optimizer start only; it does not constrain the response.
            standard = x0.copy()
            standard[:price_offset] = 0.0
            for index, (_, _importer, tp) in enumerate(price_keys, start=price_offset):
                lower, upper = bounds[index]
                cost = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))
                standard[index] = np.clip(cost, lower, upper)
            start_vectors.append(standard)
            start_labels.append("standard_zero_capacity_change_and_manufacturing_cost_prices")
        price_factors = (0.95, 1.05, 0.90)
        for factor in price_factors[: max(starts - 2, 0)]:
            alternate = x0.copy()
            for index in range(price_offset, len(alternate)):
                lower, upper = bounds[index]
                alternate[index] = np.clip(factor * alternate[index], lower, upper)
            start_vectors.append(alternate)
            start_labels.append(f"candidate_price_perturbation_factor_{factor:g}")

    attempts = []
    for start_index, start_vector in enumerate(start_vectors):
        result = minimize(
            lambda vector: scaled_objective(unscale(vector)),
            start_vector / variable_scale,
            method="SLSQP",
            bounds=scaled_bounds,
            constraints=[{"type": "ineq", "fun": scaled_capacity_feasibility}],
            options={"ftol": 1e-9, "maxiter": int(maxiter), "eps": 1e-5, "disp": False},
        )
        result_vector = unscale(result.x)
        economic_objective, optimization_objective, prox_cost, market, market_diagnostic = evaluate(
            result_vector
        )
        attempts.append(
            {
                "result": result,
                "vector": result_vector,
                "objective": float(economic_objective),
                "optimization_objective": float(optimization_objective),
                "proximal_cost": float(prox_cost),
                "market": market,
                "market_diagnostic": market_diagnostic,
                "start_index": start_index,
            }
        )
    # Failed local solves are diagnostic evidence only.  They must not drive
    # a profile update merely because they happen to report a high objective.
    successful_attempts = [attempt for attempt in attempts if attempt["result"].success]
    chosen_pool = successful_attempts if successful_attempts else attempts
    chosen = max(
        chosen_pool,
        key=lambda attempt: float(attempt["optimization_objective"]),
    )
    result = chosen["result"]
    best_objective = float(chosen["objective"])
    best_market = chosen["market"]
    market_diagnostic = chosen["market_diagnostic"]
    best_state = state_from_vector(chosen["vector"])
    diagnostics: dict[str, object] = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective_evaluations": evaluations,
        "market_failures": market_failures,
        "capacity_feasibility_min": float(
            np.min(capacity_feasibility(chosen["vector"]))
        ),
        "market": market_diagnostic,
        "reference_objective": reference_objective,
        "proximal_coefficients": prox,
        "proximal_cost": float(chosen["proximal_cost"]),
        "optimization_objective": float(chosen["optimization_objective"]),
        "economic_objective": float(chosen["objective"]),
        "chosen_start_index": int(chosen["start_index"]),
        "attempts": [
            {
                "start_index": int(attempt["start_index"]),
                "start_label": start_labels[int(attempt["start_index"])],
                "success": bool(attempt["result"].success),
                "status": int(attempt["result"].status),
                "message": str(attempt["result"].message),
                "iterations": int(attempt["result"].nit),
                "objective": float(attempt["objective"]),
                "optimization_objective": float(attempt["optimization_objective"]),
                "proximal_cost": float(attempt["proximal_cost"]),
            }
            for attempt in attempts
        ],
    }
    return float(best_objective), best_state, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or run nested economic best responses.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("workflow/zero_prox_conopt_from_block1_omega_002_manifest.json"),
    )
    parser.add_argument("--solver", default="conopt")
    parser.add_argument("--players", nargs="+", choices=["ch", "row", "apac", "us", "eu", "af"])
    parser.add_argument("--maxiter", type=int, default=250)
    args = parser.parse_args()
    _, base_cfg, candidate, warm, label = _candidate_state("search", args.manifest)
    data = _zero_prox_data(base_cfg)
    gams_obj, gams_state, gams_diag = _solve_market_reference(
        data, candidate, warm, "ch", args.solver
    )
    nested_state, nested_diag = solve_nested_market(data, candidate, gams_state)
    nested_obj = nested_economic_objective(data, candidate, nested_state, "ch")
    max_flow_error = max(
        abs(float(nested_state["x"][key]) - float(gams_state["x"][key]))
        for key in gams_state["x"]
    )
    max_lambda_error = max(
        abs(float(nested_state["lam"][key]) - float(gams_state["lam"][key]))
        for key in gams_state["lam"]
    )
    print(
        {
            "candidate": label,
            "gams_objective": gams_obj,
            "nested_objective": nested_obj,
            "objective_error": nested_obj - gams_obj,
            "max_flow_error": max_flow_error,
            "max_lambda_error": max_lambda_error,
            "gams_stationarity": gams_diag["positive_flow_stationarity_max"],
            **nested_diag,
        }
    )
    if args.players:
        records = []
        for player in args.players:
            print(f"[NESTED BR] {player}", flush=True)
            best_objective, best_state, diagnostics = nested_best_response(
                data,
                candidate,
                nested_state,
                player,
                maxiter=args.maxiter,
            )
            reference_objective = nested_economic_objective(
                data, candidate, nested_state, player
            )
            gain = best_objective - reference_objective
            relative_gain = max(gain, 0.0) / max(abs(reference_objective), 1.0)
            distance, coordinate, absolute_move = _strategy_distance(
                data, candidate, best_state, player
            )
            record = {
                "player": player,
                "reference_objective": reference_objective,
                "best_response_objective": best_objective,
                "absolute_gain": gain,
                "relative_gain": relative_gain,
                "max_scaled_strategy_distance": distance,
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": absolute_move,
                "diagnostics": diagnostics,
                "dK_net": {
                    tp: best_state["dK_net"][(player, tp)] for tp in mm._move_times(list(data.times or []))
                },
                "p_offer": {
                    f"{importer}/{tp}": best_state["p_offer"][(player, importer, tp)]
                    for importer in data.regions
                    if importer != player
                    for tp in mm._operating_times(data)
                },
            }
            records.append(record)
            print(
                f"[NESTED BR] {player}: gain={gain:.6g} rel={relative_gain:.3%} "
                f"distance={distance:.3g} success={diagnostics['success']}",
                flush=True,
            )
        output_dir = ROOT / "outputs" / "verification" / "ch-row-apac-us-eu-af"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"nested_best_responses_{stamp}.json"
        output_path.write_text(
            json.dumps(
                {
                    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "candidate": label,
                    "manifest": str(args.manifest),
                    "algorithmic_proximal_penalties": 0.0,
                    "market_validation": nested_diag,
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[NESTED BR] wrote {output_path}")


if __name__ == "__main__":
    main()
