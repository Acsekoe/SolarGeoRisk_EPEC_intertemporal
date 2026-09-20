from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from model import run_gs
from model.data_prep import load_data_from_excel
from scripts.continue_selected_equilibrium import (
    PLAYER_ORDER,
    SOURCE_ITERATION,
    SOURCE_WORKBOOK,
    _initial_model_data,
    replay_accepted_state,
)


OUTPUT_DIR = ROOT / "outputs" / "verification" / "ch-row-apac-us-eu-af"


def _checkpoint_warm_state(
    workbook: Path,
    data: mm.ModelData,
    iteration: int | None = None,
) -> dict[str, dict]:
    """Recover the saved lower-level state from a checkpoint's final player solve."""
    detail = pd.read_excel(workbook, sheet_name="detailed_iters")
    target = int(detail["iter"].max()) if iteration is None else int(iteration)
    rows = detail.loc[detail["iter"] == target].copy()
    rows["t"] = rows["t"].astype(str)
    state: dict[str, dict] = {
        name: {}
        for name in (
            "Kcap",
            "dK_net",
            "Q_offer",
            "p_offer",
            "a_bid",
            "x",
            "x_dem",
            "lam",
            "mu_offer",
            "gamma",
            "beta_dem",
            "psi_dem",
        )
    }
    for _, row in rows.iterrows():
        exporter = str(row["r"])
        tp = str(row["t"])
        for name, column in (
            ("Kcap", "Kcap"),
            ("dK_net", "net_cap_change"),
            ("Q_offer", "Q_offer"),
            ("a_bid", "a_bid"),
            ("x_dem", "x_dem"),
            ("lam", "lam"),
            ("mu_offer", "mu_offer"),
            ("beta_dem", "beta_dem"),
            ("psi_dem", "psi_dem"),
        ):
            if column in rows.columns and pd.notna(row[column]):
                state[name][(exporter, tp)] = float(row[column])
        for importer in data.regions:
            for name, prefix in (
                ("p_offer", "p_offer_to_"),
                ("x", "x_exp_to_"),
                ("gamma", "gamma_exp_to_"),
            ):
                column = f"{prefix}{importer}"
                if column in rows.columns and pd.notna(row[column]):
                    state[name][(exporter, importer, tp)] = float(row[column])
    return state


def _candidate_state(
    kind: str,
    candidate_manifest: Path | None = None,
) -> tuple[mm.ModelData, run_gs.RunConfig, dict[str, dict], dict[str, dict], str]:
    data, base_cfg, excel_initial = _initial_model_data()
    source, _ = replay_accepted_state(
        SOURCE_WORKBOOK,
        excel_initial,
        data,
        through_iteration=SOURCE_ITERATION,
    )
    if kind == "source":
        warm = _checkpoint_warm_state(SOURCE_WORKBOOK, data, SOURCE_ITERATION)
        return data, base_cfg, source, warm, f"source_iteration_{SOURCE_ITERATION}"

    if kind == "search":
        manifest_path = candidate_manifest or (
            ROOT / "workflow" / "zero_prox_conopt_search_manifest.json"
        )
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = source
        for entry in manifest.get("runs", []):
            state, _ = replay_accepted_state(ROOT / str(entry["workbook"]), state, data)
        branch = manifest_path.stem.removesuffix("_manifest")
        if manifest.get("runs"):
            warm_workbook = ROOT / str(manifest["runs"][-1]["workbook"])
            warm = _checkpoint_warm_state(warm_workbook, data)
        else:
            warm = _checkpoint_warm_state(SOURCE_WORKBOOK, data, SOURCE_ITERATION)
        return data, base_cfg, state, warm, f"{branch}_after_{len(manifest.get('runs', []))}_blocks"

    manifest_path = ROOT / "workflow" / "continuation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = source
    for entry in manifest.get("runs", []):
        state, _ = replay_accepted_state(ROOT / str(entry["workbook"]), state, data)
    if manifest.get("runs"):
        warm_workbook = ROOT / str(manifest["runs"][-1]["workbook"])
        warm = _checkpoint_warm_state(warm_workbook, data)
    else:
        warm = _checkpoint_warm_state(SOURCE_WORKBOOK, data, SOURCE_ITERATION)
    return data, base_cfg, state, warm, f"continuation_after_{len(manifest.get('runs', []))}_blocks"


def _zero_prox_data(
    base_cfg: run_gs.RunConfig,
    *,
    terminal_salvage_fraction: float = 0.0,
) -> mm.ModelData:
    cfg = run_gs.RunConfig(
        excel_path=base_cfg.excel_path,
        params_region_sheet="params_region_new",
        solver="ipopt",
        feastol=1e-4,
        opttol=1e-4,
        eps_x=1e-3,
        eps_comp=1e-3,
        c_pen_q=0.0,
        c_pen_p=0.0,
        c_pen_a=0.0,
        c_pen_dk=0.0,
        c_pen_q_mid=None,
        c_pen_p_mid=None,
        c_pen_a_mid=None,
        c_pen_dk_mid=None,
        c_pen_q_final=None,
        c_pen_p_final=None,
        c_pen_a_final=None,
        c_pen_dk_final=None,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        cap_keep_reward=0.0,
        capex_subsidy=0.0,
        terminal_salvage_fraction=terminal_salvage_fraction,
        terminal_capacity_state_only=base_cfg.terminal_capacity_state_only,
        decommission_penalty=0.0,
        fix_q_offer_to_kcap=True,
        force_mu_offer_zero=False,
        fix_a_bid_to_true_dem=True,
        discount_rate=0.02,
        base_year=2025,
    )
    data = load_data_from_excel(cfg.excel_path, params_region_sheet=cfg.params_region_sheet)
    run_gs._apply_data_overrides(data, cfg)
    return data


def _set_levels(ctx: mm.ModelContext, data: mm.ModelData, state: dict[str, dict]) -> None:
    times = list(data.times or [])
    operating_times = set(mm._operating_times(data))
    move_times = mm._move_times(times)
    d_k = state.get("dK_net", {})
    kcap = mm._implied_capacity_path(data, times, d_k)

    for (r, tp), value in kcap.items():
        ctx.vars["Kcap"].l[r, tp] = max(float(value), 0.0)
        if tp in operating_times:
            ctx.vars["Q_offer"].l[r, tp] = max(float(value), 0.0)
    for r in data.players:
        for tp in move_times:
            value = float(d_k.get((r, tp), 0.0))
            ctx.vars["Icap_pos"].l[r, tp] = max(value, 0.0)
            ctx.vars["Dcap_neg"].l[r, tp] = max(-value, 0.0)
    for (ex, im, tp), value in state.get("p_offer", {}).items():
        if tp in operating_times:
            ctx.vars["p_offer"].l[ex, im, tp] = float(value)
    for r in data.players:
        for tp in operating_times:
            ctx.vars["a_bid"].l[r, tp] = mm._true_demand_intercept(data, r, tp)

    # A reference solve can also seed all lower-level variables for the best response.
    for name in ("x", "x_dem", "lam", "mu_offer", "gamma", "beta_dem", "psi_dem"):
        variable = ctx.vars.get(name)
        for key, value in state.get(name, {}).items():
            try:
                if isinstance(key, tuple):
                    variable.l[key] = float(value)
                else:
                    variable.l[key] = float(value)
            except (KeyError, TypeError, ValueError):
                continue


def _fix_candidate_player(
    ctx: mm.ModelContext,
    data: mm.ModelData,
    candidate: dict[str, dict],
    player: str,
) -> None:
    times = list(data.times or [])
    operating_times = mm._operating_times(data)
    move_times = mm._move_times(times)
    d_k = candidate["dK_net"]
    kcap = mm._implied_capacity_path(data, times, d_k)

    for tp in operating_times:
        kval = max(float(kcap[(player, tp)]), 0.0)
        ctx.vars["Kcap"].l[player, tp] = kval
        ctx.vars["Kcap"].lo[player, tp] = kval
        ctx.vars["Kcap"].up[player, tp] = kval
        ctx.vars["Q_offer"].l[player, tp] = kval
        ctx.vars["Q_offer"].lo[player, tp] = kval
        ctx.vars["Q_offer"].up[player, tp] = kval
        aval = mm._true_demand_intercept(data, player, tp)
        ctx.vars["a_bid"].l[player, tp] = aval
        ctx.vars["a_bid"].lo[player, tp] = aval
        ctx.vars["a_bid"].up[player, tp] = aval
        for importer in data.regions:
            if importer == player:
                pval = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))
            else:
                pval = float(candidate["p_offer"][(player, importer, tp)])
            ctx.vars["p_offer"].l[player, importer, tp] = pval
            ctx.vars["p_offer"].lo[player, importer, tp] = pval
            ctx.vars["p_offer"].up[player, importer, tp] = pval

    for tp in move_times:
        value = float(d_k[(player, tp)])
        ival = max(value, 0.0)
        dval = max(-value, 0.0)
        ctx.vars["Icap_pos"].l[player, tp] = ival
        ctx.vars["Icap_pos"].lo[player, tp] = ival
        ctx.vars["Icap_pos"].up[player, tp] = ival
        ctx.vars["Dcap_neg"].l[player, tp] = dval
        ctx.vars["Dcap_neg"].lo[player, tp] = dval
        ctx.vars["Dcap_neg"].up[player, tp] = dval


def _top_symbol_violations(ctx: mm.ModelContext, limit: int = 12) -> list[dict[str, object]]:
    violations: list[dict[str, object]] = []
    for symbol_kind, symbols in (("equation", ctx.equations), ("variable", ctx.vars)):
        for name, symbol in symbols.items():
            records = symbol.records
            if records is None or records.empty or "level" not in records.columns:
                continue
            for _, row in records.iterrows():
                level = float(row["level"])
                lower = float(row.get("lower", -math.inf))
                upper = float(row.get("upper", math.inf))
                violation = max(lower - level, level - upper, 0.0)
                if violation <= 0.0 or not math.isfinite(violation):
                    continue
                keys = {
                    str(column): row[column]
                    for column in records.columns
                    if column not in {"level", "marginal", "lower", "upper", "scale"}
                }
                violations.append(
                    {
                        "kind": symbol_kind,
                        "symbol": name,
                        "keys": keys,
                        "violation": violation,
                        "level": level,
                        "lower": lower,
                        "upper": upper,
                    }
                )
    violations.sort(key=lambda item: float(item["violation"]), reverse=True)
    return violations[:limit]


def _model_diagnostics(model, ctx: mm.ModelContext) -> dict[str, object]:
    def value(name: str):
        raw = getattr(model, name, None)
        try:
            raw = raw() if callable(raw) else raw
        except Exception:
            return None
        if isinstance(raw, (int, float)):
            return float(raw) if math.isfinite(float(raw)) else str(raw)
        return None if raw is None else str(raw)

    return {
        "solve_status": value("solve_status"),
        "model_status": value("status"),
        "max_infeasibility": value("max_infeasibility"),
        "num_infeasibilities": value("num_infeasibilities"),
        "solver_time_seconds": value("solve_model_time"),
        "top_symbol_violations": _top_symbol_violations(ctx),
    }


def _diagnostic_is_feasible(diagnostic: dict[str, object], tolerance: float = 1e-4) -> bool:
    status = str(diagnostic.get("model_status", "")).lower()
    infeasibility = diagnostic.get("max_infeasibility")
    try:
        residual_ok = float(infeasibility) <= tolerance
    except (TypeError, ValueError):
        residual_ok = False
    return residual_ok and "infeasible" not in status


def _equation_marginals(equation) -> dict[tuple[str, ...], float]:
    records = equation.records
    if records is None or records.empty or "marginal" not in records.columns:
        return {}
    value_columns = {"level", "marginal", "lower", "upper", "scale"}
    domain_columns = [column for column in records.columns if column not in value_columns]
    return {
        tuple(str(row[column]) for column in domain_columns): float(row["marginal"])
        for _, row in records.iterrows()
    }


def _strategy_distance(
    data: mm.ModelData,
    candidate: dict[str, dict],
    best_response: dict[str, dict],
    player: str,
) -> tuple[float, str, float]:
    times = list(data.times or [])
    operating_times = mm._operating_times(data)
    move_times = mm._move_times(times)
    initial_capacity = mm._initial_capacity_by_region(data)
    exp_scale = float((data.g_exp_ub or {}).get(player, 0.0))
    if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
        exp_scale *= float(initial_capacity[player])
    dec_scale = float((data.g_dec_ub or {}).get(player, 0.0)) * float(initial_capacity[player])
    dk_scale = max(exp_scale, dec_scale, 1.0)

    largest = (0.0, "", 0.0)
    for tp in move_times:
        key = (player, tp)
        absolute = float(best_response["dK_net"][key]) - float(candidate["dK_net"][key])
        scaled = abs(absolute) / dk_scale
        if scaled > largest[0]:
            largest = (scaled, f"dK_net[{player},{tp}]", absolute)
    for importer in data.regions:
        if importer == player:
            continue
        scale = max(float(data.p_offer_ub[(player, importer)]), 1e-3)
        for tp in operating_times:
            key = (player, importer, tp)
            absolute = float(best_response["p_offer"][key]) - float(candidate["p_offer"][key])
            scaled = abs(absolute) / scale
            if scaled > largest[0]:
                largest = (scaled, f"p_offer[{player},{importer},{tp}]", absolute)
    return largest


def _solve_player(
    data: mm.ModelData,
    candidate: dict[str, dict],
    player: str,
    *,
    fix_player: bool,
    solver: str,
    warm_state: dict[str, dict] | None = None,
    penalty_scales: list[float] | None = None,
) -> tuple[float, dict[str, dict], dict[str, object]]:
    workdir = Path(tempfile.mkdtemp(prefix=f"solargeorisk_audit_{player}_"))
    ctx: mm.ModelContext | None = None
    try:
        ctx = mm.build_model(data, working_directory=str(workdir))
        if warm_state is not None:
            _set_levels(ctx, data, warm_state)
        # Preserve lower-level checkpoint values, but always project every
        # strategic level onto the exact replayed candidate before fixing/solving.
        _set_levels(ctx, data, candidate)
        kcap = mm._implied_capacity_path(data, list(data.times or []), candidate["dK_net"])
        mm.apply_player_fixings(
            ctx,
            data,
            candidate["Q_offer"],
            candidate["dK_net"],
            candidate["p_offer"],
            candidate["a_bid"],
            player=player,
            theta_Kcap=kcap,
        )
        if fix_player:
            _fix_candidate_player(ctx, data, candidate, player)
        model = ctx.models[player]
        stage_diagnostics = []
        if penalty_scales is not None:
            times = list(data.times or [])
            kcap = mm._implied_capacity_path(data, times, candidate["dK_net"])
            for r in data.players:
                for tp in times:
                    ctx.params["Q_offer_last"][r, tp] = float(kcap[(r, tp)])
                    ctx.params["a_bid_last"][r, tp] = mm._true_demand_intercept(data, r, tp)
                    value = float(candidate["dK_net"].get((r, tp), 0.0))
                    ctx.params["Icap_pos_last"][r, tp] = max(value, 0.0)
                    ctx.params["Dcap_neg_last"][r, tp] = max(-value, 0.0)
                    for importer in data.regions:
                        ctx.params["p_offer_last"][r, importer, tp] = float(
                            candidate["p_offer"][(r, importer, tp)]
                        )
            for scale in penalty_scales:
                for key, base in (("q", 2.0), ("p", 3.0), ("a", 2.0), ("dk", 2.0)):
                    ctx.params[f"c_pen_{key}_scalar"].setRecords(base * float(scale))
                model.solve(solver=solver)
                stage_diagnostics.append(
                    {"scale": float(scale), **_model_diagnostics(model, ctx)}
                )
        else:
            model.solve(solver=solver)
        objective = float(model.objective_value)
        state = mm.extract_state(ctx)
        diagnostics = _model_diagnostics(model, ctx)
        if stage_diagnostics:
            diagnostics["homotopy_stages"] = stage_diagnostics
        return objective, state, diagnostics
    finally:
        if ctx is not None:
            ctx.container.close()
        shutil.rmtree(workdir, ignore_errors=True)


def _economic_objective(
    data: mm.ModelData,
    candidate: dict[str, dict],
    market_state: dict[str, dict],
    player: str,
) -> float:
    times = list(data.times or [])
    operating_times = mm._operating_times(data)
    kcap = mm._implied_capacity_path(data, times, candidate["dK_net"])
    value = 0.0
    for tp in operating_times:
        beta = float((data.beta_t or {}).get(tp, 1.0))
        years = float((data.years_to_next or {}).get(tp, 1.0))
        weight = beta * years
        demand = float(market_state["x_dem"][(player, tp)])
        lam = float(market_state["lam"][(player, tp)])
        a_dem = float((data.a_dem_t or {})[(player, tp)])
        b_dem = float((data.b_dem_t or {})[(player, tp)])
        period = a_dem * demand - 0.5 * b_dem * demand * demand - lam * demand
        mu = float(market_state["mu_offer"][(player, tp)])
        c_man = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))
        for importer in data.regions:
            flow = float(market_state["x"][(player, importer, tp)])
            period += (
                float(market_state["lam"][(importer, tp)])
                - mu
                - c_man
                - float(data.c_ship[(player, importer)])
            ) * flow
            markup = float(candidate["p_offer"][(player, importer, tp)]) - c_man
            period -= 0.5 * 0.1 * markup * markup
        period -= float((data.f_hold or {})[player]) * float(kcap[(player, tp)])
        if tp in mm._move_times(times):
            investment = max(float(candidate["dK_net"][(player, tp)]), 0.0)
            period -= float((data.c_inv or {})[player]) * investment
        value += weight * period
    value += mm._terminal_salvage_credit(
        data,
        player,
        kcap[(player, times[-1])],
    )
    return value


def _solve_market_reference(
    data: mm.ModelData,
    candidate: dict[str, dict],
    checkpoint_warm: dict[str, dict],
    player: str,
    solver: str,
) -> tuple[float, dict[str, dict], dict[str, object]]:
    workdir = Path(tempfile.mkdtemp(prefix=f"solargeorisk_market_{player}_"))
    ctx: mm.ModelContext | None = None
    try:
        ctx = mm.build_model(data, working_directory=str(workdir))
        _set_levels(ctx, data, checkpoint_warm)
        _set_levels(ctx, data, candidate)
        times = list(data.times or [])
        operating_times = mm._operating_times(data)
        kcap = mm._implied_capacity_path(data, times, candidate["dK_net"])
        for exporter in data.regions:
            for tp in operating_times:
                qval = max(float(kcap[(exporter, tp)]), 0.0)
                aval = mm._true_demand_intercept(data, exporter, tp)
                for name, val in (("Q_offer", qval), ("a_bid", aval)):
                    ctx.vars[name].l[exporter, tp] = val
                    ctx.vars[name].lo[exporter, tp] = val
                    ctx.vars[name].up[exporter, tp] = val
                for importer in data.regions:
                    pval = float(candidate["p_offer"][(exporter, importer, tp)])
                    ctx.vars["p_offer"].l[exporter, importer, tp] = pval
                    ctx.vars["p_offer"].lo[exporter, importer, tp] = pval
                    ctx.vars["p_offer"].up[exporter, importer, tp] = pval

        market = mm.Model(
            ctx.container,
            "fixed_offer_market",
            equations=[
                ctx.equations["eq_obj_llp"],
                ctx.equations["eq_bal"],
                ctx.equations["eq_cap"],
            ],
            problem=mm.Problem.NLP,
            sense=mm.Sense.MIN,
            objective=ctx.container["z_llp"],
        )
        market.solve(solver=solver)
        diagnostics = _model_diagnostics(market, ctx)
        operating_time_set = set(operating_times)
        x = {
            key: value
            for key, value in ctx.vars["x"].toDict().items()
            if key[-1] in operating_time_set
        }
        x_dem = {
            key: value
            for key, value in ctx.vars["x_dem"].toDict().items()
            if key[-1] in operating_time_set
        }
        bal_marginal = _equation_marginals(ctx.equations["eq_bal"])
        cap_marginal = _equation_marginals(ctx.equations["eq_cap"])

        # GAMS marginal sign conventions vary with equation orientation.  Select
        # the sign pair that best reproduces the model's own x-stationarity on
        # strictly positive flows, while requiring nonnegative scarcity rents.
        sign_trials: list[tuple[float, float, float, dict, dict]] = []
        for bal_sign in (-1.0, 1.0):
            for cap_sign in (-1.0, 1.0):
                lam = {key: bal_sign * float(val) for key, val in bal_marginal.items()}
                mu = {key: cap_sign * float(val) for key, val in cap_marginal.items()}
                residuals = []
                for (exporter, importer, tp), flow in x.items():
                    if float(flow) <= 1e-7:
                        continue
                    residuals.append(
                        abs(
                            float(candidate["p_offer"][(exporter, importer, tp)])
                            + float(data.c_ship[(exporter, importer)])
                            + float(data.eps_x) * float(flow)
                            - float(lam[(importer, tp)])
                            + float(mu[(exporter, tp)])
                        )
                    )
                negativity = sum(max(-float(v), 0.0) for v in mu.values())
                score = (max(residuals, default=0.0) + negativity, bal_sign, cap_sign, lam, mu)
                sign_trials.append(score)
        _, bal_sign, cap_sign, lam, mu = min(sign_trials, key=lambda item: item[0])
        positive_flow_residuals = []
        for (exporter, importer, tp), flow in x.items():
            if float(flow) > 1e-7:
                positive_flow_residuals.append(
                    abs(
                        float(candidate["p_offer"][(exporter, importer, tp)])
                        + float(data.c_ship[(exporter, importer)])
                        + float(data.eps_x) * float(flow)
                        - float(lam[(importer, tp)])
                        + float(mu[(exporter, tp)])
                    )
                )
        diagnostics["dual_signs"] = {"balance": bal_sign, "capacity": cap_sign}
        diagnostics["positive_flow_stationarity_max"] = max(
            positive_flow_residuals, default=0.0
        )
        gamma = {}
        for (exporter, importer, tp), flow in x.items():
            stationarity = (
                float(candidate["p_offer"][(exporter, importer, tp)])
                + float(data.c_ship[(exporter, importer)])
                + float(data.eps_x) * float(flow)
                - float(lam[(importer, tp)])
                + float(mu[(exporter, tp)])
            )
            gamma[(exporter, importer, tp)] = max(stationarity, 0.0)
        beta_dem = {}
        psi_dem = {}
        for (importer, tp), demand in x_dem.items():
            a_bid = mm._true_demand_intercept(data, importer, tp)
            marginal_utility_gap = (
                a_bid
                - float((data.b_dem_t or {})[(importer, tp)]) * float(demand)
                - float(lam[(importer, tp)])
            )
            beta_dem[(importer, tp)] = max(marginal_utility_gap, 0.0)
            psi_dem[(importer, tp)] = max(-marginal_utility_gap, 0.0)
        market_state = {
            "x": x,
            "x_dem": x_dem,
            "lam": lam,
            "mu_offer": mu,
            "gamma": gamma,
            "beta_dem": beta_dem,
            "psi_dem": psi_dem,
        }
        objective = _economic_objective(data, candidate, market_state, player)
        return objective, market_state, diagnostics
    finally:
        if ctx is not None:
            ctx.container.close()
        shutil.rmtree(workdir, ignore_errors=True)


def audit(
    candidate_kind: str,
    threshold: float,
    solver: str,
    players: list[str],
    candidate_manifest: Path | None = None,
) -> tuple[Path, Path]:
    _, base_cfg, candidate, checkpoint_warm, label = _candidate_state(
        candidate_kind, candidate_manifest
    )
    data = _zero_prox_data(base_cfg)
    records: list[dict[str, object]] = []

    print("[AUDIT] solving common fixed-offer lower-level market", flush=True)
    _, reference_state, reference_diag = _solve_market_reference(
        data,
        candidate,
        checkpoint_warm,
        players[0],
        solver,
    )

    for player in players:
        reference_obj = _economic_objective(data, candidate, reference_state, player)
        attempts = []
        for start_name, warm_state, penalty_scales in (
            ("strategy_only", None, None),
            (
                "prox_homotopy",
                reference_state,
                [1000.0, 100.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.0],
            ),
        ):
            print(
                f"[AUDIT] {player}: zero-proximal best response start={start_name}",
                flush=True,
            )
            objective, state, diagnostic = _solve_player(
                data,
                candidate,
                player,
                fix_player=False,
                solver=solver,
                warm_state=warm_state,
                penalty_scales=penalty_scales,
            )
            attempts.append(
                {
                    "start": start_name,
                    "objective": objective,
                    "state": state,
                    "diagnostic": diagnostic,
                    "feasible": _diagnostic_is_feasible(diagnostic),
                }
            )
        feasible_attempts = [attempt for attempt in attempts if attempt["feasible"]]
        if feasible_attempts:
            chosen = max(feasible_attempts, key=lambda attempt: float(attempt["objective"]))
        else:
            chosen = min(
                attempts,
                key=lambda attempt: float(
                    attempt["diagnostic"].get("max_infeasibility") or math.inf
                ),
            )
        br_obj = float(chosen["objective"])
        br_state = chosen["state"]
        br_diag = chosen["diagnostic"]
        gain = br_obj - reference_obj
        relative_gain = max(gain, 0.0) / max(abs(reference_obj), 1.0)
        reference_feasible = _diagnostic_is_feasible(reference_diag)
        best_response_feasible = _diagnostic_is_feasible(br_diag)
        audit_valid = reference_feasible and best_response_feasible
        distance, coordinate, absolute_move = _strategy_distance(
            data, candidate, br_state, player
        )
        record = {
            "player": player,
            "reference_objective": reference_obj,
            "best_response_objective": br_obj,
            "absolute_gain": gain,
            "relative_gain": relative_gain,
            "max_scaled_strategy_distance": distance,
            "largest_move_coordinate": coordinate,
            "largest_move_absolute": absolute_move,
            "reference_feasible": reference_feasible,
            "best_response_feasible": best_response_feasible,
            "audit_valid": audit_valid,
            "passes_relative_gain": audit_valid and relative_gain <= threshold,
            "reference_solve": reference_diag,
            "best_response_solve": br_diag,
            "best_response_start": chosen["start"],
            "best_response_attempts": [
                {
                    "start": attempt["start"],
                    "objective": attempt["objective"],
                    "feasible": attempt["feasible"],
                    "diagnostic": attempt["diagnostic"],
                }
                for attempt in attempts
            ],
        }
        records.append(record)
        print(
            f"[AUDIT] {player}: gain={gain:.6g} rel={relative_gain:.3%} "
            f"distance={distance:.3g} at {coordinate} valid={audit_valid}",
            flush=True,
        )

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_label = f"{label}_{solver.lower()}"
    json_path = OUTPUT_DIR / f"audit_{run_label}_{stamp}.json"
    xlsx_path = OUTPUT_DIR / f"audit_{run_label}_{stamp}.xlsx"
    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate": label,
        "solver": solver,
        "source_workbook": str(SOURCE_WORKBOOK.relative_to(ROOT)),
        "player_order": PLAYER_ORDER,
        "algorithmic_proximal_penalties": {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0},
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "relative_gain_threshold": threshold,
        "all_players_pass": all(bool(r["passes_relative_gain"]) for r in records),
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    flat_records = []
    for record in records:
        flat = {
            k: v for k, v in record.items() if not isinstance(v, (dict, list))
        }
        for prefix in ("reference_solve", "best_response_solve"):
            for key, value in record[prefix].items():
                flat[f"{prefix}_{key}"] = value
        flat_records.append(flat)
    pd.DataFrame(flat_records).to_excel(xlsx_path, index=False)
    print(f"[AUDIT] wrote {json_path}")
    print(f"[AUDIT] wrote {xlsx_path}")
    return json_path, xlsx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a saved candidate with independent zero-proximal unilateral best responses."
    )
    parser.add_argument(
        "--candidate",
        choices=("source", "continuation", "search"),
        default="source",
        help=(
            "Candidate to audit: the paper source iteration, the latest staged "
            "continuation state, or the latest zero-proximal search state."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Maximum relative profitable unilateral deviation (default: 0.01 = 1%%).",
    )
    parser.add_argument("--solver", default="ipopt")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Search manifest to replay when --candidate search is selected.",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        choices=PLAYER_ORDER,
        default=PLAYER_ORDER,
        help="Players to audit, in the requested order.",
    )
    args = parser.parse_args()
    if args.threshold < 0.0:
        raise ValueError("--threshold must be non-negative")
    audit(args.candidate, args.threshold, args.solver, list(args.players), args.manifest)


if __name__ == "__main__":
    main()
