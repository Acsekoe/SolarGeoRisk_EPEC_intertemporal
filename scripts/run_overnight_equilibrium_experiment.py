from __future__ import annotations

"""Checkpointed parallel O1--O6 equilibrium-search experiment.

The six Gauss--Seidel trajectories are independent processes.  Player updates
inside each trajectory remain strictly sequential.  Periodic and final audits
are zero-proximal frozen-profile solves distributed as independent
(candidate, player, start) process tasks.
"""

import os

# Prevent outer process parallelism from multiplying hidden BLAS thread pools.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import copy
import csv
import json
import math
import socket
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.continue_selected_equilibrium import BASE_PENALTIES, PLAYER_ORDER, SOURCE_ITERATION, SOURCE_WORKBOOK
from scripts.nested_market_audit import nested_best_response, nested_economic_objective, solve_nested_market
from scripts.run_local_paper_equilibrium_experiment import (
    CERTIFIED_AUDIT,
    CERTIFIED_CHECKPOINT,
    clone_state,
    detect_cycle,
    full_state_payload,
    normalized_l2,
    paper_state,
    player_strategy_payload,
    strategy_metric,
)
from scripts.search_nested_equilibrium import _deserialize_state, _serialize_state, _sync_quantity


EXPERIMENT_BASE = (
    ROOT / "outputs" / "equilibrium_search" / "ch-row-apac-us-eu-af" / "overnight"
)
LOCAL_BEST_CHECKPOINT = (
    ROOT
    / "outputs/equilibrium_search/ch-row-apac-us-eu-af/local_paper_profile_zero_prox"
    / "branch_F_alpha_0p50/sweep_006.json"
)
LOCAL_BEST_AUDIT = LOCAL_BEST_CHECKPOINT.with_name("audit_sweep_006.json")
PAPER_AUDIT = (
    ROOT
    / "outputs/equilibrium_search/ch-row-apac-us-eu-af/local_paper_profile_zero_prox"
    / "audit_paper_profile.json"
)
HOMOTOPY_GAMMAS = [1.00, 0.50, 0.25, 0.10, 0.05, 0.02, 0.00]

BRANCHES: dict[str, dict[str, Any]] = {
    "O1": {
        "description": "paper profile, zero proximal, fixed alpha 0.20",
        "start": "paper",
        "mode": "fixed",
        "alpha": 0.20,
        "max_sweeps": 150,
    },
    "O2": {
        "description": "paper profile, zero proximal, fixed alpha 0.10",
        "start": "paper",
        "mode": "fixed",
        "alpha": 0.10,
        "max_sweeps": 200,
    },
    "O3": {
        "description": "paper profile, zero proximal, adaptive damping",
        "start": "paper",
        "mode": "adaptive",
        "alpha": 0.50,
        "alpha_min": 0.025,
        "alpha_max": 0.65,
        "max_sweeps": 150,
    },
    "O4": {
        "description": "paper profile, proximal continuation/homotopy",
        "start": "paper",
        "mode": "homotopy",
        "alpha": 0.50,
        "positive_stage_sweeps": 12,
        "zero_stage_sweeps": 50,
    },
    "O5": {
        "description": "Branch-F alpha-0.50 sweep-6 profile, zero proximal, fixed alpha 0.20",
        "start": "local",
        "mode": "fixed",
        "alpha": 0.20,
        "max_sweeps": 150,
    },
    "O6": {
        "description": "Branch-F alpha-0.50 sweep-6 profile, zero proximal, adaptive damping",
        "start": "local",
        "mode": "adaptive",
        "alpha": 0.30,
        "alpha_min": 0.025,
        "alpha_max": 0.50,
        "max_sweeps": 150,
    },
}

_WORKER_CONTEXT: tuple[Any, dict[str, dict], dict[str, dict]] | None = None


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{now()} pid={os.getpid()} {message}\n")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def worker_context() -> tuple[Any, dict[str, dict], dict[str, dict]]:
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        data, paper, replay_error = paper_state()
        if replay_error > 1e-10:
            raise RuntimeError(f"paper replay residual is {replay_error:.3g}")
        local_payload = json.loads(LOCAL_BEST_CHECKPOINT.read_text(encoding="utf-8"))
        local = _deserialize_state(local_payload["ending_profile"]["strategy"], data, paper)
        _WORKER_CONTEXT = data, paper, local
    return _WORKER_CONTEXT


def load_profile(path: Path, data: Any, paper: dict[str, dict]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "ending_profile" in payload:
        serialized = payload["ending_profile"]["strategy"]
    elif "state" in payload:
        serialized = payload["state"]
    elif "profile" in payload and "strategy" in payload["profile"]:
        serialized = payload["profile"]["strategy"]
    elif "strategy" in payload:
        serialized = payload["strategy"]
    else:
        raise ValueError(f"Cannot find a serialized strategy in {path}")
    return _deserialize_state(serialized, data, paper)


def economic_objectives(data: Any, state: dict[str, dict], market=None):
    if market is None:
        market, diagnostics = solve_nested_market(data, state)
    else:
        diagnostics = None
    values = {
        player: float(nested_economic_objective(data, state, market, player))
        for player in PLAYER_ORDER
    }
    return values, market, diagnostics


def update_player(data: Any, state: dict[str, dict], response: dict[str, dict], player: str, alpha: float) -> None:
    for tp in mm._move_times(list(data.times or [])):
        key = (player, tp)
        state["dK_net"][key] = (1.0 - alpha) * float(state["dK_net"][key]) + alpha * float(
            response["dK_net"][key]
        )
    for importer in data.regions:
        if importer == player:
            continue
        for tp in list(data.times or []):
            key = (player, importer, tp)
            state["p_offer"][key] = (1.0 - alpha) * float(state["p_offer"][key]) + alpha * float(
                response["p_offer"][key]
            )
    _sync_quantity(data, state)


def response_with_retry(
    data: Any,
    state: dict[str, dict],
    market: dict[str, dict],
    player: str,
    *,
    maxiter: int,
    prox: dict[str, float] | None = None,
) -> tuple[float, dict[str, dict], dict[str, Any], bool]:
    best, response, diagnostics = nested_best_response(
        data,
        state,
        market,
        player,
        maxiter=maxiter,
        starts=1,
        proximal_coefficients=prox,
        proximal_reference_state=state,
    )
    retried = False
    if not diagnostics["success"]:
        retried = True
        best, response, diagnostics = nested_best_response(
            data,
            state,
            market,
            player,
            maxiter=max(maxiter * 2, 800),
            starts=3,
            proximal_coefficients=prox,
            proximal_reference_state=state,
        )
    return best, response, diagnostics, retried


def branch_schedule(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if spec["mode"] != "homotopy":
        return [
            {"global_sweep": index, "stage": 1, "stage_sweep": index, "gamma": 0.0}
            for index in range(1, int(spec["max_sweeps"]) + 1)
        ]
    schedule = []
    global_sweep = 0
    for stage, gamma in enumerate(HOMOTOPY_GAMMAS, start=1):
        count = int(spec["zero_stage_sweeps"] if gamma == 0.0 else spec["positive_stage_sweeps"])
        for stage_sweep in range(1, count + 1):
            global_sweep += 1
            schedule.append(
                {
                    "global_sweep": global_sweep,
                    "stage": stage,
                    "stage_sweep": stage_sweep,
                    "gamma": float(gamma),
                }
            )
    return schedule


def adaptive_next_alpha(
    alpha: float,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    current: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    event = {"action": "hold", "reasons": []}
    cycle = current["cycle_diagnostics"]["two_cycle"] or current["cycle_diagnostics"]["three_cycle"]
    prior_strategy = [float(row["strategy_change_metric"]) for row in rows[-4:]]
    prior_jump = [float(row["largest_raw_best_response_move"]) for row in rows[-4:]]
    current_strategy = float(current["strategy_change_metric"])
    current_jump = float(current["largest_raw_best_response_move"])
    if cycle:
        event["reasons"].append("two_or_three_cycle_detected")
    if prior_strategy and current_strategy > 2.5 * max(float(np.median(prior_strategy)), 1e-5):
        event["reasons"].append("strategy_displacement_spike")
    if prior_jump and current_jump > max(0.20, 3.0 * max(float(np.median(prior_jump)), 1e-5)):
        event["reasons"].append("large_raw_best_response_branch_jump")
    if event["reasons"]:
        event["action"] = "halve"
        return max(0.5 * alpha, float(spec["alpha_min"])), event
    recent = rows[-3:] + [current]
    if len(recent) >= 4:
        gains = [float(row["max_raw_unilateral_gain"]) for row in recent]
        displacements = [float(row["strategy_change_metric"]) for row in recent]
        improving = all(gains[index] < gains[index - 1] for index in range(1, len(gains)))
        stable = max(displacements) <= 1.5 * max(min(displacements), 1e-5)
        if improving and stable:
            event["action"] = "increase_20pct"
            event["reasons"].append("four_stable_improving_sweeps")
            return min(1.2 * alpha, float(spec["alpha_max"])), event
    return alpha, event


def run_branch_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    branch = str(task["branch"])
    spec = dict(task["spec"])
    run_root = Path(task["run_root"])
    out = run_root / "branches" / branch
    log_path = out / "branch.log"
    append_log(log_path, f"START branch={branch} mode={spec['mode']}")
    try:
        data, paper, local = worker_context()
        initial = clone_state(paper if spec["start"] == "paper" else local)
        init_path = out / "initialization.json"
        if not init_path.exists():
            market, market_diag = solve_nested_market(data, initial)
            atomic_json(
                init_path,
                {
                    "created": now(),
                    "branch": branch,
                    "specification": spec,
                    "pid": os.getpid(),
                    "player_order": PLAYER_ORDER,
                    "source_workbook": relative(SOURCE_WORKBOOK),
                    "source_iteration": SOURCE_ITERATION,
                    "local_start_checkpoint": relative(LOCAL_BEST_CHECKPOINT) if spec["start"] == "local" else None,
                    "local_start_audit": relative(LOCAL_BEST_AUDIT) if spec["start"] == "local" else None,
                    "original_proximal_penalties": BASE_PENALTIES,
                    "profile": full_state_payload(data, initial, market),
                    "market_diagnostics": market_diag,
                    "distance_from_paper": normalized_l2(data, paper, initial),
                },
            )
        sweep_paths = sorted(out.glob("sweep_*.json"))
        schedule = branch_schedule(spec)
        summary_path = out / "summary.json"
        if len(sweep_paths) >= len(schedule) and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["resumed_no_work"] = True
            append_log(
                log_path,
                f"END branch={branch} status=already_complete sweeps={len(sweep_paths)}",
            )
            return summary
        state = load_profile(sweep_paths[-1], data, paper) if sweep_paths else initial
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sweep_paths]
        history = [initial]
        for path in sweep_paths[-4:]:
            history.append(load_profile(path, data, paper))
        alpha = float(rows[-1]["next_alpha"] if rows else spec["alpha"])
        for item in schedule[len(sweep_paths) :]:
            sweep_started = time.perf_counter()
            sweep = int(item["global_sweep"])
            gamma = float(item["gamma"])
            alpha_used = alpha
            before_sweep = clone_state(state)
            starting_objectives, _, starting_market_diag = economic_objectives(data, before_sweep)
            player_rows = []
            for player in PLAYER_ORDER:
                turn_started = time.perf_counter()
                before_player = clone_state(state)
                market, market_diag = solve_nested_market(data, state)
                reference = float(nested_economic_objective(data, state, market, player))
                prox = (
                    {name: float(value) * gamma for name, value in BASE_PENALTIES.items()}
                    if gamma > 0.0
                    else None
                )
                generated_value, generated_response, generated_diag, generated_retry = response_with_retry(
                    data, state, market, player, maxiter=int(task["maxiter"]), prox=prox
                )
                if not generated_diag["success"]:
                    raise RuntimeError(
                        f"{branch} sweep {sweep} {player}: candidate-generation solve failed: "
                        f"{generated_diag['message']}"
                    )
                if gamma > 0.0:
                    raw_value, raw_response, raw_diag, raw_retry = response_with_retry(
                        data, state, market, player, maxiter=int(task["maxiter"]), prox=None
                    )
                    if not raw_diag["success"]:
                        raise RuntimeError(
                            f"{branch} sweep {sweep} {player}: zero-proximal diagnostic failed: "
                            f"{raw_diag['message']}"
                        )
                else:
                    raw_value, raw_response, raw_diag, raw_retry = (
                        generated_value,
                        generated_response,
                        generated_diag,
                        generated_retry,
                    )
                raw_gain = max(float(raw_value) - reference, 0.0) / max(abs(reference), 1.0)
                generated_gain = max(float(generated_value) - reference, 0.0) / max(abs(reference), 1.0)
                update_player(data, state, generated_response, player, alpha_used)
                post_market, post_market_diag = solve_nested_market(data, state)
                player_rows.append(
                    {
                        "player": player,
                        "pid": os.getpid(),
                        "reference_objective": reference,
                        "raw_zero_proximal_best_response_objective": float(raw_value),
                        "raw_unilateral_gain": raw_gain,
                        "raw_best_response": player_strategy_payload(data, raw_response, player),
                        "raw_strategy_move": _strategy_distance(data, before_player, raw_response, player)[0],
                        "generated_response_economic_objective": float(generated_value),
                        "generated_response_gain": generated_gain,
                        "generated_response": player_strategy_payload(data, generated_response, player),
                        "generated_response_move": _strategy_distance(data, before_player, generated_response, player)[0],
                        "generated_response_proximal_cost": float(generated_diag.get("proximal_cost", 0.0)),
                        "generated_response_penalized_objective": float(
                            generated_diag.get("optimization_objective", generated_value)
                        ),
                        "damped_strategy": player_strategy_payload(data, state, player),
                        "damped_strategy_move": _strategy_distance(data, before_player, state, player)[0],
                        "alpha": alpha_used,
                        "gamma": gamma,
                        "solver": {
                            "generated": generated_diag,
                            "generated_retry_used": generated_retry,
                            "zero_proximal_diagnostic": raw_diag,
                            "zero_proximal_retry_used": raw_retry,
                        },
                        "feasibility": {
                            "reference_market": market_diag,
                            "post_damped_market": post_market_diag,
                        },
                        "post_damped_market": full_state_payload(data, state, post_market)["market"],
                        "elapsed_seconds": time.perf_counter() - turn_started,
                    }
                )
                append_log(
                    log_path,
                    f"branch={branch} sweep={sweep} player={player} gamma={gamma:g} "
                    f"raw_gain={raw_gain:.8g} status=success elapsed={time.perf_counter()-turn_started:.2f}s",
                )
            ending_objectives, ending_market, ending_market_diag = economic_objectives(data, state)
            rx = strategy_metric(data, before_sweep, state)
            rpi = max(
                abs(float(ending_objectives[player]) - float(starting_objectives[player]))
                / max(abs(float(starting_objectives[player])), 1.0)
                for player in PLAYER_ORDER
            )
            history.append(clone_state(state))
            history = history[-5:]
            cycle = detect_cycle(data, history)
            current = {
                "created": now(),
                "branch": branch,
                "description": spec["description"],
                "pid": os.getpid(),
                "sweep": sweep,
                "stage": int(item["stage"]),
                "stage_sweep": int(item["stage_sweep"]),
                "gamma": gamma,
                "proximal_coefficients": {
                    name: float(value) * gamma for name, value in BASE_PENALTIES.items()
                },
                "algorithmic_proximal_penalties_are_candidate_generation_only": True,
                "alpha": alpha_used,
                "player_order": PLAYER_ORDER,
                "gauss_seidel_updates_sequential": True,
                "starting_common_objectives": starting_objectives,
                "ending_common_objectives": ending_objectives,
                "strategy_change_metric": rx,
                "objective_change_metric": rpi,
                "max_raw_unilateral_gain": max(row["raw_unilateral_gain"] for row in player_rows),
                "max_raw_gain_player": max(player_rows, key=lambda row: row["raw_unilateral_gain"])["player"],
                "largest_raw_best_response_move": max(row["raw_strategy_move"] for row in player_rows),
                "players": player_rows,
                "starting_market_diagnostics": starting_market_diag,
                "ending_market_diagnostics": ending_market_diag,
                "cycle_diagnostics": cycle,
                "distance_from_paper": normalized_l2(data, paper, state),
                "periodic_audit_due": sweep % 5 == 0,
                "elapsed_seconds": time.perf_counter() - sweep_started,
                "ending_profile": full_state_payload(data, state, ending_market),
            }
            if spec["mode"] == "adaptive":
                alpha, event = adaptive_next_alpha(alpha_used, spec, rows, current)
            else:
                alpha, event = alpha_used, {"action": "hold", "reasons": []}
            current["adaptive_damping_event"] = event
            current["next_alpha"] = alpha
            atomic_json(out / f"sweep_{sweep:03d}.json", current)
            rows.append(current)
            append_log(
                log_path,
                f"CHECKPOINT branch={branch} sweep={sweep} gamma={gamma:g} alpha={alpha_used:g} "
                f"next_alpha={alpha:g} max_raw_gain={current['max_raw_unilateral_gain']:.8g} "
                f"rx={rx:.8g} elapsed={current['elapsed_seconds']:.2f}s",
            )
        summary = {
            "created": now(),
            "branch": branch,
            "pid": os.getpid(),
            "status": "complete",
            "sweeps_completed": len(rows),
            "last_checkpoint": relative(out / f"sweep_{len(rows):03d}.json") if rows else None,
            "ending_max_raw_unilateral_gain": rows[-1]["max_raw_unilateral_gain"] if rows else None,
            "ending_distance_from_paper": rows[-1]["distance_from_paper"] if rows else None,
            "cycle_ever_detected": any(
                row["cycle_diagnostics"]["two_cycle"] or row["cycle_diagnostics"]["three_cycle"]
                for row in rows
            ),
            "elapsed_seconds": sum(float(row.get("elapsed_seconds", 0.0)) for row in rows),
            "resume_segment_seconds": time.perf_counter() - started,
        }
        atomic_json(out / "summary.json", summary)
        append_log(log_path, f"END branch={branch} status=complete elapsed={summary['elapsed_seconds']:.2f}s")
        return summary
    except Exception as exc:
        failure = {
            "created": now(),
            "branch": branch,
            "pid": os.getpid(),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(out / "failure.json", failure)
        append_log(log_path, f"END branch={branch} status=failed error={failure['error']}")
        return failure


def clip_player_start(data: Any, common: dict[str, dict], start: dict[str, dict], player: str) -> None:
    times = list(data.times or [])
    candidate_caps = mm._implied_capacity_path(data, times, common["dK_net"])
    initial_capacity = float(mm._initial_capacity_by_region(data)[player])
    expansion = float((data.g_exp_ub or {}).get(player, 0.0))
    if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
        expansion *= initial_capacity
    g_dec = float((data.g_dec_ub or {}).get(player, 1.0))
    current = initial_capacity
    for tp, next_tp in zip(times[:-1], times[1:]):
        key = (player, tp)
        lower = max(-g_dec * max(float(candidate_caps[(player, tp)]), 0.0), -g_dec * max(current, 0.0))
        value = float(np.clip(float(start["dK_net"][key]), lower, expansion))
        years = float((data.years_to_next or {}).get(tp, 1.0))
        value = max(value, -max(current, 0.0) / max(years, 1e-12))
        start["dK_net"][key] = value
        current += years * value
    for importer in data.regions:
        if importer == player:
            continue
        for tp in times:
            key = (player, importer, tp)
            start["p_offer"][key] = float(
                np.clip(float(start["p_offer"][key]), 0.0, float(data.p_offer_ub[(player, importer)]))
            )
    _sync_quantity(data, start)


def make_audit_start(
    data: Any,
    paper: dict[str, dict],
    common: dict[str, dict],
    player: str,
    start_spec: dict[str, Any],
) -> dict[str, dict]:
    start = clone_state(common)
    mode = str(start_spec["mode"])
    times = list(data.times or [])
    if mode == "standard":
        for tp in mm._move_times(times):
            start["dK_net"][(player, tp)] = 0.0
        for importer in data.regions:
            if importer == player:
                continue
            for tp in times:
                start["p_offer"][(player, importer, tp)] = float(
                    (data.c_man_t or {}).get((player, tp), data.c_man[player])
                )
    elif mode == "price_factor":
        factor = float(start_spec["factor"])
        for importer in data.regions:
            if importer == player:
                continue
            for tp in times:
                key = (player, importer, tp)
                start["p_offer"][key] = factor * float(start["p_offer"][key])
    elif mode == "cost_mix":
        weight = float(start_spec["weight"])
        for importer in data.regions:
            if importer == player:
                continue
            for tp in times:
                key = (player, importer, tp)
                cost = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))
                start["p_offer"][key] = (1.0 - weight) * float(start["p_offer"][key]) + weight * cost
    elif mode == "dk_factor":
        factor = float(start_spec["factor"])
        for tp in mm._move_times(times):
            key = (player, tp)
            start["dK_net"][key] = factor * float(start["dK_net"][key])
    elif mode == "profile":
        source = load_profile(Path(start_spec["path"]), data, paper)
        for tp in mm._move_times(times):
            start["dK_net"][(player, tp)] = float(source["dK_net"][(player, tp)])
        for importer in data.regions:
            if importer == player:
                continue
            for tp in times:
                start["p_offer"][(player, importer, tp)] = float(
                    source["p_offer"][(player, importer, tp)]
                )
    elif mode == "response":
        response = start_spec["response"]
        for tp, value in response["dK_net"].items():
            start["dK_net"][(player, str(tp))] = float(value)
        for coordinate, value in response["offer_prices"].items():
            importer, tp = coordinate.split("/", maxsplit=1)
            start["p_offer"][(player, importer, tp)] = float(value)
    elif mode != "candidate":
        raise ValueError(f"Unknown audit start mode {mode}")
    clip_player_start(data, common, start, player)
    return start


def audit_start_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint = Path(task["checkpoint"])
    player = str(task["player"])
    label = str(task["start_label"])
    try:
        data, paper, _ = worker_context()
        common = load_profile(checkpoint, data, paper)
        market, market_diag = solve_nested_market(data, common)
        reference = float(nested_economic_objective(data, common, market, player))
        start = make_audit_start(data, paper, common, player, task["start_spec"])
        best, response, diagnostics = nested_best_response(
            data,
            common,
            market,
            player,
            maxiter=int(task["maxiter"]),
            starts=1,
            start_states=[start],
            start_state_labels=[label],
        )
        retry_used = False
        if not diagnostics["success"]:
            retry_used = True
            best, response, diagnostics = nested_best_response(
                data,
                common,
                market,
                player,
                maxiter=max(2 * int(task["maxiter"]), 1000),
                starts=1,
                start_states=[start],
                start_state_labels=[label],
            )
        gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
        move, coordinate, absolute = _strategy_distance(data, common, response, player)
        return {
            "created": now(),
            "status": "success" if diagnostics["success"] else "optimizer_failed",
            "pid": os.getpid(),
            "checkpoint": relative(checkpoint),
            "branch": task["branch"],
            "sweep": int(task["sweep"]),
            "player": player,
            "start_label": label,
            "start_spec": task["start_spec"],
            "reference_objective": reference,
            "best_response_objective": float(best),
            "relative_gain": gain,
            "strategy_move": move,
            "largest_move_coordinate": coordinate,
            "largest_move_absolute": absolute,
            "best_response": player_strategy_payload(data, response, player),
            "optimizer": diagnostics,
            "retry_used": retry_used,
            "reference_market_diagnostics": market_diag,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "created": now(),
            "status": "exception",
            "pid": os.getpid(),
            "checkpoint": relative(checkpoint),
            "branch": task["branch"],
            "sweep": int(task["sweep"]),
            "player": player,
            "start_label": label,
            "start_spec": task["start_spec"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }


def task_id(task: dict[str, Any]) -> str:
    safe_label = "".join(character if character.isalnum() else "_" for character in task["start_label"])
    return f"{task['branch']}_s{int(task['sweep']):03d}_{task['player']}_{safe_label}"


def run_audit_tasks(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    log_path: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    pending = []
    for task in tasks:
        path = output_dir / f"{task_id(task)}.json"
        if path.exists():
            results.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append((task, path))
    append_log(log_path, f"AUDIT_POOL start workers={workers} pending={len(pending)} cached={len(results)}")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_paths = {pool.submit(audit_start_task, task): path for task, path in pending}
            for future in as_completed(future_paths):
                path = future_paths[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "created": now(),
                        "status": "executor_exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                atomic_json(path, result)
                results.append(result)
                append_log(
                    log_path,
                    f"AUDIT_TASK id={path.stem} pid={result.get('pid')} status={result.get('status')} "
                    f"elapsed={result.get('elapsed_seconds')} error={result.get('error')}",
                )
    append_log(log_path, f"AUDIT_POOL end tasks={len(results)}")
    return results


def aggregate_audits(
    results: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    output_dir: Path,
    starts_per_player: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for result in results:
        if "branch" in result and "sweep" in result:
            grouped.setdefault((str(result["branch"]), int(result["sweep"])), []).append(result)
    audits = []
    for target in targets:
        key = (str(target["branch"]), int(target["sweep"]))
        rows = grouped.get(key, [])
        players = []
        for player in PLAYER_ORDER:
            attempts = [row for row in rows if row.get("player") == player]
            successful = [row for row in attempts if row.get("status") == "success"]
            chosen = max(successful, key=lambda row: float(row["best_response_objective"])) if successful else None
            players.append(
                {
                    "player": player,
                    "attempts": attempts,
                    "successful_attempts": len(successful),
                    "best": chosen,
                    "relative_gain": None if chosen is None else float(chosen["relative_gain"]),
                }
            )
        complete = all(player["best"] is not None for player in players)
        all_success = all(
            len(player["attempts"]) == starts_per_player
            and all(attempt.get("status") == "success" for attempt in player["attempts"])
            for player in players
        )
        max_gain = max((float(player["relative_gain"]) for player in players if player["relative_gain"] is not None), default=None)
        critical = None if max_gain is None else max(players, key=lambda player: -1.0 if player["relative_gain"] is None else float(player["relative_gain"]))["player"]
        audit = {
            "created": now(),
            "label": target["label"],
            "branch": target["branch"],
            "sweep": int(target["sweep"]),
            "checkpoint": relative(Path(target["checkpoint"])),
            "common_profile_frozen": True,
            "algorithmic_proximal_penalties": 0.0,
            "starts_per_player": starts_per_player,
            "players": players,
            "complete": complete,
            "all_attempts_successful": all_success,
            "max_relative_gain": max_gain,
            "max_gain_player": critical,
            "equilibrium_verified": bool(complete and all_success and max_gain is not None and max_gain <= 0.01),
        }
        atomic_json(output_dir / f"audit_{target['branch']}_s{int(target['sweep']):03d}.json", audit)
        audits.append(audit)
    return audits


def collect_periodic_targets(run_root: Path) -> list[dict[str, Any]]:
    targets = []
    seen: set[tuple[str, int]] = set()
    for branch in BRANCHES:
        branch_rows = []
        for checkpoint in sorted((run_root / "branches" / branch).glob("sweep_*.json")):
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            sweep = int(payload["sweep"])
            branch_rows.append((checkpoint, payload))
            if sweep % 5 == 0:
                seen.add((branch, sweep))
                targets.append(
                    {
                        "branch": branch,
                        "sweep": sweep,
                        "checkpoint": str(checkpoint.resolve()),
                        "label": f"{branch}_sweep_{sweep:03d}_lightweight",
                        "selection_reason": "periodic_fifth_sweep",
                        "trajectory_max_raw_gain": float(payload["max_raw_unilateral_gain"]),
                    }
                )
        if not branch_rows:
            continue
        additions = [
            (min(branch_rows, key=lambda item: float(item[1]["max_raw_unilateral_gain"])), "branch_best_within_sweep_diagnostic"),
            (branch_rows[-1], "terminal_profile"),
        ]
        for (checkpoint, payload), reason in additions:
            sweep = int(payload["sweep"])
            if (branch, sweep) in seen:
                continue
            seen.add((branch, sweep))
            targets.append(
                {
                    "branch": branch,
                    "sweep": sweep,
                    "checkpoint": str(checkpoint.resolve()),
                    "label": f"{branch}_sweep_{sweep:03d}_lightweight",
                    "selection_reason": reason,
                    "trajectory_max_raw_gain": float(payload["max_raw_unilateral_gain"]),
                }
            )
    # Prioritize the most promising trajectory diagnostics.  This makes the
    # best intermediate candidates useful even if a later external interruption
    # prevents the complete periodic-audit queue from finishing.
    targets.sort(key=lambda target: float(target["trajectory_max_raw_gain"]))
    return targets


def build_tasks(targets: list[dict[str, Any]], start_specs: list[tuple[str, dict[str, Any]]], maxiter: int) -> list[dict[str, Any]]:
    return [
        {
            **target,
            "player": player,
            "start_label": label,
            "start_spec": spec,
            "maxiter": maxiter,
        }
        for target in targets
        for player in PLAYER_ORDER
        for label, spec in start_specs
    ]


def select_final_targets(audits: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    eligible = [audit for audit in audits if audit.get("max_relative_gain") is not None]
    eligible.sort(key=lambda audit: (not bool(audit.get("complete")), float(audit["max_relative_gain"])))
    selected = []
    seen: set[str] = set()
    for audit in eligible:
        checkpoint = str((ROOT / audit["checkpoint"]).resolve())
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        selected.append(
            {
                "branch": audit["branch"],
                "sweep": int(audit["sweep"]),
                "checkpoint": checkpoint,
                "label": f"{audit['branch']}_sweep_{int(audit['sweep']):03d}_full",
                "lightweight_max_relative_gain": float(audit["max_relative_gain"]),
            }
        )
        if len(selected) >= count:
            break
    return selected


def branch_plots(run_root: Path) -> None:
    for branch in BRANCHES:
        paths = sorted((run_root / "branches" / branch).glob("sweep_*.json"))
        if not paths:
            continue
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        sweeps = [int(row["sweep"]) for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
        axes[0].plot(sweeps, [row["max_raw_unilateral_gain"] for row in rows], linewidth=1)
        axes[0].axhline(0.01, color="tab:red", linestyle="--", linewidth=1)
        axes[0].set_ylabel("raw gain")
        axes[1].plot(sweeps, [row["strategy_change_metric"] for row in rows], linewidth=1)
        axes[1].set_ylabel("strategy change")
        axes[2].plot(sweeps, [row["alpha"] for row in rows], linewidth=1, label="alpha")
        if branch == "O4":
            axes[2].plot(sweeps, [row["gamma"] for row in rows], linewidth=1, label="gamma")
            axes[2].legend()
        axes[2].set_ylabel("alpha / gamma")
        axes[2].set_xlabel("complete GS sweep")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle(f"{branch}: {BRANCHES[branch]['description']}")
        fig.tight_layout()
        fig.savefig(run_root / "branches" / branch / "trajectory.png", dpi=180)
        plt.close(fig)


def write_best_archive(run_root: Path, audits: list[dict[str, Any]], name: str) -> dict[str, Any]:
    complete = [audit for audit in audits if audit.get("complete") and audit.get("max_relative_gain") is not None]
    by_branch = {}
    for branch in BRANCHES:
        candidates = [audit for audit in complete if audit["branch"] == branch]
        if candidates:
            best = min(candidates, key=lambda audit: float(audit["max_relative_gain"]))
            by_branch[branch] = {
                "checkpoint": best["checkpoint"],
                "sweep": best["sweep"],
                "max_relative_gain": best["max_relative_gain"],
                "max_gain_player": best["max_gain_player"],
            }
    global_best = min(complete, key=lambda audit: float(audit["max_relative_gain"])) if complete else None
    payload = {
        "created": now(),
        "source": name,
        "by_branch": by_branch,
        "global": None
        if global_best is None
        else {
            "branch": global_best["branch"],
            "sweep": global_best["sweep"],
            "checkpoint": global_best["checkpoint"],
            "max_relative_gain": global_best["max_relative_gain"],
            "max_gain_player": global_best["max_gain_player"],
            "equilibrium_verified": global_best["equilibrium_verified"],
        },
    }
    atomic_json(run_root / f"best_so_far_{name}.json", payload)
    return payload


def stress_targets(full_audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    for audit in full_audits:
        if not audit.get("equilibrium_verified"):
            continue
        targets.append(
            {
                "branch": audit["branch"],
                "sweep": audit["sweep"],
                "checkpoint": str((ROOT / audit["checkpoint"]).resolve()),
                "label": f"{audit['branch']}_sweep_{int(audit['sweep']):03d}_stress",
                "critical_players": [
                    player["player"]
                    for player in sorted(
                        audit["players"],
                        key=lambda player: float(player["relative_gain"] or -1.0),
                        reverse=True,
                    )[:2]
                ],
                "full_audit": audit,
            }
        )
    return targets


def build_stress_tasks(targets: list[dict[str, Any]], maxiter: int) -> list[dict[str, Any]]:
    tasks = []
    generic_specs: list[tuple[str, dict[str, Any]]] = [
        ("candidate", {"mode": "candidate"}),
        ("standard", {"mode": "standard"}),
        *[(f"price_factor_{factor:g}", {"mode": "price_factor", "factor": factor}) for factor in (0.50, 0.75, 0.90, 0.95, 1.05, 1.10, 1.25, 1.50)],
        *[(f"capacity_factor_{factor:g}", {"mode": "dk_factor", "factor": factor}) for factor in (0.0, 0.5, 1.5)],
        *[(f"cost_mix_{weight:g}", {"mode": "cost_mix", "weight": weight}) for weight in (0.25, 0.50, 1.0)],
        ("paper_profile", {"mode": "profile", "path": str((ROOT / "outputs/equilibrium_search/ch-row-apac-us-eu-af/local_paper_profile_zero_prox/paper_profile.json").resolve())}),
        ("previous_local_candidate", {"mode": "profile", "path": str(LOCAL_BEST_CHECKPOINT.resolve())}),
        ("previous_cost_price_candidate", {"mode": "profile", "path": str(CERTIFIED_CHECKPOINT.resolve())}),
    ]
    for target in targets:
        for player in target["critical_players"]:
            player_record = next(row for row in target["full_audit"]["players"] if row["player"] == player)
            best_response = player_record["best"]["best_response"]
            specs = generic_specs + [("existing_best_response", {"mode": "response", "response": best_response})]
            checkpoint = Path(target["checkpoint"])
            sweep = int(target["sweep"])
            for neighbor in (sweep - 1, sweep + 1):
                neighbor_path = checkpoint.with_name(f"sweep_{neighbor:03d}.json")
                if neighbor_path.exists():
                    specs.append((f"neighbor_sweep_{neighbor:03d}", {"mode": "profile", "path": str(neighbor_path.resolve())}))
            for label, spec in specs:
                tasks.append(
                    {
                        **{key: value for key, value in target.items() if key != "full_audit"},
                        "player": player,
                        "start_label": label,
                        "start_spec": spec,
                        "maxiter": maxiter,
                    }
                )
    return tasks


def combine_stress(full_audits: list[dict[str, Any]], stress_results: list[dict[str, Any]], run_root: Path) -> list[dict[str, Any]]:
    combined = []
    for audit in full_audits:
        rows = [
            result
            for result in stress_results
            if result.get("branch") == audit["branch"] and int(result.get("sweep", -1)) == int(audit["sweep"])
        ]
        if not rows:
            continue
        players = []
        for player_record in audit["players"]:
            attempts = list(player_record["attempts"]) + [row for row in rows if row.get("player") == player_record["player"]]
            successful = [row for row in attempts if row.get("status") == "success"]
            best = max(successful, key=lambda row: float(row["best_response_objective"])) if successful else None
            players.append(
                {
                    "player": player_record["player"],
                    "attempts": attempts,
                    "successful_attempts": len(successful),
                    "best": best,
                    "relative_gain": None if best is None else float(best["relative_gain"]),
                }
            )
        max_gain = max(float(player["relative_gain"]) for player in players if player["relative_gain"] is not None)
        result = {
            **{key: value for key, value in audit.items() if key != "players"},
            "label": audit["label"].replace("_full", "_stress_combined"),
            "players": players,
            "max_relative_gain": max_gain,
            "max_gain_player": max(players, key=lambda player: float(player["relative_gain"] or -1.0))["player"],
            "stress_tested": bool(rows),
            "equilibrium_verified": bool(all(player["best"] is not None for player in players) and max_gain <= 0.01),
        }
        atomic_json(run_root / "audits" / "stress" / f"audit_{audit['branch']}_s{int(audit['sweep']):03d}.json", result)
        combined.append(result)
    return combined


def state_comparison_record(
    data: Any,
    paper: dict[str, dict],
    name: str,
    state: dict[str, dict],
    audit: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    market, diagnostics = solve_nested_market(data, state)
    objectives, _, _ = economic_objectives(data, state, market)
    capacities = mm._implied_capacity_path(data, list(data.times or []), state["dK_net"])
    regrets = {}
    for player in PLAYER_ORDER:
        row = next((row for row in audit.get("players", []) if row.get("player") == player), None)
        if row is None:
            regrets[player] = None
        elif row.get("relative_gain") is not None:
            regrets[player] = float(row["relative_gain"])
        elif row.get("best") is not None:
            regrets[player] = float(row["best"]["relative_gain"])
        else:
            regrets[player] = None
    return {
        "name": name,
        "provenance": provenance,
        "distance_from_paper": normalized_l2(data, paper, state),
        "objectives": objectives,
        "regrets": regrets,
        "max_relative_gain": audit.get("max_relative_gain"),
        "max_gain_player": audit.get("max_gain_player"),
        "equilibrium_verified": audit.get("equilibrium_verified"),
        "capacities": [
            {"player": player, "time": tp, "value": float(capacities[(player, tp)])}
            for player in PLAYER_ORDER
            for tp in list(data.times or [])
        ],
        "offer_prices": _serialize_state(state)["p_offer"],
        "clearing_prices": [
            {"region": region, "time": tp, "value": float(market["lam"][(region, tp)])}
            for region in data.regions
            for tp in list(data.times or [])
        ],
        "market_diagnostics": diagnostics,
    }


def normalize_existing_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    players = []
    for row in payload["players"]:
        players.append({"player": row["player"], "relative_gain": float(row["relative_gain"])})
    return {
        "players": players,
        "max_relative_gain": float(payload["max_relative_gain"]),
        "max_gain_player": max(players, key=lambda row: row["relative_gain"])["player"],
        "equilibrium_verified": bool(payload.get("equilibrium_verified", False)),
    }


def write_report_and_comparison(
    run_root: Path,
    periodic_audits: list[dict[str, Any]],
    final_audits: list[dict[str, Any]],
    stress_audits: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> None:
    data, paper, local = worker_context()
    certified = load_profile(CERTIFIED_CHECKPOINT, data, paper)
    paper_audit = normalize_existing_audit(PAPER_AUDIT)
    local_audit = normalize_existing_audit(LOCAL_BEST_AUDIT)
    certified_audit = normalize_existing_audit(CERTIFIED_AUDIT)
    evaluated = stress_audits or final_audits
    best = min(
        (audit for audit in evaluated if audit.get("max_relative_gain") is not None),
        key=lambda audit: float(audit["max_relative_gain"]),
        default=None,
    )
    records = [
        state_comparison_record(data, paper, "paper_profile", paper, paper_audit, {"iteration": SOURCE_ITERATION}),
        state_comparison_record(
            data,
            paper,
            "previous_best_local",
            local,
            local_audit,
            {"checkpoint": relative(LOCAL_BEST_CHECKPOINT), "branch": "F", "sweep": 6, "alpha": 0.50},
        ),
        state_comparison_record(
            data,
            paper,
            "previous_cost_price_candidate",
            certified,
            certified_audit,
            {"checkpoint": relative(CERTIFIED_CHECKPOINT)},
        ),
    ]
    if best is not None:
        best_path = ROOT / best["checkpoint"]
        best_state = load_profile(best_path, data, paper)
        sweep_payload = json.loads(best_path.read_text(encoding="utf-8"))
        records.append(
            state_comparison_record(
                data,
                paper,
                "best_overnight_candidate",
                best_state,
                best,
                {
                    "checkpoint": best["checkpoint"],
                    "branch": best["branch"],
                    "sweep": best["sweep"],
                    "alpha": sweep_payload["alpha"],
                    "gamma": sweep_payload["gamma"],
                    "stress_tested": best.get("stress_tested", False),
                },
            )
        )
    atomic_json(run_root / "comparison.json", {"created": now(), "profiles": records})
    for field, filename, columns in (
        ("capacities", "comparison_capacities.csv", ["profile", "player", "time", "value"]),
        ("offer_prices", "comparison_offer_prices.csv", ["profile", "exporter", "importer", "time", "value"]),
        ("clearing_prices", "comparison_clearing_prices.csv", ["profile", "region", "time", "value"]),
    ):
        with (run_root / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in records:
                for row in record[field]:
                    writer.writerow({"profile": record["name"], **row})
    periodic_best = min(periodic_audits, key=lambda audit: float(audit["max_relative_gain"]), default=None)
    final_best = min(final_audits, key=lambda audit: float(audit["max_relative_gain"]), default=None)
    stress_best = min(stress_audits, key=lambda audit: float(audit["max_relative_gain"]), default=None)
    lines = [
        "# Overnight equilibrium-search report",
        "",
        f"Completed: {now()}",
        "",
        "All frozen-profile audits evaluate the original economic game with zero algorithmic proximal penalties. O4's positive penalties were used only to generate candidates.",
        "",
        "## Main results",
        "",
        f"- Smallest lightweight frozen regret: {None if periodic_best is None else f'{100*periodic_best['max_relative_gain']:.4f}% at {periodic_best['branch']} sweep {periodic_best['sweep']}' }.",
        f"- Smallest three-start regret: {None if final_best is None else f'{100*final_best['max_relative_gain']:.4f}% at {final_best['branch']} sweep {final_best['sweep']}' }.",
        f"- Smallest stress-tested regret: {None if stress_best is None else f'{100*stress_best['max_relative_gain']:.4f}% at {stress_best['branch']} sweep {stress_best['sweep']}' }.",
        f"- Any three-start 1% pass: {any(audit.get('equilibrium_verified') for audit in final_audits)}.",
        f"- Any stress-test 1% pass: {any(audit.get('equilibrium_verified') for audit in stress_audits)}.",
        "",
        "## Runtime and parallelism",
        "",
        f"- Worker count: {runtime['worker_count']} process workers; solver/BLAS threads capped at one.",
        f"- Wall-clock runtime: {runtime['wall_seconds']/3600:.3f} hours.",
        f"- Sum of measured independent task runtimes: {runtime['summed_task_seconds']/3600:.3f} hours.",
        f"- Measured wall-clock time saved by parallel execution: {runtime['parallel_seconds_saved']/3600:.3f} hours ({runtime['parallel_savings_fraction']:.1%}).",
        "",
        "## Method notes",
        "",
        "- O1, O2, O3, and O4 start from the exactly replayed iteration-21 paper profile.",
        "- O5 and O6 start from the exact Branch-F alpha-0.50 sweep-6 checkpoint.",
        "- Every GS sweep uses CH -> ROW -> APAC -> US -> EU -> AF sequentially.",
        "- Adaptive damping halves alpha after detected cycles, displacement spikes, or large raw response jumps; it raises alpha by 20% after four stable improving sweeps.",
        "- Inspect `comparison.json`, the audit JSON files, and branch trajectory plots for the detailed answers to the experiment questions.",
    ]
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def system_resources() -> dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = None
    total_memory = None
    free_memory = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        total_memory = int(memory.total)
        free_memory = int(memory.available)
    except ImportError:
        if os.name == "nt":
            try:
                import ctypes

                class MemoryStatus(ctypes.Structure):
                    _fields_ = [
                        ("length", ctypes.c_ulong),
                        ("memory_load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong),
                        ("available_physical", ctypes.c_ulonglong),
                        ("total_page_file", ctypes.c_ulonglong),
                        ("available_page_file", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong),
                        ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended_virtual", ctypes.c_ulonglong),
                    ]

                memory = MemoryStatus()
                memory.length = ctypes.sizeof(MemoryStatus)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
                total_memory = int(memory.total_physical)
                free_memory = int(memory.available_physical)
            except Exception:
                pass
    if physical is None:
        # The experiment machine is a conventional SMT desktop; this estimate
        # is deliberately conservative for selecting outer process workers.
        physical = max(logical // 2, 1)
    return {
        "hostname": socket.gethostname(),
        "logical_cpus": logical,
        "physical_cores": physical,
        "total_memory_bytes": total_memory,
        "available_memory_bytes_at_launch": free_memory,
    }


def sensible_workers(requested: int | None, resources: dict[str, Any]) -> int:
    if requested is not None:
        return max(1, int(requested))
    cores = int(resources.get("physical_cores") or max(int(resources["logical_cpus"]) // 2, 1))
    cpu_limit = max(1, cores // 2)
    available = resources.get("available_memory_bytes_at_launch")
    memory_limit = max(1, int(available // (1536 * 1024**2))) if available else 3
    return min(4, cpu_limit, memory_limit)


def run_experiment(args: argparse.Namespace) -> Path:
    wall_started = time.perf_counter()
    run_id = args.run_id or datetime.now().astimezone().strftime("overnight_%Y%m%d_%H%M%S")
    run_root = EXPERIMENT_BASE / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "orchestrator.log"
    resources = system_resources()
    workers = sensible_workers(args.workers, resources)
    manifest_path = run_root / "manifest.json"
    workflow_manifest = ROOT / "workflow" / f"{run_id}_manifest.json"
    resuming = manifest_path.exists()
    if resuming:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("resume_history", []).append(
            {
                "resumed": now(),
                "previous_status": manifest.get("status"),
                "previous_phase": manifest.get("phase"),
                "previous_pid": manifest.get("pid"),
                "reason": "recover checkpointed run after external interruption",
            }
        )
        manifest.update(
            {
                "status": "running",
                "pid": os.getpid(),
                "resources": resources,
                "worker_count": workers,
                "phase": "branch_search",
            }
        )
    else:
        manifest = {
            "created": now(),
            "run_id": run_id,
            "status": "running",
            "pid": os.getpid(),
            "output_root": relative(run_root),
            "workflow_manifest": relative(workflow_manifest),
            "resources": resources,
            "worker_count": workers,
            "worker_policy": "process based; outer workers limited by physical cores and available memory; BLAS threads capped at one",
            "source_workbook": relative(SOURCE_WORKBOOK),
            "source_iteration": SOURCE_ITERATION,
            "standard_player_order": PLAYER_ORDER,
            "local_start_checkpoint": relative(LOCAL_BEST_CHECKPOINT),
            "local_start_audit": relative(LOCAL_BEST_AUDIT),
            "preserved_certified_checkpoint": relative(CERTIFIED_CHECKPOINT),
            "preserved_certified_audit": relative(CERTIFIED_AUDIT),
            "original_proximal_penalties": BASE_PENALTIES,
            "branches": BRANCHES,
            "phase": "branch_search",
            "branch_results": [],
        }
    atomic_json(manifest_path, manifest)
    atomic_json(workflow_manifest, manifest)
    append_log(
        log_path,
        f"{'RESUME' if resuming else 'START'} run_id={run_id} workers={workers} "
        f"resources={json.dumps(resources, sort_keys=True)}",
    )
    selected_branches = list(BRANCHES)
    if args.branches:
        selected_branches = list(args.branches)
    branch_tasks = [
        {
            "branch": branch,
            "spec": {
                **BRANCHES[branch],
                **({"max_sweeps": args.smoke_sweeps} if args.smoke_sweeps and BRANCHES[branch]["mode"] != "homotopy" else {}),
                **(
                    {"positive_stage_sweeps": 1, "zero_stage_sweeps": 1}
                    if args.smoke_sweeps and BRANCHES[branch]["mode"] == "homotopy"
                    else {}
                ),
            },
            "run_root": str(run_root.resolve()),
            "maxiter": args.maxiter,
        }
        for branch in selected_branches
    ]
    branch_phase_started = time.perf_counter()
    branch_result_map = {
        str(result["branch"]): result
        for result in manifest.get("branch_results", [])
        if result.get("branch") in selected_branches
    }
    with ProcessPoolExecutor(max_workers=min(workers, len(branch_tasks))) as pool:
        futures = {pool.submit(run_branch_task, task): task["branch"] for task in branch_tasks}
        for future in as_completed(futures):
            branch = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "branch": branch,
                    "status": "executor_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": 0.0,
                }
            branch_result_map[branch] = result
            branch_results = [branch_result_map[name] for name in selected_branches if name in branch_result_map]
            append_log(log_path, f"BRANCH_END branch={branch} status={result.get('status')} pid={result.get('pid')} elapsed={result.get('elapsed_seconds')}")
            manifest["branch_results"] = branch_results
            atomic_json(manifest_path, manifest)
            atomic_json(workflow_manifest, manifest)
    branch_results = [branch_result_map[name] for name in selected_branches if name in branch_result_map]
    branch_phase_wall = time.perf_counter() - branch_phase_started
    branch_plots(run_root)

    manifest["phase"] = "lightweight_audits"
    atomic_json(manifest_path, manifest)
    atomic_json(workflow_manifest, manifest)
    periodic_targets = collect_periodic_targets(run_root)
    if args.smoke_sweeps:
        # Exercise the audit pipeline even when no fifth sweep exists.
        periodic_targets = []
        for branch in selected_branches:
            paths = sorted((run_root / "branches" / branch).glob("sweep_*.json"))
            if paths:
                payload = json.loads(paths[-1].read_text(encoding="utf-8"))
                periodic_targets.append(
                    {
                        "branch": branch,
                        "sweep": int(payload["sweep"]),
                        "checkpoint": str(paths[-1].resolve()),
                        "label": f"{branch}_sweep_{int(payload['sweep']):03d}_lightweight",
                    }
                )
    light_tasks = build_tasks(periodic_targets, [("candidate", {"mode": "candidate"})], args.maxiter)
    light_results = run_audit_tasks(light_tasks, run_root / "audits" / "lightweight" / "tasks", workers, log_path)
    periodic_audits = aggregate_audits(
        light_results,
        periodic_targets,
        run_root / "audits" / "lightweight",
        starts_per_player=1,
    )
    light_archive = write_best_archive(run_root, periodic_audits, "lightweight")

    manifest["phase"] = "final_three_start_audits"
    manifest["lightweight_best"] = light_archive
    atomic_json(manifest_path, manifest)
    atomic_json(workflow_manifest, manifest)
    final_targets = select_final_targets(periodic_audits, count=min(5, len(periodic_audits)))
    final_starts = [
        ("candidate", {"mode": "candidate"}),
        ("standard", {"mode": "standard"}),
        ("price_factor_0p95", {"mode": "price_factor", "factor": 0.95}),
    ]
    final_tasks = build_tasks(final_targets, final_starts, max(args.maxiter, 600))
    final_results = run_audit_tasks(final_tasks, run_root / "audits" / "final" / "tasks", workers, log_path)
    final_audits = aggregate_audits(final_results, final_targets, run_root / "audits" / "final", starts_per_player=3)
    final_archive = write_best_archive(run_root, final_audits, "final")

    manifest["phase"] = "stress_tests"
    manifest["final_best"] = final_archive
    atomic_json(manifest_path, manifest)
    atomic_json(workflow_manifest, manifest)
    to_stress = stress_targets(final_audits)
    stress_task_list = build_stress_tasks(to_stress, max(args.maxiter, 900))
    stress_results = run_audit_tasks(stress_task_list, run_root / "audits" / "stress" / "tasks", workers, log_path)
    stress_audits = combine_stress(final_audits, stress_results, run_root)
    stress_archive = write_best_archive(run_root, stress_audits, "stress_combined")

    all_task_seconds = sum(float(result.get("elapsed_seconds", 0.0)) for result in branch_results)
    all_task_seconds += sum(float(result.get("elapsed_seconds", 0.0)) for result in light_results)
    all_task_seconds += sum(float(result.get("elapsed_seconds", 0.0)) for result in final_results)
    all_task_seconds += sum(float(result.get("elapsed_seconds", 0.0)) for result in stress_results)
    wall_seconds = time.perf_counter() - wall_started
    saved = max(all_task_seconds - wall_seconds, 0.0)
    runtime = {
        "worker_count": workers,
        "branch_phase_wall_seconds": branch_phase_wall,
        "wall_seconds": wall_seconds,
        "summed_task_seconds": all_task_seconds,
        "parallel_seconds_saved": saved,
        "parallel_savings_fraction": saved / all_task_seconds if all_task_seconds > 0 else 0.0,
    }
    write_report_and_comparison(run_root, periodic_audits, final_audits, stress_audits, runtime)
    manifest.update(
        {
            "updated": now(),
            "status": "complete" if all(result.get("status") == "complete" for result in branch_results) else "complete_with_branch_failures",
            "phase": "complete",
            "periodic_audits": len(periodic_audits),
            "final_audits": len(final_audits),
            "stress_tested_candidates": len(to_stress),
            "stress_best": stress_archive,
            "runtime": runtime,
            "report": relative(run_root / "report.md"),
            "comparison": relative(run_root / "comparison.json"),
        }
    )
    atomic_json(manifest_path, manifest)
    atomic_json(workflow_manifest, manifest)
    append_log(log_path, f"END status={manifest['status']} wall={wall_seconds:.2f}s")
    print(json.dumps({"run_root": str(run_root), "manifest": str(manifest_path), "status": manifest["status"]}, indent=2), flush=True)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--branches", nargs="+", choices=list(BRANCHES))
    parser.add_argument("--smoke-sweeps", type=int, help="Development check: shorten each selected branch and force one audit target.")
    args = parser.parse_args()
    if args.maxiter < 1:
        raise ValueError("--maxiter must be positive")
    if args.smoke_sweeps is not None and args.smoke_sweeps < 1:
        raise ValueError("--smoke-sweeps must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
