from __future__ import annotations

"""Reproducible zero-proximal local equilibrium search around the paper profile.

The runner deliberately has its own output tree and manifest.  It never reads a
mutable search manifest and never writes to any of the earlier equilibrium-search
directories.  Best responses use the exact convex lower-market solution and the
economic objective of the submitted model; damping is applied only after each
unrestricted player solve.
"""

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _strategy_distance, _zero_prox_data
from scripts.continue_selected_equilibrium import (
    PLAYER_ORDER,
    SOURCE_ITERATION,
    SOURCE_WORKBOOK,
    _initial_model_data,
    replay_accepted_state,
)
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.search_nested_equilibrium import _deserialize_state, _serialize_state, _sync_quantity


EXPERIMENT = "local_paper_profile_zero_prox"
OUTPUT_ROOT = ROOT / "outputs" / "equilibrium_search" / "ch-row-apac-us-eu-af" / EXPERIMENT
MANIFEST_PATH = ROOT / "workflow" / f"{EXPERIMENT}_manifest.json"
CERTIFIED_CHECKPOINT = (
    ROOT
    / "outputs/equilibrium_search/ch-row-apac-us-eu-af/nested_zero_prox"
    / "checkpoint_20260914_184749.json"
)
CERTIFIED_AUDIT = CERTIFIED_CHECKPOINT.with_name("audit_20260914_184948.json")

BRANCHES = {
    "A": {"description": "exact reported paper equilibrium", "kind": "paper"},
    "B": {"description": "full-strategy uniform +/-5% perturbation", "kind": "full", "size": 0.05, "seed": 520260914},
    "C": {"description": "full-strategy uniform +/-10% perturbation", "kind": "full", "size": 0.10, "seed": 1020260914},
    "D": {"description": "price-only uniform +/-10% perturbation", "kind": "price", "size": 0.10, "seed": 7520260914},
    "E": {"description": "25% interpolation from paper prices toward manufacturing cost", "kind": "cost", "weight": 0.25},
    "F": {"description": "50% interpolation from paper prices toward manufacturing cost", "kind": "cost", "weight": 0.50},
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def clone_state(state: dict[str, dict]) -> dict[str, dict]:
    return {name: dict(values) for name, values in state.items()}


def paper_state():
    data0, base_cfg, excel_initial = _initial_model_data()
    state, replay_error = replay_accepted_state(
        SOURCE_WORKBOOK, excel_initial, data0, through_iteration=SOURCE_ITERATION
    )
    if replay_error > 1e-10:
        raise RuntimeError(f"Paper-profile replay failed: residual error={replay_error:.3g}")
    data = _zero_prox_data(base_cfg)
    _sync_quantity(data, state)
    return data, state, replay_error


def full_state_payload(data, state: dict[str, dict], market: dict[str, dict] | None = None):
    capacities = mm._implied_capacity_path(data, list(data.times or []), state["dK_net"])
    payload = {
        "strategy": _serialize_state(state),
        "capacities": [
            {"player": p, "time": t, "value": float(capacities[(p, t)])}
            for p in PLAYER_ORDER for t in list(data.times or [])
        ],
    }
    if market is not None:
        payload["market"] = {
            "clearing_prices": [
                {"region": r, "time": t, "value": float(market["lam"][(r, t)])}
                for r in data.regions for t in list(data.times or [])
            ],
            "demand": [
                {"region": r, "time": t, "value": float(market["x_dem"][(r, t)])}
                for r in data.regions for t in list(data.times or [])
            ],
            "trade_flows": [
                {"exporter": e, "importer": i, "time": t, "value": float(market["x"][(e, i, t)])}
                for e in data.regions for i in data.regions for t in list(data.times or [])
            ],
        }
    return payload


def player_strategy_payload(data, state, player):
    caps = mm._implied_capacity_path(data, list(data.times or []), state["dK_net"])
    return {
        "dK_net": {t: float(state["dK_net"][(player, t)]) for t in mm._move_times(list(data.times or []))},
        "capacities": {t: float(caps[(player, t)]) for t in list(data.times or [])},
        "offer_prices": {
            f"{importer}/{t}": float(state["p_offer"][(player, importer, t)])
            for importer in data.regions if importer != player for t in list(data.times or [])
        },
    }


def objectives(data, state, market=None):
    if market is None:
        market, diagnostics = solve_nested_market(data, state)
    else:
        diagnostics = None
    return (
        {p: float(nested_economic_objective(data, state, market, p)) for p in PLAYER_ORDER},
        market,
        diagnostics,
    )


def make_initialization(data, paper, branch: str):
    spec = BRANCHES[branch]
    state = clone_state(paper)
    kind = spec["kind"]
    rng = np.random.default_rng(spec.get("seed"))
    times = list(data.times or [])
    if kind == "full":
        size = float(spec["size"])
        paper_caps = mm._implied_capacity_path(data, times, paper["dK_net"])
        initial_caps = mm._initial_capacity_by_region(data)
        for p in PLAYER_ORDER:
            current = float(initial_caps[p])
            for t, t_next in zip(times[:-1], times[1:]):
                target = float(paper_caps[(p, t_next)]) * (1.0 + rng.uniform(-size, size))
                years = float((data.years_to_next or {}).get(t, 1.0))
                proposed = (target - current) / years
                expansion = float((data.g_exp_ub or {}).get(p, 0.0))
                if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
                    expansion *= max(current, 0.0)
                decline = float((data.g_dec_ub or {}).get(p, 1.0)) * max(current, 0.0)
                state["dK_net"][(p, t)] = float(np.clip(proposed, -decline, expansion))
                current += years * state["dK_net"][(p, t)]
    if kind in {"full", "price"}:
        size = float(spec["size"])
        for e in data.regions:
            for i in data.regions:
                if e == i:
                    continue
                for t in times:
                    key = (e, i, t)
                    upper = float(data.p_offer_ub[(e, i)])
                    state["p_offer"][key] = float(
                        np.clip(float(paper["p_offer"][key]) * (1.0 + rng.uniform(-size, size)), 0.0, upper)
                    )
    if kind == "cost":
        weight = float(spec["weight"])
        for e in data.regions:
            for i in data.regions:
                if e == i:
                    continue
                for t in times:
                    key = (e, i, t)
                    cost = float((data.c_man_t or {}).get((e, t), data.c_man[e]))
                    state["p_offer"][key] = (1.0 - weight) * float(paper["p_offer"][key]) + weight * cost
    _sync_quantity(data, state)
    return state


def vector_and_scales(data, state, player: str | None = None):
    players = [player] if player else PLAYER_ORDER
    values, scales, labels = [], [], []
    initial_caps = mm._initial_capacity_by_region(data)
    for p in players:
        expansion = float((data.g_exp_ub or {}).get(p, 0.0))
        if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
            expansion *= float(initial_caps[p])
        scale = max(expansion, float((data.g_dec_ub or {}).get(p, 0.0)) * float(initial_caps[p]), 1.0)
        for t in mm._move_times(list(data.times or [])):
            values.append(float(state["dK_net"][(p, t)])); scales.append(scale); labels.append(f"dK_net[{p},{t}]")
        for i in data.regions:
            if i == p:
                continue
            for t in list(data.times or []):
                values.append(float(state["p_offer"][(p, i, t)]))
                scales.append(max(float(data.p_offer_ub[(p, i)]), 1.0)); labels.append(f"p_offer[{p},{i},{t}]")
    return np.asarray(values), np.asarray(scales), labels


def normalized_l2(data, left, right, player: str | None = None):
    a, scales, _ = vector_and_scales(data, left, player)
    b, _, _ = vector_and_scales(data, right, player)
    return float(np.linalg.norm((b - a) / scales) / max(math.sqrt(len(a)), 1.0))


def strategy_metric(data, before, after):
    return max(_strategy_distance(data, before, after, p)[0] for p in PLAYER_ORDER)


def audit_profile(data, state, *, starts: int, maxiter: int, label: str, output: Path):
    market, market_diag = solve_nested_market(data, state)
    rows, max_gain, all_success = [], 0.0, True
    for player in PLAYER_ORDER:
        reference = float(nested_economic_objective(data, state, market, player))
        best, best_state, diag = nested_best_response(data, state, market, player, maxiter=maxiter, starts=starts)
        gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
        move, coordinate, absolute = _strategy_distance(data, state, best_state, player)
        all_success = all_success and all(bool(a["success"]) for a in diag["attempts"])
        max_gain = max(max_gain, gain)
        rows.append({
            "player": player, "reference_objective": reference, "best_response_objective": float(best),
            "relative_gain": gain, "raw_best_response": player_strategy_payload(data, best_state, player),
            "strategy_move": move, "largest_move_coordinate": coordinate, "largest_move_absolute": absolute,
            "optimizer_success": bool(diag["success"]), "chosen_start_index": int(diag["chosen_start_index"]),
            "attempts": diag["attempts"], "capacity_feasibility_min": float(diag["capacity_feasibility_min"]),
            "best_response_market_diagnostics": diag["market"],
        })
        print(f"[AUDIT {label}] {player}: gain={gain:.3%} success={diag['success']}", flush=True)
    start_definitions = [
        "candidate strategy",
        "standard zero-capacity-change/manufacturing-cost-price initialization",
        "feasible 0.95x candidate-price perturbation",
    ]
    payload = {
        "created": now(), "label": label, "common_profile_frozen": True, "starts": starts,
        "start_definitions": start_definitions[:starts],
        "maxiter": maxiter,
        "algorithmic_proximal_penalties": 0.0, "economic_objective_only": True,
        "reference_market_diagnostics": market_diag, "profile": full_state_payload(data, state, market),
        "players": rows, "max_relative_gain": max_gain,
        "max_gain_player": max(rows, key=lambda x: x["relative_gain"])["player"],
        "all_attempts_successful": all_success, "relative_gain_tolerance": 0.01,
        "equilibrium_verified": bool(all_success and max_gain <= 0.01),
        "claim_wording": "solver-verified computational relative 1%-equilibrium" if all_success and max_gain <= 0.01 else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[AUDIT {label}] max={max_gain:.3%} verified={payload['equilibrium_verified']}", flush=True)
    return payload


def load_sweep_state(data, paper, path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _deserialize_state(payload["ending_profile"]["strategy"], data, paper)


def detect_cycle(data, history):
    result = {"two_cycle": False, "three_cycle": False, "details": []}
    if len(history) < 4:
        return result
    step = normalized_l2(data, history[-2], history[-1])
    for period, name in ((2, "two_cycle"), (3, "three_cycle")):
        if len(history) > period:
            recurrence = normalized_l2(data, history[-1-period], history[-1])
            if recurrence < 0.25 * max(step, 1e-12) and step > 1e-5:
                result[name] = True
                by_player = {p: normalized_l2(data, history[-1-period], history[-1], p) for p in PLAYER_ORDER}
                result["details"].append({"period": period, "recurrence_distance": recurrence, "step_distance": step,
                    "largest_player": max(by_player, key=by_player.get), "player_distances": by_player})
    return result


def run_branch(data, paper, branch: str, *, alpha: float, max_sweeps: int, starts: int, maxiter: int,
               order: list[str] | None = None, run_name: str | None = None):
    order = list(order or PLAYER_ORDER)
    run_name = run_name or f"branch_{branch}_alpha_{alpha:.2f}".replace(".", "p")
    out = OUTPUT_ROOT / run_name
    out.mkdir(parents=True, exist_ok=True)
    init_path = out / "initialization.json"
    if init_path.exists():
        initial_payload = json.loads(init_path.read_text(encoding="utf-8"))
        initial = _deserialize_state(initial_payload["profile"]["strategy"], data, paper)
    else:
        initial = make_initialization(data, paper, branch)
        market, market_diag = solve_nested_market(data, initial)
        init_payload = {
            "created": now(), "branch": branch, "specification": BRANCHES[branch], "player_order": order,
            "alpha": alpha, "source_workbook": str(SOURCE_WORKBOOK.relative_to(ROOT)), "source_iteration": SOURCE_ITERATION,
            "paper_replay_definition": "accepted damped state reconstructed through iteration 21",
            "algorithmic_proximal_penalties": 0.0, "prices_free_after_initialization": True,
            "market_diagnostics": market_diag, "profile": full_state_payload(data, initial, market),
            "distance_from_paper": normalized_l2(data, paper, initial),
        }
        init_path.write_text(json.dumps(init_payload, indent=2) + "\n", encoding="utf-8")
    sweep_paths = sorted(out.glob("sweep_*.json"))
    state = load_sweep_state(data, paper, sweep_paths[-1]) if sweep_paths else clone_state(initial)
    history = [clone_state(initial)]
    for path in sweep_paths:
        history.append(load_sweep_state(data, paper, path))
    stable_strategy = 0
    for path in sweep_paths[-3:]:
        if json.loads(path.read_text(encoding="utf-8"))["strategy_change_metric"] <= 0.01:
            stable_strategy += 1
        else:
            stable_strategy = 0
    promising_audits = []
    terminal_reason = None
    for sweep in range(len(sweep_paths) + 1, max_sweeps + 1):
        before_sweep = clone_state(state)
        before_obj, before_market, before_market_diag = objectives(data, before_sweep)
        player_rows = []
        for player in order:
            before_player = clone_state(state)
            market, market_diag = solve_nested_market(data, state)
            reference = float(nested_economic_objective(data, state, market, player))
            best, raw, diag = nested_best_response(data, state, market, player, maxiter=maxiter, starts=starts)
            retry_used = False
            if not diag["success"]:
                retry_used = True
                best, raw, diag = nested_best_response(
                    data, state, market, player, maxiter=max(maxiter * 2, 1000), starts=max(starts, 3)
                )
            if not diag["success"]:
                raise RuntimeError(f"{run_name} sweep {sweep} {player}: no successful best response after retry: {diag['message']}")
            gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
            for t in mm._move_times(list(data.times or [])):
                key = (player, t); state["dK_net"][key] = (1.0-alpha)*before_player["dK_net"][key] + alpha*raw["dK_net"][key]
            for importer in data.regions:
                if importer == player:
                    continue
                for t in list(data.times or []):
                    key = (player, importer, t); state["p_offer"][key] = (1.0-alpha)*before_player["p_offer"][key] + alpha*raw["p_offer"][key]
            _sync_quantity(data, state)
            post_market, post_market_diag = solve_nested_market(data, state)
            player_rows.append({
                "player": player, "reference_objective": reference, "raw_best_response_objective": float(best),
                "raw_relative_gain": gain, "alpha": alpha,
                "raw_best_response": player_strategy_payload(data, raw, player),
                "damped_strategy": player_strategy_payload(data, state, player),
                "raw_strategy_move": _strategy_distance(data, before_player, raw, player)[0],
                "damped_strategy_move": _strategy_distance(data, before_player, state, player)[0],
                "solver": {"success": bool(diag["success"]), "status": int(diag["status"]), "message": diag["message"],
                    "iterations": int(diag["iterations"]), "objective_evaluations": int(diag["objective_evaluations"]),
                    "chosen_start_index": int(diag["chosen_start_index"]), "retry_used": retry_used,
                    "starts_on_accepted_attempt": len(diag["attempts"]), "attempts": diag["attempts"]},
                "feasibility": {"capacity_min": float(diag["capacity_feasibility_min"]),
                    "reference_market": market_diag, "best_response_market": diag["market"], "post_damped_market": post_market_diag},
                "post_damped_market": full_state_payload(data, state, post_market)["market"],
            })
            print(f"[{run_name} S{sweep:02d}] {player}: raw_gain={gain:.3%}", flush=True)
        ending_obj, ending_market, ending_market_diag = objectives(data, state)
        rx = strategy_metric(data, before_sweep, state)
        rpi = max(abs(ending_obj[p]-before_obj[p])/max(abs(before_obj[p]), 1.0) for p in PLAYER_ORDER)
        stable_strategy = stable_strategy + 1 if rx <= 0.01 else 0
        history.append(clone_state(state))
        cycle = detect_cycle(data, history)
        max_raw = max(row["raw_relative_gain"] for row in player_rows)
        payload = {
            "created": now(), "branch": branch, "run_name": run_name, "sweep": sweep, "alpha": alpha,
            "player_order": order, "algorithmic_proximal_penalties": 0.0, "economic_objective_only": True,
            "starting_common_objectives": before_obj, "ending_common_objectives": ending_obj,
            "strategy_change_metric": rx, "objective_change_metric": rpi,
            "stable_strategy_sweeps": stable_strategy, "max_raw_unilateral_gain": max_raw,
            "max_raw_gain_player": max(player_rows, key=lambda x: x["raw_relative_gain"])["player"],
            "players": player_rows, "starting_market_diagnostics": before_market_diag,
            "ending_market_diagnostics": ending_market_diag, "cycle_diagnostics": cycle,
            "distance_from_paper": normalized_l2(data, paper, state),
            "ending_profile": full_state_payload(data, state, ending_market),
        }
        sweep_path = out / f"sweep_{sweep:03d}.json"
        sweep_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[{run_name} S{sweep:02d}] max_gain={max_raw:.3%} rx={rx:.3%} rPi={rpi:.3%} cycle={cycle['two_cycle'] or cycle['three_cycle']}", flush=True)
        should_audit = (max_raw <= 0.02 or stable_strategy >= 3) and not (out / f"audit_sweep_{sweep:03d}.json").exists()
        if should_audit:
            audit = audit_profile(data, state, starts=3, maxiter=max(maxiter, 600), label=f"{run_name}_sweep_{sweep}", output=out / f"audit_sweep_{sweep:03d}.json")
            promising_audits.append(audit)
            if audit["equilibrium_verified"]:
                terminal_reason = "solver_verified_relative_1pct_equilibrium"
                break
        if (cycle["two_cycle"] or cycle["three_cycle"]) and sweep >= 8:
            terminal_reason = "detected_recurrent_cycle"
            break
        recent_paths = sorted(out.glob("sweep_*.json"))[-10:]
        if sweep >= 15 and len(recent_paths) == 10:
            recent = [json.loads(path.read_text(encoding="utf-8")) for path in recent_paths]
            if min(r["max_raw_unilateral_gain"] for r in recent) > 0.02 and min(r["strategy_change_metric"] for r in recent) > 0.01:
                terminal_reason = "persistent_nonconvergence_over_last_10_sweeps"
                break
    make_plots(data, paper, out)
    return summarize_run(data, paper, out, terminal_reason)


def make_plots(data, paper, out: Path):
    paths = sorted(out.glob("sweep_*.json"))
    if not paths:
        return
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    sweeps = [r["sweep"] for r in rows]
    series = [
        ("max_raw_unilateral_gain", "Maximum raw unilateral gain", "Relative gain", "max_raw_gain.png"),
        ("strategy_change_metric", "Strategy change by complete sweep", "Relative strategy change", "strategy_change.png"),
        ("objective_change_metric", "Objective change by complete sweep", "Relative objective change", "objective_change.png"),
    ]
    for key, title, ylabel, filename in series:
        fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.plot(sweeps, [r[key] for r in rows], marker="o", ms=3)
        ax.axhline(0.01, color="tab:red", linestyle="--", linewidth=1, label="1%")
        ax.set(xlabel="Complete Gauss-Seidel sweep", ylabel=ylabel, title=title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        fig.savefig(out / filename, dpi=180); plt.close(fig)
    times = list(data.times or [])
    for quantity, filename, ylabel in (("capacities", "capacities_by_player.png", "Capacity (GW)"),):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for p in PLAYER_ORDER:
            vals = []
            for r in rows:
                cap = {(x["player"], x["time"]): x["value"] for x in r["ending_profile"][quantity]}
                vals.append(np.mean([cap[(p, t)] for t in times[:-1]]))
            ax.plot(sweeps, vals, marker="o", ms=2, label=p.upper())
        ax.set(xlabel="Complete Gauss-Seidel sweep", ylabel=ylabel, title="Mean 2025-2040 capacity by player"); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout()
        fig.savefig(out / filename, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for p in PLAYER_ORDER:
        vals=[]
        for r in rows:
            offers=r["ending_profile"]["strategy"]["p_offer"]
            vals.append(np.mean([x["value"] for x in offers if x["exporter"]==p and x["importer"]!=p]))
        ax.plot(sweeps, vals, marker="o", ms=2, label=p.upper())
    ax.set(xlabel="Complete Gauss-Seidel sweep", ylabel="Mean offer price (USD/kW)", title="Mean strategic offer price by player"); ax.grid(alpha=.25); ax.legend(ncol=3); fig.tight_layout()
    fig.savefig(out / "offer_prices_by_player.png", dpi=180); plt.close(fig)


def summarize_run(data, paper, out: Path, terminal_reason=None):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("sweep_*.json"))]
    audits = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("audit_*.json"))]
    best = min(audits, key=lambda a: a["max_relative_gain"]) if audits else None
    summary = {
        "run_name": out.name, "sweeps_completed": len(rows),
        "ending_max_raw_gain": rows[-1]["max_raw_unilateral_gain"] if rows else None,
        "ending_strategy_change": rows[-1]["strategy_change_metric"] if rows else None,
        "ending_objective_change": rows[-1]["objective_change_metric"] if rows else None,
        "ending_distance_from_paper": rows[-1]["distance_from_paper"] if rows else 0.0,
        "cycle_detected": any(r["cycle_diagnostics"]["two_cycle"] or r["cycle_diagnostics"]["three_cycle"] for r in rows),
        "terminal_reason": terminal_reason or "requested_sweep_limit",
        "best_audit": None if best is None else {"label": best["label"], "max_relative_gain": best["max_relative_gain"],
            "max_gain_player": best["max_gain_player"], "equilibrium_verified": best["equilibrium_verified"]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"created": now(), "experiment": EXPERIMENT, "source_workbook": str(SOURCE_WORKBOOK.relative_to(ROOT)),
        "source_iteration": SOURCE_ITERATION, "standard_player_order": PLAYER_ORDER,
        "algorithmic_proximal_penalties": 0.0, "economic_objective_only": True,
        "branches": BRANCHES, "runs": [], "paper_profile_audit": None,
        "preserved_cost_price_candidate": {"checkpoint": str(CERTIFIED_CHECKPOINT.relative_to(ROOT)), "audit": str(CERTIFIED_AUDIT.relative_to(ROOT))}}


def save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated"] = now(); MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=list(BRANCHES))
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--max-sweeps", type=int, default=50)
    parser.add_argument("--starts", type=int, default=1, help="Starts per GS best response; frozen audits always use at least 3")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--audit-paper-only", action="store_true")
    parser.add_argument("--order", nargs=6, choices=PLAYER_ORDER)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if not 0 < args.alpha <= 1: raise ValueError("alpha must be in (0,1]")
    if args.order and len(set(args.order)) != 6: raise ValueError("--order must contain each player exactly once")
    data, paper, replay_error = paper_state(); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(); manifest["paper_profile_replay_error"] = replay_error
    paper_init = OUTPUT_ROOT / "paper_profile.json"
    if not paper_init.exists():
        market, diag = solve_nested_market(data, paper)
        paper_init.write_text(json.dumps({"created": now(), "source_workbook": str(SOURCE_WORKBOOK.relative_to(ROOT)),
            "source_iteration": SOURCE_ITERATION, "player_order": PLAYER_ORDER, "replay_error": replay_error,
            "market_diagnostics": diag, "profile": full_state_payload(data, paper, market)}, indent=2)+"\n", encoding="utf-8")
    paper_audit_path = OUTPUT_ROOT / "audit_paper_profile.json"
    if not paper_audit_path.exists():
        audit = audit_profile(data, paper, starts=3, maxiter=max(args.maxiter, 600), label="paper_profile_iteration_21", output=paper_audit_path)
        manifest["paper_profile_audit"] = str(paper_audit_path.relative_to(ROOT)); manifest["paper_profile_max_relative_gain"] = audit["max_relative_gain"]
        save_manifest(manifest)
    if args.audit_paper_only:
        return
    if not args.branch: raise ValueError("--branch is required unless --audit-paper-only is used")
    summary = run_branch(data, paper, args.branch, alpha=args.alpha, max_sweeps=args.max_sweeps,
        starts=args.starts, maxiter=args.maxiter, order=args.order, run_name=args.run_name)
    manifest = load_manifest(); manifest["runs"] = [r for r in manifest.get("runs", []) if r.get("run_name") != summary["run_name"]]
    manifest["runs"].append(summary); save_manifest(manifest)


if __name__ == "__main__":
    main()
