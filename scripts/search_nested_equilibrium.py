from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _candidate_state, _strategy_distance, _zero_prox_data
from scripts.continue_selected_equilibrium import PLAYER_ORDER
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)


SEED_MANIFEST = Path("workflow/zero_prox_conopt_from_block1_omega_002_manifest.json")
MANIFEST_PATH = ROOT / "workflow" / "nested_equilibrium_search_manifest.json"
OUTPUT_DIR = ROOT / "outputs" / "equilibrium_search" / "ch-row-apac-us-eu-af" / "nested_zero_prox"
GAIN_TOLERANCE = 0.01


def _usable_runs(manifest: dict[str, object]) -> list[dict[str, object]]:
    return [
        entry
        for entry in manifest.get("runs", [])
        if entry.get("status") not in {"invalid", "rejected"}
    ]


def _serialize_state(state: dict[str, dict]) -> dict[str, list[dict[str, object]]]:
    return {
        "dK_net": [
            {"region": region, "time": tp, "value": float(value)}
            for (region, tp), value in sorted(state["dK_net"].items())
        ],
        "p_offer": [
            {"exporter": exporter, "importer": importer, "time": tp, "value": float(value)}
            for (exporter, importer, tp), value in sorted(state["p_offer"].items())
        ],
    }


def _deserialize_state(payload: dict[str, list[dict[str, object]]], data, template):
    state = {
        "dK_net": dict(template["dK_net"]),
        "p_offer": dict(template["p_offer"]),
        "a_bid": dict(template["a_bid"]),
        "Q_offer": dict(template["Q_offer"]),
    }
    for row in payload["dK_net"]:
        state["dK_net"][(str(row["region"]), str(row["time"]))] = float(row["value"])
    for row in payload["p_offer"]:
        state["p_offer"][(str(row["exporter"]), str(row["importer"]), str(row["time"]))] = float(
            row["value"]
        )
    _sync_quantity(data, state)
    return state


def _sync_quantity(data, state) -> None:
    times = list(data.times or [])
    kcap = mm._implied_capacity_path(data, times, state["dK_net"])
    state["Q_offer"] = {
        (region, tp): max(float(kcap[(region, tp)]), 0.0)
        for region in data.players
        for tp in times
    }


def _set_prices_at_cost(data, state) -> None:
    for exporter in data.regions:
        for importer in data.regions:
            for tp in list(data.times or []):
                state["p_offer"][(exporter, importer, tp)] = float(
                    (data.c_man_t or {}).get((exporter, tp), data.c_man[exporter])
                )


def _load(*, prices_at_cost: bool = False) -> tuple[object, dict[str, dict], dict[str, object]]:
    _, base_cfg, seed, _, seed_label = _candidate_state("search", SEED_MANIFEST)
    data = _zero_prox_data(base_cfg)
    if not MANIFEST_PATH.exists():
        manifest = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed_manifest": str(SEED_MANIFEST),
            "seed_label": seed_label,
            "player_order": PLAYER_ORDER,
            "algorithmic_proximal_penalties": 0.0,
            "runs": [],
        }
        if prices_at_cost:
            _set_prices_at_cost(data, seed)
        return data, seed, manifest
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    usable_runs = _usable_runs(manifest)
    if not usable_runs:
        return data, seed, manifest
    checkpoint = ROOT / str(usable_runs[-1]["checkpoint"])
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    state = _deserialize_state(payload["state"], data, seed)
    if prices_at_cost:
        _set_prices_at_cost(data, state)
    return data, state, manifest


def run(
    sweeps: int,
    omega: float,
    maxiter: int,
    starts: int,
    max_damped_move: float | None,
    update_threshold: float,
    gain_tolerance: float,
    stable_sweeps_required: int,
    prices_at_cost: bool,
) -> None:
    data, state, manifest = _load(prices_at_cost=prices_at_cost)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    sweep_records = []
    stable_gain_sweeps = 0
    for sweep in range(1, sweeps + 1):
        player_records = []
        max_gain = 0.0
        max_applied_move = 0.0
        for player in PLAYER_ORDER:
            market, market_diag = solve_nested_market(data, state)
            reference = nested_economic_objective(data, state, market, player)
            best, best_state, diagnostics = nested_best_response(
                data, state, market, player, maxiter=maxiter, starts=starts
            )
            if not diagnostics["success"]:
                raise RuntimeError(
                    f"No successful best-response solve for {player} in sweep {sweep}: "
                    f"{diagnostics['message']}"
                )
            gain = max(best - reference, 0.0)
            relative_gain = gain / max(abs(reference), 1.0)
            before = {
                "dK_net": dict(state["dK_net"]),
                "p_offer": dict(state["p_offer"]),
                "a_bid": dict(state["a_bid"]),
                "Q_offer": dict(state["Q_offer"]),
            }
            full_move, _, _ = _strategy_distance(data, before, best_state, player)
            effective_omega = omega
            if relative_gain <= update_threshold or not bool(diagnostics["success"]):
                effective_omega = 0.0
            if max_damped_move is not None and full_move > max_damped_move:
                effective_omega = min(effective_omega, max_damped_move / full_move)
            for tp in mm._move_times(list(data.times or [])):
                key = (player, tp)
                state["dK_net"][key] = (1.0 - effective_omega) * float(state["dK_net"][key]) + effective_omega * float(
                    best_state["dK_net"][key]
                )
            for importer in data.regions:
                if importer == player:
                    continue
                for tp in list(data.times or []):
                    key = (player, importer, tp)
                    state["p_offer"][key] = (1.0 - effective_omega) * float(state["p_offer"][key]) + effective_omega * float(
                        best_state["p_offer"][key]
                    )
            _sync_quantity(data, state)
            damped_move, coordinate, absolute_move = _strategy_distance(
                data, before, state, player
            )
            max_gain = max(max_gain, relative_gain)
            max_applied_move = max(max_applied_move, damped_move)
            record = {
                "player": player,
                "reference_objective": reference,
                "best_response_objective": best,
                "relative_gain": relative_gain,
                "damped_strategy_move": damped_move,
                "full_best_response_move": full_move,
                "effective_omega": effective_omega,
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": absolute_move,
                "optimizer_success": bool(diagnostics["success"]),
                "optimizer_message": str(diagnostics["message"]),
                "outer_iterations": int(diagnostics["iterations"]),
                "chosen_start_index": int(diagnostics["chosen_start_index"]),
                "market_kkt_residual": float(
                    diagnostics["market"]["max_positive_flow_stationarity"]
                ),
                "reference_market_kkt_residual": float(
                    market_diag["max_positive_flow_stationarity"]
                ),
            }
            player_records.append(record)
            print(
                f"[NESTED GS {sweep}] {player}: gain={relative_gain:.3%} "
                f"damped_move={damped_move:.3g} success={diagnostics['success']}",
                flush=True,
            )
        sweep_record = {
            "sweep": sweep,
            "max_turn_relative_gain": max_gain,
            "max_damped_strategy_move": max_applied_move,
            "objective_gain_tolerance": gain_tolerance,
            "objective_gain_pass": max_gain <= gain_tolerance,
            "players": player_records,
        }
        stable_gain_sweeps = stable_gain_sweeps + 1 if max_gain <= gain_tolerance else 0
        sweep_record["stable_objective_gain_sweeps"] = stable_gain_sweeps
        sweep_records.append(sweep_record)
        print(
            f"[NESTED GS {sweep}] max_gain={max_gain:.3%} "
            f"objective_stable={stable_gain_sweeps}/{stable_sweeps_required} "
            f"max_move_diagnostic={max_applied_move:.3g}",
            flush=True,
        )
        if stable_gain_sweeps >= stable_sweeps_required:
            print(
                f"[NESTED GS] objective-gain stopping rule met after {sweep} sweeps; "
                "a common-profile audit is still required",
                flush=True,
            )
            break

    finished = datetime.now().astimezone()
    stamp = finished.strftime("%Y%m%d_%H%M%S")
    checkpoint = OUTPUT_DIR / f"checkpoint_{stamp}.json"
    checkpoint.write_text(
        json.dumps(
            {
                "created": finished.isoformat(timespec="seconds"),
                "omega": omega,
                "max_damped_move": max_damped_move,
                "state": _serialize_state(state),
                "sweeps": sweep_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["runs"].append(
        {
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "omega": omega,
            "sweeps_requested": sweeps,
            "sweeps_completed": len(sweep_records),
            "maxiter": maxiter,
            "starts": starts,
            "max_damped_move": max_damped_move,
            "update_threshold": update_threshold,
            "prices_at_cost_start": prices_at_cost,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "ending_max_turn_relative_gain": sweep_records[-1]["max_turn_relative_gain"],
            "ending_max_damped_strategy_move": sweep_records[-1]["max_damped_strategy_move"],
            "status": "active",
            "equilibrium_criterion": "final common-profile zero-proximal objective audit",
            "objective_gain_tolerance": gain_tolerance,
            "stable_objective_gain_sweeps_required": stable_sweeps_required,
            "ending_stable_objective_gain_sweeps": stable_gain_sweeps,
            "turn_objective_rule_met": stable_gain_sweeps >= stable_sweeps_required,
        }
    )
    manifest["updated"] = finished.isoformat(timespec="seconds")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[NESTED GS] wrote {checkpoint}")
    print(f"[NESTED GS] updated {MANIFEST_PATH}")


def audit(
    maxiter: int,
    starts: int,
    gain_tolerance: float,
    prices_at_cost: bool = False,
) -> None:
    data, state, manifest = _load(prices_at_cost=prices_at_cost)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    market, market_diag = solve_nested_market(data, state)
    records = []
    max_gain = 0.0
    all_successful = True
    for player in PLAYER_ORDER:
        reference = nested_economic_objective(data, state, market, player)
        best, best_state, diagnostics = nested_best_response(
            data, state, market, player, maxiter=maxiter, starts=starts
        )
        gain = max(best - reference, 0.0)
        relative_gain = gain / max(abs(reference), 1.0)
        move, coordinate, absolute_move = _strategy_distance(
            data, state, best_state, player
        )
        max_gain = max(max_gain, relative_gain)
        all_successful = all_successful and all(
            bool(attempt["success"]) for attempt in diagnostics["attempts"]
        )
        records.append(
            {
                "player": player,
                "reference_objective": reference,
                "best_response_objective": best,
                "relative_gain": relative_gain,
                "strategy_move": move,
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": absolute_move,
                "chosen_start_index": int(diagnostics["chosen_start_index"]),
                "optimizer_success": bool(diagnostics["success"]),
                "attempts": diagnostics["attempts"],
                "market_kkt_residual": float(
                    diagnostics["market"]["max_positive_flow_stationarity"]
                ),
            }
        )
        print(
            f"[NESTED AUDIT] {player}: gain={relative_gain:.3%} "
            f"move={move:.3g} chosen_start={diagnostics['chosen_start_index']} "
            f"success={diagnostics['success']}",
            flush=True,
        )
    finished = datetime.now().astimezone()
    stamp = finished.strftime("%Y%m%d_%H%M%S")
    output = OUTPUT_DIR / f"audit_{stamp}.json"
    base_profile_checkpoint = _usable_runs(manifest)[-1]["checkpoint"]
    payload = {
        "created": finished.isoformat(timespec="seconds"),
        "profile_checkpoint": (
            f"{base_profile_checkpoint}#prices_at_cost"
            if prices_at_cost
            else base_profile_checkpoint
        ),
        "base_profile_checkpoint": base_profile_checkpoint,
        "starts": starts,
        "maxiter": maxiter,
        "algorithmic_proximal_penalties": 0.0,
        "common_profile_frozen": True,
        "prices_at_cost_profile": prices_at_cost,
        "reference_market": market_diag,
        "max_relative_gain": max_gain,
        "all_attempts_successful": all_successful,
        "relative_gain_tolerance": gain_tolerance,
        "equilibrium_verified": all_successful and max_gain <= gain_tolerance,
        "players": records,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    manifest.setdefault("audits", []).append(
        {
            "created": payload["created"],
            "profile_checkpoint": payload["profile_checkpoint"],
            "starts": starts,
            "maxiter": maxiter,
            "max_relative_gain": max_gain,
            "all_attempts_successful": all_successful,
            "relative_gain_tolerance": gain_tolerance,
            "equilibrium_verified": payload["equilibrium_verified"],
            "output": str(output.relative_to(ROOT)),
        }
    )
    for entry in reversed(manifest["runs"]):
        if entry.get("checkpoint") == payload["profile_checkpoint"]:
            entry["status"] = (
                "equilibrium_verified" if payload["equilibrium_verified"] else "active"
            )
            entry["audited_max_relative_gain"] = max_gain
            entry["audit_output"] = str(output.relative_to(ROOT))
            break
    manifest["updated"] = payload["created"]
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[NESTED AUDIT] max_gain={max_gain:.3%} "
        f"tolerance={gain_tolerance:.3%} verified={payload['equilibrium_verified']} "
        f"all_attempts_successful={all_successful}"
    )
    print(f"[NESTED AUDIT] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested zero-proximal equilibrium search.")
    parser.add_argument("--sweeps", type=int, default=3)
    parser.add_argument("--omega", type=float, default=0.25)
    parser.add_argument("--maxiter", type=int, default=250)
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--gain-tol", type=float, default=GAIN_TOLERANCE)
    parser.add_argument("--stable-sweeps", type=int, default=3)
    parser.add_argument(
        "--audit-after",
        action="store_true",
        help="Run the common-profile multistart objective audit after the sweep block.",
    )
    parser.add_argument(
        "--max-damped-move",
        type=float,
        help="Cap each player's normalized update while allowing smaller responses in full.",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=0.0,
        help="Do not update a successful player whose relative gain is at or below this value.",
    )
    parser.add_argument(
        "--prices-at-cost",
        action="store_true",
        help="Replace all offer prices by exporter-period manufacturing cost before the run.",
    )
    args = parser.parse_args()
    if args.sweeps < 1:
        raise ValueError("--sweeps must be positive")
    if not 0.0 < args.omega <= 1.0:
        raise ValueError("--omega must be in (0, 1]")
    if args.starts < 1 or args.starts > 5:
        raise ValueError("--starts must be in [1, 5]")
    if args.max_damped_move is not None and args.max_damped_move <= 0.0:
        raise ValueError("--max-damped-move must be positive")
    if args.update_threshold < 0.0:
        raise ValueError("--update-threshold must be nonnegative")
    if args.gain_tol < 0.0:
        raise ValueError("--gain-tol must be non-negative")
    if args.stable_sweeps < 1:
        raise ValueError("--stable-sweeps must be positive")
    if args.audit_only:
        audit(args.maxiter, args.starts, args.gain_tol, args.prices_at_cost)
    else:
        run(
            args.sweeps,
            args.omega,
            args.maxiter,
            args.starts,
            args.max_damped_move,
            args.update_threshold,
            args.gain_tol,
            args.stable_sweeps,
            args.prices_at_cost,
        )
        if args.audit_after:
            audit(args.maxiter, args.starts, args.gain_tol)


if __name__ == "__main__":
    main()
