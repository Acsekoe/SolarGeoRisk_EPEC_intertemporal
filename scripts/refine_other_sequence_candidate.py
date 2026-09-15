from __future__ import annotations

"""Selectively refine a near-pass frozen-profile equilibrium candidate.

Players at or below the one-percent unilateral-gain threshold are not updated.
Every saved sweep is audited at one common frozen profile with one
unregularized solve per player.  A three-start audit is retained separately as
a diagnostic after a passing profile is found.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.nested_market_audit import nested_best_response, nested_economic_objective, solve_nested_market
from scripts.run_local_paper_equilibrium_experiment import audit_profile, clone_state, full_state_payload
from scripts.run_other_sequence_o6_search import SEQUENCES, configure_modules, relative, write_json
from scripts.search_nested_equilibrium import _deserialize_state, _sync_quantity


BRANCHES = {
    "selective_a010_cap010": {"omega": 0.10, "max_move": 0.010},
    "selective_a005_cap005": {"omega": 0.05, "max_move": 0.005},
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_checkpoint(path: Path, data: Any, template: dict[str, dict]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _deserialize_state(payload["ending_profile"]["strategy"], data, template)


def update_player(
    data: Any,
    state: dict[str, dict],
    response: dict[str, dict],
    player: str,
    weight: float,
) -> None:
    for tp in mm._move_times(list(data.times or [])):
        key = (player, tp)
        state["dK_net"][key] = (1.0 - weight) * float(state["dK_net"][key]) + weight * float(
            response["dK_net"][key]
        )
    for importer in data.regions:
        if importer == player:
            continue
        for tp in list(data.times or []):
            key = (player, importer, tp)
            state["p_offer"][key] = (1.0 - weight) * float(state["p_offer"][key]) + weight * float(
                response["p_offer"][key]
            )
    _sync_quantity(data, state)


def run_branch(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    branch = str(task["branch"])
    spec = BRANCHES[branch]
    order = list(SEQUENCES[sequence]["order"])
    source_checkpoint = ROOT / str(task["source_checkpoint"])
    output = ROOT / "outputs" / "equilibrium_search" / sequence / "selective_refinement" / str(task["run_id"]) / branch
    status_path = output / "status.json"
    try:
        data, paper, replay_error = configure_modules(sequence, SEQUENCES[sequence])
        initial = load_checkpoint(source_checkpoint, data, paper)
        output.mkdir(parents=True, exist_ok=True)
        checkpoints = sorted(output.glob("sweep_*.json"))
        state = load_checkpoint(checkpoints[-1], data, paper) if checkpoints else clone_state(initial)
        best_record = None
        selected_record = None
        for sweep in range(len(checkpoints) + 1, int(task["max_sweeps"]) + 1):
            before_sweep = clone_state(state)
            player_rows = []
            for player in order:
                before_player = clone_state(state)
                market, market_diag = solve_nested_market(data, state)
                reference = float(nested_economic_objective(data, state, market, player))
                best, response, diagnostics = nested_best_response(
                    data,
                    state,
                    market,
                    player,
                    maxiter=int(task["maxiter"]),
                    starts=1,
                )
                if not diagnostics["success"]:
                    best, response, diagnostics = nested_best_response(
                        data,
                        state,
                        market,
                        player,
                        maxiter=max(2 * int(task["maxiter"]), 800),
                        starts=3,
                    )
                if not diagnostics["success"]:
                    raise RuntimeError(f"{branch} sweep {sweep} {player}: best-response solve failed")
                gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
                full_move = _strategy_distance(data, before_player, response, player)[0]
                weight = 0.0
                if gain > float(task["gain_tolerance"]):
                    weight = float(spec["omega"])
                    if full_move > 0.0:
                        weight = min(weight, float(spec["max_move"]) / full_move)
                    update_player(data, state, response, player, weight)
                player_rows.append(
                    {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best),
                        "relative_gain": gain,
                        "raw_strategy_move": full_move,
                        "effective_weight": weight,
                        "optimizer_success": bool(diagnostics["success"]),
                        "chosen_start_index": int(diagnostics["chosen_start_index"]),
                        "attempts": diagnostics["attempts"],
                        "reference_market_diagnostics": market_diag,
                    }
                )
            ending_market, ending_market_diag = solve_nested_market(data, state)
            checkpoint = output / f"sweep_{sweep:03d}.json"
            write_json(
                checkpoint,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": branch,
                    "specification": spec,
                    "sweep": sweep,
                    "player_order": order,
                    "source_checkpoint": relative(source_checkpoint),
                    "gain_tolerance": task["gain_tolerance"],
                    "algorithmic_proximal_penalties": 0.0,
                    "players": player_rows,
                    "updated_players": [row["player"] for row in player_rows if row["effective_weight"] > 0.0],
                    "max_sequential_relative_gain": max(row["relative_gain"] for row in player_rows),
                    "strategy_change_metric": max(
                        _strategy_distance(data, before_sweep, state, player)[0] for player in order
                    ),
                    "ending_market_diagnostics": ending_market_diag,
                    "ending_profile": full_state_payload(data, state, ending_market),
                },
            )
            audit_path = output / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
            frozen = audit_profile(
                data,
                state,
                starts=1,
                maxiter=int(task["maxiter"]),
                label=f"{sequence}_{branch}_sweep_{sweep:03d}_one_start",
                output=audit_path,
            )
            record = {
                **frozen,
                "checkpoint": relative(checkpoint),
                "audit_path": relative(audit_path),
                "sweep": sweep,
            }
            if best_record is None or float(record["max_relative_gain"]) < float(best_record["max_relative_gain"]):
                best_record = record
            print(
                f"[{sequence} {branch}] sweep {sweep}: frozen max gain "
                f"{100.0 * float(record['max_relative_gain']):.4f}% "
                f"({record['max_gain_player']}) updated={checkpoint.name}",
                flush=True,
            )
            if frozen["equilibrium_verified"]:
                selected_record = record
                break

        chosen = selected_record or best_record
        if chosen is None:
            raise RuntimeError("No selective-refinement audit was produced")
        chosen_state = load_checkpoint(ROOT / chosen["checkpoint"], data, paper)
        diagnostic_path = output / "audits" / f"audit_sweep_{int(chosen['sweep']):03d}_three_start.json"
        diagnostic = audit_profile(
            data,
            chosen_state,
            starts=3,
            maxiter=max(int(task["maxiter"]), 600),
            label=f"{sequence}_{branch}_sweep_{int(chosen['sweep']):03d}_three_start_diagnostic",
            output=diagnostic_path,
        )
        result = {
            "created": now(),
            "status": "accepted" if selected_record is not None else "no_pass_within_limit",
            "sequence": sequence,
            "branch": branch,
            "specification": spec,
            "source_checkpoint": relative(source_checkpoint),
            "source_replay_error": replay_error,
            "selected_checkpoint": chosen["checkpoint"],
            "selected_sweep": int(chosen["sweep"]),
            "one_start_audit": chosen["audit_path"],
            "one_start_max_relative_gain": chosen["max_relative_gain"],
            "one_start_max_gain_player": chosen["max_gain_player"],
            "one_start_equilibrium_verified": chosen["equilibrium_verified"],
            "three_start_diagnostic": relative(diagnostic_path),
            "three_start_max_relative_gain": diagnostic["max_relative_gain"],
            "three_start_max_gain_player": diagnostic["max_gain_player"],
            "three_start_equilibrium_verified": diagnostic["equilibrium_verified"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        failure = {
            "created": now(),
            "status": "failed",
            "sequence": sequence,
            "branch": branch,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", choices=list(SEQUENCES), required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--branches", nargs="+", choices=list(BRANCHES), default=list(BRANCHES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--gain-tolerance", type=float, default=0.01)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or datetime.now().astimezone().strftime("selective_%Y%m%d_%H%M%S")
    manifest_path = ROOT / "workflow" / f"{run_id}_manifest.json"
    manifest = {
        "created": now(),
        "status": "running",
        "run_id": run_id,
        "sequence": args.sequence,
        "source_checkpoint": args.source_checkpoint,
        "gain_tolerance": args.gain_tolerance,
        "max_sweeps": args.max_sweeps,
        "branches": {branch: BRANCHES[branch] for branch in args.branches},
        "results": [],
    }
    write_json(manifest_path, manifest)
    tasks = [
        {
            "sequence": args.sequence,
            "source_checkpoint": args.source_checkpoint,
            "branch": branch,
            "run_id": run_id,
            "max_sweeps": args.max_sweeps,
            "maxiter": args.maxiter,
            "gain_tolerance": args.gain_tolerance,
        }
        for branch in args.branches
    ]
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_branch, task): task["branch"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            manifest["results"] = results
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)
    manifest["updated"] = now()
    manifest["status"] = "complete" if all(r["status"] in {"accepted", "no_pass_within_limit"} for r in results) else "complete_with_failures"
    manifest["results"] = sorted(results, key=lambda row: row["branch"])
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
