from __future__ import annotations

"""Diversified zero-proximal equilibrium search from corrected Stage-1 endpoints.

The search deliberately avoids inheriting a stalled adaptive-O6 state.  For
each manuscript player order it explores three reproducible basins:

* the exact corrected Stage-1 endpoint with selective low-step updates;
* a 50% interpolation of its bilateral offers toward manufacturing cost with
  the same selective low-step updates; and
* manufacturing-cost offers with the staged move caps that produced the prior
  strong cost-price-basin equilibrium.

Only players whose current sequential unilateral economic gain exceeds 1% are
updated.  Every saved sweep is then tested again at one common frozen profile.
Algorithmic proximal penalties remain zero throughout.
"""

import argparse
import hashlib
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
from scripts.run_corrected_equilibrium_search import SEQUENCES, configure_modules, relative, sha256, write_json
from scripts.run_local_paper_equilibrium_experiment import audit_profile, clone_state, full_state_payload
from scripts.run_overnight_equilibrium_experiment import load_profile
from scripts.search_nested_equilibrium import _sync_quantity


BRANCHES: dict[str, dict[str, Any]] = {
    "stage1_selective": {
        "description": "exact corrected Stage-1 endpoint; selective low-step zero-proximal updates",
        "cost_weight": 0.0,
        "phases": [{"sweeps": 30, "omega": 0.10, "max_move": 0.010}],
    },
    "halfcost_selective": {
        "description": "Stage-1 capacities and offers interpolated 50% toward cost; selective low-step updates",
        "cost_weight": 0.5,
        "phases": [{"sweeps": 30, "omega": 0.10, "max_move": 0.010}],
    },
    "cost_staged": {
        "description": "Stage-1 capacities and manufacturing-cost offers; tight caps, one broad capped sweep, then low-step refinement",
        "cost_weight": 1.0,
        "phases": [
            {"sweeps": 4, "omega": 1.00, "max_move": 0.002},
            {"sweeps": 1, "omega": 1.00, "max_move": 0.060},
            {"sweeps": 25, "omega": 0.10, "max_move": 0.010},
        ],
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def code_hashes() -> dict[str, str]:
    names = [
        "scripts/run_corrected_diversified_search.py",
        "scripts/run_corrected_equilibrium_search.py",
        "scripts/nested_market_audit.py",
        "scripts/run_local_paper_equilibrium_experiment.py",
        "scripts/run_overnight_equilibrium_experiment.py",
        "model/data_prep.py",
        "model/model_main.py",
    ]
    return {name: sha256(ROOT / name) for name in names}


def schedule(specification: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sweep = 0
    for phase, phase_specification in enumerate(specification["phases"], start=1):
        for phase_sweep in range(1, int(phase_specification["sweeps"]) + 1):
            sweep += 1
            rows.append(
                {
                    "sweep": sweep,
                    "phase": phase,
                    "phase_sweep": phase_sweep,
                    "omega": float(phase_specification["omega"]),
                    "max_move": float(phase_specification["max_move"]),
                }
            )
    return rows


def initialize_basin(data: Any, source: dict[str, dict], cost_weight: float) -> dict[str, dict]:
    state = clone_state(source)
    for exporter in data.regions:
        for importer in data.regions:
            if exporter == importer:
                continue
            for period in mm._operating_times(data):
                key = (exporter, importer, period)
                cost = float((data.c_man_t or {}).get((exporter, period), data.c_man[exporter]))
                state["p_offer"][key] = (
                    (1.0 - cost_weight) * float(source["p_offer"][key]) + cost_weight * cost
                )
    _sync_quantity(data, state)
    return state


def update_player(
    data: Any,
    state: dict[str, dict],
    response: dict[str, dict],
    player: str,
    weight: float,
) -> None:
    for period in mm._move_times(list(data.times or [])):
        key = (player, period)
        state["dK_net"][key] = (
            (1.0 - weight) * float(state["dK_net"][key])
            + weight * float(response["dK_net"][key])
        )
    for importer in data.regions:
        if importer == player:
            continue
        for period in mm._operating_times(data):
            key = (player, importer, period)
            state["p_offer"][key] = (
                (1.0 - weight) * float(state["p_offer"][key])
                + weight * float(response["p_offer"][key])
            )
    _sync_quantity(data, state)


def run_audit(
    data: Any,
    state: dict[str, dict],
    *,
    starts: int,
    maxiter: int,
    label: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    return audit_profile(
        data,
        state,
        starts=starts,
        maxiter=maxiter,
        label=label,
        output=output,
    )


def record_for(
    audit: dict[str, Any],
    *,
    profile_path: Path,
    audit_path: Path,
    sweep: int,
) -> dict[str, Any]:
    return {
        "max_relative_gain": float(audit["max_relative_gain"]),
        "max_gain_player": str(audit["max_gain_player"]),
        "equilibrium_verified": bool(audit["equilibrium_verified"]),
        "all_attempts_successful": bool(audit["all_attempts_successful"]),
        "profile_path": relative(profile_path),
        "audit_path": relative(audit_path),
        "sweep": sweep,
    }


def run_branch(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    branch = str(task["branch"])
    specification = BRANCHES[branch]
    order = list(SEQUENCES[sequence]["order"])
    branch_root = Path(task["output_root"]).resolve() / sequence / branch
    status_path = branch_root / "status.json"
    try:
        branch_root.mkdir(parents=True, exist_ok=True)
        write_json(
            status_path,
            {
                "created": now(),
                "status": "initializing",
                "pid": os.getpid(),
                "sequence": sequence,
                "branch": branch,
            },
        )
        data, source, replay_error, workbook, _ = configure_modules(task)
        initialization_path = branch_root / "initialization.json"
        if initialization_path.exists():
            state = load_profile(initialization_path, data, source)
        else:
            state = initialize_basin(data, source, float(specification["cost_weight"]))
            market, market_diagnostics = solve_nested_market(data, state)
            write_json(
                initialization_path,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": branch,
                    "specification": specification,
                    "player_order": order,
                    "source_workbook": relative(workbook),
                    "source_workbook_sha256": sha256(workbook),
                    "source_iteration": SEQUENCES[sequence]["source_iteration"],
                    "source_replay_error": replay_error,
                    "algorithmic_proximal_penalties": 0.0,
                    "economic_objective_only": True,
                    "players_at_or_below_gain_tolerance_are_frozen": True,
                    "gain_tolerance": float(task["gain_tolerance"]),
                    "profile": full_state_payload(data, state, market),
                    "market_diagnostics": market_diagnostics,
                },
            )

        completed = sorted(branch_root.glob("sweep_*.json"))
        if completed:
            state = load_profile(completed[-1], data, source)
        initial_audit_path = branch_root / "audits" / "audit_initial_one_start.json"
        initial_audit = run_audit(
            data,
            load_profile(initialization_path, data, source),
            starts=1,
            maxiter=int(task["maxiter"]),
            label=f"corrected_{sequence}_{branch}_initial_one_start",
            output=initial_audit_path,
        )
        best = record_for(
            initial_audit,
            profile_path=initialization_path,
            audit_path=initial_audit_path,
            sweep=0,
        )
        selected = best if best["equilibrium_verified"] else None

        # Recover the best completed audited checkpoint when resuming.
        for checkpoint in completed:
            sweep = int(checkpoint.stem.split("_")[-1])
            audit_path = branch_root / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
            if audit_path.exists():
                existing = json.loads(audit_path.read_text(encoding="utf-8"))
                record = record_for(existing, profile_path=checkpoint, audit_path=audit_path, sweep=sweep)
                if record["max_relative_gain"] < best["max_relative_gain"]:
                    best = record
                if record["equilibrium_verified"] and selected is None:
                    selected = record

        branch_schedule = schedule(specification)
        for item in ([] if selected is not None else branch_schedule[len(completed) :]):
            sweep = int(item["sweep"])
            before_sweep = clone_state(state)
            player_rows: list[dict[str, Any]] = []
            for player in order:
                before_player = clone_state(state)
                market, market_diagnostics = solve_nested_market(data, state)
                reference = float(nested_economic_objective(data, state, market, player))
                best_value, response, diagnostics = nested_best_response(
                    data,
                    state,
                    market,
                    player,
                    maxiter=int(task["maxiter"]),
                    starts=1,
                )
                retried = False
                if not diagnostics["success"]:
                    retried = True
                    best_value, response, diagnostics = nested_best_response(
                        data,
                        state,
                        market,
                        player,
                        maxiter=max(2 * int(task["maxiter"]), 800),
                        starts=3,
                    )
                if not diagnostics["success"]:
                    raise RuntimeError(f"{branch} sweep {sweep} {player}: best-response solve failed")
                gain = max(float(best_value) - reference, 0.0) / max(abs(reference), 1.0)
                full_move = float(_strategy_distance(data, before_player, response, player)[0])
                effective_weight = 0.0
                if gain > float(task["gain_tolerance"]):
                    effective_weight = float(item["omega"])
                    if full_move > 0.0:
                        effective_weight = min(effective_weight, float(item["max_move"]) / full_move)
                    update_player(data, state, response, player, effective_weight)
                player_rows.append(
                    {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best_value),
                        "relative_gain": gain,
                        "raw_strategy_move": full_move,
                        "effective_weight": effective_weight,
                        "applied_strategy_move": float(
                            _strategy_distance(data, before_player, state, player)[0]
                        ),
                        "optimizer_success": bool(diagnostics["success"]),
                        "retry_used": retried,
                        "chosen_start_index": int(diagnostics["chosen_start_index"]),
                        "attempts": diagnostics["attempts"],
                        "reference_market_diagnostics": market_diagnostics,
                    }
                )
            ending_market, ending_market_diagnostics = solve_nested_market(data, state)
            checkpoint = branch_root / f"sweep_{sweep:03d}.json"
            write_json(
                checkpoint,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": branch,
                    "specification": specification,
                    **item,
                    "player_order": order,
                    "source_workbook": relative(workbook),
                    "algorithmic_proximal_penalties": 0.0,
                    "economic_objective_only": True,
                    "gain_tolerance": float(task["gain_tolerance"]),
                    "players": player_rows,
                    "updated_players": [row["player"] for row in player_rows if row["effective_weight"] > 0.0],
                    "max_sequential_relative_gain": max(row["relative_gain"] for row in player_rows),
                    "strategy_change_metric": max(
                        float(_strategy_distance(data, before_sweep, state, player)[0]) for player in order
                    ),
                    "ending_market_diagnostics": ending_market_diagnostics,
                    "ending_profile": full_state_payload(data, state, ending_market),
                },
            )
            audit_path = branch_root / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
            frozen = run_audit(
                data,
                state,
                starts=1,
                maxiter=int(task["maxiter"]),
                label=f"corrected_{sequence}_{branch}_s{sweep:03d}_one_start",
                output=audit_path,
            )
            current = record_for(frozen, profile_path=checkpoint, audit_path=audit_path, sweep=sweep)
            if current["max_relative_gain"] < best["max_relative_gain"]:
                best = current
            print(
                f"[{sequence} {branch}] sweep {sweep}: frozen max gain "
                f"{100.0 * current['max_relative_gain']:.4f}% "
                f"({current['max_gain_player']}) pass={current['equilibrium_verified']}",
                flush=True,
            )
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "branch": branch,
                    "sweep": sweep,
                    "phase": item["phase"],
                    "latest_frozen_max_relative_gain": current["max_relative_gain"],
                    "best_frozen_max_relative_gain": best["max_relative_gain"],
                    "best_sweep": best["sweep"],
                },
            )
            if current["equilibrium_verified"]:
                selected = current
                break

        chosen = selected or best
        chosen_path = ROOT / str(chosen["profile_path"])
        chosen_state = load_profile(chosen_path, data, source)
        diagnostic_path = branch_root / "audits" / f"audit_selected_s{int(chosen['sweep']):03d}_three_start.json"
        diagnostic = run_audit(
            data,
            chosen_state,
            starts=3,
            maxiter=max(int(task["maxiter"]), 600),
            label=f"corrected_{sequence}_{branch}_selected_s{int(chosen['sweep']):03d}_three_start",
            output=diagnostic_path,
        )
        result = {
            "created": now(),
            "status": "accepted" if selected is not None else "no_pass_within_schedule",
            "pid": os.getpid(),
            "sequence": sequence,
            "branch": branch,
            "specification": specification,
            "player_order": order,
            "input": relative(Path(task["input_path"])),
            "input_sha256": sha256(Path(task["input_path"])),
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": SEQUENCES[sequence]["source_iteration"],
            "source_replay_error": replay_error,
            "selected_profile": chosen["profile_path"],
            "selected_sweep": chosen["sweep"],
            "one_start_audit": chosen["audit_path"],
            "one_start_max_relative_gain": chosen["max_relative_gain"],
            "one_start_max_gain_player": chosen["max_gain_player"],
            "one_start_equilibrium_verified": chosen["equilibrium_verified"],
            "three_start_diagnostic": relative(diagnostic_path),
            "three_start_max_relative_gain": diagnostic["max_relative_gain"],
            "three_start_max_gain_player": diagnostic["max_gain_player"],
            "three_start_equilibrium_verified": diagnostic["equilibrium_verified"],
            "criterion": "one-start common frozen-profile zero-proximal maximum relative gain <= 1%, all six solves successful",
            "claim_scope": "local computational criterion, not a global Nash-equilibrium certificate",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        failure = {
            "created": now(),
            "status": "failed",
            "pid": os.getpid(),
            "sequence": sequence,
            "branch": branch,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def validate(task: dict[str, Any]) -> dict[str, Any]:
    data, source, replay_error, workbook, _ = configure_modules(task)
    branch = str(task["branch"])
    state = initialize_basin(data, source, float(BRANCHES[branch]["cost_weight"]))
    market, diagnostics = solve_nested_market(data, state)
    return {
        "sequence": task["sequence"],
        "branch": branch,
        "source_workbook": relative(workbook),
        "source_replay_error": replay_error,
        "market_solve_successful": market is not None,
        "market_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cold-start-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument("--branches", nargs="+", choices=list(BRANCHES), default=list(BRANCHES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--gain-tolerance", type=float, default=0.01)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.maxiter < 1:
        raise ValueError("workers and maxiter must be positive")
    if not 0.0 < args.gain_tolerance < 1.0:
        raise ValueError("gain tolerance must be between zero and one")
    input_path = args.input.resolve()
    cold_start_root = args.cold_start_root.resolve()
    output_root = args.output_root.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not cold_start_root.is_dir():
        raise FileNotFoundError(cold_start_root)

    branch_priority = [
        branch
        for branch in ("cost_staged", "stage1_selective", "halfcost_selective")
        if branch in args.branches
    ]
    tasks = [
        {
            "sequence": sequence,
            "branch": branch,
            "input_path": str(input_path),
            "cold_start_root": str(cold_start_root),
            "output_root": str(output_root),
            "maxiter": args.maxiter,
            "gain_tolerance": args.gain_tolerance,
        }
        for branch in branch_priority
        for sequence in args.sequences
    ]
    if args.validate_only:
        print(json.dumps([validate(task) for task in tasks], indent=2), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "diversified selective zero-proximal search from corrected Stage-1 endpoints",
        "acceptance_criterion": "one-start common frozen-profile maximum relative gain <= 1%, all six solves successful",
        "three_start_role": "stronger diagnostic retained separately",
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "output_root": relative(output_root),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "branches": {name: BRANCHES[name] for name in args.branches},
        "workers": min(args.workers, len(tasks)),
        "maxiter": args.maxiter,
        "gain_tolerance": args.gain_tolerance,
        "code_sha256": code_hashes(),
        "results": [],
    }
    write_json(manifest_path, manifest)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_branch, task): (task["sequence"], task["branch"]) for task in tasks}
        for future in as_completed(futures):
            sequence, branch = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "status": "executor_exception",
                    "sequence": sequence,
                    "branch": branch,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            manifest["results"] = sorted(results, key=lambda row: (row["sequence"], row["branch"]))
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)
    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(result["status"] in {"accepted", "no_pass_within_schedule"} for result in results)
        else "complete_with_failures"
    )
    manifest["results"] = sorted(results, key=lambda row: (row["sequence"], row["branch"]))
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
