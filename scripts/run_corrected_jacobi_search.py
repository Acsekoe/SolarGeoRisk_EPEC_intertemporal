from __future__ import annotations

"""Run fixed-damping Jacobi best-response sweeps from corrected Stage-1 endpoints.

Every player's best response in a sweep is computed against the same frozen
start-of-sweep profile.  Once all six solves succeed, their strategic decisions
are damped simultaneously with the requested alpha.  The runner uses zero
algorithmic proximal penalties, no move caps, no gain filtering, and a separate
common-frozen-profile one-start deviation audit after every completed sweep.

Unlike the Gauss--Seidel search runners, this experiment always completes the
declared number of sweeps.  Passing profiles are recorded but do not stop the
trajectory, which makes transient and sustained 1% passes distinguishable.
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

from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.run_corrected_a030_search import run_audit
from scripts.run_corrected_diversified_search import update_player
from scripts.run_corrected_equilibrium_search import (
    SEQUENCES,
    configure_modules,
    relative,
    sha256,
    write_json,
)
from scripts.run_local_paper_equilibrium_experiment import (
    audit_profile,
    clone_state,
    full_state_payload,
)
from scripts.run_overnight_equilibrium_experiment import load_profile


BRANCH = "stage1"
BRANCH_SPECIFICATION = {
    "description": "exact corrected Stage-1 endpoint",
    "kind": "exact_stage1_endpoint",
    "cost_weight": 0.0,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def code_hashes() -> dict[str, str]:
    names = [
        "scripts/run_corrected_jacobi_search.py",
        "scripts/run_corrected_a030_search.py",
        "scripts/run_corrected_diversified_search.py",
        "scripts/run_corrected_equilibrium_search.py",
        "scripts/nested_market_audit.py",
        "scripts/run_local_paper_equilibrium_experiment.py",
        "scripts/run_overnight_equilibrium_experiment.py",
        "model/data_prep.py",
        "model/model_main.py",
    ]
    return {name: sha256(ROOT / name) for name in names}


def audit_record(
    audit: dict[str, Any],
    *,
    profile_path: Path,
    audit_path: Path,
    sweep: int,
) -> dict[str, Any]:
    maximum = float(audit["max_relative_gain"])
    successful = bool(audit["all_attempts_successful"])
    return {
        "sweep": sweep,
        "profile_path": relative(profile_path),
        "audit_path": relative(audit_path),
        "max_relative_gain": maximum,
        "max_gain_player": str(audit["max_gain_player"]),
        "all_six_solves_successful": successful,
        "strict_one_percent_equilibrium": successful and maximum < 0.01,
        "audit_equilibrium_verified": bool(audit["equilibrium_verified"]),
    }


def apply_jacobi_responses(
    data: Any,
    frozen_state: dict[str, dict],
    responses: dict[str, dict[str, dict]],
    players: list[str],
    alpha: float,
) -> dict[str, dict]:
    """Apply disjoint player responses to one copy of the frozen profile."""
    next_state = clone_state(frozen_state)
    for player in players:
        update_player(data, next_state, responses[player], player, alpha)
    return next_state


def solve_frozen_response(
    data: Any,
    frozen_state: dict[str, dict],
    frozen_market: dict[str, dict],
    player: str,
    maxiter: int,
) -> tuple[float, dict[str, dict], dict[str, Any], bool]:
    best_value, response, diagnostics = nested_best_response(
        data,
        frozen_state,
        frozen_market,
        player,
        maxiter=maxiter,
        starts=1,
    )
    retry_used = False
    if not diagnostics["success"]:
        retry_used = True
        best_value, response, diagnostics = nested_best_response(
            data,
            frozen_state,
            frozen_market,
            player,
            maxiter=max(2 * maxiter, 1000),
            starts=1,
        )
    return float(best_value), response, diagnostics, retry_used


def run_branch(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    order = list(SEQUENCES[sequence]["order"])
    alpha = float(task["alpha"])
    maxiter = int(task["maxiter"])
    max_sweeps = int(task["max_sweeps"])
    branch_root = Path(task["output_root"]).resolve() / sequence / BRANCH
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
                "branch": BRANCH,
                "update_scheme": "jacobi",
                "alpha": alpha,
            },
        )
        data, source, replay_error, workbook, _ = configure_modules(task)
        initialization_path = branch_root / "initialization.json"
        if initialization_path.exists():
            initial_state = load_profile(initialization_path, data, source)
        else:
            initial_state = clone_state(source)
            initial_market, initial_market_diagnostics = solve_nested_market(
                data, initial_state
            )
            write_json(
                initialization_path,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": BRANCH,
                    "branch_specification": BRANCH_SPECIFICATION,
                    "player_order_for_reporting": order,
                    "update_scheme": "jacobi",
                    "update_definition": (
                        "all six best responses use the same frozen start-of-sweep "
                        "profile; all damped player strategies are committed together"
                    ),
                    "alpha": alpha,
                    "move_cap": None,
                    "gain_filter": None,
                    "source_workbook": relative(workbook),
                    "source_workbook_sha256": sha256(workbook),
                    "source_iteration": SEQUENCES[sequence]["source_iteration"],
                    "source_replay_error": replay_error,
                    "algorithmic_proximal_penalties": 0.0,
                    "audit_starts": 1,
                    "profile": full_state_payload(
                        data, initial_state, initial_market
                    ),
                    "market_diagnostics": initial_market_diagnostics,
                },
            )

        completed = sorted(branch_root.glob("sweep_*.json"))
        state = (
            load_profile(completed[-1], data, source)
            if completed
            else clone_state(initial_state)
        )
        records: list[dict[str, Any]] = []

        initial_audit_path = (
            branch_root / "audits" / "audit_initial_one_start.json"
        )
        initial_audit = run_audit(
            data,
            initial_state,
            maxiter=maxiter,
            label=f"corrected_jacobi_{sequence}_initial_one_start",
            output=initial_audit_path,
        )
        records.append(
            audit_record(
                initial_audit,
                profile_path=initialization_path,
                audit_path=initial_audit_path,
                sweep=0,
            )
        )

        for checkpoint in completed:
            sweep = int(checkpoint.stem.split("_")[-1])
            checkpoint_state = load_profile(checkpoint, data, source)
            audit_path = (
                branch_root
                / "audits"
                / f"audit_sweep_{sweep:03d}_one_start.json"
            )
            audit = run_audit(
                data,
                checkpoint_state,
                maxiter=maxiter,
                label=f"corrected_jacobi_{sequence}_s{sweep:03d}_one_start",
                output=audit_path,
            )
            records.append(
                audit_record(
                    audit,
                    profile_path=checkpoint,
                    audit_path=audit_path,
                    sweep=sweep,
                )
            )

        first_new_sweep = len(completed) + 1
        for sweep in range(first_new_sweep, max_sweeps + 1):
            frozen_state = clone_state(state)
            frozen_market, frozen_market_diagnostics = solve_nested_market(
                data, frozen_state
            )
            responses: dict[str, dict[str, dict]] = {}
            player_rows: list[dict[str, Any]] = []

            for player in order:
                reference = float(
                    nested_economic_objective(
                        data, frozen_state, frozen_market, player
                    )
                )
                best_value, response, diagnostics, retry_used = (
                    solve_frozen_response(
                        data,
                        frozen_state,
                        frozen_market,
                        player,
                        maxiter,
                    )
                )
                if not diagnostics["success"]:
                    raise RuntimeError(
                        f"Jacobi sweep {sweep} {player}: one-start "
                        "best-response solve failed"
                    )
                responses[player] = response
                gain = max(best_value - reference, 0.0) / max(
                    abs(reference), 1.0
                )
                player_rows.append(
                    {
                        "player": player,
                        "common_frozen_profile": True,
                        "reference_objective": reference,
                        "best_response_objective": best_value,
                        "relative_gain": gain,
                        "raw_strategy_move": float(
                            _strategy_distance(
                                data, frozen_state, response, player
                            )[0]
                        ),
                        "damping_weight": alpha,
                        "move_cap": None,
                        "gain_filter": None,
                        "optimizer_success": bool(diagnostics["success"]),
                        "retry_used": retry_used,
                        "starts": 1,
                        "chosen_start_index": int(
                            diagnostics["chosen_start_index"]
                        ),
                        "attempts": diagnostics["attempts"],
                    }
                )

            state = apply_jacobi_responses(
                data, frozen_state, responses, order, alpha
            )
            for row in player_rows:
                row["applied_strategy_move"] = float(
                    _strategy_distance(
                        data, frozen_state, state, row["player"]
                    )[0]
                )

            ending_market, ending_market_diagnostics = solve_nested_market(
                data, state
            )
            checkpoint = branch_root / f"sweep_{sweep:03d}.json"
            write_json(
                checkpoint,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": BRANCH,
                    "branch_specification": BRANCH_SPECIFICATION,
                    "sweep": sweep,
                    "player_order_for_reporting": order,
                    "update_scheme": "jacobi",
                    "common_start_of_sweep_profile": True,
                    "simultaneous_commit": True,
                    "alpha": alpha,
                    "move_cap": None,
                    "gain_filter": None,
                    "source_workbook": relative(workbook),
                    "algorithmic_proximal_penalties": 0.0,
                    "players": player_rows,
                    "max_start_of_sweep_relative_gain": max(
                        row["relative_gain"] for row in player_rows
                    ),
                    "strategy_change_metric": max(
                        float(
                            _strategy_distance(
                                data, frozen_state, state, player
                            )[0]
                        )
                        for player in order
                    ),
                    "start_of_sweep_market_diagnostics": (
                        frozen_market_diagnostics
                    ),
                    "ending_market_diagnostics": ending_market_diagnostics,
                    "ending_profile": full_state_payload(
                        data, state, ending_market
                    ),
                },
            )

            audit_path = (
                branch_root
                / "audits"
                / f"audit_sweep_{sweep:03d}_one_start.json"
            )
            frozen_audit = run_audit(
                data,
                state,
                maxiter=maxiter,
                label=f"corrected_jacobi_{sequence}_s{sweep:03d}_one_start",
                output=audit_path,
            )
            current = audit_record(
                frozen_audit,
                profile_path=checkpoint,
                audit_path=audit_path,
                sweep=sweep,
            )
            records.append(current)
            best = min(records, key=lambda row: row["max_relative_gain"])
            accepted_sweeps = [
                row["sweep"]
                for row in records
                if row["strict_one_percent_equilibrium"]
            ]
            print(
                f"[{sequence} jacobi] sweep {sweep}: frozen max gain "
                f"{100.0 * current['max_relative_gain']:.4f}% "
                f"({current['max_gain_player']}) strict_pass="
                f"{current['strict_one_percent_equilibrium']}",
                flush=True,
            )
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "branch": BRANCH,
                    "update_scheme": "jacobi",
                    "sweep": sweep,
                    "alpha": alpha,
                    "latest_one_start_max_relative_gain": current[
                        "max_relative_gain"
                    ],
                    "latest_all_six_solves_successful": current[
                        "all_six_solves_successful"
                    ],
                    "best_one_start_max_relative_gain": best[
                        "max_relative_gain"
                    ],
                    "best_sweep": best["sweep"],
                    "strict_pass_sweeps": accepted_sweeps,
                },
            )

        best = min(records, key=lambda row: row["max_relative_gain"])
        accepted = [
            row for row in records if row["strict_one_percent_equilibrium"]
        ]
        result = {
            "created": now(),
            "status": "complete",
            "pid": os.getpid(),
            "sequence": sequence,
            "branch": BRANCH,
            "branch_specification": BRANCH_SPECIFICATION,
            "player_order_for_reporting": order,
            "update_scheme": "jacobi",
            "update_definition": (
                "all best responses evaluated at one common frozen "
                "start-of-sweep profile, followed by one simultaneous "
                "damped commit"
            ),
            "input": relative(Path(task["input_path"])),
            "input_sha256": sha256(Path(task["input_path"])),
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": SEQUENCES[sequence]["source_iteration"],
            "source_replay_error": replay_error,
            "alpha": alpha,
            "completed_sweeps": max_sweeps,
            "continued_after_pass": True,
            "move_cap": None,
            "gain_filter": None,
            "algorithmic_proximal_penalties": 0.0,
            "audit_starts": 1,
            "strict_one_percent_equilibrium_found": bool(accepted),
            "strict_pass_sweeps": [row["sweep"] for row in accepted],
            "first_strict_pass_sweep": (
                accepted[0]["sweep"] if accepted else None
            ),
            "best_profile": best["profile_path"],
            "best_sweep": best["sweep"],
            "best_one_start_audit": best["audit_path"],
            "best_one_start_max_relative_gain": best[
                "max_relative_gain"
            ],
            "best_one_start_max_gain_player": best["max_gain_player"],
            "criterion": (
                "one-start common frozen-profile zero-proximal maximum "
                "relative gain < 1%, all six solves successful"
            ),
            "claim_scope": (
                "local computational 1%-equilibrium criterion; no "
                "multistart or global claim"
            ),
            "audit_trajectory": records,
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
            "branch": BRANCH,
            "update_scheme": "jacobi",
            "alpha": alpha,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def validate(task: dict[str, Any]) -> dict[str, Any]:
    data, source, replay_error, workbook, _ = configure_modules(task)
    market, diagnostics = solve_nested_market(data, source)
    return {
        "sequence": task["sequence"],
        "source_iteration": SEQUENCES[str(task["sequence"])][
            "source_iteration"
        ],
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
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=list(SEQUENCES),
        default=list(SEQUENCES),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--max-sweeps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.maxiter < 1 or args.max_sweeps < 1:
        raise ValueError("workers, maxiter, and max-sweeps must be positive")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    input_path = args.input.resolve()
    cold_start_root = args.cold_start_root.resolve()
    output_root = args.output_root.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not cold_start_root.is_dir():
        raise FileNotFoundError(cold_start_root)

    tasks = [
        {
            "sequence": sequence,
            "input_path": str(input_path),
            "cold_start_root": str(cold_start_root),
            "output_root": str(output_root),
            "maxiter": args.maxiter,
            "max_sweeps": args.max_sweeps,
            "alpha": args.alpha,
        }
        for sequence in args.sequences
    ]
    if args.validate_only:
        print(
            json.dumps([validate(task) for task in tasks], indent=2),
            flush=True,
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    protocol_path = args.protocol.resolve() if args.protocol else None
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": (
            "fixed-alpha simultaneous Jacobi best-response experiment "
            "from exact corrected Stage-1 endpoints"
        ),
        "update_scheme": "jacobi",
        "update_definition": (
            "all six best responses are computed against the same frozen "
            "start-of-sweep profile; their alpha-damped strategies are "
            "then committed together"
        ),
        "acceptance_criterion": (
            "one-start common frozen-profile maximum relative gain < 1%, "
            "all six solves successful"
        ),
        "continue_after_acceptance": True,
        "acceptance_audit_starts": 1,
        "multistart_used": False,
        "alpha": args.alpha,
        "move_cap": None,
        "gain_filter": None,
        "players_frozen": False,
        "algorithmic_proximal_penalties": 0.0,
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "output_root": relative(output_root),
        "protocol": relative(protocol_path) if protocol_path else None,
        "protocol_sha256": (
            sha256(protocol_path) if protocol_path else None
        ),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "branch": BRANCH_SPECIFICATION,
        "workers": min(args.workers, len(tasks)),
        "maxiter": args.maxiter,
        "max_sweeps": args.max_sweeps,
        "code_sha256": code_hashes(),
        "results": [],
    }
    write_json(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(tasks))
    ) as pool:
        futures = {
            pool.submit(run_branch, task): task["sequence"] for task in tasks
        }
        for future in as_completed(futures):
            sequence = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "status": "executor_exception",
                    "sequence": sequence,
                    "branch": BRANCH,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            manifest["results"] = sorted(
                results, key=lambda row: row["sequence"]
            )
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(result["status"] == "complete" for result in results)
        else "complete_with_failures"
    )
    manifest["results"] = sorted(
        results, key=lambda row: row["sequence"]
    )
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": relative(manifest_path),
                "status": manifest["status"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
