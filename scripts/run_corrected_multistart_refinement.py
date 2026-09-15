from __future__ import annotations

"""Refine corrected local equilibria using three-start best responses.

Each task starts from an accepted corrected cost-staged profile.  Sequential
updates and frozen-profile audits both use three optimizer starts per player.
Only players whose identified relative gain exceeds the requested tolerance
are moved, and every applied move is damped and normalized-cap limited.
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

from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
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


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def code_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def cached_audit(
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


def record(
    audit: dict[str, Any],
    checkpoint: Path,
    audit_path: Path,
    sweep: int,
) -> dict[str, Any]:
    return {
        "sweep": sweep,
        "checkpoint": relative(checkpoint),
        "audit": relative(audit_path),
        "max_relative_gain": float(audit["max_relative_gain"]),
        "max_gain_player": str(audit["max_gain_player"]),
        "equilibrium_verified": bool(audit["equilibrium_verified"]),
        "all_attempts_successful": bool(audit["all_attempts_successful"]),
    }


def source_for(source_root: Path, sequence: str) -> tuple[Path, dict[str, Any]]:
    status_path = source_root / sequence / "cost_staged" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "accepted":
        raise RuntimeError(f"Source profile is not accepted: {status_path}")
    checkpoint = resolve_recorded_path(str(status["selected_profile"]))
    return checkpoint, status


def run_sequence(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    order = list(SEQUENCES[sequence]["order"])
    output_root = Path(task["output_root"]).resolve()
    sequence_root = output_root / sequence
    audit_root = sequence_root / "audits"
    status_path = sequence_root / "status.json"
    try:
        sequence_root.mkdir(parents=True, exist_ok=True)
        write_json(
            status_path,
            {
                "created": now(),
                "status": "initializing",
                "pid": os.getpid(),
                "sequence": sequence,
                "player_order": order,
            },
        )
        source_checkpoint, source_status = source_for(
            Path(task["source_root"]).resolve(), sequence
        )
        data, source_template, replay_error, workbook, _ = configure_modules(task)
        initial_state = load_profile(source_checkpoint, data, source_template)

        completed = sorted(sequence_root.glob("sweep_*.json"))
        state = (
            load_profile(completed[-1], data, source_template)
            if completed
            else clone_state(initial_state)
        )
        initial_audit_path = audit_root / "audit_initial_three_start.json"
        initial_audit = cached_audit(
            data,
            initial_state,
            starts=int(task["starts"]),
            maxiter=int(task["maxiter"]),
            label=f"corrected_{sequence}_multistart_initial",
            output=initial_audit_path,
        )
        best = record(initial_audit, source_checkpoint, initial_audit_path, 0)
        selected = best if best["equilibrium_verified"] else None

        # Restore the best fully audited checkpoint when resuming.
        for checkpoint in completed:
            sweep = int(checkpoint.stem.split("_")[-1])
            audit_path = audit_root / f"audit_sweep_{sweep:03d}_three_start.json"
            if not audit_path.exists():
                continue
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            candidate = record(existing, checkpoint, audit_path, sweep)
            if candidate["max_relative_gain"] < best["max_relative_gain"]:
                best = candidate
            if candidate["equilibrium_verified"] and selected is None:
                selected = candidate

        start_sweep = len(completed) + 1
        for sweep in range(start_sweep, int(task["max_sweeps"]) + 1):
            if selected is not None:
                break
            before_sweep = clone_state(state)
            player_rows = []
            for player in order:
                before_player = clone_state(state)
                market, market_diagnostics = solve_nested_market(data, state)
                reference = float(
                    nested_economic_objective(data, state, market, player)
                )
                best_objective, response, diagnostics = nested_best_response(
                    data,
                    state,
                    market,
                    player,
                    maxiter=int(task["maxiter"]),
                    starts=int(task["starts"]),
                )
                retried = False
                if not diagnostics["success"]:
                    retried = True
                    best_objective, response, diagnostics = nested_best_response(
                        data,
                        state,
                        market,
                        player,
                        maxiter=max(2 * int(task["maxiter"]), 1000),
                        starts=max(int(task["starts"]), 4),
                    )
                if not diagnostics["success"]:
                    raise RuntimeError(
                        f"multistart sweep {sweep} {player}: best-response solve failed"
                    )
                gain = max(float(best_objective) - reference, 0.0) / max(
                    abs(reference), 1.0
                )
                full_move = float(
                    _strategy_distance(data, before_player, response, player)[0]
                )
                weight = 0.0
                if gain > float(task["gain_tolerance"]):
                    weight = float(task["omega"])
                    if full_move > 0.0:
                        weight = min(
                            weight, float(task["max_move"]) / full_move
                        )
                    update_player(data, state, response, player, weight)
                player_rows.append(
                    {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best_objective),
                        "relative_gain": gain,
                        "raw_strategy_move": full_move,
                        "effective_weight": weight,
                        "applied_strategy_move": float(
                            _strategy_distance(
                                data, before_player, state, player
                            )[0]
                        ),
                        "optimizer_success": bool(diagnostics["success"]),
                        "all_update_attempts_successful": all(
                            bool(attempt["success"])
                            for attempt in diagnostics["attempts"]
                        ),
                        "chosen_start_index": int(
                            diagnostics["chosen_start_index"]
                        ),
                        "retry_used": retried,
                        "attempts": diagnostics["attempts"],
                        "reference_market_diagnostics": market_diagnostics,
                    }
                )

            ending_market, ending_market_diagnostics = solve_nested_market(data, state)
            checkpoint = sequence_root / f"sweep_{sweep:03d}.json"
            write_json(
                checkpoint,
                {
                    "created": now(),
                    "sequence": sequence,
                    "method": "three-start selective zero-proximal refinement",
                    "sweep": sweep,
                    "player_order": order,
                    "source_checkpoint": relative(source_checkpoint),
                    "source_workbook": relative(workbook),
                    "source_replay_error": replay_error,
                    "algorithmic_proximal_penalties": 0.0,
                    "starts_per_update": int(task["starts"]),
                    "gain_tolerance": float(task["gain_tolerance"]),
                    "omega": float(task["omega"]),
                    "max_move": float(task["max_move"]),
                    "players": player_rows,
                    "updated_players": [
                        row["player"]
                        for row in player_rows
                        if row["effective_weight"] > 0.0
                    ],
                    "max_sequential_relative_gain": max(
                        row["relative_gain"] for row in player_rows
                    ),
                    "strategy_change_metric": max(
                        float(
                            _strategy_distance(
                                data, before_sweep, state, player
                            )[0]
                        )
                        for player in order
                    ),
                    "distance_from_starting_equilibrium": max(
                        float(
                            _strategy_distance(
                                data, initial_state, state, player
                            )[0]
                        )
                        for player in order
                    ),
                    "ending_market_diagnostics": ending_market_diagnostics,
                    "ending_profile": full_state_payload(
                        data, state, ending_market
                    ),
                },
            )
            audit_path = audit_root / f"audit_sweep_{sweep:03d}_three_start.json"
            frozen = cached_audit(
                data,
                state,
                starts=int(task["starts"]),
                maxiter=int(task["maxiter"]),
                label=f"corrected_{sequence}_multistart_s{sweep:03d}",
                output=audit_path,
            )
            current = record(frozen, checkpoint, audit_path, sweep)
            if current["max_relative_gain"] < best["max_relative_gain"]:
                best = current
            if current["equilibrium_verified"]:
                selected = current
            print(
                f"[{sequence}] sweep {sweep}: three-start frozen max gain "
                f"{100.0 * current['max_relative_gain']:.4f}% "
                f"({current['max_gain_player']}) "
                f"pass={current['equilibrium_verified']}",
                flush=True,
            )
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "sweep": sweep,
                    "latest_three_start_max_relative_gain": current[
                        "max_relative_gain"
                    ],
                    "latest_three_start_max_gain_player": current[
                        "max_gain_player"
                    ],
                    "best_three_start_max_relative_gain": best[
                        "max_relative_gain"
                    ],
                    "best_sweep": best["sweep"],
                    "all_latest_audit_attempts_successful": current[
                        "all_attempts_successful"
                    ],
                },
            )

        chosen = selected or best
        chosen_state = load_profile(
            resolve_recorded_path(chosen["checkpoint"]), data, source_template
        )
        one_start_path = audit_root / (
            f"audit_selected_s{chosen['sweep']:03d}_one_start.json"
        )
        one_start = cached_audit(
            data,
            chosen_state,
            starts=1,
            maxiter=int(task["maxiter"]),
            label=f"corrected_{sequence}_multistart_selected_one_start",
            output=one_start_path,
        )
        result = {
            "created": now(),
            "status": (
                "accepted_three_start_local_1pct_equilibrium"
                if selected is not None
                else "no_three_start_pass_within_limit"
            ),
            "pid": os.getpid(),
            "sequence": sequence,
            "player_order": order,
            "input": relative(Path(task["input_path"])),
            "input_sha256": sha256(Path(task["input_path"])),
            "source_workbook": relative(workbook),
            "source_checkpoint": relative(source_checkpoint),
            "source_one_start_max_relative_gain": source_status[
                "one_start_max_relative_gain"
            ],
            "source_three_start_max_relative_gain": source_status[
                "three_start_max_relative_gain"
            ],
            "selected_checkpoint": chosen["checkpoint"],
            "selected_sweep": chosen["sweep"],
            "three_start_audit": chosen["audit"],
            "three_start_max_relative_gain": chosen["max_relative_gain"],
            "three_start_max_gain_player": chosen["max_gain_player"],
            "three_start_equilibrium_verified": chosen[
                "equilibrium_verified"
            ],
            "three_start_all_attempts_successful": chosen[
                "all_attempts_successful"
            ],
            "one_start_audit": relative(one_start_path),
            "one_start_max_relative_gain": one_start["max_relative_gain"],
            "one_start_max_gain_player": one_start["max_gain_player"],
            "one_start_equilibrium_verified": one_start[
                "equilibrium_verified"
            ],
            "criterion": (
                "three-start common frozen-profile zero-proximal maximum "
                "relative gain <= 1%, all 18 attempts successful"
            ),
            "claim_scope": (
                "stronger local multistart computational criterion; not a "
                "global Nash-equilibrium certificate"
            ),
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
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cold-start-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        nargs="+",
        choices=list(SEQUENCES),
        default=list(SEQUENCES),
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-sweeps", type=int, default=60)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--gain-tolerance", type=float, default=0.01)
    parser.add_argument("--omega", type=float, default=0.05)
    parser.add_argument("--max-move", type=float, default=0.005)
    args = parser.parse_args()

    if args.workers < 1 or args.max_sweeps < 1 or args.maxiter < 1:
        raise ValueError("workers, max-sweeps, and maxiter must be positive")
    if args.starts < 2:
        raise ValueError("starts must be at least two for multistart refinement")
    if not 0.0 < args.gain_tolerance < 1.0:
        raise ValueError("gain-tolerance must lie between zero and one")
    if not 0.0 < args.omega <= 1.0 or args.max_move <= 0.0:
        raise ValueError("omega must be in (0,1] and max-move must be positive")

    input_path = args.input.resolve()
    cold_start_root = args.cold_start_root.resolve()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    for path in (input_path,):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (cold_start_root, source_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "sequence": sequence,
            "input_path": str(input_path),
            "cold_start_root": str(cold_start_root),
            "source_root": str(source_root),
            "output_root": str(output_root),
            "max_sweeps": args.max_sweeps,
            "maxiter": args.maxiter,
            "starts": args.starts,
            "gain_tolerance": args.gain_tolerance,
            "omega": args.omega,
            "max_move": args.max_move,
        }
        for sequence in args.sequences
    ]
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": (
            "three-start selective zero-proximal refinement from accepted "
            "corrected cost-staged profiles"
        ),
        "acceptance_criterion": (
            "three-start common frozen-profile maximum relative gain <= 1%, "
            "all attempts successful"
        ),
        "claim_scope": (
            "stronger local multistart computational criterion; not a global "
            "Nash-equilibrium certificate"
        ),
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "source_root": relative(source_root),
        "output_root": relative(output_root),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "workers": min(args.workers, len(tasks)),
        "max_sweeps": args.max_sweeps,
        "maxiter": args.maxiter,
        "starts": args.starts,
        "gain_tolerance": args.gain_tolerance,
        "omega": args.omega,
        "max_move": args.max_move,
        "code_sha256": {
            relative(Path(__file__)): code_hash(Path(__file__)),
            "scripts/nested_market_audit.py": code_hash(
                ROOT / "scripts/nested_market_audit.py"
            ),
            "model/data_prep.py": code_hash(ROOT / "model/data_prep.py"),
            "model/model_main.py": code_hash(ROOT / "model/model_main.py"),
        },
        "results": [],
    }
    write_json(manifest_path, manifest)

    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {
            pool.submit(run_sequence, task): task["sequence"] for task in tasks
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
        if all(
            result["status"]
            in {
                "accepted_three_start_local_1pct_equilibrium",
                "no_three_start_pass_within_limit",
            }
            for result in results
        )
        else "complete_with_failures"
    )
    manifest["results"] = sorted(results, key=lambda row: row["sequence"])
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"manifest": relative(manifest_path), "status": manifest["status"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
