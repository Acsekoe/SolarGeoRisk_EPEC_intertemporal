from __future__ import annotations

"""Fixed-alpha 0.30 local-equilibrium search under corrected demand.

Every branch performs sequential, zero-proximal best-response updates with an
exact convex damping weight of 0.30.  No normalized move cap is applied.  A
separate one-start audit is run after every full sweep while holding the common
profile fixed.  A checkpoint is accepted exactly when all six audit solves
succeed and the largest relative unilateral gain is at most one percent.

Multistart solves are deliberately not used by this runner: neither candidate
generation nor acceptance depends on them.
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
from scripts.nested_market_audit import nested_best_response, nested_economic_objective, solve_nested_market
from scripts.run_corrected_diversified_search import initialize_basin, update_player
from scripts.run_corrected_equilibrium_search import SEQUENCES, configure_modules, relative, sha256, write_json
from scripts.run_local_paper_equilibrium_experiment import audit_profile, clone_state, full_state_payload
from scripts.run_overnight_equilibrium_experiment import load_profile
from scripts.search_nested_equilibrium import _sync_quantity


MIN_ALPHA = 0.30
BRANCHES: dict[str, dict[str, Any]] = {
    "stage1": {
        "description": "exact corrected Stage-1 endpoint",
        "kind": "cost_interpolation",
        "cost_weight": 0.0,
    },
    "halfcost": {
        "description": "corrected Stage-1 capacities; offers halfway toward manufacturing cost",
        "kind": "cost_interpolation",
        "cost_weight": 0.5,
    },
    "cost": {
        "description": "corrected Stage-1 capacities; manufacturing-cost offers",
        "kind": "cost_interpolation",
        "cost_weight": 1.0,
    },
    "historical_o6": {
        "description": "historical O6 sweep 15 strategy used only as a corrected-demand initialization",
        "kind": "checkpoint",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def code_hashes() -> dict[str, str]:
    names = [
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


def make_initial_state(
    data: Any,
    source: dict[str, dict],
    branch: str,
    historical_o6: Path,
) -> dict[str, dict]:
    specification = BRANCHES[branch]
    if specification["kind"] == "checkpoint":
        state = load_profile(historical_o6, data, source)
        _sync_quantity(data, state)
        return state
    return initialize_basin(data, source, float(specification["cost_weight"]))


def run_audit(
    data: Any,
    state: dict[str, dict],
    *,
    maxiter: int,
    label: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    return audit_profile(data, state, starts=1, maxiter=maxiter, label=label, output=output)


def audit_record(
    audit: dict[str, Any],
    *,
    profile_path: Path,
    audit_path: Path,
    sweep: int,
) -> dict[str, Any]:
    return {
        "sweep": sweep,
        "profile_path": relative(profile_path),
        "audit_path": relative(audit_path),
        "max_relative_gain": float(audit["max_relative_gain"]),
        "max_gain_player": str(audit["max_gain_player"]),
        "all_six_solves_successful": bool(audit["all_attempts_successful"]),
        "local_one_percent_equilibrium": bool(audit["equilibrium_verified"]),
    }


def run_branch(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    branch = str(task["branch"])
    order = list(SEQUENCES[sequence]["order"])
    branch_root = Path(task["output_root"]).resolve() / sequence / branch
    status_path = branch_root / "status.json"
    alpha = float(task["alpha"])
    update_gain_threshold = task.get("update_gain_threshold")
    if update_gain_threshold is not None:
        update_gain_threshold = float(update_gain_threshold)
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
                "alpha": alpha,
                "update_gain_threshold": update_gain_threshold,
            },
        )
        data, source, replay_error, workbook, _ = configure_modules(task)
        historical_o6 = Path(task["historical_o6"]).resolve()
        initialization_path = branch_root / "initialization.json"
        if initialization_path.exists():
            initial_state = load_profile(initialization_path, data, source)
        else:
            initial_state = make_initial_state(data, source, branch, historical_o6)
            initial_market, initial_market_diagnostics = solve_nested_market(data, initial_state)
            write_json(
                initialization_path,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": branch,
                    "branch_specification": BRANCHES[branch],
                    "player_order": order,
                    "alpha": alpha,
                    "minimum_allowed_alpha": MIN_ALPHA,
                    "move_cap": None,
                    "source_workbook": relative(workbook),
                    "source_workbook_sha256": sha256(workbook),
                    "source_iteration": SEQUENCES[sequence]["source_iteration"],
                    "source_replay_error": replay_error,
                    "historical_o6_initialization": (
                        relative(historical_o6) if branch == "historical_o6" else None
                    ),
                    "algorithmic_proximal_penalties": 0.0,
                    "terminal_salvage_fraction": float(
                        task.get("terminal_salvage_fraction", 0.0)
                    ),
                    "multistart_used": False,
                    "profile": full_state_payload(data, initial_state, initial_market),
                    "market_diagnostics": initial_market_diagnostics,
                },
            )

        completed = sorted(branch_root.glob("sweep_*.json"))
        state = load_profile(completed[-1], data, source) if completed else clone_state(initial_state)
        initial_audit_path = branch_root / "audits" / "audit_initial_one_start.json"
        initial_audit = run_audit(
            data,
            initial_state,
            maxiter=int(task["maxiter"]),
            label=f"corrected_{sequence}_{branch}_initial_one_start",
            output=initial_audit_path,
        )
        best = audit_record(
            initial_audit,
            profile_path=initialization_path,
            audit_path=initial_audit_path,
            sweep=0,
        )
        selected: dict[str, Any] | None = (
            best if best["local_one_percent_equilibrium"] else None
        )

        # Complete or recover the one-start audit for every existing checkpoint.
        for checkpoint in completed:
            sweep = int(checkpoint.stem.split("_")[-1])
            checkpoint_state = load_profile(checkpoint, data, source)
            audit_path = branch_root / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
            audit = run_audit(
                data,
                checkpoint_state,
                maxiter=int(task["maxiter"]),
                label=f"corrected_{sequence}_{branch}_s{sweep:03d}_one_start",
                output=audit_path,
            )
            record = audit_record(
                audit,
                profile_path=checkpoint,
                audit_path=audit_path,
                sweep=sweep,
            )
            if record["max_relative_gain"] < best["max_relative_gain"]:
                best = record
            if selected is None and record["local_one_percent_equilibrium"]:
                selected = record

        first_new_sweep = len(completed) + 1
        if selected is None:
            for sweep in range(first_new_sweep, int(task["max_sweeps"]) + 1):
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
                    retry_used = False
                    if not diagnostics["success"]:
                        retry_used = True
                        best_value, response, diagnostics = nested_best_response(
                            data,
                            state,
                            market,
                            player,
                            maxiter=max(2 * int(task["maxiter"]), 1000),
                            starts=1,
                        )
                    if not diagnostics["success"]:
                        raise RuntimeError(
                            f"{branch} sweep {sweep} {player}: one-start best-response solve failed"
                        )
                    gain = max(float(best_value) - reference, 0.0) / max(abs(reference), 1.0)
                    raw_move = float(_strategy_distance(data, before_player, response, player)[0])
                    effective_weight = (
                        alpha
                        if update_gain_threshold is None or gain > update_gain_threshold
                        else 0.0
                    )
                    update_player(data, state, response, player, effective_weight)
                    player_rows.append(
                        {
                            "player": player,
                            "reference_objective": reference,
                            "best_response_objective": float(best_value),
                            "relative_gain": gain,
                            "raw_strategy_move": raw_move,
                            "nominal_damping_weight": alpha,
                            "damping_weight": effective_weight,
                            "update_applied": bool(effective_weight > 0.0),
                            "update_gain_threshold": update_gain_threshold,
                            "move_cap": None,
                            "applied_strategy_move": float(
                                _strategy_distance(data, before_player, state, player)[0]
                            ),
                            "optimizer_success": bool(diagnostics["success"]),
                            "retry_used": retry_used,
                            "starts": 1,
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
                        "branch_specification": BRANCHES[branch],
                        "sweep": sweep,
                        "player_order": order,
                        "alpha": alpha,
                        "minimum_allowed_alpha": MIN_ALPHA,
                        "update_gain_threshold": update_gain_threshold,
                        "players_at_or_below_gain_threshold_are_frozen": (
                            update_gain_threshold is not None
                        ),
                        "move_cap": None,
                        "source_workbook": relative(workbook),
                        "algorithmic_proximal_penalties": 0.0,
                        "terminal_salvage_fraction": float(
                            task.get("terminal_salvage_fraction", 0.0)
                        ),
                        "multistart_used": False,
                        "players": player_rows,
                        "updated_players": [
                            row["player"] for row in player_rows if row["update_applied"]
                        ],
                        "max_sequential_relative_gain": max(
                            row["relative_gain"] for row in player_rows
                        ),
                        "strategy_change_metric": max(
                            float(_strategy_distance(data, before_sweep, state, player)[0])
                            for player in order
                        ),
                        "ending_market_diagnostics": ending_market_diagnostics,
                        "ending_profile": full_state_payload(data, state, ending_market),
                    },
                )
                audit_path = branch_root / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
                frozen = run_audit(
                    data,
                    state,
                    maxiter=int(task["maxiter"]),
                    label=f"corrected_{sequence}_{branch}_s{sweep:03d}_one_start",
                    output=audit_path,
                )
                current = audit_record(
                    frozen,
                    profile_path=checkpoint,
                    audit_path=audit_path,
                    sweep=sweep,
                )
                if current["max_relative_gain"] < best["max_relative_gain"]:
                    best = current
                print(
                    f"[{sequence} {branch}] sweep {sweep}: one-start frozen max gain "
                    f"{100.0 * current['max_relative_gain']:.4f}% "
                    f"({current['max_gain_player']}) pass="
                    f"{current['local_one_percent_equilibrium']}",
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
                        "alpha": alpha,
                        "update_gain_threshold": update_gain_threshold,
                        "latest_one_start_max_relative_gain": current["max_relative_gain"],
                        "latest_all_six_solves_successful": current[
                            "all_six_solves_successful"
                        ],
                        "best_one_start_max_relative_gain": best["max_relative_gain"],
                        "best_sweep": best["sweep"],
                    },
                )
                if current["local_one_percent_equilibrium"]:
                    selected = current
                    break

        chosen = selected or best
        result = {
            "created": now(),
            "status": "accepted" if selected is not None else "no_pass_within_schedule",
            "pid": os.getpid(),
            "sequence": sequence,
            "branch": branch,
            "branch_specification": BRANCHES[branch],
            "player_order": order,
            "input": relative(Path(task["input_path"])),
            "input_sha256": sha256(Path(task["input_path"])),
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": SEQUENCES[sequence]["source_iteration"],
            "source_replay_error": replay_error,
            "alpha": alpha,
            "minimum_allowed_alpha": MIN_ALPHA,
            "update_gain_threshold": update_gain_threshold,
            "players_at_or_below_gain_threshold_are_frozen": (
                update_gain_threshold is not None
            ),
            "move_cap": None,
            "algorithmic_proximal_penalties": 0.0,
            "terminal_salvage_fraction": float(
                task.get("terminal_salvage_fraction", 0.0)
            ),
            "multistart_used": False,
            "selected_profile": chosen["profile_path"],
            "selected_sweep": chosen["sweep"],
            "one_start_audit": chosen["audit_path"],
            "one_start_max_relative_gain": chosen["max_relative_gain"],
            "one_start_max_gain_player": chosen["max_gain_player"],
            "all_six_solves_successful": chosen["all_six_solves_successful"],
            "local_one_percent_equilibrium": chosen["local_one_percent_equilibrium"],
            "criterion": "one-start common frozen-profile zero-proximal maximum relative gain <= 1%, all six solves successful",
            "claim_scope": "local computational 1%-equilibrium criterion; no multistart or global claim",
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
            "alpha": alpha,
            "update_gain_threshold": update_gain_threshold,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def validate(task: dict[str, Any]) -> dict[str, Any]:
    data, source, replay_error, workbook, _ = configure_modules(task)
    historical_o6 = Path(task["historical_o6"]).resolve()
    state = make_initial_state(data, source, str(task["branch"]), historical_o6)
    market, diagnostics = solve_nested_market(data, state)
    return {
        "sequence": task["sequence"],
        "branch": task["branch"],
        "alpha": task["alpha"],
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
        "--historical-o6",
        type=Path,
        default=ROOT
        / "outputs/equilibrium_search/ch-row-apac-us-eu-af/overnight/overnight_20260914_224500/branches/O6/sweep_015.json",
    )
    parser.add_argument("--sequences", nargs="+", choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument("--branches", nargs="+", choices=list(BRANCHES), default=list(BRANCHES))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--max-sweeps", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=MIN_ALPHA)
    parser.add_argument(
        "--update-gain-threshold",
        type=float,
        default=None,
        help="Apply a player's update only when its sequential relative gain exceeds this threshold.",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.maxiter < 1 or args.max_sweeps < 1:
        raise ValueError("workers, maxiter, and max-sweeps must be positive")
    if not MIN_ALPHA <= args.alpha <= 1.0:
        raise ValueError(f"alpha must be between the allowed floor {MIN_ALPHA:.2f} and 1.0")
    if args.update_gain_threshold is not None and not 0.0 <= args.update_gain_threshold < 1.0:
        raise ValueError("update-gain-threshold must be in [0, 1)")

    input_path = args.input.resolve()
    cold_start_root = args.cold_start_root.resolve()
    output_root = args.output_root.resolve()
    historical_o6 = args.historical_o6.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not cold_start_root.is_dir():
        raise FileNotFoundError(cold_start_root)
    if "historical_o6" in args.branches and not historical_o6.is_file():
        raise FileNotFoundError(historical_o6)

    branch_priority = [
        branch
        for branch in ("historical_o6", "halfcost", "cost", "stage1")
        if branch in args.branches
    ]
    tasks: list[dict[str, Any]] = []
    for branch in branch_priority:
        for sequence in args.sequences:
            if branch == "historical_o6" and sequence != "ch-row-apac-us-eu-af":
                continue
            tasks.append(
                {
                    "sequence": sequence,
                    "branch": branch,
                    "input_path": str(input_path),
                    "cold_start_root": str(cold_start_root),
                    "output_root": str(output_root),
                    "historical_o6": str(historical_o6),
                    "maxiter": args.maxiter,
                    "max_sweeps": args.max_sweeps,
                    "alpha": args.alpha,
                    "update_gain_threshold": args.update_gain_threshold,
                }
            )
    if not tasks:
        raise ValueError("No compatible sequence/branch tasks were selected")
    if args.validate_only:
        print(json.dumps([validate(task) for task in tasks], indent=2), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": (
            "fixed-alpha, one-start, zero-proximal selective Gauss--Seidel search under corrected demand"
            if args.update_gain_threshold is not None
            else "fixed-alpha, one-start, zero-proximal Gauss--Seidel search under corrected demand"
        ),
        "acceptance_criterion": "one-start common frozen-profile maximum relative gain <= 1%, all six solves successful",
        "acceptance_audit_starts": 1,
        "multistart_used": False,
        "alpha": args.alpha,
        "minimum_allowed_alpha": MIN_ALPHA,
        "update_gain_threshold": args.update_gain_threshold,
        "players_at_or_below_gain_threshold_are_frozen": (
            args.update_gain_threshold is not None
        ),
        "move_cap": None,
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "historical_o6": relative(historical_o6),
        "output_root": relative(output_root),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "branches": {name: BRANCHES[name] for name in branch_priority},
        "workers": min(args.workers, len(tasks)),
        "maxiter": args.maxiter,
        "max_sweeps": args.max_sweeps,
        "code_sha256": code_hashes(),
        "results": [],
    }
    write_json(manifest_path, manifest)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {
            pool.submit(run_branch, task): (task["sequence"], task["branch"])
            for task in tasks
        }
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
            manifest["results"] = sorted(
                results, key=lambda row: (row["sequence"], row["branch"])
            )
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)
    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(
            result["status"] in {"accepted", "no_pass_within_schedule"}
            for result in results
        )
        else "complete_with_failures"
    )
    manifest["results"] = sorted(
        results, key=lambda row: (row["sequence"], row["branch"])
    )
    write_json(manifest_path, manifest)
    print(
        json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
