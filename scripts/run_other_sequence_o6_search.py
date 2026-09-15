from __future__ import annotations

"""Repeat the accepted O6 search path for the other manuscript sequences.

Each sequence is reconstructed at its first endpoint with three consecutive
reported strategy-change residuals below one percent.  The candidate path is
then identical to the accepted O6 path:

1. interpolate the manuscript-profile offer prices 50% toward manufacturing cost;
2. run six zero-proximal Gauss--Seidel sweeps with alpha=0.50;
3. start the adaptive O6 schedule at alpha=0.30 (bounds 0.025--0.50);
4. every fifth sweep, audit one unregularized deviation solve per player from
   the common frozen candidate profile and stop at the first 1% pass;
5. run a separate three-start diagnostic for the selected candidate.

The two sequences are independent and may be run in separate worker processes.
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

from scripts import continue_selected_equilibrium as continuation
from scripts import run_local_paper_equilibrium_experiment as local_search
from scripts import run_overnight_equilibrium_experiment as overnight
from scripts.search_nested_equilibrium import _deserialize_state


SEQUENCES: dict[str, dict[str, Any]] = {
    "ch-af-apac-eu-row-us": {
        "equilibrium_number": 1,
        "order": ["ch", "af", "apac", "eu", "row", "us"],
        "source_iteration": 24,
    },
    "ch-af-eu-us-row-apac": {
        "equilibrium_number": 2,
        "order": ["ch", "af", "eu", "us", "row", "apac"],
        "source_iteration": 15,
    },
}

SEARCH_BASE = ROOT / "outputs" / "equilibrium_search"
WORKFLOW = ROOT / "workflow"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


def configure_modules(sequence: str, specification: dict[str, Any]) -> tuple[Any, dict[str, dict], float]:
    order = list(specification["order"])
    source_iteration = int(specification["source_iteration"])
    source_workbook = ROOT / "outputs" / "sens" / "converged" / f"sens_{sequence}.xlsx"

    data0, base_cfg, excel_initial = continuation._initial_model_data()
    paper, replay_error = continuation.replay_accepted_state(
        source_workbook,
        excel_initial,
        data0,
        through_iteration=source_iteration,
        expected_player_order=order,
    )
    if replay_error > 1e-10:
        raise RuntimeError(f"{sequence}: manuscript-profile replay residual is {replay_error:.3g}")
    data = local_search._zero_prox_data(base_cfg)
    overnight._sync_quantity(data, paper)

    local_root = SEARCH_BASE / sequence / "local_paper_profile_zero_prox"
    local_checkpoint = local_root / "branch_F_alpha_0p50" / "sweep_006.json"
    local_audit = local_root / "branch_F_alpha_0p50" / "audit_sweep_006_one_start.json"

    continuation.PLAYER_ORDER = order
    continuation.SOURCE_WORKBOOK = source_workbook
    continuation.SOURCE_ITERATION = source_iteration
    local_search.PLAYER_ORDER = order
    local_search.SOURCE_WORKBOOK = source_workbook
    local_search.SOURCE_ITERATION = source_iteration
    local_search.OUTPUT_ROOT = local_root
    overnight.PLAYER_ORDER = order
    overnight.SOURCE_WORKBOOK = source_workbook
    overnight.SOURCE_ITERATION = source_iteration
    overnight.LOCAL_BEST_CHECKPOINT = local_checkpoint
    overnight.LOCAL_BEST_AUDIT = local_audit

    return data, paper, replay_error


def audit(
    data: Any,
    state: dict[str, dict],
    *,
    starts: int,
    maxiter: int,
    label: str,
    output: Path,
) -> dict[str, Any]:
    return local_search.audit_profile(
        data,
        state,
        starts=starts,
        maxiter=maxiter,
        label=label,
        output=output,
    )


def run_sequence(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    specification = dict(SEQUENCES[sequence])
    order = list(specification["order"])
    run_id = str(task["run_id"])
    sequence_root = SEARCH_BASE / sequence
    local_root = sequence_root / "local_paper_profile_zero_prox"
    run_root = sequence_root / "o6_analogue" / run_id
    status_path = run_root / "status.json"
    try:
        data, paper, replay_error = configure_modules(sequence, specification)
        run_root.mkdir(parents=True, exist_ok=True)
        source_workbook = ROOT / "outputs" / "sens" / "converged" / f"sens_{sequence}.xlsx"
        source_profile = local_root / "paper_profile.json"
        if not source_profile.exists():
            source_market, source_market_diag = overnight.solve_nested_market(data, paper)
            write_json(
                source_profile,
                {
                    "created": now(),
                    "sequence": sequence,
                    "equilibrium_number": specification["equilibrium_number"],
                    "source_workbook": relative(source_workbook),
                    "source_iteration": specification["source_iteration"],
                    "selection_rule": "first endpoint of three consecutive reported strategy-change residuals below 1%",
                    "player_order": order,
                    "replay_error": replay_error,
                    "profile": local_search.full_state_payload(data, paper, source_market),
                    "market_diagnostics": source_market_diag,
                },
            )

        source_audit_path = local_root / "audit_paper_profile_one_start.json"
        if source_audit_path.exists():
            source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        else:
            source_audit = audit(
                data,
                paper,
                starts=1,
                maxiter=int(task["maxiter"]),
                label=f"{sequence}_source_iteration_{specification['source_iteration']}_one_start",
                output=source_audit_path,
            )

        local_search.run_branch(
            data,
            paper,
            "F",
            alpha=0.50,
            max_sweeps=6,
            starts=1,
            maxiter=int(task["maxiter"]),
            order=order,
            run_name="branch_F_alpha_0p50",
        )
        local_checkpoint = local_root / "branch_F_alpha_0p50" / "sweep_006.json"
        local_payload = json.loads(local_checkpoint.read_text(encoding="utf-8"))
        local_state = _deserialize_state(local_payload["ending_profile"]["strategy"], data, paper)
        local_audit_path = local_root / "branch_F_alpha_0p50" / "audit_sweep_006_one_start.json"
        if local_audit_path.exists():
            local_audit = json.loads(local_audit_path.read_text(encoding="utf-8"))
        else:
            local_audit = audit(
                data,
                local_state,
                starts=1,
                maxiter=int(task["maxiter"]),
                label=f"{sequence}_branch_F_sweep_006_one_start",
                output=local_audit_path,
            )

        overnight._WORKER_CONTEXT = (data, paper, local_state)
        local_record = {
            **local_audit,
            "checkpoint": relative(local_checkpoint),
            "audit_path": relative(local_audit_path),
            "sequence": sequence,
            "stage": "branch_F",
            "sweep": 6,
        }
        selected_audit = local_record if local_audit["equilibrium_verified"] else None
        best_audit = local_record
        max_sweeps = int(task["max_sweeps"])
        audit_interval = int(task["audit_interval"])
        for sweep_limit in ([] if selected_audit is not None else range(audit_interval, max_sweeps + 1, audit_interval)):
            branch_result = overnight.run_branch_task(
                {
                    "branch": "O6",
                    "spec": {
                        "description": "sequence-specific Branch-F sweep-6 profile, zero proximal, adaptive damping",
                        "start": "local",
                        "mode": "adaptive",
                        "alpha": 0.30,
                        "alpha_min": 0.025,
                        "alpha_max": 0.50,
                        "max_sweeps": sweep_limit,
                    },
                    "run_root": str(run_root.resolve()),
                    "maxiter": int(task["maxiter"]),
                }
            )
            if branch_result.get("status") != "complete":
                raise RuntimeError(f"O6 branch failed: {branch_result}")
            checkpoint = run_root / "branches" / "O6" / f"sweep_{sweep_limit:03d}.json"
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            state = _deserialize_state(checkpoint_payload["ending_profile"]["strategy"], data, paper)
            audit_path = run_root / "audits" / "one_start" / f"audit_O6_s{sweep_limit:03d}.json"
            if audit_path.exists():
                current_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            else:
                current_audit = audit(
                    data,
                    state,
                    starts=1,
                    maxiter=int(task["maxiter"]),
                    label=f"{sequence}_O6_sweep_{sweep_limit:03d}_one_start",
                    output=audit_path,
                )
            current_record = {
                **current_audit,
                "checkpoint": relative(checkpoint),
                "audit_path": relative(audit_path),
                "sequence": sequence,
                "stage": "O6",
                "sweep": sweep_limit,
            }
            if best_audit is None or float(current_record["max_relative_gain"]) < float(best_audit["max_relative_gain"]):
                best_audit = current_record
            print(
                f"[{sequence}] O6 sweep {sweep_limit}: frozen one-start max gain "
                f"{100.0 * float(current_audit['max_relative_gain']):.4f}% "
                f"({current_audit['max_gain_player']}) pass={current_audit['equilibrium_verified']}",
                flush=True,
            )
            if current_record["equilibrium_verified"]:
                selected_audit = current_record
                break

        chosen = selected_audit or best_audit
        if chosen is None:
            raise RuntimeError("No O6 frozen-profile audit was produced")
        chosen_checkpoint = ROOT / chosen["checkpoint"]
        chosen_payload = json.loads(chosen_checkpoint.read_text(encoding="utf-8"))
        chosen_state = _deserialize_state(chosen_payload["ending_profile"]["strategy"], data, paper)
        existing_local_three_start = local_root / "branch_F_alpha_0p50" / "audit_sweep_006.json"
        three_start_path = (
            existing_local_three_start
            if chosen["stage"] == "branch_F" and existing_local_three_start.exists()
            else run_root / "audits" / "three_start" / f"audit_{chosen['stage']}_s{int(chosen['sweep']):03d}.json"
        )
        if three_start_path.exists():
            three_start = json.loads(three_start_path.read_text(encoding="utf-8"))
        else:
            three_start = audit(
                data,
                chosen_state,
                starts=3,
                maxiter=max(int(task["maxiter"]), 600),
                label=f"{sequence}_O6_sweep_{int(chosen['sweep']):03d}_three_start_diagnostic",
                output=three_start_path,
            )
        result = {
            "created": now(),
            "status": "accepted" if selected_audit is not None else "no_pass_within_limit",
            "sequence": sequence,
            "equilibrium_number": specification["equilibrium_number"],
            "player_order": order,
            "source_workbook": relative(source_workbook),
            "source_iteration": specification["source_iteration"],
            "source_replay_error": replay_error,
            "source_one_start_audit": relative(source_audit_path),
            "source_one_start_max_relative_gain": source_audit["max_relative_gain"],
            "branch_F_sweep_006": relative(local_checkpoint),
            "branch_F_one_start_audit": relative(local_audit_path),
            "branch_F_one_start_max_relative_gain": local_audit["max_relative_gain"],
            "selected_checkpoint": relative(chosen_checkpoint),
            "selected_stage": chosen["stage"],
            "selected_sweep": int(chosen["sweep"]),
            "one_start_audit": chosen["audit_path"],
            "one_start_max_relative_gain": chosen["max_relative_gain"],
            "one_start_max_gain_player": chosen["max_gain_player"],
            "one_start_equilibrium_verified": chosen["equilibrium_verified"],
            "three_start_diagnostic": relative(three_start_path),
            "three_start_max_relative_gain": three_start["max_relative_gain"],
            "three_start_max_gain_player": three_start["max_gain_player"],
            "three_start_equilibrium_verified": three_start["equilibrium_verified"],
            "criterion": "computational local 1%-equilibrium: one unregularized solve per player from the candidate strategy against one common frozen rival profile",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        failure = {
            "created": now(),
            "status": "failed",
            "sequence": sequence,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", nargs="+", choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--max-sweeps", type=int, default=300)
    parser.add_argument("--audit-interval", type=int, default=5)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.maxiter < 1 or args.max_sweeps < 1 or args.audit_interval < 1:
        raise ValueError("iteration and sweep arguments must be positive")
    if args.max_sweeps % args.audit_interval:
        raise ValueError("--max-sweeps must be divisible by --audit-interval")

    run_id = args.run_id or datetime.now().astimezone().strftime("other_sequences_%Y%m%d_%H%M%S")
    manifest_path = WORKFLOW / f"{run_id}_manifest.json"
    manifest = {
        "created": now(),
        "run_id": run_id,
        "status": "running",
        "method": "O6 analogue with sequence-specific source profile and update order",
        "acceptance_criterion": "one-start frozen-profile zero-proximal maximum relative gain <= 1%, with all six solves successful",
        "three_start_role": "separate stronger diagnostic; retained but not used for acceptance",
        "max_sweeps": args.max_sweeps,
        "audit_interval": args.audit_interval,
        "maxiter": args.maxiter,
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "results": [],
    }
    write_json(manifest_path, manifest)
    tasks = [
        {
            "sequence": sequence,
            "run_id": run_id,
            "maxiter": args.maxiter,
            "max_sweeps": args.max_sweeps,
            "audit_interval": args.audit_interval,
        }
        for sequence in args.sequences
    ]
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_sequence, task): task["sequence"] for task in tasks}
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
            manifest["results"] = results
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = "complete" if all(r["status"] in {"accepted", "no_pass_within_limit"} for r in results) else "complete_with_failures"
    manifest["results"] = sorted(results, key=lambda row: row["sequence"])
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
