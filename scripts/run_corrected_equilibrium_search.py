from __future__ import annotations

"""Run the established zero-proximal equilibrium search on corrected cold starts.

For each manuscript player order this runner:

1. replays the terminal accepted state from the corrected paper-algorithm run;
2. records a one-start, common frozen-profile zero-proximal audit;
3. follows the established Branch-F path (prices halfway to manufacturing cost,
   six zero-proximal Gauss--Seidel sweeps at alpha 0.50);
4. if needed, continues with the adaptive O6 schedule from alpha 0.30 and
   audits every fifth sweep; and
5. retains a separate three-start diagnostic for the selected candidate.

The one-start frozen-profile maximum relative gain <= 1%, with all six solves
successful, is the acceptance rule used for the prior packaged equilibria.  A
three-start audit is diagnostic and does not broaden that local claim.
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

from model import run_gs
from model.data_prep import load_data_from_excel
from scripts import continue_selected_equilibrium as continuation
from scripts import run_local_paper_equilibrium_experiment as local_search
from scripts import run_overnight_equilibrium_experiment as overnight
from scripts.run_corrected_cold_start import _build_fresh_state
from scripts.search_nested_equilibrium import _deserialize_state


PARAMS_SHEET = "params_region_new"
SEQUENCES: dict[str, dict[str, Any]] = {
    "ch-af-apac-eu-row-us": {
        "equilibrium_number": 1,
        "order": ["ch", "af", "apac", "eu", "row", "us"],
        "source_iteration": 23,
    },
    "ch-af-eu-us-row-apac": {
        "equilibrium_number": 2,
        "order": ["ch", "af", "eu", "us", "row", "apac"],
        "source_iteration": 20,
    },
    "ch-row-apac-us-eu-af": {
        "equilibrium_number": 3,
        "order": ["ch", "row", "apac", "us", "eu", "af"],
        "source_iteration": 24,
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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


def source_workbook(cold_start_root: Path, sequence: str) -> Path:
    matches = sorted((cold_start_root / sequence / "results").glob("results_*.xlsx"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one corrected result workbook for {sequence}; found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def base_configuration(input_path: Path) -> run_gs.RunConfig:
    return run_gs.RunConfig(
        excel_path=str(input_path.resolve()),
        params_region_sheet=PARAMS_SHEET,
        discount_rate=0.02,
        base_year=2025,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        fix_q_offer_to_kcap=True,
        fix_a_bid_to_true_dem=True,
        force_mu_offer_zero=False,
    )


def configure_modules(task: dict[str, Any]) -> tuple[Any, dict[str, dict], float, Path, Path]:
    sequence = str(task["sequence"])
    specification = SEQUENCES[sequence]
    order = list(specification["order"])
    iteration = int(specification["source_iteration"])
    input_path = Path(task["input_path"]).resolve()
    cold_start_root = Path(task["cold_start_root"]).resolve()
    output_root = Path(task["output_root"]).resolve()
    workbook = source_workbook(cold_start_root, sequence)

    cfg = base_configuration(input_path)
    replay_data = load_data_from_excel(str(input_path), params_region_sheet=PARAMS_SHEET)
    run_gs._apply_data_overrides(replay_data, cfg)
    fresh_initial = _build_fresh_state(replay_data)
    candidate, replay_error = continuation.replay_accepted_state(
        workbook,
        fresh_initial,
        replay_data,
        through_iteration=iteration,
        expected_player_order=order,
    )
    if replay_error > 1e-10:
        raise RuntimeError(f"{sequence}: corrected endpoint replay residual is {replay_error:.3g}")

    data = local_search._zero_prox_data(cfg)
    overnight._sync_quantity(data, candidate)
    sequence_root = output_root / sequence
    local_root = sequence_root / "local_candidate_zero_prox"
    local_checkpoint = local_root / "branch_F_alpha_0p50" / "sweep_006.json"
    local_audit = local_root / "branch_F_alpha_0p50" / "audit_sweep_006_one_start.json"

    # These modules implement the previously used and reviewed search mechanics.
    # Each sequence runs in its own process, so the per-process globals are isolated.
    continuation.PLAYER_ORDER = order
    continuation.SOURCE_WORKBOOK = workbook
    continuation.SOURCE_ITERATION = iteration
    local_search.PLAYER_ORDER = order
    local_search.SOURCE_WORKBOOK = workbook
    local_search.SOURCE_ITERATION = iteration
    local_search.OUTPUT_ROOT = local_root
    overnight.PLAYER_ORDER = order
    overnight.SOURCE_WORKBOOK = workbook
    overnight.SOURCE_ITERATION = iteration
    overnight.LOCAL_BEST_CHECKPOINT = local_checkpoint
    overnight.LOCAL_BEST_AUDIT = local_audit
    overnight._WORKER_CONTEXT = None
    return data, candidate, replay_error, workbook, local_root


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


def candidate_record(payload: dict[str, Any], checkpoint: Path, audit_path: Path, stage: str, sweep: int) -> dict[str, Any]:
    return {
        **payload,
        "checkpoint": relative(checkpoint),
        "audit_path": relative(audit_path),
        "stage": stage,
        "sweep": sweep,
    }


def run_sequence(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    specification = SEQUENCES[sequence]
    order = list(specification["order"])
    sequence_root = Path(task["output_root"]).resolve() / sequence
    local_root = sequence_root / "local_candidate_zero_prox"
    o6_root = sequence_root / "o6_analogue"
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
        data, source, replay_error, workbook, local_root = configure_modules(task)
        input_path = Path(task["input_path"]).resolve()
        write_json(
            status_path,
            {
                "created": now(),
                "status": "running",
                "stage": "source_frozen_audit",
                "pid": os.getpid(),
                "sequence": sequence,
                "player_order": order,
                "input": relative(input_path),
                "input_sha256": sha256(input_path),
                "source_workbook": relative(workbook),
                "source_workbook_sha256": sha256(workbook),
                "source_iteration": specification["source_iteration"],
                "source_replay_error": replay_error,
            },
        )

        source_profile_path = local_root / "source_profile.json"
        if not source_profile_path.exists():
            source_market, source_market_diag = overnight.solve_nested_market(data, source)
            write_json(
                source_profile_path,
                {
                    "created": now(),
                    "sequence": sequence,
                    "equilibrium_number": specification["equilibrium_number"],
                    "source_workbook": relative(workbook),
                    "source_workbook_sha256": sha256(workbook),
                    "source_iteration": specification["source_iteration"],
                    "selection_rule": "terminal endpoint satisfying three consecutive paper strategy-movement residuals below 1%",
                    "player_order": order,
                    "replay_error": replay_error,
                    "profile": local_search.full_state_payload(data, source, source_market),
                    "market_diagnostics": source_market_diag,
                },
            )

        source_audit_path = local_root / "audit_source_profile_one_start.json"
        source_audit = (
            json.loads(source_audit_path.read_text(encoding="utf-8"))
            if source_audit_path.exists()
            else audit(
                data,
                source,
                starts=1,
                maxiter=int(task["maxiter"]),
                label=f"corrected_{sequence}_source_s{specification['source_iteration']}_one_start",
                output=source_audit_path,
            )
        )

        write_json(
            status_path,
            {
                "updated": now(),
                "status": "running",
                "stage": "branch_F_six_sweeps",
                "pid": os.getpid(),
                "sequence": sequence,
                "source_one_start_max_relative_gain": source_audit["max_relative_gain"],
                "source_one_start_equilibrium_verified": source_audit["equilibrium_verified"],
            },
        )
        local_search.run_branch(
            data,
            source,
            "F",
            alpha=0.50,
            max_sweeps=6,
            starts=1,
            maxiter=int(task["maxiter"]),
            order=order,
            run_name="branch_F_alpha_0p50",
        )
        local_checkpoint = local_root / "branch_F_alpha_0p50" / "sweep_006.json"
        if not local_checkpoint.exists():
            raise RuntimeError("Branch F terminated before producing sweep_006.json")
        local_payload = json.loads(local_checkpoint.read_text(encoding="utf-8"))
        local_state = _deserialize_state(local_payload["ending_profile"]["strategy"], data, source)
        local_audit_path = local_root / "branch_F_alpha_0p50" / "audit_sweep_006_one_start.json"
        local_audit = (
            json.loads(local_audit_path.read_text(encoding="utf-8"))
            if local_audit_path.exists()
            else audit(
                data,
                local_state,
                starts=1,
                maxiter=int(task["maxiter"]),
                label=f"corrected_{sequence}_branch_F_s006_one_start",
                output=local_audit_path,
            )
        )

        overnight._WORKER_CONTEXT = (data, source, local_state)
        best = candidate_record(local_audit, local_checkpoint, local_audit_path, "branch_F", 6)
        selected = best if bool(local_audit["equilibrium_verified"]) else None
        max_sweeps = int(task["max_sweeps"])
        audit_interval = int(task["audit_interval"])
        for sweep_limit in ([] if selected is not None else range(audit_interval, max_sweeps + 1, audit_interval)):
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "stage": "adaptive_O6",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "target_sweep": sweep_limit,
                    "best_frozen_max_relative_gain": best["max_relative_gain"],
                    "best_frozen_max_gain_player": best["max_gain_player"],
                },
            )
            branch_result = overnight.run_branch_task(
                {
                    "branch": "O6",
                    "spec": {
                        "description": "corrected Branch-F sweep-6 profile, zero proximal, adaptive damping",
                        "start": "local",
                        "mode": "adaptive",
                        "alpha": 0.30,
                        "alpha_min": 0.025,
                        "alpha_max": 0.50,
                        "max_sweeps": sweep_limit,
                    },
                    "run_root": str(o6_root),
                    "maxiter": int(task["maxiter"]),
                }
            )
            if branch_result.get("status") != "complete":
                raise RuntimeError(f"O6 branch failed: {branch_result}")
            checkpoint = o6_root / "branches" / "O6" / f"sweep_{sweep_limit:03d}.json"
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            state = _deserialize_state(checkpoint_payload["ending_profile"]["strategy"], data, source)
            audit_path = o6_root / "audits" / "one_start" / f"audit_O6_s{sweep_limit:03d}.json"
            current_audit = (
                json.loads(audit_path.read_text(encoding="utf-8"))
                if audit_path.exists()
                else audit(
                    data,
                    state,
                    starts=1,
                    maxiter=int(task["maxiter"]),
                    label=f"corrected_{sequence}_O6_s{sweep_limit:03d}_one_start",
                    output=audit_path,
                )
            )
            current = candidate_record(current_audit, checkpoint, audit_path, "O6", sweep_limit)
            if float(current["max_relative_gain"]) < float(best["max_relative_gain"]):
                best = current
            print(
                f"[{sequence}] O6 sweep {sweep_limit}: frozen one-start max gain "
                f"{100.0 * float(current_audit['max_relative_gain']):.4f}% "
                f"({current_audit['max_gain_player']}) pass={current_audit['equilibrium_verified']}",
                flush=True,
            )
            if bool(current["equilibrium_verified"]):
                selected = current
                break

        chosen = selected or best
        chosen_checkpoint = ROOT / str(chosen["checkpoint"])
        chosen_payload = json.loads(chosen_checkpoint.read_text(encoding="utf-8"))
        chosen_state = _deserialize_state(chosen_payload["ending_profile"]["strategy"], data, source)
        three_start_path = (
            local_root / "branch_F_alpha_0p50" / "audit_sweep_006_three_start.json"
            if chosen["stage"] == "branch_F"
            else o6_root / "audits" / "three_start" / f"audit_O6_s{int(chosen['sweep']):03d}.json"
        )
        three_start = (
            json.loads(three_start_path.read_text(encoding="utf-8"))
            if three_start_path.exists()
            else audit(
                data,
                chosen_state,
                starts=3,
                maxiter=max(int(task["maxiter"]), 600),
                label=f"corrected_{sequence}_{chosen['stage']}_s{int(chosen['sweep']):03d}_three_start",
                output=three_start_path,
            )
        )
        result = {
            "created": now(),
            "status": "accepted" if selected is not None else "no_pass_within_limit",
            "pid": os.getpid(),
            "sequence": sequence,
            "equilibrium_number": specification["equilibrium_number"],
            "player_order": order,
            "input": relative(Path(task["input_path"])),
            "input_sha256": sha256(Path(task["input_path"])),
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": specification["source_iteration"],
            "source_replay_error": replay_error,
            "source_one_start_audit": relative(source_audit_path),
            "source_one_start_max_relative_gain": source_audit["max_relative_gain"],
            "branch_F_sweep_006": relative(local_checkpoint),
            "branch_F_one_start_audit": relative(local_audit_path),
            "branch_F_one_start_max_relative_gain": local_audit["max_relative_gain"],
            "selected_checkpoint": chosen["checkpoint"],
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
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def validate_task(task: dict[str, Any]) -> dict[str, Any]:
    data, candidate, replay_error, workbook, _ = configure_modules(task)
    market, diagnostics = overnight.solve_nested_market(data, candidate)
    return {
        "sequence": task["sequence"],
        "source_workbook": relative(workbook),
        "source_workbook_sha256": sha256(workbook),
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
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--max-sweeps", type=int, default=300)
    parser.add_argument("--audit-interval", type=int, default=5)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.maxiter < 1 or args.max_sweeps < 1 or args.audit_interval < 1:
        raise ValueError("iteration and sweep arguments must be positive")
    if args.max_sweeps % args.audit_interval:
        raise ValueError("--max-sweeps must be divisible by --audit-interval")
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
            "audit_interval": args.audit_interval,
        }
        for sequence in args.sequences
    ]
    if args.validate_only:
        print(json.dumps([validate_task(task) for task in tasks], indent=2), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "corrected endpoint -> Branch F alpha 0.50 for six sweeps -> adaptive O6",
        "acceptance_criterion": "one-start common frozen-profile zero-proximal maximum relative gain <= 1%, all six solves successful",
        "three_start_role": "stronger diagnostic retained separately; not the acceptance rule",
        "claim_scope": "local computational criterion, not a global Nash-equilibrium certificate",
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "output_root": relative(output_root),
        "max_sweeps": args.max_sweeps,
        "audit_interval": args.audit_interval,
        "maxiter": args.maxiter,
        "workers": min(args.workers, len(tasks)),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "results": [],
    }
    write_json(manifest_path, manifest)
    results: list[dict[str, Any]] = []
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
            manifest["results"] = sorted(results, key=lambda row: row["sequence"])
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(result["status"] in {"accepted", "no_pass_within_limit"} for result in results)
        else "complete_with_failures"
    )
    manifest["results"] = sorted(results, key=lambda row: row["sequence"])
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
