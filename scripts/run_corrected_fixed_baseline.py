from __future__ import annotations

"""Thirty-sweep fixed-alpha baseline from corrected Stage-1 profiles.

Each player order starts from its exact corrected Stage-1 terminal profile and
runs 30 sequential zero-algorithmic-proximal Gauss--Seidel sweeps at fixed
alpha=0.50. Every fifth checkpoint receives a one-start common frozen-profile
audit. The trajectory always completes all 30 sweeps so it remains a clean
baseline; the best checkpoint then receives a separate three-start diagnostic.
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

from scripts import run_overnight_equilibrium_experiment as overnight
from scripts.run_corrected_equilibrium_search import SEQUENCES, configure_modules, relative, sha256, write_json
from scripts.run_local_paper_equilibrium_experiment import audit_profile
from scripts.search_nested_equilibrium import _deserialize_state


ALPHA = 0.50
SWEEPS = 30
AUDIT_INTERVAL = 5


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_sequence(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    order = list(SEQUENCES[sequence]["order"])
    sequence_root = Path(task["output_root"]).resolve() / sequence
    branch_root = sequence_root / "branches" / "fixed_alpha_0p50"
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
            },
        )
        data, source, replay_error, workbook, _ = configure_modules(task)
        overnight._WORKER_CONTEXT = (data, source, source)
        best: dict[str, Any] | None = None
        audit_records: list[dict[str, Any]] = []
        for target in range(AUDIT_INTERVAL, SWEEPS + 1, AUDIT_INTERVAL):
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "target_sweep": target,
                    "last_audited_sweep": None if not audit_records else audit_records[-1]["sweep"],
                    "best_frozen_max_relative_gain": None if best is None else best["max_relative_gain"],
                },
            )
            result = overnight.run_branch_task(
                {
                    "branch": "fixed_alpha_0p50",
                    "spec": {
                        "description": "corrected Stage-1 endpoint, zero proximal, fixed alpha 0.50 baseline",
                        "start": "paper",
                        "mode": "fixed",
                        "alpha": ALPHA,
                        "max_sweeps": target,
                    },
                    "run_root": str(sequence_root),
                    "maxiter": int(task["maxiter"]),
                }
            )
            if result.get("status") != "complete":
                raise RuntimeError(f"Fixed baseline failed: {result}")
            checkpoint = branch_root / f"sweep_{target:03d}.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            state = _deserialize_state(payload["ending_profile"]["strategy"], data, source)
            audit_path = sequence_root / "audits" / f"audit_fixed_a050_s{target:03d}_one_start.json"
            frozen = (
                json.loads(audit_path.read_text(encoding="utf-8"))
                if audit_path.exists()
                else audit_profile(
                    data,
                    state,
                    starts=1,
                    maxiter=int(task["maxiter"]),
                    label=f"corrected_{sequence}_fixed_a050_s{target:03d}_one_start",
                    output=audit_path,
                )
            )
            record = {
                "sweep": target,
                "checkpoint": relative(checkpoint),
                "audit": relative(audit_path),
                "max_relative_gain": float(frozen["max_relative_gain"]),
                "max_gain_player": str(frozen["max_gain_player"]),
                "equilibrium_verified": bool(frozen["equilibrium_verified"]),
                "all_attempts_successful": bool(frozen["all_attempts_successful"]),
            }
            audit_records.append(record)
            if best is None or record["max_relative_gain"] < best["max_relative_gain"]:
                best = record
            print(
                f"[{sequence}] fixed alpha 0.50 sweep {target}: frozen max gain "
                f"{100.0 * record['max_relative_gain']:.4f}% "
                f"({record['max_gain_player']}) pass={record['equilibrium_verified']}",
                flush=True,
            )
        if best is None:
            raise RuntimeError("No baseline audit was produced")
        best_checkpoint = ROOT / best["checkpoint"]
        best_payload = json.loads(best_checkpoint.read_text(encoding="utf-8"))
        best_state = _deserialize_state(best_payload["ending_profile"]["strategy"], data, source)
        diagnostic_path = sequence_root / "audits" / f"audit_fixed_a050_s{int(best['sweep']):03d}_three_start.json"
        diagnostic = (
            json.loads(diagnostic_path.read_text(encoding="utf-8"))
            if diagnostic_path.exists()
            else audit_profile(
                data,
                best_state,
                starts=3,
                maxiter=max(int(task["maxiter"]), 600),
                label=f"corrected_{sequence}_fixed_a050_s{int(best['sweep']):03d}_three_start",
                output=diagnostic_path,
            )
        )
        output = {
            "created": now(),
            "status": "complete",
            "pid": os.getpid(),
            "sequence": sequence,
            "player_order": order,
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": SEQUENCES[sequence]["source_iteration"],
            "source_replay_error": replay_error,
            "algorithmic_proximal_penalties": 0.0,
            "alpha": ALPHA,
            "sweeps_completed": SWEEPS,
            "audit_interval": AUDIT_INTERVAL,
            "audits": audit_records,
            "best_checkpoint": best["checkpoint"],
            "best_sweep": best["sweep"],
            "best_one_start_max_relative_gain": best["max_relative_gain"],
            "best_one_start_max_gain_player": best["max_gain_player"],
            "best_one_start_equilibrium_verified": best["equilibrium_verified"],
            "three_start_diagnostic": relative(diagnostic_path),
            "three_start_max_relative_gain": diagnostic["max_relative_gain"],
            "three_start_max_gain_player": diagnostic["max_gain_player"],
            "three_start_equilibrium_verified": diagnostic["equilibrium_verified"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, output)
        return output
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", choices=list(SEQUENCES), default=list(SEQUENCES))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=500)
    args = parser.parse_args()
    if args.workers < 1 or args.maxiter < 1:
        raise ValueError("workers and maxiter must be positive")
    input_path = args.input.resolve()
    cold_start_root = args.cold_start_root.resolve()
    output_root = args.output_root.resolve()
    tasks = [
        {
            "sequence": sequence,
            "input_path": str(input_path),
            "cold_start_root": str(cold_start_root),
            "output_root": str(output_root),
            "maxiter": args.maxiter,
        }
        for sequence in args.sequences
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "30-sweep zero-proximal Gauss--Seidel baseline from corrected Stage-1 endpoints",
        "alpha": ALPHA,
        "adaptive_damping": False,
        "sweeps": SWEEPS,
        "audit_interval": AUDIT_INTERVAL,
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "output_root": relative(output_root),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "workers": min(args.workers, len(tasks)),
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
    manifest["status"] = "complete" if all(row["status"] == "complete" for row in results) else "complete_with_failures"
    manifest["results"] = sorted(results, key=lambda row: row["sequence"])
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
