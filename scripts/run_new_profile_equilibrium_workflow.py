from __future__ import annotations

"""Audit continued profiles and, when needed, run the established basin grid."""

import argparse
import csv
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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import run_gs
from model.data_prep import load_data_from_excel
from scripts import continue_selected_equilibrium as continuation
from scripts import run_corrected_a030_search as base_search
from scripts import run_corrected_paper_profile_factorial as factorial
from scripts import run_local_paper_equilibrium_experiment as local_search
from scripts import run_overnight_equilibrium_experiment as overnight
from scripts.nested_market_audit import solve_nested_market
from scripts.run_corrected_cold_start import _build_fresh_state, _state_sha256
from scripts.run_corrected_equilibrium_search import (
    base_configuration,
    relative,
    sha256,
    write_json,
)


PARAMS_SHEET = "params_region_new"
SOURCE_ITERATION = 30
ABSOLUTE_FINAL_SWEEP = 40
TERMINAL_SALVAGE_FRACTION = 0.5
THRESHOLD = 0.01

PROFILES: dict[str, list[str]] = {
    "af-eu-us-apac-row-ch": ["af", "eu", "us", "apac", "row", "ch"],
    "eu-us-af-row-apac-ch": ["eu", "us", "af", "row", "apac", "ch"],
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def result_workbook(root: Path, sequence: str) -> Path:
    matches = sorted((root / sequence / "results").glob("results_*.xlsx"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one workbook for {sequence} below {root}; found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def continuation_iteration(workbook: Path) -> int:
    rows = pd.read_excel(
        workbook,
        sheet_name="iters",
        usecols=["iter", "all_solves_acceptable"],
    ).sort_values("iter")
    if rows.empty:
        raise RuntimeError(f"No continuation sweeps in {workbook}")
    if not rows["all_solves_acceptable"].fillna(False).astype(bool).all():
        failed = rows.loc[
            ~rows["all_solves_acceptable"].fillna(False).astype(bool), "iter"
        ].astype(int).tolist()
        raise RuntimeError(f"Continuation profile contains failed solves: {failed}")
    return int(rows["iter"].max())


def reconstruct_profile(task: dict[str, Any]) -> dict[str, Any]:
    sequence = str(task["sequence"])
    order = list(task["order"])
    input_path = Path(task["input_path"]).resolve()
    source_root = Path(task["source_root"]).resolve()
    continuation_root = Path(task["continuation_root"]).resolve()
    source_book = result_workbook(source_root, sequence)
    continued_book = result_workbook(continuation_root, sequence)
    continued_iteration = continuation_iteration(continued_book)

    cfg = base_configuration(input_path)
    replay_data = load_data_from_excel(
        str(input_path), params_region_sheet=PARAMS_SHEET
    )
    run_gs._apply_data_overrides(replay_data, cfg)
    fresh = _build_fresh_state(replay_data)
    source, source_error = continuation.replay_accepted_state(
        source_book,
        fresh,
        replay_data,
        through_iteration=SOURCE_ITERATION,
        expected_player_order=order,
    )
    candidate, continuation_error = continuation.replay_accepted_state(
        continued_book,
        source,
        replay_data,
        through_iteration=continued_iteration,
        expected_player_order=order,
    )
    replay_error = max(float(source_error), float(continuation_error))
    if replay_error > 1e-10:
        raise RuntimeError(f"{sequence}: replay residual is {replay_error:.3g}")
    return {
        "configuration": cfg,
        "candidate": candidate,
        "source_workbook": source_book,
        "continuation_workbook": continued_book,
        "continuation_iteration": continued_iteration,
        "absolute_sweep": SOURCE_ITERATION + continued_iteration,
        "source_replay_error": float(source_error),
        "continuation_replay_error": float(continuation_error),
        "replay_error": replay_error,
        "strategy_sha256": _state_sha256(candidate),
    }


def configure_continued_modules(
    task: dict[str, Any],
) -> tuple[Any, dict[str, dict], float, Path, Path]:
    prepared = reconstruct_profile(task)
    sequence = str(task["sequence"])
    order = list(task["order"])
    cfg = prepared["configuration"]
    candidate = prepared["candidate"]
    continued_book = Path(prepared["continuation_workbook"])

    data = local_search._zero_prox_data(
        cfg,
        terminal_salvage_fraction=float(
            task.get("terminal_salvage_fraction", TERMINAL_SALVAGE_FRACTION)
        ),
    )
    overnight._sync_quantity(data, candidate)

    base_search.SEQUENCES[sequence] = {
        "equilibrium_number": None,
        "order": order,
        "source_iteration": int(prepared["absolute_sweep"]),
    }
    continuation.PLAYER_ORDER = order
    continuation.SOURCE_WORKBOOK = continued_book
    continuation.SOURCE_ITERATION = int(prepared["continuation_iteration"])
    local_search.PLAYER_ORDER = order
    local_search.SOURCE_WORKBOOK = continued_book
    local_search.SOURCE_ITERATION = int(prepared["absolute_sweep"])
    overnight.PLAYER_ORDER = order
    overnight.SOURCE_WORKBOOK = continued_book
    overnight.SOURCE_ITERATION = int(prepared["absolute_sweep"])
    overnight._WORKER_CONTEXT = None
    local_root = Path(task["output_root"]).resolve() / sequence
    return data, candidate, float(prepared["replay_error"]), continued_book, local_root


def direct_audit(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    output_root = Path(task["output_root"]).resolve()
    audit_root = output_root / "direct_audits" / sequence
    status_path = audit_root / "status.json"
    audit_path = audit_root / "audit_one_start.json"
    try:
        audit_root.mkdir(parents=True, exist_ok=False)
        prepared = reconstruct_profile(task)
        cfg = prepared["configuration"]
        candidate = prepared["candidate"]
        data = local_search._zero_prox_data(
            cfg, terminal_salvage_fraction=TERMINAL_SALVAGE_FRACTION
        )
        overnight._sync_quantity(data, candidate)
        local_search.PLAYER_ORDER = list(task["order"])
        audit = local_search.audit_profile(
            data,
            candidate,
            starts=1,
            maxiter=int(task["maxiter"]),
            label=f"continued_{sequence}_absolute_sweep_{prepared['absolute_sweep']}_one_start",
            output=audit_path,
        )
        result = {
            "created": now(),
            "status": "completed",
            "sequence": sequence,
            "player_order": list(task["order"]),
            "source_workbook": relative(Path(prepared["source_workbook"])),
            "source_workbook_sha256": sha256(Path(prepared["source_workbook"])),
            "continuation_workbook": relative(
                Path(prepared["continuation_workbook"])
            ),
            "continuation_workbook_sha256": sha256(
                Path(prepared["continuation_workbook"])
            ),
            "source_iteration": SOURCE_ITERATION,
            "continuation_iteration": int(prepared["continuation_iteration"]),
            "absolute_sweep": int(prepared["absolute_sweep"]),
            "source_replay_error": prepared["source_replay_error"],
            "continuation_replay_error": prepared["continuation_replay_error"],
            "strategy_sha256": prepared["strategy_sha256"],
            "terminal_salvage_fraction": TERMINAL_SALVAGE_FRACTION,
            "algorithmic_proximal_penalties": 0.0,
            "audit_starts": 1,
            "audit_path": relative(audit_path),
            "all_six_solves_successful": bool(audit["all_attempts_successful"]),
            "max_relative_gain": float(audit["max_relative_gain"]),
            "max_gain_player": str(audit["max_gain_player"]),
            "relative_gain_tolerance": THRESHOLD,
            "local_one_percent_equilibrium": bool(audit["equilibrium_verified"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        result = {
            "created": now(),
            "status": "failed",
            "sequence": sequence,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result


def prepare_factorial_worker(task: dict[str, Any]) -> None:
    sequence = str(task["sequence"])
    base_search.SEQUENCES[sequence] = {
        "equilibrium_number": None,
        "order": list(task["order"]),
        "source_iteration": ABSOLUTE_FINAL_SWEEP,
    }
    base_search.BRANCHES[str(task["branch"])] = dict(
        task["branch_specification"]
    )
    base_search.make_initial_state = factorial.make_factorial_initial_state
    base_search.configure_modules = configure_continued_modules


def run_factorial_branch(task: dict[str, Any]) -> dict[str, Any]:
    prepare_factorial_worker(task)
    return base_search.run_branch(task)


def validate_profile(task: dict[str, Any]) -> dict[str, Any]:
    prepared = reconstruct_profile(task)
    data = local_search._zero_prox_data(
        prepared["configuration"],
        terminal_salvage_fraction=TERMINAL_SALVAGE_FRACTION,
    )
    candidate = prepared["candidate"]
    overnight._sync_quantity(data, candidate)
    market, diagnostics = solve_nested_market(data, candidate)
    return {
        "sequence": task["sequence"],
        "source_workbook": relative(Path(prepared["source_workbook"])),
        "continuation_workbook": relative(
            Path(prepared["continuation_workbook"])
        ),
        "absolute_sweep": prepared["absolute_sweep"],
        "replay_error": prepared["replay_error"],
        "strategy_sha256": prepared["strategy_sha256"],
        "market_solve_successful": market is not None,
        "market_diagnostics": diagnostics,
    }


def protocol_text(args: argparse.Namespace, output_root: Path) -> str:
    price_factors = ", ".join(f"{value:g}" for value in args.price_factors)
    return f"""# Continued-profile equilibrium workflow

Created {now()}.

## Direct check

- Profiles: `{list(PROFILES)}`
- Source checkpoint: sweep {SOURCE_ITERATION}
- Continued checkpoint: absolute sweep {ABSOLUTE_FINAL_SWEEP}
- Common frozen profile during every audit
- One candidate-initialized best-response solve per player
- Zero algorithmic proximal penalties in the audit
- Acceptance: all six solves successful and maximum normalized unilateral gain <= 1%

## Conditional basin search

The established reinitialization grid is run only for profiles that fail the
direct check. Off-diagonal offer prices start at one of `[{price_factors}]` times the
exporter's manufacturing cost; the continued net-capacity-change path receives
weight 0.5 or 1.0; and sequential Gauss--Seidel damping is fixed at 0.3 or 0.4.
All updates and audits use zero proximal penalties. Each branch is audited at
initialization and after every sweep, stops at its first accepted profile, and
runs for at most {args.max_sweeps} sweeps.

- Corrected input: `{relative(args.input.resolve())}`
- Original sweep-30 root: `{relative(args.source_root.resolve())}`
- Penalized continuation root: `{relative(args.continuation_root.resolve())}`
- Output root: `{relative(output_root)}`
- Direct/grid best-response maximum iterations: {args.maxiter}
- Parallel workers: {args.workers}
- Terminal salvage fraction: {TERMINAL_SALVAGE_FRACTION}
"""


def write_summary(output_root: Path, direct: list[dict[str, Any]], grid: list[dict[str, Any]]) -> None:
    fields = [
        "record_type",
        "sequence",
        "branch",
        "status",
        "selected_sweep",
        "max_relative_gain",
        "max_gain_player",
        "all_six_solves_successful",
        "local_one_percent_equilibrium",
        "profile",
        "audit",
    ]
    rows: list[dict[str, Any]] = []
    for item in direct:
        rows.append(
            {
                "record_type": "direct_audit",
                "sequence": item.get("sequence"),
                "branch": "",
                "status": item.get("status"),
                "selected_sweep": item.get("absolute_sweep"),
                "max_relative_gain": item.get("max_relative_gain"),
                "max_gain_player": item.get("max_gain_player"),
                "all_six_solves_successful": item.get(
                    "all_six_solves_successful"
                ),
                "local_one_percent_equilibrium": item.get(
                    "local_one_percent_equilibrium"
                ),
                "profile": item.get("continuation_workbook"),
                "audit": item.get("audit_path"),
            }
        )
    for item in grid:
        rows.append(
            {
                "record_type": "basin_grid",
                "sequence": item.get("sequence"),
                "branch": item.get("branch"),
                "status": item.get("status"),
                "selected_sweep": item.get("selected_sweep"),
                "max_relative_gain": item.get("one_start_max_relative_gain"),
                "max_gain_player": item.get("one_start_max_gain_player"),
                "all_six_solves_successful": item.get(
                    "all_six_solves_successful"
                ),
                "local_one_percent_equilibrium": item.get(
                    "local_one_percent_equilibrium"
                ),
                "profile": item.get("selected_profile"),
                "audit": item.get("one_start_audit"),
            }
        )
    with (output_root / "results_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--max-sweeps", type=int, default=15)
    parser.add_argument(
        "--price-factors", nargs="+", type=float, default=[1.0, 1.2]
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.maxiter < 1 or args.max_sweeps < 1:
        raise ValueError("workers, maxiter, and max-sweeps must be positive")
    if len(set(args.price_factors)) != len(args.price_factors):
        raise ValueError("price factors must not contain duplicates")
    if any(value <= 0.0 for value in args.price_factors):
        raise ValueError("price factors must be positive")
    for path in (args.input, args.source_root, args.continuation_root):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())

    output_root = args.output_root.resolve()
    base_tasks = [
        {
            "sequence": sequence,
            "order": order,
            "input_path": str(args.input.resolve()),
            "source_root": str(args.source_root.resolve()),
            "continuation_root": str(args.continuation_root.resolve()),
            "output_root": str(output_root),
            "maxiter": args.maxiter,
        }
        for sequence, order in PROFILES.items()
    ]
    if args.validate_only:
        print(
            json.dumps([validate_profile(task) for task in base_tasks], indent=2),
            flush=True,
        )
        return

    output_root.mkdir(parents=True, exist_ok=False)
    protocol_path = output_root / "PROTOCOL.md"
    protocol_path.write_text(protocol_text(args, output_root), encoding="utf-8")
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running_direct_audits",
        "pid": os.getpid(),
        "input": relative(args.input.resolve()),
        "input_sha256": sha256(args.input.resolve()),
        "source_root": relative(args.source_root.resolve()),
        "continuation_root": relative(args.continuation_root.resolve()),
        "output_root": relative(output_root),
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "profiles": PROFILES,
        "direct_audit_starts": 1,
        "relative_gain_tolerance": THRESHOLD,
        "terminal_salvage_fraction": TERMINAL_SALVAGE_FRACTION,
        "grid": {
            "conditional_on_direct_failure": True,
            "price_factors": list(args.price_factors),
            "capacity_weights": [0.5, 1.0],
            "alphas": [0.3, 0.4],
            "max_sweeps": args.max_sweeps,
            "maxiter": args.maxiter,
            "algorithmic_proximal_penalties": 0.0,
        },
        "code_sha256": {
            "scripts/run_new_profile_equilibrium_workflow.py": sha256(Path(__file__)),
            "scripts/run_corrected_paper_profile_factorial.py": sha256(
                ROOT / "scripts" / "run_corrected_paper_profile_factorial.py"
            ),
            "scripts/run_corrected_a030_search.py": sha256(
                ROOT / "scripts" / "run_corrected_a030_search.py"
            ),
            "scripts/run_local_paper_equilibrium_experiment.py": sha256(
                ROOT / "scripts" / "run_local_paper_equilibrium_experiment.py"
            ),
        },
        "direct_audits": [],
        "grid_results": [],
    }
    write_json(manifest_path, manifest)

    direct_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(base_tasks))) as pool:
        futures = {pool.submit(direct_audit, task): task["sequence"] for task in base_tasks}
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
            direct_results.append(result)
            manifest["direct_audits"] = sorted(
                direct_results, key=lambda row: row["sequence"]
            )
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)

    if not all(result.get("status") == "completed" for result in direct_results):
        manifest["status"] = "direct_audit_failure"
        manifest["updated"] = now()
        write_json(manifest_path, manifest)
        write_summary(output_root, direct_results, [])
        raise RuntimeError("At least one direct audit failed to complete")

    failed_sequences = [
        str(result["sequence"])
        for result in direct_results
        if not bool(result["local_one_percent_equilibrium"])
    ]
    manifest["profiles_requiring_grid"] = failed_sequences
    manifest["status"] = (
        "running_conditional_grid" if failed_sequences else "complete"
    )
    write_json(manifest_path, manifest)

    grid_results: list[dict[str, Any]] = []
    grid_tasks: list[dict[str, Any]] = []
    for sequence in failed_sequences:
        order = PROFILES[sequence]
        for price_factor in args.price_factors:
            for capacity_weight in (0.5, 1.0):
                for alpha in (0.3, 0.4):
                    branch = factorial.branch_name(
                        price_factor, capacity_weight, alpha
                    )
                    grid_tasks.append(
                        {
                            "sequence": sequence,
                            "order": order,
                            "branch": branch,
                            "branch_specification": factorial.branch_specification(
                                price_factor, capacity_weight, alpha
                            ),
                            "price_factor": price_factor,
                            "capacity_weight": capacity_weight,
                            "alpha": alpha,
                            "update_gain_threshold": None,
                            "input_path": str(args.input.resolve()),
                            "source_root": str(args.source_root.resolve()),
                            "continuation_root": str(
                                args.continuation_root.resolve()
                            ),
                            "cold_start_root": str(args.source_root.resolve()),
                            "output_root": str(output_root / "basin_search"),
                            "historical_o6": str(args.input.resolve()),
                            "maxiter": args.maxiter,
                            "max_sweeps": args.max_sweeps,
                            "terminal_salvage_fraction": TERMINAL_SALVAGE_FRACTION,
                        }
                    )

    if grid_tasks:
        (output_root / "basin_search").mkdir(parents=True, exist_ok=False)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(grid_tasks))) as pool:
            futures = {
                pool.submit(run_factorial_branch, task): (
                    task["sequence"],
                    task["branch"],
                )
                for task in grid_tasks
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
                grid_results.append(result)
                manifest["grid_results"] = sorted(
                    grid_results,
                    key=lambda row: (row["sequence"], row["branch"]),
                )
                write_json(manifest_path, manifest)
                print(json.dumps(result, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(
            result.get("status") in {"accepted", "no_pass_within_schedule"}
            for result in grid_results
        )
        else "complete_with_failures"
    )
    manifest["grid_results"] = sorted(
        grid_results, key=lambda row: (row["sequence"], row["branch"])
    )
    write_json(manifest_path, manifest)
    write_summary(output_root, direct_results, grid_results)
    print(
        json.dumps(
            {
                "manifest": relative(manifest_path),
                "status": manifest["status"],
                "profiles_requiring_grid": failed_sequences,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
