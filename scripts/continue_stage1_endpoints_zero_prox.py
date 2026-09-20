from __future__ import annotations

"""Continue the three corrected Stage-1 endpoints with zero proximal penalty.

The terminal clean Stage-1 state for each player order is replayed directly
from its workbook. Saved capacity changes and bilateral offer prices are not
reinitialized or interpolated. Each state receives a fixed number of
candidate-strategy-start Gauss--Seidel sweeps with all algorithmic proximal
coefficients set to zero.
"""

import argparse
import csv
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

import pandas as pd


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


ZERO_PROX = {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0}
ECONOMIC_QUADRATIC = {"q": 0.1, "p": 0.1, "a": 0.1}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def strategy_fingerprint(state: dict[str, dict]) -> str:
    payload = {
        "dK_net": [
            [*key, float(value)]
            for key, value in sorted(state["dK_net"].items())
        ],
        "p_offer": [
            [*key, float(value)]
            for key, value in sorted(state["p_offer"].items())
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def terminal_stage1_row(workbook: Path, iteration: int) -> dict[str, Any]:
    rows = pd.read_excel(workbook, sheet_name="iters")
    selected = rows.loc[rows["iter"].astype(int) == int(iteration)]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one Stage-1 iteration {iteration} in {workbook}; found {len(selected)}"
        )
    row = selected.iloc[0]
    if not bool(row["all_solves_acceptable"]):
        raise RuntimeError(f"Selected Stage-1 iteration {iteration} is not clean")
    return {
        "iteration": int(iteration),
        "all_solves_acceptable": bool(row["all_solves_acceptable"]),
        "omega": float(row["omega"]),
        "omega_next": float(row["omega_next"]),
        "algorithmic_proximal_penalties": {
            "q": float(row["c_pen_q"]),
            "p": float(row["c_pen_p"]),
            "a": float(row["c_pen_a"]),
            "dk": float(row["c_pen_dk"]),
        },
    }


def validate_model_configuration(data: Any) -> None:
    settings = data.settings or {}
    actual_quadratic = {
        name: float(settings.get(f"c_quad_{name}", float("nan")))
        for name in ECONOMIC_QUADRATIC
    }
    if actual_quadratic != ECONOMIC_QUADRATIC:
        raise RuntimeError(
            f"Economic quadratic coefficients changed unexpectedly: {actual_quadratic}"
        )
    salvage = float(settings.get("terminal_salvage_fraction", float("nan")))
    if salvage != 0.5:
        raise RuntimeError(f"Expected terminal salvage fraction 0.5; got {salvage}")


def cached_audit(
    data: Any,
    state: dict[str, dict],
    *,
    maxiter: int,
    label: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    return audit_profile(
        data,
        state,
        starts=1,
        maxiter=maxiter,
        label=label,
        output=output,
    )


def configure_task(
    task: dict[str, Any],
) -> tuple[Any, dict[str, dict], float, Path, dict[str, Any]]:
    data, source, replay_error, workbook, _ = configure_modules(task)
    validate_model_configuration(data)
    iteration = int(SEQUENCES[str(task["sequence"])]["source_iteration"])
    terminal_row = terminal_stage1_row(workbook, iteration)
    if replay_error > 1e-10:
        raise RuntimeError(
            f"{task['sequence']}: endpoint replay residual {replay_error:.3g}"
        )
    if abs(terminal_row["omega_next"] - float(task["alpha"])) > 1e-12:
        raise RuntimeError(
            f"Requested alpha {task['alpha']} does not continue the Stage-1 "
            f"terminal omega {terminal_row['omega_next']}"
        )
    return data, source, replay_error, workbook, terminal_row


def validate_task(task: dict[str, Any]) -> dict[str, Any]:
    data, source, replay_error, workbook, terminal_row = configure_task(task)
    market, market_diagnostics = solve_nested_market(data, source)
    return {
        "sequence": task["sequence"],
        "player_order": list(SEQUENCES[str(task["sequence"])]["order"]),
        "source_workbook": relative(workbook),
        "source_workbook_sha256": sha256(workbook),
        "source_iteration": terminal_row["iteration"],
        "source_replay_error": replay_error,
        "source_strategy_sha256": strategy_fingerprint(source),
        "source_penalties": terminal_row["algorithmic_proximal_penalties"],
        "continuation_penalties": ZERO_PROX,
        "continuation_alpha": task["alpha"],
        "market_solve_successful": market is not None,
        "market_diagnostics": market_diagnostics,
    }


def run_sequence(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    order = list(SEQUENCES[sequence]["order"])
    alpha = float(task["alpha"])
    total_sweeps = int(task["sweeps"])
    sequence_root = Path(task["output_root"]).resolve() / sequence
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
            },
        )
        data, source, replay_error, workbook, terminal_row = configure_task(task)
        source_hash = strategy_fingerprint(source)
        initial_state = clone_state(source)
        initial_hash = strategy_fingerprint(initial_state)
        if initial_hash != source_hash:
            raise RuntimeError("Stage-1 strategy changed before initialization")

        initialization_path = sequence_root / "initialization.json"
        if not initialization_path.exists():
            market, market_diagnostics = solve_nested_market(data, initial_state)
            write_json(
                initialization_path,
                {
                    "created": now(),
                    "sequence": sequence,
                    "method": "exact terminal clean Stage-1 endpoint replay",
                    "source_workbook": relative(workbook),
                    "source_workbook_sha256": sha256(workbook),
                    "source_iteration": terminal_row["iteration"],
                    "source_replay_error": replay_error,
                    "source_strategy_sha256": source_hash,
                    "initialized_strategy_sha256": initial_hash,
                    "exact_strategy_replay_verified": source_hash == initial_hash,
                    "capacity_or_offer_reinitialization": False,
                    "player_order": order,
                    "source_omega": terminal_row["omega"],
                    "continuation_alpha": alpha,
                    "source_algorithmic_proximal_penalties": terminal_row[
                        "algorithmic_proximal_penalties"
                    ],
                    "continuation_algorithmic_proximal_penalties": ZERO_PROX,
                    "economic_quadratic_coefficients": ECONOMIC_QUADRATIC,
                    "terminal_salvage_fraction": 0.5,
                    "scheduled_sweeps": total_sweeps,
                    "market_diagnostics": market_diagnostics,
                    "profile": full_state_payload(data, initial_state, market),
                },
            )
        else:
            existing = json.loads(initialization_path.read_text(encoding="utf-8"))
            if existing["source_strategy_sha256"] != source_hash:
                raise RuntimeError("Existing initialization has different Stage-1 source")

        initial_audit_path = audit_root / "audit_initial_one_start.json"
        initial_audit = cached_audit(
            data,
            initial_state,
            maxiter=int(task["maxiter"]),
            label=f"stage1_exact_zero_prox_{sequence}_initial",
            output=initial_audit_path,
        )

        completed = sorted(sequence_root.glob("sweep_*.json"))
        completed_numbers = [int(path.stem.split("_")[-1]) for path in completed]
        if completed_numbers != list(range(1, len(completed_numbers) + 1)):
            raise RuntimeError(f"Non-contiguous sweep checkpoints: {completed_numbers}")
        if len(completed) > total_sweeps:
            raise RuntimeError(
                f"Found {len(completed)} checkpoints for a {total_sweeps}-sweep run"
            )
        state = (
            load_profile(completed[-1], data, source)
            if completed
            else clone_state(initial_state)
        )

        for sweep in range(len(completed) + 1, total_sweeps + 1):
            sweep_started = time.perf_counter()
            before_sweep = clone_state(state)
            player_rows: list[dict[str, Any]] = []
            for player in order:
                before_player = clone_state(state)
                market, market_diagnostics = solve_nested_market(data, state)
                reference = float(
                    nested_economic_objective(data, state, market, player)
                )
                best_value, response, diagnostics = nested_best_response(
                    data,
                    state,
                    market,
                    player,
                    maxiter=int(task["maxiter"]),
                    starts=1,
                    proximal_coefficients=None,
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
                        proximal_coefficients=None,
                    )
                if not diagnostics["success"]:
                    raise RuntimeError(
                        f"sweep {sweep} {player}: current-profile best response failed"
                    )
                gain = max(float(best_value) - reference, 0.0) / max(
                    abs(reference), 1.0
                )
                raw_move = float(
                    _strategy_distance(data, before_player, response, player)[0]
                )
                update_player(data, state, response, player, alpha)
                player_rows.append(
                    {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best_value),
                        "relative_gain": gain,
                        "raw_strategy_move": raw_move,
                        "damping_weight": alpha,
                        "applied_strategy_move": float(
                            _strategy_distance(data, before_player, state, player)[0]
                        ),
                        "optimizer_start": "current strategy profile only",
                        "optimizer_success": bool(diagnostics["success"]),
                        "retry_used": retry_used,
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
                    "method": "exact Stage-1 endpoint zero-proximal continuation",
                    "sweep": sweep,
                    "scheduled_sweeps": total_sweeps,
                    "source_workbook": relative(workbook),
                    "source_iteration": terminal_row["iteration"],
                    "source_strategy_sha256": source_hash,
                    "capacity_or_offer_reinitialization": False,
                    "optimizer_multistart_used": False,
                    "player_order": order,
                    "alpha": alpha,
                    "algorithmic_proximal_penalties": ZERO_PROX,
                    "economic_quadratic_coefficients": ECONOMIC_QUADRATIC,
                    "terminal_salvage_fraction": 0.5,
                    "players": player_rows,
                    "max_sequential_relative_gain": max(
                        row["relative_gain"] for row in player_rows
                    ),
                    "max_sequential_gain_player": max(
                        player_rows, key=lambda row: row["relative_gain"]
                    )["player"],
                    "strategy_change_metric": max(
                        float(
                            _strategy_distance(
                                data, before_sweep, state, player
                            )[0]
                        )
                        for player in order
                    ),
                    "ending_market_diagnostics": ending_market_diagnostics,
                    "elapsed_seconds": time.perf_counter() - sweep_started,
                    "ending_profile": full_state_payload(data, state, ending_market),
                },
            )
            latest = json.loads(checkpoint.read_text(encoding="utf-8"))
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "pid": os.getpid(),
                    "sequence": sequence,
                    "completed_sweeps": sweep,
                    "scheduled_sweeps": total_sweeps,
                    "latest_max_sequential_relative_gain": latest[
                        "max_sequential_relative_gain"
                    ],
                    "latest_strategy_change_metric": latest[
                        "strategy_change_metric"
                    ],
                },
            )
            print(
                f"[{sequence}] sweep {sweep:02d}/{total_sweeps}: "
                f"max sequential gain="
                f"{100.0 * latest['max_sequential_relative_gain']:.4f}%",
                flush=True,
            )

        final_profile = sequence_root / f"sweep_{total_sweeps:03d}.json"
        final_state = load_profile(final_profile, data, source)
        final_audit_path = audit_root / "audit_final_one_start.json"
        final_audit = cached_audit(
            data,
            final_state,
            maxiter=int(task["maxiter"]),
            label=f"stage1_exact_zero_prox_{sequence}_final",
            output=final_audit_path,
        )
        checkpoint_rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(sequence_root.glob("sweep_*.json"))
        ]
        best_sequential_row = min(
            checkpoint_rows,
            key=lambda row: float(row["max_sequential_relative_gain"]),
        )
        best_sequential_sweep = int(best_sequential_row["sweep"])
        best_sequential_profile = (
            sequence_root / f"sweep_{best_sequential_sweep:03d}.json"
        )
        best_sequential_state = load_profile(
            best_sequential_profile, data, source
        )
        best_sequential_audit_path = audit_root / (
            f"audit_best_sequential_s{best_sequential_sweep:03d}_one_start.json"
        )
        best_sequential_audit = cached_audit(
            data,
            best_sequential_state,
            maxiter=int(task["maxiter"]),
            label=(
                f"stage1_exact_zero_prox_{sequence}_"
                f"best_sequential_s{best_sequential_sweep:03d}"
            ),
            output=best_sequential_audit_path,
        )
        audited_candidates = [
            {
                "label": "initial",
                "profile": initialization_path,
                "audit": initial_audit_path,
                "payload": initial_audit,
            },
            {
                "label": f"best_sequential_s{best_sequential_sweep:03d}",
                "profile": best_sequential_profile,
                "audit": best_sequential_audit_path,
                "payload": best_sequential_audit,
            },
            {
                "label": f"final_s{total_sweeps:03d}",
                "profile": final_profile,
                "audit": final_audit_path,
                "payload": final_audit,
            },
        ]
        selected = min(
            audited_candidates,
            key=lambda item: float(item["payload"]["max_relative_gain"]),
        )
        equilibrium_found = bool(selected["payload"]["equilibrium_verified"])
        result = {
            "created": now(),
            "status": (
                "accepted_one_start_local_1pct_equilibrium"
                if equilibrium_found
                else "completed_15_sweeps_without_audited_one_start_1pct_pass"
            ),
            "pid": os.getpid(),
            "sequence": sequence,
            "player_order": order,
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": terminal_row["iteration"],
            "source_replay_error": replay_error,
            "source_strategy_sha256": source_hash,
            "capacity_or_offer_reinitialization": False,
            "optimizer_multistart_used": False,
            "alpha": alpha,
            "sweeps_completed": total_sweeps,
            "source_algorithmic_proximal_penalties": terminal_row[
                "algorithmic_proximal_penalties"
            ],
            "continuation_algorithmic_proximal_penalties": ZERO_PROX,
            "economic_quadratic_coefficients": ECONOMIC_QUADRATIC,
            "terminal_salvage_fraction": 0.5,
            "initialization": relative(initialization_path),
            "initial_one_start_audit": relative(initial_audit_path),
            "initial_one_start_max_relative_gain": float(
                initial_audit["max_relative_gain"]
            ),
            "best_sequential_checkpoint": relative(best_sequential_profile),
            "best_sequential_checkpoint_sweep": best_sequential_sweep,
            "best_sequential_checkpoint_live_max_relative_gain": float(
                best_sequential_row["max_sequential_relative_gain"]
            ),
            "best_sequential_checkpoint_audit": relative(
                best_sequential_audit_path
            ),
            "best_sequential_checkpoint_frozen_max_relative_gain": float(
                best_sequential_audit["max_relative_gain"]
            ),
            "final_profile": relative(final_profile),
            "final_one_start_audit": relative(final_audit_path),
            "final_one_start_max_relative_gain": float(
                final_audit["max_relative_gain"]
            ),
            "final_max_gain_player": str(final_audit["max_gain_player"]),
            "all_final_attempts_successful": bool(
                final_audit["all_attempts_successful"]
            ),
            "local_one_percent_equilibrium": bool(
                equilibrium_found
            ),
            "selected_audited_candidate": selected["label"],
            "selected_profile": relative(selected["profile"]),
            "selected_audit": relative(selected["audit"]),
            "selected_max_relative_gain": float(
                selected["payload"]["max_relative_gain"]
            ),
            "criterion": (
                "one-start common frozen-profile zero-proximal maximum relative "
                "gain <= 1%, all six solves successful"
            ),
            "claim_scope": (
                "local computational result; not a multistart or global "
                "Nash-equilibrium certificate"
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


def protocol_text(
    *,
    input_path: Path,
    cold_start_root: Path,
    output_root: Path,
    sweeps: int,
    maxiter: int,
    workers: int,
) -> str:
    orders = "\n".join(
        f"- `{name}`: `{'-'.join(SEQUENCES[name]['order'])}`"
        for name in SEQUENCES
    )
    return f"""# Exact Stage-1 endpoint continuation with zero proximal penalty

Created {now()}.

{orders}

- Corrected input: `{relative(input_path)}`
- Stage-1 workbook root: `{relative(cold_start_root)}`
- Output root: `{relative(output_root)}`
- Further sweeps per profile: {sweeps}
- Best-response maximum iterations: {maxiter}
- Parallel workers: {workers}
- Continuation damping: 0.4, equal to each terminal Stage-1 `omega_next`
- Stage-1 terminal proximal coefficients: q=2, p=3, a=2, dk=2
- Continuation proximal coefficients: q=0, p=0, a=0, dk=0
- Economic quadratic coefficients retained: q=0.1, p=0.1, a=0.1
- Terminal salvage fraction retained: 0.5

For each player order, the runner uses the last contiguous Stage-1 iteration
for which every player solve was acceptable. It replays the complete damped
strategy through that iteration directly from the Stage-1 workbook. The saved
`dK_net` and bilateral `p_offer` values are the sweep-0 strategy; no capacity
or price initialization, interpolation, or randomization is performed.

All {sweeps} sequential Gauss--Seidel sweeps use one optimizer start at the
current strategy profile, no move cap, no gain filter, no player freezing, and
zero algorithmic proximal cost. Initial and final common-frozen-profile audits
use the same one-start rule. The checkpoint with the lowest live sequential
gain is also audited so an intermediate low-gain point is not overlooked. The
1% criterion is local computational evidence, not a multistart or global
certificate.
"""


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "sequence",
        "source_iteration",
        "alpha",
        "sweeps_completed",
        "initial_one_start_max_relative_gain",
        "best_sequential_checkpoint_sweep",
        "best_sequential_checkpoint_frozen_max_relative_gain",
        "final_one_start_max_relative_gain",
        "selected_audited_candidate",
        "selected_max_relative_gain",
        "final_max_gain_player",
        "local_one_percent_equilibrium",
        "status",
        "source_workbook",
        "final_profile",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["sequence"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cold-start-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=15)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.sweeps < 1 or args.maxiter < 1 or args.workers < 1:
        raise ValueError("sweeps, maxiter, and workers must be positive")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
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
            "terminal_salvage_fraction": 0.5,
            "alpha": args.alpha,
            "sweeps": args.sweeps,
            "maxiter": args.maxiter,
        }
        for sequence in SEQUENCES
    ]
    if args.validate_only:
        print(
            json.dumps([validate_task(task) for task in tasks], indent=2),
            flush=True,
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers, len(tasks))
    protocol_path = output_root / "PROTOCOL.md"
    protocol_path.write_text(
        protocol_text(
            input_path=input_path,
            cold_start_root=cold_start_root,
            output_root=output_root,
            sweeps=args.sweeps,
            maxiter=args.maxiter,
            workers=workers,
        ),
        encoding="utf-8",
    )
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "three exact Stage-1 endpoint continuations after removing algorithmic proximal penalties",
        "task_count": len(tasks),
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "output_root": relative(output_root),
        "sweeps_per_profile": args.sweeps,
        "maxiter": args.maxiter,
        "workers": workers,
        "alpha": args.alpha,
        "capacity_or_offer_reinitialization": False,
        "optimizer_multistart_used": False,
        "continuation_algorithmic_proximal_penalties": ZERO_PROX,
        "economic_quadratic_coefficients": ECONOMIC_QUADRATIC,
        "terminal_salvage_fraction": 0.5,
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "code_sha256": {
            "scripts/continue_stage1_endpoints_zero_prox.py": sha256(Path(__file__)),
            "scripts/nested_market_audit.py": sha256(
                ROOT / "scripts" / "nested_market_audit.py"
            ),
            "scripts/run_corrected_equilibrium_search.py": sha256(
                ROOT / "scripts" / "run_corrected_equilibrium_search.py"
            ),
        },
        "results": [],
    }
    write_json(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
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
            manifest["results"] = sorted(results, key=lambda row: row["sequence"])
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(row["status"] != "failed" and row["status"] != "executor_exception" for row in results)
        else "complete_with_failures"
    )
    manifest["results"] = sorted(results, key=lambda row: row["sequence"])
    write_json(manifest_path, manifest)
    write_summary(output_root / "results_summary.csv", results)
    print(
        json.dumps(
            {"manifest": relative(manifest_path), "status": manifest["status"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
