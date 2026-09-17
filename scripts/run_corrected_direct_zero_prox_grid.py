from __future__ import annotations

"""Direct-from-primitives, zero-proximal equilibrium search grid.

This runner deliberately bypasses every Stage-1 endpoint.  Each independent
branch starts from observed capacity, zero capacity change, true demand, and a
low or high multiple of time-specific manufacturing costs.  It then performs
sequential one-start best responses with fixed damping and no move cap, player
freezing, algorithmic proximal penalty, or multistart solve.

After every full Gauss--Seidel sweep, a separate one-start audit evaluates all
six unilateral deviations against one common frozen profile.  The first
checkpoint for which all six solves succeed and maximum relative gain is at
most one percent is accepted.
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
from scripts import run_local_paper_equilibrium_experiment as local_search
from scripts.audit_selected_equilibrium import _strategy_distance, _zero_prox_data
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.run_corrected_diversified_search import update_player
from scripts.run_corrected_equilibrium_search import (
    PARAMS_SHEET,
    base_configuration,
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
from scripts.search_nested_equilibrium import _sync_quantity


MIN_ALPHA = 0.30
GAIN_TOLERANCE = 0.01

# Three paper-converged orders followed by the four orders reported as not
# converged in the paper experiments.
ORDERS: dict[str, list[str]] = {
    "ch-af-apac-eu-row-us": ["ch", "af", "apac", "eu", "row", "us"],
    "ch-af-eu-us-row-apac": ["ch", "af", "eu", "us", "row", "apac"],
    "ch-row-apac-us-eu-af": ["ch", "row", "apac", "us", "eu", "af"],
    "af-eu-us-apac-row-ch": ["af", "eu", "us", "apac", "row", "ch"],
    "eu-us-af-row-apac-ch": ["eu", "us", "af", "row", "apac", "ch"],
    "us-apac-af-row-eu-ch": ["us", "apac", "af", "row", "eu", "ch"],
    "us-row-eu-apac-af-ch": ["us", "row", "eu", "apac", "af", "ch"],
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def alpha_label(alpha: float) -> str:
    return f"a{int(round(100.0 * alpha)):03d}"


def code_hashes() -> dict[str, str]:
    names = [
        "scripts/run_corrected_direct_zero_prox_grid.py",
        "scripts/nested_market_audit.py",
        "scripts/run_corrected_diversified_search.py",
        "scripts/run_corrected_equilibrium_search.py",
        "scripts/run_local_paper_equilibrium_experiment.py",
        "model/data_prep.py",
        "model/model_main.py",
        "model/run_gs.py",
    ]
    return {name: sha256(ROOT / name) for name in names}


def task_id(order_name: str, start_name: str, alpha: float) -> str:
    return f"{order_name}/{start_name}_{alpha_label(alpha)}"


def direct_initial_state(data: Any, price_factor: float) -> dict[str, dict]:
    times = list(data.times or [])
    move_times = mm._move_times(times)
    initial_capacity = mm._initial_capacity_by_region(data)
    state: dict[str, dict] = {
        "Q_offer": {
            (region, period): float(initial_capacity[region])
            for region in data.players
            for period in times
        },
        "dK_net": {
            (region, period): 0.0
            for region in data.players
            for period in move_times
        },
        "p_offer": {},
        "a_bid": {
            (region, period): float(data.a_dem_t[(region, period)])
            for region in data.regions
            for period in times
        },
    }
    for exporter in data.regions:
        for importer in data.regions:
            for period in times:
                cost = float(
                    (data.c_man_t or {}).get(
                        (exporter, period), data.c_man[exporter]
                    )
                )
                upper = float(data.p_offer_ub[(exporter, importer)])
                state["p_offer"][(exporter, importer, period)] = min(
                    max(price_factor * cost, 0.0), upper
                )
    _sync_quantity(data, state)
    return state


def load_task_data(input_path: Path, order: list[str]) -> Any:
    data = _zero_prox_data(base_configuration(input_path))
    if set(order) != set(data.players) or len(order) != len(set(order)):
        raise ValueError(
            f"Order must contain every player exactly once: order={order}, "
            f"players={list(data.players)}"
        )
    # audit_profile and full_state_payload use this process-local ordering.
    local_search.PLAYER_ORDER = list(order)
    return data


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
    return audit_profile(
        data,
        state,
        starts=1,
        maxiter=maxiter,
        label=label,
        output=output,
    )


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


def state_hash(state: dict[str, dict]) -> str:
    rows: list[list[Any]] = []
    for name in sorted(state):
        for key, value in sorted(state[name].items()):
            rows.append([name, list(key), float(value)])
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def run_branch(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    order_name = str(task["order_name"])
    order = list(ORDERS[order_name])
    start_name = str(task["start_name"])
    price_factor = float(task["price_factor"])
    alpha = float(task["alpha"])
    branch_name = f"{start_name}_{alpha_label(alpha)}"
    branch_root = Path(task["output_root"]).resolve() / order_name / branch_name
    status_path = branch_root / "status.json"
    try:
        branch_root.mkdir(parents=True, exist_ok=True)
        write_json(
            status_path,
            {
                "created": now(),
                "status": "initializing",
                "pid": os.getpid(),
                "order_name": order_name,
                "player_order": order,
                "start_name": start_name,
                "price_factor": price_factor,
                "alpha": alpha,
            },
        )
        input_path = Path(task["input_path"]).resolve()
        data = load_task_data(input_path, order)
        initialization_path = branch_root / "initialization.json"
        if initialization_path.exists():
            initial_state = load_profile(initialization_path, data, direct_initial_state(data, price_factor))
        else:
            initial_state = direct_initial_state(data, price_factor)
            initial_market, initial_market_diagnostics = solve_nested_market(data, initial_state)
            write_json(
                initialization_path,
                {
                    "created": now(),
                    "order_name": order_name,
                    "player_order": order,
                    "start_name": start_name,
                    "price_factor": price_factor,
                    "price_initialization": "price_factor times exporter and period specific manufacturing cost, clipped to model bounds",
                    "capacity_initialization": "observed capacity in every period; zero net capacity change",
                    "demand_initialization": "corrected true-demand intercept",
                    "initial_state_sha256": state_hash(initial_state),
                    "alpha": alpha,
                    "minimum_allowed_alpha": MIN_ALPHA,
                    "move_cap": None,
                    "players_frozen": False,
                    "algorithmic_proximal_penalties": 0.0,
                    "multistart_used": False,
                    "stage1_profile_used": False,
                    "source_workbook_used": False,
                    "input": relative(input_path),
                    "input_sha256": sha256(input_path),
                    "parameter_sheet": PARAMS_SHEET,
                    "profile": full_state_payload(data, initial_state, initial_market),
                    "market_diagnostics": initial_market_diagnostics,
                },
            )

        completed = sorted(branch_root.glob("sweep_*.json"))
        state = (
            load_profile(completed[-1], data, initial_state)
            if completed
            else clone_state(initial_state)
        )
        initial_audit_path = branch_root / "audits" / "audit_initial_one_start.json"
        initial_audit = run_audit(
            data,
            initial_state,
            maxiter=int(task["maxiter"]),
            label=f"direct_{order_name}_{branch_name}_initial",
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

        # Make interrupted branches exactly restartable, including an audit
        # that may not have finished after the latest saved sweep.
        for checkpoint in completed:
            sweep = int(checkpoint.stem.split("_")[-1])
            checkpoint_state = load_profile(checkpoint, data, initial_state)
            audit_path = branch_root / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
            audit = run_audit(
                data,
                checkpoint_state,
                maxiter=int(task["maxiter"]),
                label=f"direct_{order_name}_{branch_name}_s{sweep:03d}",
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

        if selected is None:
            for sweep in range(len(completed) + 1, int(task["max_sweeps"]) + 1):
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
                            f"{order_name}/{branch_name} sweep {sweep} {player}: "
                            "one-start best-response solve failed"
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
                            "move_cap": None,
                            "applied_strategy_move": float(
                                _strategy_distance(data, before_player, state, player)[0]
                            ),
                            "optimizer_success": bool(diagnostics["success"]),
                            "retry_used": retry_used,
                            "starts": 1,
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
                        "order_name": order_name,
                        "player_order": order,
                        "start_name": start_name,
                        "price_factor": price_factor,
                        "alpha": alpha,
                        "minimum_allowed_alpha": MIN_ALPHA,
                        "sweep": sweep,
                        "move_cap": None,
                        "players_frozen": False,
                        "algorithmic_proximal_penalties": 0.0,
                        "multistart_used": False,
                        "stage1_profile_used": False,
                        "players": player_rows,
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
                frozen = run_audit(
                    data,
                    state,
                    maxiter=int(task["maxiter"]),
                    label=f"direct_{order_name}_{branch_name}_s{sweep:03d}",
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
                write_json(
                    status_path,
                    {
                        "updated": now(),
                        "status": "running",
                        "pid": os.getpid(),
                        "order_name": order_name,
                        "player_order": order,
                        "start_name": start_name,
                        "price_factor": price_factor,
                        "alpha": alpha,
                        "sweep": sweep,
                        "latest_one_start_max_relative_gain": current[
                            "max_relative_gain"
                        ],
                        "latest_max_gain_player": current["max_gain_player"],
                        "latest_all_six_solves_successful": current[
                            "all_six_solves_successful"
                        ],
                        "best_one_start_max_relative_gain": best[
                            "max_relative_gain"
                        ],
                        "best_sweep": best["sweep"],
                    },
                )
                print(
                    f"[{order_name} {branch_name}] sweep {sweep}: "
                    f"frozen max gain {100.0 * current['max_relative_gain']:.4f}% "
                    f"({current['max_gain_player']}) pass="
                    f"{current['local_one_percent_equilibrium']}",
                    flush=True,
                )
                if current["local_one_percent_equilibrium"]:
                    selected = current
                    break

        chosen = selected or best
        result = {
            "created": now(),
            "status": "accepted" if selected is not None else "no_pass_within_schedule",
            "pid": os.getpid(),
            "order_name": order_name,
            "player_order": order,
            "start_name": start_name,
            "price_factor": price_factor,
            "alpha": alpha,
            "minimum_allowed_alpha": MIN_ALPHA,
            "max_sweeps": int(task["max_sweeps"]),
            "move_cap": None,
            "players_frozen": False,
            "algorithmic_proximal_penalties": 0.0,
            "multistart_used": False,
            "stage1_profile_used": False,
            "source_workbook_used": False,
            "input": relative(input_path),
            "input_sha256": sha256(input_path),
            "selected_profile": chosen["profile_path"],
            "selected_sweep": chosen["sweep"],
            "one_start_audit": chosen["audit_path"],
            "one_start_max_relative_gain": chosen["max_relative_gain"],
            "one_start_max_gain_player": chosen["max_gain_player"],
            "all_six_solves_successful": chosen["all_six_solves_successful"],
            "local_one_percent_equilibrium": chosen[
                "local_one_percent_equilibrium"
            ],
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
            "order_name": order_name,
            "player_order": order,
            "start_name": start_name,
            "price_factor": price_factor,
            "alpha": alpha,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def validate(task: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(task["input_path"]).resolve()
    order_name = str(task["order_name"])
    data = load_task_data(input_path, ORDERS[order_name])
    state = direct_initial_state(data, float(task["price_factor"]))
    market, diagnostics = solve_nested_market(data, state)
    return {
        "task": task_id(order_name, str(task["start_name"]), float(task["alpha"])),
        "price_factor": task["price_factor"],
        "alpha": task["alpha"],
        "stage1_profile_used": False,
        "algorithmic_proximal_penalties": 0.0,
        "market_solve_successful": market is not None,
        "market_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--orders", nargs="+", choices=["all", *ORDERS], default=["all"]
    )
    parser.add_argument("--low-price-factor", type=float, default=0.90)
    parser.add_argument("--high-price-factor", type=float, default=1.20)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.40, 0.70])
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--max-sweeps", type=int, default=80)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.maxiter < 1 or args.max_sweeps < 1:
        raise ValueError("workers, maxiter, and max-sweeps must be positive")
    if not 0.0 < args.low_price_factor < args.high_price_factor:
        raise ValueError("price factors must satisfy 0 < low < high")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("alphas must not contain duplicates")
    for alpha in args.alphas:
        if not MIN_ALPHA <= alpha <= 1.0:
            raise ValueError(
                f"every alpha must be between the allowed floor {MIN_ALPHA:.2f} and 1.0"
            )

    input_path = args.input.resolve()
    output_root = args.output_root.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    selected_orders = list(ORDERS) if "all" in args.orders else list(args.orders)
    if "all" in args.orders and len(args.orders) != 1:
        raise ValueError("use --orders all alone, or list explicit order names")

    starts = {
        "low": float(args.low_price_factor),
        "high": float(args.high_price_factor),
    }
    tasks = [
        {
            "order_name": order_name,
            "start_name": start_name,
            "price_factor": price_factor,
            "alpha": float(alpha),
            "input_path": str(input_path),
            "output_root": str(output_root),
            "maxiter": int(args.maxiter),
            "max_sweeps": int(args.max_sweeps),
        }
        for order_name in selected_orders
        for start_name, price_factor in starts.items()
        for alpha in args.alphas
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
        "method": "direct-from-primitives fixed-damping zero-proximal Gauss--Seidel grid",
        "task_count": len(tasks),
        "acceptance_criterion": "one-start common frozen-profile maximum relative gain <= 1%, all six solves successful",
        "acceptance_audit_starts": 1,
        "algorithmic_proximal_penalties": 0.0,
        "multistart_used": False,
        "stage1_profile_used": False,
        "source_workbook_used": False,
        "move_cap": None,
        "players_frozen": False,
        "orders": {name: ORDERS[name] for name in selected_orders},
        "starts": starts,
        "alphas": [float(alpha) for alpha in args.alphas],
        "minimum_allowed_alpha": MIN_ALPHA,
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "output_root": relative(output_root),
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
            pool.submit(run_branch, task): (
                task["order_name"],
                task["start_name"],
                task["alpha"],
            )
            for task in tasks
        }
        for future in as_completed(futures):
            order_name, start_name, alpha = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "status": "executor_exception",
                    "order_name": order_name,
                    "start_name": start_name,
                    "alpha": alpha,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            manifest["results"] = sorted(
                results,
                key=lambda row: (
                    row["order_name"],
                    row["start_name"],
                    row["alpha"],
                ),
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
        results,
        key=lambda row: (
            row["order_name"],
            row["start_name"],
            row["alpha"],
        ),
    )
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
