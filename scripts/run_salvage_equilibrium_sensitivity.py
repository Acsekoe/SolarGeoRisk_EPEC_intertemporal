from __future__ import annotations

"""Restart E1--E3 at the zero-proximal equilibrium stage with terminal salvage.

This is deliberately a one-start local sensitivity.  It does not perform or
claim a multistart audit.  Existing accepted artifacts are read-only inputs and
all new checkpoints are written beneath a separate timestamped output root.
"""

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.audit_selected_equilibrium import _strategy_distance, _zero_prox_data
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.run_corrected_diversified_search import update_player
from scripts.run_corrected_equilibrium_search import base_configuration
from scripts.run_local_paper_equilibrium_experiment import (
    audit_profile,
    clone_state,
    full_state_payload,
)
from scripts.run_overnight_equilibrium_experiment import load_profile
from scripts.search_nested_equilibrium import _sync_quantity


DEFAULT_INPUT = (
    ROOT
    / "outputs/demand_calibration/correction_20260915_131409"
    / "input_data_intertemporal_corrected_20260915_131409.xlsx"
)
DEFAULT_SOURCE_ROOT = ROOT / "outputs/equilibria/corrected_demand_20260916"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def blank_template(data: Any) -> dict[str, dict]:
    times = list(data.times or [])
    initial_capacity = mm._initial_capacity_by_region(data)
    state: dict[str, dict] = {
        "dK_net": {
            (player, period): 0.0
            for player in data.players
            for period in mm._move_times(times)
        },
        "p_offer": {},
        "a_bid": {
            (region, period): float((data.a_dem_t or {})[(region, period)])
            for region in data.regions
            for period in times
        },
        "Q_offer": {
            (player, period): float(initial_capacity[player])
            for player in data.players
            for period in times
        },
    }
    for exporter in data.regions:
        for importer in data.regions:
            for period in times:
                state["p_offer"][(exporter, importer, period)] = float(
                    (data.c_man_t or {}).get(
                        (exporter, period), data.c_man[exporter]
                    )
                )
    _sync_quantity(data, state)
    return state


def capacity_comparison(data: Any, source: dict[str, dict], candidate: dict[str, dict]) -> dict[str, object]:
    times = list(data.times or [])
    source_capacity = mm._implied_capacity_path(data, times, source["dK_net"])
    candidate_capacity = mm._implied_capacity_path(data, times, candidate["dK_net"])
    rows = []
    for player in data.players:
        for period in times:
            before = float(source_capacity[(player, period)])
            after = float(candidate_capacity[(player, period)])
            rows.append(
                {
                    "player": player,
                    "time": period,
                    "source": before,
                    "salvage": after,
                    "change": after - before,
                }
            )
    largest = max(rows, key=lambda row: abs(float(row["change"])))
    return {
        "maximum_absolute_change": abs(float(largest["change"])),
        "largest_change_coordinate": f"Kcap[{largest['player']},{largest['time']}]",
        "rows": rows,
    }


def salvage_metadata(data: Any, fraction: float) -> dict[str, object]:
    times = list(data.times or [])
    terminal = times[-1]
    terminal_year = None
    try:
        terminal_year = int(float(terminal)) + int(
            round(float((data.years_to_next or {}).get(terminal, 0.0)))
        )
    except (TypeError, ValueError):
        pass
    discount = mm._terminal_salvage_discount_factor(data)
    return {
        "terminal_salvage_fraction": fraction,
        "terminal_stock_period": terminal,
        "valuation_year": terminal_year,
        "discount_factor": discount,
        "formula": "discount_factor * terminal_salvage_fraction * c_inv[player] * Kcap[player,terminal]",
        "discounted_credit_per_gw": {
            player: discount * fraction * float((data.c_inv or {})[player])
            for player in data.players
        },
    }


def run_equilibrium(
    *,
    equilibrium_id: str,
    input_path: Path,
    source_root: Path,
    output_root: Path,
    salvage_fraction: float,
    alpha: float,
    max_sweeps: int,
    maxiter: int,
    gain_tolerance: float,
) -> dict[str, object]:
    started = time.perf_counter()
    source_dir = source_root / equilibrium_id
    source_profile_path = source_dir / "equilibrium_profile.json"
    source_status_path = source_dir / "status.json"
    output_dir = output_root / equilibrium_id
    status_path = output_dir / "status.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        order = [str(player) for player in source_status["player_order"]]
        cfg = base_configuration(input_path)
        data = _zero_prox_data(
            cfg,
            terminal_salvage_fraction=salvage_fraction,
        )
        if set(order) != set(data.players) or len(order) != len(set(order)):
            raise ValueError(f"Invalid player order for {equilibrium_id}: {order}")

        # The audit helper intentionally keeps player order process-local.
        import scripts.run_local_paper_equilibrium_experiment as local_search

        local_search.PLAYER_ORDER = order
        template = blank_template(data)
        source = load_profile(source_profile_path, data, template)
        _sync_quantity(data, source)
        state = clone_state(source)

        initial_market, initial_market_diag = solve_nested_market(data, state)
        initial_capacities = mm._implied_capacity_path(
            data, list(data.times or []), state["dK_net"]
        )
        initial_objectives = {
            player: float(nested_economic_objective(data, state, initial_market, player))
            for player in order
        }
        terminal = list(data.times or [])[-1]
        initial_credits = {
            player: mm._terminal_salvage_credit(
                data, player, initial_capacities[(player, terminal)]
            )
            for player in order
        }
        write_json(
            output_dir / "initial_profile.json",
            {
                "created": now(),
                "equilibrium_id": equilibrium_id,
                "source_profile": relative(source_profile_path),
                "source_profile_sha256": sha256(source_profile_path),
                "player_order": order,
                "salvage": salvage_metadata(data, salvage_fraction),
                "objectives_with_salvage": initial_objectives,
                "objectives_without_salvage_at_same_profile": {
                    player: initial_objectives[player] - initial_credits[player]
                    for player in order
                },
                "terminal_salvage_credit": initial_credits,
                "market_diagnostics": initial_market_diag,
                "profile": full_state_payload(data, state, initial_market),
            },
        )

        initial_audit_path = output_dir / "audit_initial_one_start.json"
        initial_audit = audit_profile(
            data,
            state,
            starts=1,
            maxiter=maxiter,
            label=f"salvage_{equilibrium_id}_initial",
            output=initial_audit_path,
        )
        accepted_audit = initial_audit if bool(initial_audit["equilibrium_verified"]) else None
        accepted_profile_path = output_dir / "initial_profile.json" if accepted_audit else None
        sweeps_completed = 0
        terminal_reason = "initial_profile_passed_one_start_audit" if accepted_audit else None

        for sweep in range(1, max_sweeps + 1):
            if accepted_audit is not None:
                break
            before_sweep = clone_state(state)
            player_rows: list[dict[str, object]] = []
            failed = False
            for player in order:
                before_player = clone_state(state)
                market, market_diag = solve_nested_market(data, state)
                reference = float(
                    nested_economic_objective(data, state, market, player)
                )
                best, response, diagnostics = nested_best_response(
                    data,
                    state,
                    market,
                    player,
                    maxiter=maxiter,
                    starts=1,
                )
                retry_used = False
                if not diagnostics["success"]:
                    retry_used = True
                    best, response, diagnostics = nested_best_response(
                        data,
                        state,
                        market,
                        player,
                        maxiter=max(2 * maxiter, 1000),
                        starts=1,
                    )
                relative_gain = max(float(best) - reference, 0.0) / max(
                    abs(reference), 1.0
                )
                if diagnostics["success"]:
                    update_player(data, state, response, player, alpha)
                else:
                    failed = True
                move, coordinate, absolute = _strategy_distance(
                    data, before_player, response, player
                )
                player_rows.append(
                    {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best),
                        "relative_gain": relative_gain,
                        "raw_strategy_move": move,
                        "largest_move_coordinate": coordinate,
                        "largest_move_absolute": absolute,
                        "alpha": alpha,
                        "optimizer_success": bool(diagnostics["success"]),
                        "retry_same_start_with_higher_iteration_cap": retry_used,
                        "optimizer": diagnostics,
                        "reference_market_diagnostics": market_diag,
                    }
                )
                print(
                    f"[{equilibrium_id} S{sweep:03d}] {player}: "
                    f"gain={relative_gain:.3%} success={diagnostics['success']}",
                    flush=True,
                )
                if failed:
                    break

            sweeps_completed = sweep
            ending_market, ending_market_diag = solve_nested_market(data, state)
            strategy_change = max(
                _strategy_distance(data, before_sweep, state, player)[0]
                for player in order
            )
            sweep_path = output_dir / f"sweep_{sweep:03d}.json"
            write_json(
                sweep_path,
                {
                    "created": now(),
                    "equilibrium_id": equilibrium_id,
                    "sweep": sweep,
                    "player_order": order,
                    "alpha": alpha,
                    "starts_per_best_response": 1,
                    "algorithmic_proximal_penalties": 0.0,
                    "terminal_salvage_fraction": salvage_fraction,
                    "strategy_change_metric": strategy_change,
                    "players": player_rows,
                    "all_player_updates_successful": not failed,
                    "ending_market_diagnostics": ending_market_diag,
                    "ending_profile": full_state_payload(data, state, ending_market),
                },
            )
            if failed:
                terminal_reason = "one_start_player_solve_failed"
                break

            audit_path = output_dir / f"audit_sweep_{sweep:03d}_one_start.json"
            audit = audit_profile(
                data,
                state,
                starts=1,
                maxiter=maxiter,
                label=f"salvage_{equilibrium_id}_sweep_{sweep:03d}",
                output=audit_path,
            )
            if bool(audit["equilibrium_verified"]):
                accepted_audit = audit
                accepted_profile_path = sweep_path
                terminal_reason = "one_start_relative_gain_at_or_below_tolerance"
                break

        if terminal_reason is None:
            terminal_reason = "maximum_sweeps_reached"

        final_state = state
        comparison = capacity_comparison(data, source, final_state)
        result: dict[str, object] = {
            "created": now(),
            "status": "accepted" if accepted_audit is not None else "not_accepted",
            "equilibrium_id": equilibrium_id,
            "pid": os.getpid(),
            "source_profile": relative(source_profile_path),
            "source_profile_sha256": sha256(source_profile_path),
            "source_status": relative(source_status_path),
            "input": relative(input_path),
            "input_sha256": sha256(input_path),
            "player_order": order,
            "salvage": salvage_metadata(data, salvage_fraction),
            "alpha": alpha,
            "starts_per_best_response": 1,
            "multistart_used": False,
            "algorithmic_proximal_penalties": 0.0,
            "gain_tolerance": gain_tolerance,
            "sweeps_completed": sweeps_completed,
            "terminal_reason": terminal_reason,
            "accepted_profile": None if accepted_profile_path is None else relative(accepted_profile_path),
            "accepted_audit_max_relative_gain": None if accepted_audit is None else float(accepted_audit["max_relative_gain"]),
            "accepted_audit_max_gain_player": None if accepted_audit is None else str(accepted_audit["max_gain_player"]),
            "capacity_comparison_to_source": comparison,
            "claim_scope": "one-start local computational sensitivity; no multistart or global claim",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        failure: dict[str, object] = {
            "created": now(),
            "status": "failed",
            "equilibrium_id": equilibrium_id,
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--equilibria", nargs="+", default=["E1", "E2", "E3"])
    parser.add_argument("--salvage-fraction", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.30)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--gain-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    if args.salvage_fraction < 0.0:
        raise ValueError("--salvage-fraction must be non-negative")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be in (0,1]")
    if args.max_sweeps < 1 or args.maxiter < 1:
        raise ValueError("--max-sweeps and --maxiter must be positive")
    if not 0.0 < args.gain_tolerance < 1.0:
        raise ValueError("--gain-tolerance must lie in (0,1)")

    input_path = args.input.resolve()
    source_root = args.source_root.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    for equilibrium_id in args.equilibria:
        if not (source_root / equilibrium_id / "equilibrium_profile.json").is_file():
            raise FileNotFoundError(source_root / equilibrium_id / "equilibrium_profile.json")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (ROOT / "outputs/equilibria" / f"salvage_phi{args.salvage_fraction:.2f}_{stamp}")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, object] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "one-start zero-proximal equilibrium-stage restart from packaged E1-E3 profiles",
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "source_root": relative(source_root),
        "output_root": relative(output_root),
        "equilibria": list(args.equilibria),
        "salvage_fraction": args.salvage_fraction,
        "alpha": args.alpha,
        "max_sweeps": args.max_sweeps,
        "maxiter": args.maxiter,
        "gain_tolerance": args.gain_tolerance,
        "starts_per_best_response": 1,
        "multistart_used": False,
        "code_sha256": {
            relative(Path(__file__)): sha256(Path(__file__)),
            "model/model_main.py": sha256(ROOT / "model/model_main.py"),
            "model/run_gs.py": sha256(ROOT / "model/run_gs.py"),
            "scripts/audit_selected_equilibrium.py": sha256(
                ROOT / "scripts/audit_selected_equilibrium.py"
            ),
            "scripts/nested_market_audit.py": sha256(
                ROOT / "scripts/nested_market_audit.py"
            ),
        },
        "results": [],
    }
    write_json(manifest_path, manifest)

    results = []
    for equilibrium_id in args.equilibria:
        result = run_equilibrium(
            equilibrium_id=equilibrium_id,
            input_path=input_path,
            source_root=source_root,
            output_root=output_root,
            salvage_fraction=args.salvage_fraction,
            alpha=args.alpha,
            max_sweeps=args.max_sweeps,
            maxiter=args.maxiter,
            gain_tolerance=args.gain_tolerance,
        )
        results.append(result)
        manifest["results"] = results
        write_json(manifest_path, manifest)

    manifest["status"] = (
        "completed" if all(result.get("status") != "failed" for result in results) else "completed_with_failures"
    )
    manifest["completed"] = now()
    write_json(manifest_path, manifest)
    print(str(output_root), flush=True)


if __name__ == "__main__":
    main()
