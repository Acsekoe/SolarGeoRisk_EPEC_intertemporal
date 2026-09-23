from __future__ import annotations

"""Continue selected profiles under the no-mu, no-penalty objective."""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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

from model import data_prep, model_main as mm, run_gs
from scripts.audit_selected_equilibrium import _strategy_distance, _zero_prox_data
from scripts.continue_selected_equilibrium import _initial_model_data
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.run_local_paper_equilibrium_experiment import clone_state, full_state_payload
from scripts.search_nested_equilibrium import _deserialize_state, _sync_quantity


CANDIDATE_ROOT = ROOT / "outputs" / "new_equilibria" / "candidates"
DEFAULT_PROFILES = (
    "ch-af-apac-eu-row-us/pf080_k100_a030",
    "af-eu-us-apac-row-ch/pf100_k050_a040",
    "eu-us-af-row-apac-ch/pf100_k050_a030",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def clean_data_and_state(profile_id: str):
    profile_path = CANDIDATE_ROOT / profile_id / "profile.json"
    document = json.loads(profile_path.read_text(encoding="utf-8"))

    # Preserve the explicit demand coefficients with which these profiles were
    # generated while bypassing the later p_full/elasticity metadata guard.
    original_tolerance = data_prep._DEMAND_CALIBRATION_ABS_TOL
    data_prep._DEMAND_CALIBRATION_ABS_TOL = float("inf")
    try:
        _, base_cfg, template = _initial_model_data()
        salvage_fraction = float(document.get("terminal_salvage_fraction", 0.0))
        data = _zero_prox_data(
            base_cfg,
            terminal_salvage_fraction=salvage_fraction,
        )
    finally:
        data_prep._DEMAND_CALIBRATION_ABS_TOL = original_tolerance

    clean_cfg = replace(
        base_cfg,
        objective_mode=run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
        terminal_salvage_fraction=salvage_fraction,
        terminal_capacity_state_only=True,
    )
    run_gs._apply_data_overrides(data, clean_cfg)
    state = _deserialize_state(
        document["ending_profile"]["strategy"],
        data,
        template,
    )
    return data, state, document, profile_path


def update_player(
    data: mm.ModelData,
    state: dict[str, dict],
    response: dict[str, dict],
    player: str,
    alpha: float,
) -> None:
    for tp in mm._move_times(list(data.times or [])):
        key = (player, tp)
        state["dK_net"][key] = (
            (1.0 - alpha) * float(state["dK_net"][key])
            + alpha * float(response["dK_net"][key])
        )
    for importer in data.regions:
        if importer == player:
            continue
        for tp in mm._operating_times(data):
            key = (player, importer, tp)
            state["p_offer"][key] = (
                (1.0 - alpha) * float(state["p_offer"][key])
                + alpha * float(response["p_offer"][key])
            )
    _sync_quantity(data, state)


def common_objectives(
    data: mm.ModelData,
    state: dict[str, dict],
) -> tuple[dict[str, float], dict[str, dict], dict[str, float]]:
    market, diagnostics = solve_nested_market(data, state)
    values = {
        player: float(nested_economic_objective(data, state, market, player))
        for player in data.players
    }
    return values, market, diagnostics


def final_audit(
    data: mm.ModelData,
    state: dict[str, dict],
    *,
    maxiter: int,
) -> dict[str, Any]:
    market, market_diagnostics = solve_nested_market(data, state)
    rows: list[dict[str, Any]] = []
    for player in data.players:
        reference = float(nested_economic_objective(data, state, market, player))
        best, response, diagnostics = nested_best_response(
            data,
            state,
            market,
            player,
            maxiter=maxiter,
            starts=1,
        )
        gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
        move, coordinate, absolute = _strategy_distance(
            data, state, response, player
        )
        rows.append(
            {
                "player": player,
                "reference_objective": reference,
                "best_response_objective": float(best),
                "relative_gain": gain,
                "relative_gain_percent": 100.0 * gain,
                "strategy_move": float(move),
                "largest_move_coordinate": coordinate,
                "largest_move_absolute": float(absolute),
                "optimizer_success": bool(diagnostics["success"]),
                "all_attempts_successful": all(
                    bool(attempt["success"])
                    for attempt in diagnostics["attempts"]
                ),
                "attempts": diagnostics["attempts"],
            }
        )
    limiting = max(rows, key=lambda row: float(row["relative_gain"]))
    all_successful = all(bool(row["all_attempts_successful"]) for row in rows)
    return {
        "created": now(),
        "common_profile_frozen": True,
        "starts_per_best_response": 1,
        "maxiter": maxiter,
        "objective_mode": run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
        "relative_gain_tolerance": 0.01,
        "reference_market_diagnostics": market_diagnostics,
        "players": rows,
        "max_relative_gain": float(limiting["relative_gain"]),
        "max_gain_player": str(limiting["player"]),
        "all_attempts_successful": all_successful,
        "equilibrium_verified": bool(
            all_successful and float(limiting["relative_gain"]) <= 0.01
        ),
    }


def run_profile(task: dict[str, Any]) -> dict[str, Any]:
    profile_id = str(task["profile_id"])
    output_root = Path(task["output_root"])
    branch_root = output_root / profile_id
    status_path = branch_root / "status.json"
    started = time.perf_counter()
    try:
        branch_root.mkdir(parents=True, exist_ok=False)
        data, state, document, profile_path = clean_data_and_state(profile_id)
        order = [str(player) for player in document["player_order"]]
        alpha = float(document["alpha"])
        source_sweep = int(document.get("sweep", 0))
        initial = clone_state(state)
        initial_objectives, initial_market, initial_market_diagnostics = common_objectives(
            data, state
        )
        write_json(
            branch_root / "initialization.json",
            {
                "created": now(),
                "profile_id": profile_id,
                "source_profile": str(profile_path.relative_to(ROOT)),
                "source_sweep": source_sweep,
                "player_order": order,
                "alpha": alpha,
                "objective_mode": run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
                "subtract_mu_offer_from_producer_margin": False,
                "economic_quadratic_penalties": 0.0,
                "algorithmic_proximal_penalties": 0.0,
                "terminal_salvage_fraction": float(
                    document.get("terminal_salvage_fraction", 0.0)
                ),
                "terminal_capacity_state_only": True,
                "common_objectives": initial_objectives,
                "market_diagnostics": initial_market_diagnostics,
                "profile": full_state_payload(data, state, initial_market),
            },
        )

        sweep_rows: list[dict[str, Any]] = []
        for continuation_sweep in range(1, int(task["sweeps"]) + 1):
            before_sweep = clone_state(state)
            starting_objectives, _, starting_market_diagnostics = common_objectives(
                data, before_sweep
            )
            player_rows: list[dict[str, Any]] = []
            for player in order:
                before_player = clone_state(state)
                market, market_diagnostics = solve_nested_market(data, state)
                reference = float(
                    nested_economic_objective(data, state, market, player)
                )
                best, response, diagnostics = nested_best_response(
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
                    best, response, diagnostics = nested_best_response(
                        data,
                        state,
                        market,
                        player,
                        maxiter=max(2 * int(task["maxiter"]), 1000),
                        starts=1,
                    )
                if not diagnostics["success"]:
                    raise RuntimeError(
                        f"continuation sweep {continuation_sweep} {player}: "
                        f"best-response solve failed: {diagnostics['message']}"
                    )
                gain = max(float(best) - reference, 0.0) / max(
                    abs(reference), 1.0
                )
                raw_move = float(
                    _strategy_distance(data, before_player, response, player)[0]
                )
                update_player(data, state, response, player, alpha)
                applied_move = float(
                    _strategy_distance(data, before_player, state, player)[0]
                )
                player_rows.append(
                    {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best),
                        "relative_gain": gain,
                        "relative_gain_percent": 100.0 * gain,
                        "raw_strategy_move": raw_move,
                        "applied_strategy_move": applied_move,
                        "alpha": alpha,
                        "optimizer_success": bool(diagnostics["success"]),
                        "retry_used": retry_used,
                        "attempts": diagnostics["attempts"],
                        "reference_market_diagnostics": market_diagnostics,
                    }
                )

            ending_objectives, ending_market, ending_market_diagnostics = common_objectives(
                data, state
            )
            strategy_change = max(
                float(_strategy_distance(data, before_sweep, state, player)[0])
                for player in order
            )
            objective_change = max(
                abs(ending_objectives[player] - starting_objectives[player])
                / max(abs(starting_objectives[player]), 1.0)
                for player in order
            )
            max_sequential_gain = max(
                float(row["relative_gain"]) for row in player_rows
            )
            limiting_player = max(
                player_rows, key=lambda row: float(row["relative_gain"])
            )["player"]
            sweep_payload = {
                "created": now(),
                "profile_id": profile_id,
                "continuation_sweep": continuation_sweep,
                "absolute_sweep_label": source_sweep + continuation_sweep,
                "player_order": order,
                "alpha": alpha,
                "objective_mode": run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
                "strategy_change_metric": strategy_change,
                "objective_change_metric": objective_change,
                "max_sequential_relative_gain": max_sequential_gain,
                "max_sequential_gain_player": limiting_player,
                "starting_common_objectives": starting_objectives,
                "ending_common_objectives": ending_objectives,
                "players": player_rows,
                "starting_market_diagnostics": starting_market_diagnostics,
                "ending_market_diagnostics": ending_market_diagnostics,
                "ending_profile": full_state_payload(data, state, ending_market),
            }
            write_json(
                branch_root / f"sweep_{continuation_sweep:03d}.json",
                sweep_payload,
            )
            sweep_rows.append(
                {
                    "profile_id": profile_id,
                    "continuation_sweep": continuation_sweep,
                    "absolute_sweep_label": source_sweep + continuation_sweep,
                    "strategy_change_metric": strategy_change,
                    "objective_change_metric": objective_change,
                    "max_sequential_relative_gain": max_sequential_gain,
                    "max_sequential_gain_player": limiting_player,
                }
            )
            print(
                f"[CLEAN {profile_id}] sweep {continuation_sweep:02d}/"
                f"{task['sweeps']}: move={strategy_change:.4%} "
                f"objective_change={objective_change:.4%} "
                f"max_sequential_gain={max_sequential_gain:.4%} "
                f"({limiting_player})",
                flush=True,
            )
            write_json(
                status_path,
                {
                    "updated": now(),
                    "status": "running",
                    "profile_id": profile_id,
                    "completed_sweeps": continuation_sweep,
                    "requested_sweeps": int(task["sweeps"]),
                    "latest": sweep_rows[-1],
                },
            )

        audit = final_audit(data, state, maxiter=int(task["maxiter"]))
        write_json(branch_root / "final_audit_one_start.json", audit)
        initial_to_final_change = max(
            float(_strategy_distance(data, initial, state, player)[0])
            for player in order
        )
        result = {
            "created": now(),
            "status": "completed",
            "profile_id": profile_id,
            "source_sweep": source_sweep,
            "continued_sweeps": int(task["sweeps"]),
            "player_order": order,
            "alpha": alpha,
            "objective_mode": run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
            "initial_to_final_strategy_change": initial_to_final_change,
            "first_strategy_change_metric": float(
                sweep_rows[0]["strategy_change_metric"]
            ),
            "final_strategy_change_metric": float(
                sweep_rows[-1]["strategy_change_metric"]
            ),
            "minimum_strategy_change_metric": min(
                float(row["strategy_change_metric"]) for row in sweep_rows
            ),
            "last_three_max_strategy_change_metric": max(
                float(row["strategy_change_metric"]) for row in sweep_rows[-3:]
            ),
            "first_max_sequential_relative_gain": float(
                sweep_rows[0]["max_sequential_relative_gain"]
            ),
            "final_max_sequential_relative_gain": float(
                sweep_rows[-1]["max_sequential_relative_gain"]
            ),
            "final_audit_max_relative_gain": float(audit["max_relative_gain"]),
            "final_audit_max_gain_player": audit["max_gain_player"],
            "final_audit_all_attempts_successful": audit["all_attempts_successful"],
            "final_equilibrium_verified": audit["equilibrium_verified"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        result = {
            "created": now(),
            "status": "failed",
            "profile_id": profile_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    parser.add_argument("--sweeps", type=int, default=15)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.sweeps < 1 or args.maxiter < 1 or args.workers < 1:
        raise ValueError("sweeps, maxiter, and workers must be positive")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (
        ROOT
        / "outputs"
        / "new_equilibria"
        / f"clean_objective_continuation_{stamp}"
    )
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "created": now(),
        "status": "running",
        "profiles": list(args.profiles),
        "continued_sweeps": args.sweeps,
        "maxiter": args.maxiter,
        "workers": min(args.workers, len(args.profiles)),
        "objective_mode": run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES,
        "objective_changes": {
            "subtract_mu_offer_from_producer_margin": False,
            "economic_quadratic_penalties": 0.0,
            "algorithmic_proximal_penalties": 0.0,
        },
        "terminal_capacity_state_only": True,
        "results": [],
    }
    write_json(manifest_path, manifest)

    tasks = [
        {
            "profile_id": profile_id,
            "output_root": str(output_root),
            "sweeps": args.sweeps,
            "maxiter": args.maxiter,
        }
        for profile_id in args.profiles
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        futures = {pool.submit(run_profile, task): task["profile_id"] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            manifest["results"] = sorted(results, key=lambda row: row["profile_id"])
            manifest["status"] = (
                "failed" if any(row["status"] == "failed" for row in results) else "running"
            )
            write_json(manifest_path, manifest)
            print(
                f"[CLEAN {result['profile_id']}] {result['status']}",
                flush=True,
            )

    results.sort(key=lambda row: row["profile_id"])
    manifest["results"] = results
    manifest["status"] = (
        "completed"
        if all(row["status"] == "completed" for row in results)
        else "failed"
    )
    manifest["completed"] = now()
    write_json(manifest_path, manifest)

    fields = [
        "profile_id",
        "status",
        "alpha",
        "first_strategy_change_metric",
        "final_strategy_change_metric",
        "minimum_strategy_change_metric",
        "last_three_max_strategy_change_metric",
        "first_max_sequential_relative_gain",
        "final_max_sequential_relative_gain",
        "final_audit_max_relative_gain",
        "final_audit_max_gain_player",
        "final_audit_all_attempts_successful",
        "final_equilibrium_verified",
        "elapsed_seconds",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[CLEAN] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
