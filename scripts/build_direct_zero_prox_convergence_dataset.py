"""Package the saved 28-branch direct-grid convergence record as strict JSON.

This utility only reads existing profiles, audits, statuses, and the archived
post-run analysis.  It does not import or execute the economic model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_RELATIVE = Path(
    "outputs/old/demand_calibration/direct_zero_prox_grid_20260915_232405"
)
OUTPUT = (
    ROOT
    / "outputs/equilibria"
    / "direct_zero_prox_28_run_convergence_20260915.json"
)
AUDIT_NAME = re.compile(
    r"audit_(?:(initial)|sweep_(\d{3}))_one_start\.json$"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def source_path(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def audit_sweep(path: Path) -> int:
    match = AUDIT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected audit filename: {path}")
    return 0 if match.group(1) else int(match.group(2))


def audit_objectives(audit: dict[str, Any]) -> dict[str, float]:
    return {
        str(player["player"]): float(player["reference_objective"])
        for player in audit.get("players", [])
        if player.get("reference_objective") is not None
    }


def objective_change(
    current: dict[str, float], previous: dict[str, float]
) -> float | None:
    players = sorted(set(current) & set(previous))
    if not players:
        return None
    return max(
        abs(current[player] - previous[player]) / max(abs(previous[player]), 1.0)
        for player in players
    )


def compact_player(player: dict[str, Any]) -> dict[str, Any]:
    gain = player.get("relative_gain")
    return {
        "player": player.get("player"),
        "relative_gain": gain,
        "relative_gain_percent": None if gain is None else 100.0 * float(gain),
        "reference_objective": player.get("reference_objective"),
        "best_response_objective": player.get("best_response_objective"),
        "strategy_move": player.get("strategy_move"),
        "optimizer_success": player.get("optimizer_success"),
        "chosen_start_index": player.get("chosen_start_index"),
    }


def point_summary(point: dict[str, Any] | None) -> dict[str, Any] | None:
    if point is None:
        return None
    return {
        "sweep": point["sweep"],
        "audited_max_relative_gain": point["audited_max_relative_gain"],
        "audited_max_relative_gain_percent": point[
            "audited_max_relative_gain_percent"
        ],
        "audit_max_gain_player": point["audit_max_gain_player"],
        "all_six_solves_successful": point["all_six_solves_successful"],
        "equilibrium_verified": point["equilibrium_verified"],
    }


def build_run(
    run_root: Path,
    branch_root: Path,
    source_root: Path,
    analysis: dict[str, Any],
    max_sweeps: int,
) -> dict[str, Any]:
    status = read_json(branch_root / "status.json")
    initialization = read_json(branch_root / "initialization.json")
    audit_paths = sorted(
        (branch_root / "audits").glob("audit_*_one_start.json"), key=audit_sweep
    )
    if not audit_paths:
        raise RuntimeError(f"No audits found in {branch_root}")

    points: list[dict[str, Any]] = []
    previous_objectives: dict[str, float] = {}
    for audit_path in audit_paths:
        sweep = audit_sweep(audit_path)
        profile_path = (
            branch_root / "initialization.json"
            if sweep == 0
            else branch_root / f"sweep_{sweep:03d}.json"
        )
        if not profile_path.is_file():
            raise FileNotFoundError(profile_path)
        audit = read_json(audit_path)
        profile = read_json(profile_path)
        objectives = audit_objectives(audit)
        change = (
            objective_change(objectives, previous_objectives)
            if previous_objectives
            else None
        )
        previous_objectives = objectives or previous_objectives
        gain = audit.get("max_relative_gain")
        sequential_gain = profile.get("max_sequential_relative_gain")
        diagnostics = (
            profile.get("ending_market_diagnostics")
            or profile.get("market_diagnostics")
            or audit.get("reference_market_diagnostics")
            or {}
        )
        points.append(
            {
                "sweep": sweep,
                "stage": "initialization" if sweep == 0 else "after_sweep",
                "profile_file": source_path(profile_path, source_root),
                "audit_file": source_path(audit_path, source_root),
                "audited_max_relative_gain": gain,
                "audited_max_relative_gain_percent": (
                    None if gain is None else 100.0 * float(gain)
                ),
                "audit_max_gain_player": audit.get("max_gain_player"),
                "all_six_solves_successful": bool(
                    audit.get("all_attempts_successful")
                ),
                "equilibrium_verified": bool(audit.get("equilibrium_verified")),
                "strategy_change_metric": profile.get("strategy_change_metric"),
                "reference_objective_change": change,
                "max_sequential_relative_gain": sequential_gain,
                "max_sequential_relative_gain_percent": (
                    None
                    if sequential_gain is None
                    else 100.0 * float(sequential_gain)
                ),
                "market_diagnostics": diagnostics,
                "players": [
                    compact_player(player) for player in audit.get("players", [])
                ],
            }
        )

    eligible = [
        point
        for point in points
        if point["all_six_solves_successful"]
        and point["audited_max_relative_gain"] is not None
    ]
    best_point = min(
        eligible, key=lambda point: point["audited_max_relative_gain"]
    )
    selected_sweep = status.get("selected_sweep")
    selected_point = next(
        (point for point in points if point["sweep"] == selected_sweep), None
    )
    relative_branch = branch_root.relative_to(run_root).as_posix()

    return {
        "run_id": relative_branch,
        "outcome": status.get("status"),
        "classification": analysis.get("classification"),
        "termination_reason": analysis.get("termination_reason"),
        "source_directory": source_path(branch_root, source_root),
        "order": status.get("order_name", analysis.get("order")),
        "player_order": status.get(
            "player_order", analysis.get("player_order", [])
        ),
        "start": status.get("start_name", analysis.get("start")),
        "price_factor": status.get(
            "price_factor", initialization.get("price_factor")
        ),
        "alpha": status.get("alpha", initialization.get("alpha")),
        "max_sweeps": max_sweeps,
        "completed_sweeps": points[-1]["sweep"],
        "accepted_sweep": selected_sweep if status.get("status") == "accepted" else None,
        "status_selected_point": point_summary(selected_point),
        "best_valid_point": point_summary(best_point),
        "runtime_seconds": status.get(
            "elapsed_seconds", analysis.get("runtime_seconds")
        ),
        "initialization": {
            "price_initialization": initialization.get("price_initialization"),
            "capacity_initialization": initialization.get(
                "capacity_initialization"
            ),
            "demand_initialization": initialization.get("demand_initialization"),
            "initial_state_sha256": initialization.get("initial_state_sha256"),
            "stage1_profile_used": bool(
                initialization.get("stage1_profile_used", False)
            ),
            "source_workbook_used": bool(
                initialization.get("source_workbook_used", False)
            ),
        },
        "algorithm": {
            "algorithmic_proximal_penalties": initialization.get(
                "algorithmic_proximal_penalties", 0.0
            ),
            "move_cap": initialization.get("move_cap"),
            "players_frozen": initialization.get("players_frozen", False),
            "multistart_used": initialization.get("multistart_used", False),
        },
        "postrun_diagnostics": {
            "all_saved_audits_successful": analysis.get(
                "all_saved_audits_successful"
            ),
            "cycle": analysis.get("cycle"),
            "tail_movement_driver": analysis.get("tail_movement_driver"),
            "feasibility": analysis.get("feasibility"),
            "final_strategy_change": analysis.get("final_strategy_change"),
            "final_objective_change": analysis.get("final_objective_change"),
            "minimum_strategy_change": analysis.get("minimum_strategy_change"),
            "minimum_strategy_sweep": analysis.get("minimum_strategy_sweep"),
            "minimum_objective_change": analysis.get("minimum_objective_change"),
            "minimum_objective_sweep": analysis.get("minimum_objective_sweep"),
            "solver_failures": analysis.get("solver_failures"),
        },
        "path_point_count": len(points),
        "path": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT / "_MOVE",
        help=(
            "Directory containing outputs/old/.../direct_zero_prox_grid. "
            "After relocation, pass the external archive handoff directory."
        ),
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    run_root = source_root / RUN_RELATIVE
    manifest = read_json(run_root / "manifest.json")
    postrun = read_json(run_root / "postrun_convergence_analysis.json")
    analysis_by_branch = {
        str(row["branch"]): row for row in postrun.get("inventory", [])
    }

    status_paths = sorted(run_root.glob("*/*/status.json"))
    runs: list[dict[str, Any]] = []
    for status_path in status_paths:
        branch_root = status_path.parent
        branch_name = branch_root.relative_to(run_root).as_posix()
        if branch_name not in analysis_by_branch:
            raise RuntimeError(f"Missing post-run analysis for {branch_name}")
        runs.append(
            build_run(
                run_root,
                branch_root,
                source_root,
                analysis_by_branch[branch_name],
                int(manifest["max_sweeps"]),
            )
        )

    counts = {
        outcome: sum(run["outcome"] == outcome for run in runs)
        for outcome in ("accepted", "no_pass_within_schedule", "failed")
    }
    if (len(runs), counts) != (
        28,
        {"accepted": 1, "no_pass_within_schedule": 17, "failed": 10},
    ):
        raise RuntimeError(f"Unexpected run accounting: {len(runs)} {counts}")
    accepted = [run for run in runs if run["outcome"] == "accepted"]
    if len(accepted) != 1 or accepted[0]["accepted_sweep"] != 10:
        raise RuntimeError("Expected the sole accepted profile at sweep 10")
    accepted_point = next(
        point
        for point in accepted[0]["path"]
        if point["sweep"] == accepted[0]["accepted_sweep"]
    )

    payload = sanitize(
        {
            "schema_version": "1.0",
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "title": "Direct-from-primitives zero-proximal 28-run convergence record",
            "description": (
                "Self-contained, plot-ready convergence trajectories for the "
                "28-branch direct zero-proximal Gauss--Seidel control experiment."
            ),
            "source_experiment": {
                "id": "direct_zero_prox_grid_20260915_232405",
                "historical_root": RUN_RELATIVE.as_posix(),
                "manifest": source_path(run_root / "manifest.json", source_root),
                "manifest_sha256": sha256(run_root / "manifest.json"),
                "postrun_analysis": source_path(
                    run_root / "postrun_convergence_analysis.json", source_root
                ),
                "postrun_analysis_sha256": sha256(
                    run_root / "postrun_convergence_analysis.json"
                ),
                "corrected_input_sha256": manifest.get("input_sha256"),
                "code_sha256": manifest.get("code_sha256"),
                "path_note": (
                    "Paths are historical archive-relative provenance paths. All "
                    "quantities needed for convergence plots are embedded here."
                ),
            },
            "method": {
                "name": manifest.get("method"),
                "stage1_profile_used": manifest.get("stage1_profile_used"),
                "source_workbook_used": manifest.get("source_workbook_used"),
                "algorithmic_proximal_penalties": manifest.get(
                    "algorithmic_proximal_penalties"
                ),
                "move_cap": manifest.get("move_cap"),
                "players_frozen": manifest.get("players_frozen"),
                "multistart_used": manifest.get("multistart_used"),
                "orders": manifest.get("orders"),
                "starts": manifest.get("starts"),
                "alphas": manifest.get("alphas"),
                "max_sweeps": manifest.get("max_sweeps"),
                "maxiter": manifest.get("maxiter"),
            },
            "acceptance": {
                "criterion": manifest.get("acceptance_criterion"),
                "relative_gain_threshold": 0.01,
                "relative_gain_threshold_percent": 1.0,
                "audit_starts": manifest.get("acceptance_audit_starts"),
                "claim_scope": (
                    "Local one-start computational 1%-equilibrium criterion; "
                    "not a global Nash certificate or fixed-point convergence proof."
                ),
            },
            "accounting": {
                "configured_branches": len(runs),
                "accepted": counts["accepted"],
                "exhausted_80_sweeps_without_pass": counts[
                    "no_pass_within_schedule"
                ],
                "failed_before_schedule_end": counts["failed"],
                "saved_path_points": sum(run["path_point_count"] for run in runs),
                "classification_counts": postrun.get("classification_counts"),
                "interpretation": (
                    "One intermediate profile passed the audit at sweep 10, but "
                    "no branch established sustained fixed-point convergence."
                ),
            },
            "accepted_candidate": point_summary(accepted_point)
            | {
                "run_id": accepted[0]["run_id"],
                "profile_file": accepted_point["profile_file"],
                "audit_file": accepted_point["audit_file"],
            },
            "summaries": {
                "alpha": postrun.get("alpha_summary"),
                "order": postrun.get("order_summary"),
                "methodological_assessment": postrun.get(
                    "methodological_assessment"
                ),
            },
            "runs": runs,
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
