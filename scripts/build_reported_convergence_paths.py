"""Build one plot-ready JSON file for the 24 reported convergence paths.

This is a data-packaging utility only.  It reads existing saved profiles and
one-start audits; it does not import or execute the model.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_ROOT = (
    ROOT / "outputs/demand_calibration/paper_profile_factorial_20260916_182347"
)
E1_ROOT = (
    ROOT
    / "outputs/old/demand_calibration/paper_profile_3x3_a030_20260916_110336"
    / "ch-af-apac-eu-row-us/cost"
)
OUTPUT = (
    ROOT
    / "outputs/equilibria/corrected_demand_20260916"
    / "convergence_paths_24_reported_runs.json"
)

THRESHOLD = 0.01
AUDIT_NAME = re.compile(
    r"audit_(?:(initial)|sweep_(\d{3}))_one_start\.json$"
)
REPORTED_FACTORIAL_EQUILIBRIA = {
    ("ch-row-apac-us-eu-af", "pf120_k050_a030"): "E2",
    ("ch-row-apac-us-eu-af", "pf120_k100_a030"): "E3",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def audit_sweep(path: Path) -> int:
    match = AUDIT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected audit filename: {path}")
    return 0 if match.group(1) else int(match.group(2))


def compact_player_audit(player: dict[str, Any]) -> dict[str, Any]:
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


def build_point(run_root: Path, audit_path: Path) -> dict[str, Any]:
    sweep = audit_sweep(audit_path)
    audit = read_json(audit_path)
    profile_path = (
        run_root / "initialization.json"
        if sweep == 0
        else run_root / f"sweep_{sweep:03d}.json"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)

    profile = read_json(profile_path)
    gain = audit.get("max_relative_gain")
    all_successful = bool(audit.get("all_attempts_successful"))
    verified = bool(audit.get("equilibrium_verified"))

    point: dict[str, Any] = {
        "sweep": sweep,
        "stage": "initialization" if sweep == 0 else "after_sweep",
        "profile_file": relative(profile_path),
        "audit_file": relative(audit_path),
        "audited_max_relative_gain": gain,
        "audited_max_relative_gain_percent": (
            None if gain is None else 100.0 * float(gain)
        ),
        "audit_max_gain_player": audit.get("max_gain_player"),
        "all_six_solves_successful": all_successful,
        "equilibrium_verified": verified,
        "valid_acceptance_point": all_successful and verified,
        "players": [compact_player_audit(row) for row in audit.get("players", [])],
        "max_sequential_relative_gain": None,
        "max_sequential_relative_gain_percent": None,
        "strategy_change_metric": None,
    }

    if sweep > 0:
        sequential_gain = profile.get("max_sequential_relative_gain")
        point["max_sequential_relative_gain"] = sequential_gain
        point["max_sequential_relative_gain_percent"] = (
            None if sequential_gain is None else 100.0 * float(sequential_gain)
        )
        point["strategy_change_metric"] = profile.get("strategy_change_metric")

    return point


def best_valid_point(points: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        point
        for point in points
        if point["all_six_solves_successful"]
        and point["audited_max_relative_gain"] is not None
    ]
    if not eligible:
        return None
    best = min(eligible, key=lambda point: point["audited_max_relative_gain"])
    return {
        "sweep": best["sweep"],
        "audited_max_relative_gain": best["audited_max_relative_gain"],
        "audited_max_relative_gain_percent": best[
            "audited_max_relative_gain_percent"
        ],
        "audit_max_gain_player": best["audit_max_gain_player"],
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
    *,
    source_experiment: str,
    reported_equilibrium_id: str | None,
    fallback_branch_specification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = read_json(run_root / "status.json")
    audit_paths = sorted(
        (run_root / "audits").glob("audit_*_one_start.json"), key=audit_sweep
    )
    if not audit_paths:
        raise RuntimeError(f"No one-start audits found in {run_root}")
    points = [build_point(run_root, path) for path in audit_paths]

    sequence = str(status["sequence"])
    branch = str(status["branch"])
    branch_specification = status.get("branch_specification")
    if branch_specification is None:
        branch_specification = fallback_branch_specification

    accepted = reported_equilibrium_id is not None
    selected_sweep = status.get("selected_sweep")
    if accepted and selected_sweep is None:
        selected_sweep = next(
            point["sweep"] for point in points if point["equilibrium_verified"]
        )
    selected_point = next(
        (point for point in points if point["sweep"] == selected_sweep), None
    )

    return {
        "run_id": reported_equilibrium_id or f"{sequence}__{branch}",
        "reported_equilibrium_id": reported_equilibrium_id,
        "plot_group": "accepted" if accepted else "completed_no_pass",
        "outcome": status.get("status"),
        "source_experiment": source_experiment,
        "source_directory": relative(run_root),
        "sequence": sequence,
        "player_order": status.get("player_order", sequence.split("-")),
        "source_iteration": status.get("source_iteration"),
        "branch": branch,
        "configuration": branch_specification,
        "alpha": status.get("alpha"),
        "algorithmic_proximal_penalties": status.get(
            "algorithmic_proximal_penalties", 0.0
        ),
        "accepted_sweep": selected_sweep if accepted else None,
        "status_selected_point": point_summary(selected_point),
        "terminal_sweep": points[-1]["sweep"],
        "path_point_count": len(points),
        "best_valid_point": best_valid_point(points),
        "path": points,
    }


def main() -> None:
    factorial_manifest = read_json(FACTORIAL_ROOT / "manifest.json")
    branch_specs = factorial_manifest["branches"]

    included_runs: list[dict[str, Any]] = []
    excluded_partial_runs: list[dict[str, Any]] = []

    for status_path in sorted(FACTORIAL_ROOT.glob("*/*/status.json")):
        run_root = status_path.parent
        status = read_json(status_path)
        key = (str(status["sequence"]), str(status["branch"]))
        equilibrium_id = REPORTED_FACTORIAL_EQUILIBRIA.get(key)
        if status.get("status") == "failed":
            failed_run = build_run(
                run_root,
                source_experiment="paper_profile_factorial_20260916_182347",
                reported_equilibrium_id=None,
                fallback_branch_specification=branch_specs.get(status["branch"]),
            )
            failed_run["plot_group"] = "excluded_partial_failure"
            failed_run["failure"] = {
                "error": status.get("error"),
                "elapsed_seconds": status.get("elapsed_seconds"),
            }
            excluded_partial_runs.append(failed_run)
            continue

        included_runs.append(
            build_run(
                run_root,
                source_experiment="paper_profile_factorial_20260916_182347",
                reported_equilibrium_id=equilibrium_id,
                fallback_branch_specification=branch_specs.get(status["branch"]),
            )
        )

    included_runs.append(
        build_run(
            E1_ROOT,
            source_experiment="paper_profile_3x3_a030_20260916_110336",
            reported_equilibrium_id="E1",
        )
    )
    included_runs.sort(
        key=lambda run: (
            run["plot_group"] != "accepted",
            run["run_id"],
        )
    )

    accepted_count = sum(run["plot_group"] == "accepted" for run in included_runs)
    no_pass_count = sum(
        run["plot_group"] == "completed_no_pass" for run in included_runs
    )
    if (len(included_runs), accepted_count, no_pass_count) != (24, 3, 21):
        raise RuntimeError(
            "Expected 24 included paths (3 accepted, 21 completed no-pass); "
            f"found {len(included_runs)}, {accepted_count}, {no_pass_count}"
        )
    if len(excluded_partial_runs) != 1:
        raise RuntimeError(
            f"Expected one excluded partial failure; found {len(excluded_partial_runs)}"
        )

    payload = {
        "schema_version": "1.0",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": "Corrected-demand audited convergence paths",
        "description": (
            "Plot-ready audit trajectories for the 21 completed factorial branches "
            "that did not pass and the three reported accepted equilibria E1-E3."
        ),
        "metric": {
            "primary": "audited_max_relative_gain",
            "definition": (
                "Maximum normalized unilateral profit gain in the saved one-start, "
                "common-frozen, zero-proximal six-player audit."
            ),
            "acceptance_threshold": THRESHOLD,
            "acceptance_threshold_percent": 100.0 * THRESHOLD,
            "criterion": (
                "All six one-start solves successful and maximum normalized "
                "relative gain <= 1%."
            ),
        },
        "accounting": {
            "included_plot_paths": 24,
            "included_completed_no_pass": no_pass_count,
            "included_accepted": accepted_count,
            "factorial_total_branches": 24,
            "factorial_completed_no_pass": 21,
            "factorial_accepted": 2,
            "factorial_failed_partial": 1,
            "earlier_accepted_path_added": "E1",
            "note": (
                "E2 and E3 are accepted factorial branches. E1 is the accepted "
                "path from the earlier 3x3 alpha=0.30 experiment. The factorial "
                "branch that failed during sweep 12 is retained under "
                "excluded_partial_runs and is not counted among the 24 plotting paths."
            ),
            "data_quality_note": (
                "best_valid_point is recomputed using only audits in which all six "
                "solves succeeded. This matters for one no-pass branch whose status "
                "selected a sub-1% audit with a failed solve."
            ),
        },
        "source_artifacts": {
            "factorial_manifest": relative(FACTORIAL_ROOT / "manifest.json"),
            "factorial_protocol": relative(FACTORIAL_ROOT / "PROTOCOL.md"),
            "reported_equilibria_package": (
                "outputs/equilibria/corrected_demand_20260916"
            ),
        },
        "runs": included_runs,
        "excluded_partial_runs": excluded_partial_runs,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(relative(OUTPUT))


if __name__ == "__main__":
    main()
