from __future__ import annotations

"""Audit the seven Figure-2 Stage-1 player-order endpoints.

For every player order, the script selects the longest contiguous prefix of
Stage-1 sweeps for which every regional MPEC solve was acceptable.  It then
reconstructs the exact damped strategy profile and evaluates unilateral,
zero-proximal best responses against one common frozen profile.

One three-start audit supplies two diagnostics:

* ``candidate-start`` uses only the optimizer attempt initialized at the
  candidate strategy; and
* ``best-of-three`` takes the best response found from all three declared
  optimizer starts.

Neither diagnostic is a global Nash-equilibrium certificate.
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

import pandas as pd
import matplotlib.pyplot as plt


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
from scripts.run_corrected_equilibrium_search import (
    base_configuration,
    relative,
    sha256,
    source_workbook,
    terminal_source_iteration,
    write_json,
)


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


def _source_root(slug: str, existing_root: Path, additional_root: Path) -> Path:
    candidates = [root for root in (existing_root, additional_root) if (root / slug).is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage-1 source root for {slug}; found {candidates}"
        )
    return candidates[0]


def _movement_record(workbook: Path, clean_iteration: int) -> dict[str, Any]:
    iterations = pd.read_excel(workbook, sheet_name="iters").sort_values("iter")
    terminal_iteration = int(iterations["iter"].max())
    row = iterations.loc[iterations["iter"] == clean_iteration].iloc[-1]
    stable_count = int(row.get("stable_count", 0) or 0)
    failures = iterations.loc[
        ~iterations["all_solves_acceptable"].fillna(False).astype(bool),
        ["iter", "solve_failures"],
    ]
    return {
        "recorded_terminal_iteration": terminal_iteration,
        "selected_clean_iteration": clean_iteration,
        "selected_is_recorded_terminal": clean_iteration == terminal_iteration,
        "movement_converged": stable_count >= 3,
        "stable_count": stable_count,
        "damped_strategy_movement": float(row["r_strat"]),
        "raw_best_response_movement": float(row["r_raw_br"]),
        "all_selected_prefix_solves_acceptable": True,
        "first_unacceptable_sweep": (
            None
            if failures.empty
            else {
                "iteration": int(failures.iloc[0]["iter"]),
                "failures": str(failures.iloc[0]["solve_failures"]),
            }
        ),
    }


def _candidate_start_result(audit: dict[str, Any], threshold: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_successful = True
    for player in audit["players"]:
        attempt = player["attempts"][0]
        successful = bool(attempt["success"])
        all_successful = all_successful and successful
        reference = float(player["reference_objective"])
        objective = float(attempt["objective"])
        gain = max(objective - reference, 0.0) / max(abs(reference), 1.0)
        rows.append(
            {
                "player": str(player["player"]),
                "reference_objective": reference,
                "best_response_objective": objective,
                "relative_gain": gain,
                "optimizer_success": successful,
                "start_label": str(attempt["start_label"]),
            }
        )
    maximum = max(rows, key=lambda row: float(row["relative_gain"]))
    return {
        "starts": 1,
        "start_definition": "candidate strategy",
        "all_attempts_successful": all_successful,
        "max_relative_gain": float(maximum["relative_gain"]),
        "max_gain_player": str(maximum["player"]),
        "relative_gain_tolerance": threshold,
        "equilibrium_verified": bool(
            all_successful and float(maximum["relative_gain"]) <= threshold
        ),
        "players": rows,
    }


def audit_order(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    slug = str(task["slug"])
    order = list(task["order"])
    input_path = Path(task["input_path"]).resolve()
    source_root = Path(task["source_root"]).resolve()
    output_root = Path(task["output_root"]).resolve()
    threshold = float(task["threshold"])
    starts = int(task["starts"])
    maxiter = int(task["maxiter"])
    status_path = output_root / slug / "status.json"
    audit_path = output_root / slug / f"audit_{starts}_start.json"

    try:
        workbook = source_workbook(source_root, slug)
        clean_iteration = terminal_source_iteration(workbook)
        cfg = base_configuration(input_path)
        replay_data = load_data_from_excel(
            str(input_path), params_region_sheet="params_region_new"
        )
        run_gs._apply_data_overrides(replay_data, cfg)
        fresh_state = _build_fresh_state(replay_data)
        candidate, replay_error = continuation.replay_accepted_state(
            workbook,
            fresh_state,
            replay_data,
            through_iteration=clean_iteration,
            expected_player_order=order,
        )
        if replay_error > 1e-10:
            raise RuntimeError(f"{slug}: replay residual {replay_error:.3g}")

        data = local_search._zero_prox_data(
            cfg, terminal_salvage_fraction=float(task["terminal_salvage_fraction"])
        )
        overnight._sync_quantity(data, candidate)
        local_search.PLAYER_ORDER = order
        audit = local_search.audit_profile(
            data,
            candidate,
            starts=starts,
            maxiter=maxiter,
            label=f"stage1_{slug}_sweep_{clean_iteration}_{starts}_start",
            output=audit_path,
        )
        candidate_start = _candidate_start_result(audit, threshold)
        best_of_starts_verified = bool(
            audit["all_attempts_successful"]
            and float(audit["max_relative_gain"]) <= threshold
        )
        record = {
            "created": now(),
            "status": "completed",
            "slug": slug,
            "player_order": order,
            "source_root": relative(source_root),
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_replay_error": replay_error,
            "terminal_salvage_fraction": float(task["terminal_salvage_fraction"]),
            "movement": _movement_record(workbook, clean_iteration),
            "candidate_start": candidate_start,
            "best_of_starts": {
                "starts": starts,
                "all_attempts_successful": bool(audit["all_attempts_successful"]),
                "max_relative_gain": float(audit["max_relative_gain"]),
                "max_gain_player": str(audit["max_gain_player"]),
                "relative_gain_tolerance": threshold,
                "equilibrium_verified": best_of_starts_verified,
            },
            "audit_path": relative(audit_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, record)
        return record
    except Exception as exc:
        failure = {
            "created": now(),
            "status": "failed",
            "slug": slug,
            "player_order": order,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def _pct(value: float) -> str:
    return f"{100.0 * value:.6f}%"


def results_markdown(records: list[dict[str, Any]], threshold: float) -> str:
    complete = [record for record in records if record["status"] == "completed"]
    movement_passes = sum(bool(record["movement"]["movement_converged"]) for record in complete)
    one_passes = sum(bool(record["candidate_start"]["equilibrium_verified"]) for record in complete)
    three_passes = sum(bool(record["best_of_starts"]["equilibrium_verified"]) for record in complete)
    lines = [
        "# Seven-order Stage-1 equilibrium audit",
        "",
        f"Created: {now()}.",
        "",
        "## Outcome",
        "",
        f"- Completed order audits: {len(complete)} of {len(records)}.",
        f"- Stage-1 movement passes: {movement_passes}.",
        f"- Candidate-start local {_pct(threshold)} passes: {one_passes}.",
        f"- Best-of-three local {_pct(threshold)} passes: {three_passes}.",
        "",
        "Movement convergence is a Stage-1 stabilization diagnostic only. Equilibrium",
        "status is evaluated independently with zero proximal penalties against one",
        "common frozen profile. The best-of-three test is still local and is not a",
        "global Nash-equilibrium certificate.",
        "",
        "## Player-order comparison",
        "",
        "| Player order | Selected sweep | Movement | Damped / raw movement | Candidate-start gain | Best-of-three gain | Result |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for record in records:
        if record["status"] != "completed":
            lines.append(
                f"| {record['slug'].upper()} | -- | failed | -- | -- | -- | {record['error']} |"
            )
            continue
        movement = record["movement"]
        one = record["candidate_start"]
        three = record["best_of_starts"]
        result = (
            "best-of-three pass"
            if three["equilibrium_verified"]
            else "candidate-start only"
            if one["equilibrium_verified"]
            else "not an audited 1% equilibrium"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    record["slug"].upper(),
                    str(movement["selected_clean_iteration"]),
                    "pass" if movement["movement_converged"] else "no pass",
                    f"{_pct(movement['damped_strategy_movement'])} / {_pct(movement['raw_best_response_movement'])}",
                    f"{_pct(one['max_relative_gain'])} ({one['max_gain_player'].upper()})",
                    f"{_pct(three['max_relative_gain'])} ({three['max_gain_player'].upper()})",
                    result,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "A Stage-1 endpoint is never accepted because its damped movement is small.",
            "Only a successful common-frozen-profile audit with maximum normalized",
            "unilateral gain at or below the declared tolerance passes the computational",
            "equilibrium test.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolved_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_figures(
    records: list[dict[str, Any]], output_root: Path, threshold: float
) -> None:
    complete = [record for record in records if record["status"] == "completed"]
    path_rows: list[pd.DataFrame] = []
    colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#666666",
        "#888888",
        "#aaaaaa",
        "#c0c0c0",
    ]
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    for index, record in enumerate(complete):
        workbook = _resolved_record_path(str(record["source_workbook"]))
        iterations = pd.read_excel(workbook, sheet_name="iters")
        selected = int(record["movement"]["selected_clean_iteration"])
        iterations = iterations.loc[iterations["iter"] <= selected].copy()
        iterations.insert(0, "player_order", record["slug"])
        path_rows.append(
            iterations[
                [
                    "player_order",
                    "iter",
                    "r_strat",
                    "r_raw_br",
                    "stable_count",
                    "all_solves_acceptable",
                ]
            ]
        )
        converged = bool(record["movement"]["movement_converged"])
        ax.plot(
            iterations["iter"],
            iterations["r_strat"],
            color=colors[index],
            linestyle="-" if converged else "--",
            marker="o",
            markersize=3.5,
            linewidth=2.0 if converged else 1.5,
            label=f"({index + 1}) {record['slug'].upper()}",
        )
    ax.axhline(
        threshold,
        color="#b2182b",
        linestyle=":",
        linewidth=1.5,
        label=f"movement threshold ({100.0 * threshold:g}%)",
    )
    ax.set_xlabel("Complete Gauss--Seidel sweep")
    ax.set_ylabel(r"Damped strategy movement $\Delta\theta$")
    ax.set_title("Stage-1 movement under the seven Figure-2 player orders")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_root / "stage1_movement_figure2.png", dpi=220)
    plt.close(fig)
    if path_rows:
        pd.concat(path_rows, ignore_index=True).to_csv(
            output_root / "stage1_movement_paths.csv", index=False
        )

    labels = [record["slug"].upper() for record in complete]
    candidate_values = [
        100.0 * float(record["candidate_start"]["max_relative_gain"])
        for record in complete
    ]
    multistart_values = [
        100.0 * float(record["best_of_starts"]["max_relative_gain"])
        for record in complete
    ]
    positions = list(range(len(complete)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    ax.bar(
        [position - width / 2 for position in positions],
        candidate_values,
        width,
        label="Candidate-start audit",
        color="#67a9cf",
    )
    ax.bar(
        [position + width / 2 for position in positions],
        multistart_values,
        width,
        label="Best-of-three audit",
        color="#ef8a62",
    )
    ax.axhline(
        100.0 * threshold,
        color="#b2182b",
        linestyle="--",
        linewidth=1.5,
        label=f"equilibrium tolerance ({100.0 * threshold:g}%)",
    )
    ax.set_xticks(positions, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Maximum normalized unilateral gain (%)")
    ax.set_title("Independent zero-proximal audit of Stage-1 endpoints")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "stage1_endpoint_audit.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--additional-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--terminal-salvage-fraction", type=float, default=0.5)
    args = parser.parse_args()

    if args.workers < 1 or args.starts < 1 or args.starts > 3 or args.maxiter < 1:
        raise ValueError("workers/maxiter must be positive and starts must be in [1, 3]")
    input_path = args.input.resolve()
    existing_root = args.existing_root.resolve()
    additional_root = args.additional_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "slug": slug,
            "order": order,
            "input_path": str(input_path),
            "source_root": str(_source_root(slug, existing_root, additional_root)),
            "output_root": str(output_root),
            "starts": args.starts,
            "maxiter": args.maxiter,
            "threshold": args.threshold,
            "terminal_salvage_fraction": args.terminal_salvage_fraction,
        }
        for slug, order in ORDERS.items()
    ]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = {executor.submit(audit_order, task): task["slug"] for task in tasks}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                f"[RESULT] {record['slug']}: {record['status']}",
                flush=True,
            )

    index = {slug: position for position, slug in enumerate(ORDERS)}
    records.sort(key=lambda record: index[record["slug"]])
    manifest = {
        "created": now(),
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "existing_stage1_root": relative(existing_root),
        "additional_stage1_root": relative(additional_root),
        "output_root": relative(output_root),
        "threshold": args.threshold,
        "starts": args.starts,
        "maxiter": args.maxiter,
        "terminal_salvage_fraction": args.terminal_salvage_fraction,
        "records": records,
    }
    write_json(output_root / "manifest.json", manifest)
    (output_root / "RESULTS.md").write_text(
        results_markdown(records, args.threshold), encoding="utf-8"
    )
    build_figures(records, output_root, args.threshold)
    print(results_markdown(records, args.threshold), flush=True)


if __name__ == "__main__":
    main()
