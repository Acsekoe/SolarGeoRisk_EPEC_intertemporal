from __future__ import annotations

"""Continue selected sweep-30 profiles with their terminal penalties and damping."""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
from scripts.run_corrected_cold_start import _build_fresh_state, _state_sha256
from scripts.run_corrected_equilibrium_search import relative, sha256, write_json


PARAMS_SHEET = "params_region_new"
SOURCE_ITERATION = 30
CONTINUATION_SWEEPS = 10
TOLERANCE = 0.01
FINAL_PENALTIES = {"q": 2.0, "p": 3.0, "a": 2.0, "dk": 2.0}
CONTINUATION_DAMPING = 0.4

SEQUENCES: dict[str, dict[str, Any]] = {
    "ch-af-eu-us-row-apac": {
        "order": ["ch", "af", "eu", "us", "row", "apac"],
        "source_group": "existing",
    },
    "ch-row-apac-us-eu-af": {
        "order": ["ch", "row", "apac", "us", "eu", "af"],
        "source_group": "existing",
    },
    "af-eu-us-apac-row-ch": {
        "order": ["af", "eu", "us", "apac", "row", "ch"],
        "source_group": "additional",
    },
    "eu-us-af-row-apac-ch": {
        "order": ["eu", "us", "af", "row", "apac", "ch"],
        "source_group": "additional",
    },
    "us-apac-af-row-eu-ch": {
        "order": ["us", "apac", "af", "row", "eu", "ch"],
        "source_group": "additional",
    },
    "us-row-eu-apac-af-ch": {
        "order": ["us", "row", "eu", "apac", "af", "ch"],
        "source_group": "additional",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def source_workbook(source_root: Path, sequence: str) -> Path:
    matches = sorted((source_root / sequence / "results").glob("results_*.xlsx"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one source workbook for {sequence}; found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def source_row(workbook: Path) -> dict[str, Any]:
    rows = pd.read_excel(workbook, sheet_name="iters")
    selected = rows.loc[rows["iter"].astype(int) == SOURCE_ITERATION]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one row for sweep {SOURCE_ITERATION} in {workbook}; found {len(selected)}"
        )
    row = selected.iloc[0]
    penalties = {
        "q": float(row["c_pen_q"]),
        "p": float(row["c_pen_p"]),
        "a": float(row["c_pen_a"]),
        "dk": float(row["c_pen_dk"]),
    }
    if any(abs(penalties[key] - FINAL_PENALTIES[key]) > 1e-12 for key in penalties):
        raise RuntimeError(f"Unexpected sweep-30 penalties in {workbook}: {penalties}")
    if abs(float(row["omega_next"]) - CONTINUATION_DAMPING) > 1e-12:
        raise RuntimeError(
            f"Unexpected sweep-30 next damping in {workbook}: {row['omega_next']}"
        )
    return {
        "iteration": SOURCE_ITERATION,
        "r_strat": float(row["r_strat"]),
        "r_raw_br": float(row["r_raw_br"]),
        "stable_count": int(row["stable_count"]),
        "all_solves_acceptable": bool(row["all_solves_acceptable"]),
        "solve_failures": str(row.get("solve_failures", "")),
        "omega": float(row["omega"]),
        "omega_next": float(row["omega_next"]),
        "penalties": penalties,
    }


def base_config(
    *,
    input_path: Path,
    output_dir: Path,
    order: list[str],
    initial_state: dict[str, dict],
) -> run_gs.RunConfig:
    return run_gs.RunConfig(
        excel_path=str(input_path),
        out_dir=str(output_dir / "results"),
        plots_dir=str(output_dir / "plots"),
        params_region_sheet=PARAMS_SHEET,
        solver="ipopt",
        feastol=1e-4,
        opttol=1e-4,
        iters=CONTINUATION_SWEEPS,
        omega=CONTINUATION_DAMPING,
        adaptive_omega=True,
        omega_min=CONTINUATION_DAMPING,
        omega_aggressive_sweeps=5,
        omega_ramp_iters=10,
        tol_strat=TOLERANCE,
        tol_raw_br=None,
        # Force all ten diagnostic sweeps; convergence is assessed afterward.
        stable_iters=999,
        eps_x=1e-3,
        eps_comp=1e-3,
        keep_workdir=False,
        c_pen_q=FINAL_PENALTIES["q"],
        c_pen_p=FINAL_PENALTIES["p"],
        c_pen_a=FINAL_PENALTIES["a"],
        c_pen_dk=FINAL_PENALTIES["dk"],
        c_pen_q_mid=FINAL_PENALTIES["q"],
        c_pen_p_mid=FINAL_PENALTIES["p"],
        c_pen_a_mid=FINAL_PENALTIES["a"],
        c_pen_dk_mid=FINAL_PENALTIES["dk"],
        c_pen_q_final=FINAL_PENALTIES["q"],
        c_pen_p_final=FINAL_PENALTIES["p"],
        c_pen_a_final=FINAL_PENALTIES["a"],
        c_pen_dk_final=FINAL_PENALTIES["dk"],
        c_pen_ramp_iters=1,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        cap_keep_reward=0.0,
        capex_subsidy=0.0,
        terminal_salvage_fraction=0.5,
        terminal_capacity_state_only=True,
        decommission_penalty=0.0,
        fix_q_offer_to_kcap=True,
        force_mu_offer_zero=False,
        fix_a_bid_to_true_dem=True,
        discount_rate=0.02,
        base_year=2025,
        initial_state_override=initial_state,
        player_order=order,
    )


def prepare_task(task: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(task["input_path"]).resolve()
    source_root = Path(task["source_root"]).resolve()
    workbook = source_workbook(source_root, str(task["sequence"]))
    order = list(task["order"])
    row = source_row(workbook)

    cfg0 = base_config(
        input_path=input_path,
        output_dir=Path(task["output_root"]).resolve() / str(task["sequence"]),
        order=order,
        initial_state={},
    )
    data = load_data_from_excel(str(input_path), params_region_sheet=PARAMS_SHEET)
    run_gs._apply_data_overrides(data, cfg0)
    fresh = _build_fresh_state(data)
    state, replay_error = continuation.replay_accepted_state(
        workbook,
        fresh,
        data,
        through_iteration=SOURCE_ITERATION,
        expected_player_order=order,
    )
    return {
        "workbook": workbook,
        "source_row": row,
        "state": state,
        "state_sha256": _state_sha256(state),
        "replay_error": float(replay_error),
    }


def continuation_metrics(
    frame: pd.DataFrame,
    source_stable_count: int,
) -> dict[str, Any]:
    stable = int(source_stable_count)
    first_combined_convergence: int | None = None
    first_fresh_convergence: int | None = None
    fresh_stable = 0
    clean_stable = 0
    first_clean_convergence: int | None = None
    paths: list[dict[str, Any]] = []
    for _, row in frame.sort_values("iter").iterrows():
        continuation_sweep = int(row["iter"])
        absolute_sweep = SOURCE_ITERATION + continuation_sweep
        movement = float(row["r_strat"])
        acceptable = bool(row["all_solves_acceptable"])
        below = movement <= TOLERANCE
        stable = stable + 1 if below else 0
        fresh_stable = fresh_stable + 1 if below else 0
        clean_stable = clean_stable + 1 if below and acceptable else 0
        if stable >= 3 and first_combined_convergence is None:
            first_combined_convergence = absolute_sweep
        if fresh_stable >= 3 and first_fresh_convergence is None:
            first_fresh_convergence = absolute_sweep
        if clean_stable >= 3 and first_clean_convergence is None:
            first_clean_convergence = absolute_sweep
        paths.append(
            {
                "continuation_sweep": continuation_sweep,
                "absolute_sweep": absolute_sweep,
                "r_strat": movement,
                "r_raw_br": float(row["r_raw_br"]),
                "stable_count_continuing_source": stable,
                "stable_count_continuation_only": fresh_stable,
                "clean_stable_count": clean_stable,
                "all_solves_acceptable": acceptable,
                "solve_failures": str(row.get("solve_failures", "")),
                "omega": float(row["omega"]),
                "omega_next": float(row["omega_next"]),
                "c_pen_q": float(row["c_pen_q"]),
                "c_pen_p": float(row["c_pen_p"]),
                "c_pen_a": float(row["c_pen_a"]),
                "c_pen_dk": float(row["c_pen_dk"]),
            }
        )
    return {
        "first_combined_movement_convergence_sweep": first_combined_convergence,
        "first_continuation_only_movement_convergence_sweep": first_fresh_convergence,
        "first_clean_movement_convergence_sweep": first_clean_convergence,
        "minimum_r_strat": min(item["r_strat"] for item in paths),
        "minimum_r_strat_sweep": min(paths, key=lambda item: item["r_strat"])[
            "absolute_sweep"
        ],
        "final_r_strat": paths[-1]["r_strat"],
        "final_r_raw_br": paths[-1]["r_raw_br"],
        "all_continuation_solves_acceptable": all(
            item["all_solves_acceptable"] for item in paths
        ),
        "path": paths,
    }


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    sequence_root = Path(task["output_root"]).resolve() / sequence
    sequence_root.mkdir(parents=True, exist_ok=False)
    provenance_path = sequence_root / "provenance.json"
    try:
        prepared = prepare_task(task)
        workbook = Path(prepared["workbook"])
        source = prepared["source_row"]
        cfg = base_config(
            input_path=Path(task["input_path"]).resolve(),
            output_dir=sequence_root,
            order=list(task["order"]),
            initial_state=prepared["state"],
        )
        provenance: dict[str, Any] = {
            "created": now(),
            "status": "running",
            "pid": os.getpid(),
            "sequence": sequence,
            "player_order": list(task["order"]),
            "source_workbook": relative(workbook),
            "source_workbook_sha256": sha256(workbook),
            "source_iteration": SOURCE_ITERATION,
            "source_row": source,
            "source_replay_error": prepared["replay_error"],
            "source_strategy_sha256": prepared["state_sha256"],
            "source_warning": (
                None
                if source["all_solves_acceptable"]
                else (
                    f"Requested sweep-{SOURCE_ITERATION} source follows an "
                    "unacceptable solve and is diagnostic only."
                )
            ),
            "configuration": {
                **asdict(cfg),
                "initial_state_override": (
                    f"exact replay of source sweep {SOURCE_ITERATION}"
                ),
            },
        }
        write_json(provenance_path, provenance)
        log_path = sequence_root / "continuation.log"
        run_gs.write_default_plots = None
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(
            log
        ), contextlib.redirect_stderr(log):
            output_path = Path(run_gs.run(cfg)).resolve()
        frame = pd.read_excel(output_path, sheet_name="iters")
        metrics = continuation_metrics(frame, int(source["stable_count"]))
        result = {
            "created": now(),
            "status": "complete",
            "sequence": sequence,
            "source_workbook": relative(workbook),
            "source_iteration": SOURCE_ITERATION,
            "source_all_solves_acceptable": source["all_solves_acceptable"],
            "source_r_strat": source["r_strat"],
            "source_r_raw_br": source["r_raw_br"],
            "source_stable_count": source["stable_count"],
            "source_replay_error": prepared["replay_error"],
            "continuation_workbook": relative(output_path),
            "continuation_log": relative(log_path),
            "continuation_sweeps": CONTINUATION_SWEEPS,
            "damping": CONTINUATION_DAMPING,
            "penalties": FINAL_PENALTIES,
            **metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        provenance.update(result)
        write_json(provenance_path, provenance)
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
        write_json(provenance_path, result)
        return result


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "sequence",
        "source_all_solves_acceptable",
        "source_r_strat",
        "source_r_raw_br",
        "source_stable_count",
        "minimum_r_strat",
        "minimum_r_strat_sweep",
        "final_r_strat",
        "final_r_raw_br",
        "first_combined_movement_convergence_sweep",
        "first_continuation_only_movement_convergence_sweep",
        "first_clean_movement_convergence_sweep",
        "all_continuation_solves_acceptable",
        "status",
        "continuation_workbook",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["sequence"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--additional-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    input_path = args.input.resolve()
    existing_root = args.existing_root.resolve()
    additional_root = args.additional_root.resolve()
    output_root = args.output_root.resolve()
    for path in (input_path, existing_root, additional_root):
        if not path.exists():
            raise FileNotFoundError(path)

    roots = {"existing": existing_root, "additional": additional_root}
    tasks = [
        {
            "sequence": sequence,
            "order": specification["order"],
            "input_path": str(input_path),
            "source_root": str(roots[specification["source_group"]]),
            "output_root": str(output_root),
        }
        for sequence, specification in SEQUENCES.items()
    ]
    if args.validate_only:
        validations = []
        for task in tasks:
            prepared = prepare_task(task)
            validations.append(
                {
                    "sequence": task["sequence"],
                    "source_workbook": relative(Path(prepared["workbook"])),
                    "source_row": prepared["source_row"],
                    "source_replay_error": prepared["replay_error"],
                    "source_strategy_sha256": prepared["state_sha256"],
                }
            )
        print(json.dumps(validations, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": (
            f"{CONTINUATION_SWEEPS}-sweep continuation from recorded "
            f"sweep-{SOURCE_ITERATION} profiles with terminal damping and "
            "proximal penalties"
        ),
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "existing_source_root": relative(existing_root),
        "additional_source_root": relative(additional_root),
        "output_root": relative(output_root),
        "source_iteration": SOURCE_ITERATION,
        "continuation_sweeps": CONTINUATION_SWEEPS,
        "damping": CONTINUATION_DAMPING,
        "penalties": FINAL_PENALTIES,
        "strategy_movement_tolerance": TOLERANCE,
        "workers": min(int(args.workers), len(tasks)),
        "code_sha256": {
            "scripts/continue_penalized_sweep30_profiles.py": sha256(Path(__file__)),
            "model/gauss_seidel.py": sha256(ROOT / "model" / "gauss_seidel.py"),
            "model/run_gs.py": sha256(ROOT / "model" / "run_gs.py"),
            "scripts/continue_selected_equilibrium.py": sha256(
                ROOT / "scripts" / "continue_selected_equilibrium.py"
            ),
        },
        "results": [],
    }
    write_json(manifest_path, manifest)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=manifest["workers"]) as pool:
        futures = {pool.submit(run_task, task): task["sequence"] for task in tasks}
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
            print(json.dumps({k: result.get(k) for k in (
                "sequence", "status", "minimum_r_strat", "final_r_strat",
                "first_clean_movement_convergence_sweep", "error"
            )}, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(result["status"] == "complete" for result in results)
        else "complete_with_failures"
    )
    manifest["results"] = sorted(results, key=lambda row: row["sequence"])
    write_json(manifest_path, manifest)
    write_summary(output_root / "results_summary.csv", results)
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
