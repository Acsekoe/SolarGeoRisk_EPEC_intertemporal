"""Run the competitive-offer counterfactual through both search stages.

Stage 1 uses the existing penalized, damped cold-start algorithm for all seven
player orders. Stage 2 restarts every movement-converged Stage-1 endpoint with
the clean objective and one-start frozen-profile deviation audits. In both
stages, every bilateral offer is fixed to exporter-period manufacturing cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_stage1_player_orders import ORDERS
from scripts.run_corrected_equilibrium_search import terminal_source_iteration


DEFAULT_INPUT = (
    ROOT.parent / "_MOVE" / "demand_calibration" / "correction_20260915_131409"
    / "input_data_intertemporal_corrected_20260915_131409.xlsx"
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage1_workbook(run_dir: Path) -> Path:
    matches = sorted((run_dir / "results").glob("results_*.xlsx"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Stage-1 workbook in {run_dir}; found {matches}")
    return matches[0]


def run_stage1(slug: str, order: list[str], input_path: Path, stage1_root: Path,
               max_sweeps: int, salvage_fraction: float) -> dict:
    run_dir = stage1_root / slug
    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") == "complete" and stage1_workbook(run_dir).is_file():
            return {"sequence": slug, "status": "complete", "resumed": True}
        raise RuntimeError(f"Existing incomplete Stage-1 directory: {run_dir}")
    log_path = stage1_root / f"{slug}.log"
    command = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_corrected_cold_start.py"),
        "--input", str(input_path), "--output-dir", str(run_dir),
        "--order", ",".join(order), "--iters", str(max_sweeps),
        "--terminal-salvage-fraction", str(salvage_fraction),
        "--fix-offers-to-cost", "--log-path", str(log_path),
    ]
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    return {
        "sequence": slug,
        "status": "complete" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "log": str(log_path),
    }


def build_stage1_specs(stage1_root: Path) -> tuple[dict, list[dict]]:
    selected: dict[str, dict] = {}
    records: list[dict] = []
    for slug, order in ORDERS.items():
        run_dir = stage1_root / slug
        provenance_path = run_dir / "provenance.json"
        if not provenance_path.is_file():
            records.append({"sequence": slug, "status": "missing"})
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete":
            records.append({"sequence": slug, "status": provenance.get("status")})
            continue
        try:
            workbook = stage1_workbook(run_dir)
            iteration = terminal_source_iteration(workbook)
        except RuntimeError as exc:
            records.append({"sequence": slug, "status": "no_clean_prefix", "error": str(exc)})
            continue
        history = pd.read_excel(workbook, sheet_name="iters")
        row = history.loc[history["iter"].astype(int) == iteration].iloc[-1]
        movement_converged = int(row["stable_count"]) >= 3
        record = {
            "sequence": slug,
            "status": "complete",
            "source_sweep": iteration,
            "movement_converged": movement_converged,
            "stable_count": int(row["stable_count"]),
            "strategy_movement": float(row["r_strat"]),
            "raw_best_response_movement": float(row["r_raw_br"]),
            "workbook": str(workbook),
            "workbook_sha256": sha256(workbook),
        }
        records.append(record)
        if movement_converged:
            selected[slug] = {
                "order": order,
                "global_sweep": iteration,
                "replay": [[str(workbook.relative_to(run_dir)), iteration]],
            }
    return selected, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stage1-workers", type=int, default=3)
    parser.add_argument("--stage2-workers", type=int, default=6)
    parser.add_argument("--stage1-sweeps", type=int, default=40)
    parser.add_argument("--stage2-sweeps", type=int, default=15)
    parser.add_argument("--stage2-maxiter", type=int, default=600)
    parser.add_argument("--terminal-salvage-fraction", type=float, default=0.5)
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument(
        "--proceed-with-converged-stage1", action="store_true",
        help="Start Stage 2 from completed, movement-converged orders while other Stage 1 orders remain incomplete.",
    )
    args = parser.parse_args()
    if min(args.stage1_workers, args.stage2_workers, args.stage1_sweeps,
           args.stage2_sweeps, args.stage2_maxiter) < 1:
        raise ValueError("Worker, sweep, and iteration counts must be positive")
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_root = (
        args.output_root.resolve() if args.output_root else
        ROOT / "outputs" / f"capacity_only_chain_{datetime.now().astimezone():%Y%m%d_%H%M%S}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    stage1_root = output_root / "stage1"
    stage1_root.mkdir(exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "created": now(), "status": "stage1_running", "pid": os.getpid(),
        "input": str(input_path), "input_sha256": sha256(input_path),
        "counterfactual": "capacity-only regional strategy; all bilateral offers fixed to exporter-period marginal manufacturing cost",
        "stage1": {
            "objective": "with-mu-and-penalties", "economic_penalties": True,
            "proximal_penalties": True, "orders": ORDERS,
            "maximum_sweeps": args.stage1_sweeps,
            "terminal_salvage_fraction": args.terminal_salvage_fraction,
        },
        "stage2": {
            "objective": "without-mu-and-penalties", "economic_penalties": False,
            "proximal_penalties": False, "capacity_weights": [0.5, 1.0],
            "damping": [0.3, 0.4], "maximum_sweeps": args.stage2_sweeps,
            "maximum_optimizer_iterations": args.stage2_maxiter,
            "one_start_relative_gain_tolerance": 0.01,
        },
    }
    write_json(manifest_path, manifest)
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.stage1_workers, len(ORDERS))) as pool:
        futures = {
            pool.submit(run_stage1, slug, order, input_path, stage1_root,
                        args.stage1_sweeps, args.terminal_salvage_fraction): slug
            for slug, order in ORDERS.items()
        }
        for future in as_completed(futures):
            slug = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"sequence": slug, "status": "failed", "error": str(exc)}
            print(f"[STAGE1] {record}", flush=True)
            if record["status"] != "complete":
                failures.append(record)
    specs, records = build_stage1_specs(stage1_root)
    specs_path = output_root / "stage1_specs.json"
    write_json(specs_path, specs)
    manifest["stage1"]["results"] = records
    manifest["stage1"]["selected_sequences"] = list(specs)
    manifest["stage1"]["failures"] = failures
    manifest["stage1"]["proceeded_with_incomplete_orders"] = bool(
        failures and args.proceed_with_converged_stage1
    )
    manifest["status"] = "stage1_complete"
    write_json(manifest_path, manifest)
    print(f"[STAGE1] movement-converged anchors: {list(specs)}", flush=True)
    if args.stage1_only:
        return 0 if not failures else 1
    if (failures and not args.proceed_with_converged_stage1) or not specs:
        manifest["status"] = "stage2_not_started"
        write_json(manifest_path, manifest)
        raise RuntimeError("Stage 1 failed or produced no movement-converged anchors")

    stage2_root = output_root / "stage2"
    log_path = output_root / "stage2.log"
    command = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_clean_stage2_factorial.py"),
        "--input", str(input_path), "--stage1-root", str(stage1_root),
        "--stage1-specs", str(specs_path), "--fix-offers-to-cost",
        "--price-factors", "1.0", "--workers", str(args.stage2_workers),
        "--max-sweeps", str(args.stage2_sweeps),
        "--maxiter", str(args.stage2_maxiter),
        "--terminal-salvage-fraction", str(args.terminal_salvage_fraction),
        "--output-root", str(stage2_root), "--progress", "sweep",
    ]
    manifest["status"] = "stage2_running"
    manifest["stage2"]["command"] = command
    write_json(manifest_path, manifest)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    stage2_manifest = json.loads((stage2_root / "manifest.json").read_text(encoding="utf-8")) if (stage2_root / "manifest.json").is_file() else {}
    stage2_ok = result.returncode == 0 and stage2_manifest.get("status") == "completed"
    manifest["status"] = "stage2_complete" if stage2_ok else "stage2_failed"
    manifest["finished"] = now()
    manifest["stage2"]["returncode"] = result.returncode
    manifest["stage2"]["log"] = str(log_path)
    manifest["stage2"]["accepted_runs"] = stage2_manifest.get("accepted_runs")
    manifest["stage2"]["failed_runs"] = stage2_manifest.get("failed_runs")
    write_json(manifest_path, manifest)
    print(f"[STAGE2] returncode={result.returncode} log={log_path}", flush=True)
    if not stage2_ok:
        return 1

    if int(stage2_manifest.get("accepted_runs", 0)) > 0:
        planner_path = ROOT.parent / "_MOVE" / "15_equilibria" / "planner_benchmark_corrected.xlsx"
        plot_command = [
            sys.executable, "-u", str(ROOT / "plots" / "plot_equilibrium_paper_figures.py"),
            "--workflow-manifest", str(stage2_root / "manifest.json"),
            "--plots-dir", str(stage2_root),
            "--planner-path", str(planner_path),
            "--plots-in-profile-dirs",
        ]
        with (output_root / "plots.log").open("w", encoding="utf-8") as stream:
            plot_result = subprocess.run(plot_command, cwd=ROOT, stdout=stream,
                                         stderr=subprocess.STDOUT, check=False)
        manifest["plots_returncode"] = plot_result.returncode
        write_json(manifest_path, manifest)
    else:
        plot_result = None

    compare_command = [
        sys.executable, "-u", str(ROOT / "scripts" / "compare_capacity_only_results.py"),
        "--counterfactual-root", str(stage2_root),
        "--output-dir", str(stage2_root / "comparison"),
    ]
    with (output_root / "comparison.log").open("w", encoding="utf-8") as stream:
        compare_result = subprocess.run(compare_command, cwd=ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
    manifest["comparison_returncode"] = compare_result.returncode
    manifest["status"] = (
        "complete" if compare_result.returncode == 0 and
        (plot_result is None or plot_result.returncode == 0) else "postprocessing_failed"
    )
    write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
