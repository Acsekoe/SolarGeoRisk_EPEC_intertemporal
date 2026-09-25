"""Generate comparison and plots when a capacity-only chain finishes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], path: Path) -> int:
    with path.open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=28800)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    stage2 = run_root / "stage2"
    stage2_manifest_path = stage2 / "manifest.json"
    top_manifest_path = run_root / "manifest.json"
    final_path = run_root / "postprocessing.json"
    started = time.monotonic()
    while True:
        stage2_manifest = (
            json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
            if stage2_manifest_path.is_file() else None
        )
        if stage2_manifest and stage2_manifest.get("status") in ("completed", "failed") and (stage2 / "summary.csv").is_file():
            break
        if top_manifest_path.is_file():
            top = json.loads(top_manifest_path.read_text(encoding="utf-8"))
            if top.get("status") == "stage2_not_started":
                write_json(final_path, {"status": "stage2_not_started", "updated": now()})
                return 1
        if not args.wait or time.monotonic() - started > args.timeout_seconds:
            write_json(final_path, {"status": "waiting_timeout", "updated": now()})
            return 1
        time.sleep(args.poll_seconds)

    accepted = int(stage2_manifest.get("accepted_runs", 0))
    result = {
        "started": now(), "stage2_status": stage2_manifest["status"],
        "accepted_runs": accepted, "status": "running",
    }
    write_json(final_path, result)
    if accepted:
        planner = ROOT.parent / "_MOVE" / "15_equilibria" / "planner_benchmark_corrected.xlsx"
        plot_command = [
            sys.executable, "-u", str(ROOT / "plots" / "plot_equilibrium_paper_figures.py"),
            "--workflow-manifest", str(stage2_manifest_path),
            "--plots-dir", str(stage2), "--planner-path", str(planner),
            "--plots-in-profile-dirs",
        ]
        result["plots_returncode"] = run_logged(plot_command, run_root / "plots.log")
    compare_command = [
        sys.executable, "-u", str(ROOT / "scripts" / "compare_capacity_only_results.py"),
        "--counterfactual-root", str(stage2),
        "--output-dir", str(stage2 / "comparison"),
    ]
    result["comparison_returncode"] = run_logged(compare_command, run_root / "comparison.log")
    result["status"] = (
        "complete" if result.get("plots_returncode", 0) == 0 and
        result["comparison_returncode"] == 0 else "failed"
    )
    result["finished"] = now()
    write_json(final_path, result)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
