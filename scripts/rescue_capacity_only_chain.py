"""Continue the active chain if the separately launched fourth worker races its queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDERS = (
    "ch-af-apac-eu-row-us", "ch-af-eu-us-row-apac", "ch-row-apac-us-eu-af",
    "af-eu-us-apac-row-ch", "eu-us-af-row-apac-ch", "us-apac-af-row-eu-ch",
    "us-row-eu-apac-af-ch",
)


def all_stage1_complete(stage1_root: Path) -> bool:
    for order in ORDERS:
        run_dir = stage1_root / order
        provenance = run_dir / "provenance.json"
        if not provenance.is_file():
            return False
        if json.loads(provenance.read_text(encoding="utf-8")).get("status") != "complete":
            return False
        if len(list((run_dir / "results").glob("results_*.xlsx"))) != 1:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=28800)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    started = time.monotonic()
    while time.monotonic() - started <= args.timeout_seconds:
        if (run_root / "stage2" / "manifest.json").is_file():
            print("Original chain started Stage 2; no rescue needed.", flush=True)
            return 0
        top_path = run_root / "manifest.json"
        if top_path.is_file():
            top = json.loads(top_path.read_text(encoding="utf-8"))
            if top.get("status") == "stage2_not_started" and all_stage1_complete(run_root / "stage1"):
                print("Restarting chain controller from seven completed Stage 1 workbooks.", flush=True)
                command = [
                    sys.executable, "-u", str(ROOT / "scripts" / "run_capacity_only_chain.py"),
                    "--output-root", str(run_root), "--stage1-workers", "4",
                    "--stage2-workers", "6", "--stage1-sweeps", "40",
                    "--stage2-sweeps", "15", "--stage2-maxiter", "600",
                ]
                return subprocess.run(command, cwd=ROOT, check=False).returncode
        time.sleep(args.poll_seconds)
    print("Rescue watcher timed out.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
