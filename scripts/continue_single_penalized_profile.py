from __future__ import annotations

"""Continue one archived penalized profile from an exact recorded sweep."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import continue_penalized_sweep30_profiles as runner
from scripts.run_corrected_equilibrium_search import relative, sha256, write_json


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_order(value: str) -> list[str]:
    order = [item.strip().lower() for item in value.split(",") if item.strip()]
    if len(order) != len(set(order)):
        raise argparse.ArgumentTypeError("Player order must not contain duplicates")
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--source-iteration", type=int, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--order", type=parse_order, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.4)
    parser.add_argument("--penalty-q", type=float, default=2.0)
    parser.add_argument("--penalty-p", type=float, default=3.0)
    parser.add_argument("--penalty-a", type=float, default=2.0)
    parser.add_argument("--penalty-dk", type=float, default=2.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    source_workbook = args.source_workbook.resolve()
    output_root = args.output_root.resolve()
    for path in (input_path, source_workbook):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists() and not args.validate_only:
        raise FileExistsError(output_root)

    runner.SOURCE_ITERATION = int(args.source_iteration)
    runner.CONTINUATION_SWEEPS = int(args.sweeps)
    runner.CONTINUATION_DAMPING = float(args.damping)
    runner.FINAL_PENALTIES = {
        "q": float(args.penalty_q),
        "p": float(args.penalty_p),
        "a": float(args.penalty_a),
        "dk": float(args.penalty_dk),
    }

    # Reuse the established continuation implementation, but point its source
    # lookup at this explicitly supplied archived workbook.
    runner.source_workbook = lambda _root, _sequence: source_workbook
    task = {
        "sequence": args.sequence,
        "order": args.order,
        "input_path": str(input_path),
        "source_root": str(source_workbook.parent),
        "output_root": str(output_root),
    }

    prepared = runner.prepare_task(task)
    if prepared["replay_error"] > 1e-10:
        raise RuntimeError(
            "Source replay check failed: "
            f"max r_strat error={prepared['replay_error']:.3g}"
        )
    validation = {
        "sequence": args.sequence,
        "source_workbook": relative(source_workbook),
        "source_workbook_sha256": sha256(source_workbook),
        "source_iteration": args.source_iteration,
        "source_row": prepared["source_row"],
        "source_replay_error": prepared["replay_error"],
        "source_strategy_sha256": prepared["state_sha256"],
    }
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return

    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "created": now(),
        "status": "running",
        "method": (
            f"{args.sweeps}-sweep continuation from exact replay of archived "
            f"penalized sweep {args.source_iteration}"
        ),
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "output_root": relative(output_root),
        "source": validation,
        "continuation_sweeps": args.sweeps,
        "damping": args.damping,
        "penalties": runner.FINAL_PENALTIES,
        "strategy_movement_tolerance": runner.TOLERANCE,
        "code_sha256": {
            "scripts/continue_single_penalized_profile.py": sha256(Path(__file__)),
            "scripts/continue_penalized_sweep30_profiles.py": sha256(
                ROOT / "scripts" / "continue_penalized_sweep30_profiles.py"
            ),
            "model/gauss_seidel.py": sha256(ROOT / "model" / "gauss_seidel.py"),
            "model/run_gs.py": sha256(ROOT / "model" / "run_gs.py"),
            "scripts/continue_selected_equilibrium.py": sha256(
                ROOT / "scripts" / "continue_selected_equilibrium.py"
            ),
        },
        "results": [],
    }
    write_json(manifest_path, manifest)
    result = runner.run_task(task)
    manifest["updated"] = now()
    manifest["status"] = (
        "complete" if result.get("status") == "complete" else "complete_with_failures"
    )
    manifest["results"] = [result]
    write_json(manifest_path, manifest)
    runner.write_summary(output_root / "results_summary.csv", [result])
    print(json.dumps(result, indent=2))
    if result.get("status") != "complete":
        raise RuntimeError(result.get("error", "Continuation failed"))


if __name__ == "__main__":
    main()
