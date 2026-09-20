from __future__ import annotations

"""Run a separate three-start frozen-profile audit on salvage candidates.

The source factorial and its one-start acceptance artifacts are treated as
read-only.  Each accepted source profile is reloaded under the recorded input
and terminal-salvage configuration, then every player's unrestricted
zero-proximal best response is solved from three established initializations.
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

from scripts.run_corrected_equilibrium_search import (  # noqa: E402
    audit,
    configure_modules,
    relative,
    sha256,
    write_json,
)
from scripts.run_overnight_equilibrium_experiment import load_profile  # noqa: E402


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_recorded(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def result_id(row: dict[str, Any]) -> str:
    return f"{row['sequence']}/{row['branch']}"


def run_candidate(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    branch = str(task["branch"])
    identifier = f"{sequence}/{branch}"
    candidate_root = Path(task["output_root"]).resolve() / sequence / branch
    audit_path = candidate_root / "audit_three_start.json"
    status_path = candidate_root / "status.json"
    try:
        candidate_root.mkdir(parents=True, exist_ok=True)
        write_json(
            status_path,
            {
                "created": now(),
                "status": "running",
                "pid": os.getpid(),
                "candidate": identifier,
            },
        )
        data, source_template, replay_error, workbook, _ = configure_modules(task)
        profile_path = Path(task["profile_path"]).resolve()
        state = load_profile(profile_path, data, source_template)
        if audit_path.exists():
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            payload = audit(
                data,
                state,
                starts=int(task["starts"]),
                maxiter=int(task["maxiter"]),
                label=f"terminal_salvage_{sequence}_{branch}_three_start",
                output=audit_path,
            )

        players = list(payload["players"])
        result = {
            "created": now(),
            "status": "complete",
            "pid": os.getpid(),
            "candidate": identifier,
            "sequence": sequence,
            "branch": branch,
            "source_profile": relative(profile_path),
            "source_profile_sha256": sha256(profile_path),
            "source_one_start_audit": str(task["one_start_audit"]),
            "source_one_start_max_relative_gain": float(
                task["one_start_max_relative_gain"]
            ),
            "source_replay_error": replay_error,
            "source_workbook": relative(workbook),
            "starts_per_player": int(task["starts"]),
            "attempt_count": sum(len(row["attempts"]) for row in players),
            "all_attempts_successful": bool(payload["all_attempts_successful"]),
            "three_start_max_relative_gain": float(payload["max_relative_gain"]),
            "three_start_max_gain_player": str(payload["max_gain_player"]),
            "three_start_equilibrium_verified": bool(payload["equilibrium_verified"]),
            "players_selecting_non_candidate_start": [
                row["player"] for row in players if int(row["chosen_start_index"]) != 0
            ],
            "audit": relative(audit_path),
            "audit_sha256": sha256(audit_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        return result
    except Exception as exc:
        failure = {
            "created": now(),
            "status": "failed",
            "pid": os.getpid(),
            "candidate": identifier,
            "sequence": sequence,
            "branch": branch,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        return failure


def protocol_text(
    source_manifest: Path,
    output_root: Path,
    task_count: int,
    starts: int,
    maxiter: int,
    workers: int,
) -> str:
    return f"""# Terminal-salvage multistart audit protocol

Created: {now()}.

- Source factorial: `{relative(source_manifest)}`
- Output root: `{relative(output_root)}`
- Source profiles: all {task_count} profiles accepted by the source one-start
  local 1% criterion
- Starts per player: {starts}
- Maximum optimizer iterations per start: {maxiter}
- Parallel candidate workers: {workers}
- Algorithmic proximal penalties: zero
- Common profile frozen during every player audit
- No equilibrium profile is updated by this diagnostic

The three starts are the candidate strategy, the standard zero-capacity-change
and manufacturing-cost-price initialization, and a feasible 0.95-times
candidate-price perturbation. A candidate passes the stronger diagnostic only
when all 18 optimizer attempts succeed and the largest relative unilateral
gain over all six players is at most 1%. This is a stronger local computational
test, not a global Nash-equilibrium certificate.
"""


def results_text(manifest: dict[str, Any]) -> str:
    completed = [row for row in manifest["results"] if row["status"] == "complete"]
    survivors = [row for row in completed if row["three_start_equilibrium_verified"]]
    failed = [row for row in manifest["results"] if row["status"] != "complete"]
    ranked = sorted(
        completed,
        key=lambda row: (
            not row["three_start_equilibrium_verified"],
            row["three_start_max_relative_gain"],
            row["candidate"],
        ),
    )
    lines = [
        "# Terminal-salvage three-start audit results",
        "",
        f"Created: {now()}.",
        "",
        "## Outcome",
        "",
        f"- Audited candidates: {len(completed)}",
        f"- Three-start local 1% survivors: {len(survivors)}",
        f"- Candidates failing the stronger diagnostic: {len(completed) - len(survivors)}",
        f"- Failed audit tasks: {len(failed)}",
        "- Each complete candidate has 18 frozen-profile optimizer attempts.",
        "- No source profile was changed.",
        "",
        "Lower maximum gain is stronger within this diagnostic, provided all",
        "attempts succeeded. It is not a global optimality ranking.",
        "",
        "| Rank | Candidate | One-start max gain | Three-start max gain | Worst player | Non-candidate starts selected | Result |",
        "|---:|---|---:|---:|---|---|---|",
    ]
    for index, row in enumerate(ranked, start=1):
        selected = ", ".join(row["players_selecting_non_candidate_start"]) or "none"
        outcome = "survives" if row["three_start_equilibrium_verified"] else "does not survive"
        lines.append(
            f"| {index} | `{row['candidate']}` | "
            f"{100.0 * row['source_one_start_max_relative_gain']:.6f}% | "
            f"{100.0 * row['three_start_max_relative_gain']:.6f}% | "
            f"{str(row['three_start_max_gain_player']).upper()} | {selected.upper()} | {outcome} |"
        )
    if failed:
        lines.extend(["", "## Failed tasks", ""])
        for row in failed:
            lines.append(f"- `{row['candidate']}`: {row.get('error', 'unknown error')}")
    lines.extend(
        [
            "",
            "## Claim scope",
            "",
            "A survivor is a stronger local computational candidate than a profile",
            "that passes only the one-start audit. The diagnostic still samples a",
            "finite set of local optimizer initializations and therefore does not",
            "establish a global Nash equilibrium.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    args = parser.parse_args()

    if args.workers < 1 or args.starts < 2 or args.maxiter < 1:
        raise ValueError("workers and maxiter must be positive; starts must be at least two")
    if args.starts != 3:
        raise ValueError("this protocol is intentionally fixed to the established three starts")

    source_manifest = args.source_manifest.resolve()
    output_root = args.output_root.resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise RuntimeError("source factorial is not complete")

    input_path = resolve_recorded(source["input"])
    cold_start_root = resolve_recorded(source["cold_start_root"])
    accepted = [row for row in source["results"] if row["status"] == "accepted"]
    if not accepted:
        raise RuntimeError("source factorial has no accepted candidates")
    workers = min(int(args.workers), len(accepted))
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "PROTOCOL.md"
    protocol_path.write_text(
        protocol_text(
            source_manifest,
            output_root,
            len(accepted),
            args.starts,
            args.maxiter,
            workers,
        ),
        encoding="utf-8",
    )

    tasks = []
    for row in accepted:
        tasks.append(
            {
                "sequence": row["sequence"],
                "branch": row["branch"],
                "profile_path": str(resolve_recorded(row["selected_profile"])),
                "one_start_audit": row["one_start_audit"],
                "one_start_max_relative_gain": row["one_start_max_relative_gain"],
                "input_path": str(input_path),
                "cold_start_root": str(cold_start_root),
                "output_root": str(output_root),
                "terminal_salvage_fraction": float(source["terminal_salvage_fraction"]),
                "starts": int(args.starts),
                "maxiter": int(args.maxiter),
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "three-start common-frozen-profile zero-proximal diagnostic",
        "claim_scope": "stronger local computational test; not a global Nash certificate",
        "source_manifest": relative(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "terminal_salvage_fraction": float(source["terminal_salvage_fraction"]),
        "task_count": len(tasks),
        "starts_per_player": int(args.starts),
        "attempts_per_candidate": 6 * int(args.starts),
        "maxiter": int(args.maxiter),
        "workers": workers,
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "code_sha256": {
            "scripts/audit_salvage_candidates_multistart.py": sha256(Path(__file__)),
            "scripts/run_corrected_equilibrium_search.py": sha256(
                ROOT / "scripts/run_corrected_equilibrium_search.py"
            ),
            "scripts/run_local_paper_equilibrium_experiment.py": sha256(
                ROOT / "scripts/run_local_paper_equilibrium_experiment.py"
            ),
            "scripts/nested_market_audit.py": sha256(
                ROOT / "scripts/nested_market_audit.py"
            ),
            "model/model_main.py": sha256(ROOT / "model/model_main.py"),
            "model/data_prep.py": sha256(ROOT / "model/data_prep.py"),
        },
        "results": [],
    }
    write_json(manifest_path, manifest)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_candidate, task): result_id(task) for task in tasks}
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "status": "executor_exception",
                    "candidate": identifier,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            manifest["results"] = sorted(results, key=lambda row: row["candidate"])
            write_json(manifest_path, manifest)
            if result["status"] == "complete":
                print(
                    f"[{identifier}] three-start max gain "
                    f"{100.0 * result['three_start_max_relative_gain']:.6f}% "
                    f"pass={result['three_start_equilibrium_verified']}",
                    flush=True,
                )
            else:
                print(f"[{identifier}] {result['status']}: {result.get('error')}", flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(row["status"] == "complete" for row in results)
        else "complete_with_failures"
    )
    manifest["results"] = sorted(results, key=lambda row: row["candidate"])
    write_json(manifest_path, manifest)
    (output_root / "RESULTS.md").write_text(results_text(manifest), encoding="utf-8")
    print(json.dumps({"manifest": relative(manifest_path), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
