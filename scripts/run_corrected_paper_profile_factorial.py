from __future__ import annotations

"""Run the corrected-demand paper-profile initialization factorial.

The experiment crosses three corrected Stage-1 paper-profile anchors with:

* bilateral export offers initialized at a fixed manufacturing-cost multiple;
* the anchor's net-capacity-change path scaled toward the zero-change path; and
* a fixed Gauss--Seidel damping factor.

Candidate generation uses all-player, zero-proximal sequential best responses.
There is no move cap, player freezing, or gain filter.  A separate one-start
unilateral-deviation audit evaluates every common frozen profile.
"""

import argparse
import json
import os
import sys
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

from model import model_main as mm
from scripts import run_corrected_a030_search as base_search
from scripts.run_corrected_equilibrium_search import (
    SEQUENCES,
    configure_modules,
    relative,
    sha256,
    source_workbook,
    terminal_source_iteration,
    write_json,
)
from scripts.run_local_paper_equilibrium_experiment import clone_state
from scripts.search_nested_equilibrium import _sync_quantity


MIN_ALPHA = 0.30


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def factor_label(prefix: str, value: float) -> str:
    return f"{prefix}{int(round(100.0 * value)):03d}"


def branch_name(price_factor: float, capacity_weight: float, alpha: float) -> str:
    return "_".join(
        (
            factor_label("pf", price_factor),
            factor_label("k", capacity_weight),
            factor_label("a", alpha),
        )
    )


def branch_specification(
    price_factor: float,
    capacity_weight: float,
    alpha: float,
) -> dict[str, Any]:
    return {
        "description": (
            "corrected Stage-1 paper-profile anchor; bilateral export offers "
            f"initialized at {price_factor:.2f} times period-specific manufacturing "
            f"cost; anchor net capacity changes scaled by {capacity_weight:.2f}; "
            f"fixed all-player Gauss--Seidel damping {alpha:.2f}"
        ),
        "kind": "paper_profile_price_capacity_factorial",
        "price_offer_factor_to_manufacturing_cost": price_factor,
        "capacity_change_path_weight": capacity_weight,
        "capacity_reference_path": "corrected Stage-1 endpoint",
        "capacity_zero_weight_endpoint": "observed initial capacity with zero net changes",
        "domestic_offer_initialization": "period-specific manufacturing cost",
        "bilateral_export_offer_initialization": (
            "price factor times period-specific manufacturing cost, clipped to model bounds"
        ),
        "alpha": alpha,
        "algorithmic_proximal_penalties": 0.0,
        "move_cap": None,
        "gain_filter": None,
        "players_frozen": False,
    }


def make_factorial_initial_state(
    data: Any,
    source: dict[str, dict],
    branch: str,
    _unused_historical_profile: Path,
) -> dict[str, dict]:
    specification = base_search.BRANCHES[branch]
    price_factor = float(specification["price_offer_factor_to_manufacturing_cost"])
    capacity_weight = float(specification["capacity_change_path_weight"])
    state = clone_state(source)

    # Scaling net changes preserves the fixed initial capacity.  Because the
    # other endpoint is the feasible zero-change path and the capacity-change
    # constraints are linear, weights in [0, 1] remain within the source bounds.
    for key, value in source["dK_net"].items():
        state["dK_net"][key] = capacity_weight * float(value)
    _sync_quantity(data, state)

    # Domestic offers are not strategic export offers and remain at cost.
    # Every off-diagonal bilateral offer receives the common experimental factor.
    for exporter in data.regions:
        for importer in data.regions:
            upper = float(data.p_offer_ub[(exporter, importer)])
            for period in mm._operating_times(data):
                cost = float(
                    (data.c_man_t or {}).get(
                        (exporter, period), data.c_man[exporter]
                    )
                )
                target = cost if exporter == importer else price_factor * cost
                state["p_offer"][(exporter, importer, period)] = min(
                    max(target, 0.0), upper
                )

    _sync_quantity(data, state)
    return state


def prepare_base_search(task: dict[str, Any]) -> None:
    branch = str(task["branch"])
    base_search.BRANCHES[branch] = dict(task["branch_specification"])
    base_search.make_initial_state = make_factorial_initial_state


def run_factorial_branch(task: dict[str, Any]) -> dict[str, Any]:
    prepare_base_search(task)
    return base_search.run_branch(task)


def validate_factorial_branch(task: dict[str, Any]) -> dict[str, Any]:
    prepare_base_search(task)
    result = base_search.validate(task)
    result.update(
        {
            "price_factor": task["price_factor"],
            "capacity_weight": task["capacity_weight"],
            "branch_specification": task["branch_specification"],
        }
    )
    return result


def code_hashes() -> dict[str, str]:
    names = [
        "scripts/run_corrected_paper_profile_factorial.py",
        "scripts/run_corrected_a030_search.py",
        "scripts/run_corrected_diversified_search.py",
        "scripts/run_corrected_equilibrium_search.py",
        "scripts/nested_market_audit.py",
        "scripts/run_local_paper_equilibrium_experiment.py",
        "scripts/run_overnight_equilibrium_experiment.py",
        "model/data_prep.py",
        "model/model_main.py",
    ]
    return {name: sha256(ROOT / name) for name in names}


def protocol_text(
    *,
    input_path: Path,
    cold_start_root: Path,
    output_root: Path,
    sequences: list[str],
    price_factors: list[float],
    capacity_weights: list[float],
    alphas: list[float],
    max_sweeps: int,
    maxiter: int,
    workers: int,
    terminal_salvage_fraction: float,
) -> str:
    sequence_lines = "\n".join(
        f"- `{name}`: Stage-1 iteration {SEQUENCES[name]['source_iteration']}; "
        f"order `{'-'.join(SEQUENCES[name]['order'])}`"
        for name in sequences
    )
    return f"""# Corrected-demand paper-profile factorial protocol

Created {now()}.

## Fixed design

{sequence_lines}

- Corrected input: `{relative(input_path)}`
- Archived Stage-1 source root: `{relative(cold_start_root)}`
- Output root: `{relative(output_root)}`
- Export-offer manufacturing-cost factors: `{price_factors}`
- Stage-1 net-capacity-change weights: `{capacity_weights}`
- Fixed damping factors: `{alphas}`
- Maximum sweeps per branch: `{max_sweeps}`
- Best-response maximum iterations: `{maxiter}`
- Parallel workers: `{workers}`
- Terminal salvage fraction: `{terminal_salvage_fraction}`
- Operating market periods: `2025, 2030, 2035, 2040`
- Terminal state: `2045` installed capacity only; no 2045 market or operating payoff
- Terminal salvage definition: one credit on 2045 installed capacity, discounted
  to 2045; no all-period investment subsidy
- Stage-1 source rule: last contiguous checkpoint for which every player solve
  was acceptable; any state at or after an interrupted solve is excluded

The capacity weight multiplies the complete Stage-1 `dK_net` path and the
corresponding capacity path is reconstructed from the model's accounting
identity. A weight of zero would be the observed-capacity, zero-change path;
the present design uses the predeclared weights listed above.

Every off-diagonal bilateral offer is initialized to its exporter's
period-specific manufacturing cost times the branch price factor. Domestic
offers remain at manufacturing cost. Values are clipped only to the model's
feasible offer bounds during initialization.

## Candidate generation

- all-player sequential Gauss--Seidel best responses;
- fixed branch-specific damping;
- zero algorithmic proximal penalties;
- no move cap;
- no gain filter or player freezing;
- one solver start per sequential best response.

## Acceptance

After the initialization and every completed sweep, all six players receive
an independent zero-proximal best-response solve against one common frozen
profile. Damping, the capacity-path weight, and the initialization price factor
do not enter this audit. A profile is accepted when all six audit solves
succeed and the maximum relative unilateral gain is at most 1%. Candidate
generation stops at the first accepted sweep; otherwise the lowest-gain audited
profile is retained as a diagnostic. Multistart is not used by this run.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cold-start-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sequences", nargs="+", choices=list(SEQUENCES), default=list(SEQUENCES)
    )
    parser.add_argument("--price-factors", nargs="+", type=float, default=[1.0, 1.2])
    parser.add_argument(
        "--capacity-weights", nargs="+", type=float, default=[0.5, 1.0]
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.30, 0.40])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--max-sweeps", type=int, default=20)
    parser.add_argument("--terminal-salvage-fraction", type=float, default=0.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.maxiter < 1 or args.max_sweeps < 1:
        raise ValueError("workers, maxiter, and max-sweeps must be positive")
    if args.terminal_salvage_fraction < 0.0:
        raise ValueError("terminal-salvage-fraction must be non-negative")
    if len(set(args.sequences)) != len(args.sequences):
        raise ValueError("sequences must not contain duplicates")
    for label, values in (
        ("price factors", args.price_factors),
        ("capacity weights", args.capacity_weights),
        ("alphas", args.alphas),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} must not contain duplicates")
    if any(value <= 0.0 for value in args.price_factors):
        raise ValueError("price factors must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.capacity_weights):
        raise ValueError("capacity weights must be in [0, 1]")
    if any(not MIN_ALPHA <= value <= 1.0 for value in args.alphas):
        raise ValueError(f"alphas must be in [{MIN_ALPHA:.2f}, 1]")

    input_path = args.input.resolve()
    cold_start_root = args.cold_start_root.resolve()
    output_root = args.output_root.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not cold_start_root.is_dir():
        raise FileNotFoundError(cold_start_root)

    # Resolve the new Stage-1 endpoints from the workbooks themselves.  The
    # historical fixed iteration numbers are not reusable after changing data,
    # horizon treatment, and terminal value.
    for sequence in args.sequences:
        workbook = source_workbook(cold_start_root, sequence)
        SEQUENCES[sequence]["source_iteration"] = terminal_source_iteration(
            workbook
        )

    tasks: list[dict[str, Any]] = []
    for sequence in args.sequences:
        for price_factor in args.price_factors:
            for capacity_weight in args.capacity_weights:
                for alpha in args.alphas:
                    branch = branch_name(price_factor, capacity_weight, alpha)
                    specification = branch_specification(
                        price_factor, capacity_weight, alpha
                    )
                    tasks.append(
                        {
                            "sequence": sequence,
                            "branch": branch,
                            "branch_specification": specification,
                            "price_factor": price_factor,
                            "capacity_weight": capacity_weight,
                            "alpha": alpha,
                            "update_gain_threshold": None,
                            "input_path": str(input_path),
                            "cold_start_root": str(cold_start_root),
                            "output_root": str(output_root),
                            # Required by the reusable branch engine but unused by
                            # this initialization type. It is an existing path so
                            # provenance and path resolution remain deterministic.
                            "historical_o6": str(input_path),
                            "maxiter": args.maxiter,
                            "max_sweeps": args.max_sweeps,
                            "terminal_salvage_fraction": args.terminal_salvage_fraction,
                        }
                    )
    branch_ids = [(task["sequence"], task["branch"]) for task in tasks]
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("rounded branch labels are not unique")

    if args.validate_only:
        validations = [validate_factorial_branch(task) for task in tasks]
        print(json.dumps(validations, indent=2), flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers, len(tasks))
    protocol_path = output_root / "PROTOCOL.md"
    protocol_path.write_text(
        protocol_text(
            input_path=input_path,
            cold_start_root=cold_start_root,
            output_root=output_root,
            sequences=list(args.sequences),
            price_factors=list(args.price_factors),
            capacity_weights=list(args.capacity_weights),
            alphas=list(args.alphas),
            max_sweeps=args.max_sweeps,
            maxiter=args.maxiter,
            workers=workers,
            terminal_salvage_fraction=args.terminal_salvage_fraction,
        ),
        encoding="utf-8",
    )
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {
        "created": now(),
        "status": "running",
        "pid": os.getpid(),
        "method": "paper-profile price/capacity/damping factorial under corrected demand with optional terminal salvage",
        "task_count": len(tasks),
        "acceptance_criterion": (
            "one-start common frozen-profile maximum relative gain <= 1%, "
            "all six solves successful"
        ),
        "acceptance_audit_starts": 1,
        "multistart_used": False,
        "algorithmic_proximal_penalties": 0.0,
        "terminal_salvage_fraction": args.terminal_salvage_fraction,
        "move_cap": None,
        "gain_filter": None,
        "players_frozen": False,
        "input": relative(input_path),
        "input_sha256": sha256(input_path),
        "cold_start_root": relative(cold_start_root),
        "output_root": relative(output_root),
        "protocol": relative(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "sequences": {name: SEQUENCES[name] for name in args.sequences},
        "price_factors": list(args.price_factors),
        "capacity_weights": list(args.capacity_weights),
        "alphas": list(args.alphas),
        "workers": workers,
        "maxiter": args.maxiter,
        "max_sweeps": args.max_sweeps,
        "branches": {
            task["branch"]: task["branch_specification"] for task in tasks
        },
        "code_sha256": code_hashes(),
        "results": [],
    }
    write_json(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_factorial_branch, task): (
                task["sequence"],
                task["branch"],
            )
            for task in tasks
        }
        for future in as_completed(futures):
            sequence, branch = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "status": "executor_exception",
                    "sequence": sequence,
                    "branch": branch,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            manifest["results"] = sorted(
                results, key=lambda row: (row["sequence"], row["branch"])
            )
            write_json(manifest_path, manifest)
            print(json.dumps(result, indent=2), flush=True)

    manifest["updated"] = now()
    manifest["status"] = (
        "complete"
        if all(
            result["status"] in {"accepted", "no_pass_within_schedule"}
            for result in results
        )
        else "complete_with_failures"
    )
    manifest["results"] = sorted(
        results, key=lambda row: (row["sequence"], row["branch"])
    )
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"manifest": relative(manifest_path), "status": manifest["status"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
