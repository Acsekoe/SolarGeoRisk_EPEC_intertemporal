from __future__ import annotations

"""Restart the Stage-2 profile factorial under the clean objective.

The runner replays the three retained Stage-1 endpoints, constructs the same
price/capacity/damping branches, and applies the clean objective consistently
in both the sequential search and every frozen-profile audit.  By default it
runs six branches in parallel and prints one labelled line per completed sweep.
"""

import argparse
import contextlib
import csv
import hashlib
import io
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm, run_gs
from model.data_prep import load_data_from_excel
from scripts.audit_selected_equilibrium import _strategy_distance
from scripts.continue_selected_equilibrium import replay_accepted_state
from scripts.nested_market_audit import (
    nested_best_response,
    nested_economic_objective,
    solve_nested_market,
)
from scripts.run_corrected_cold_start import _build_fresh_state
from scripts.run_local_paper_equilibrium_experiment import clone_state, full_state_payload
from scripts.search_nested_equilibrium import _deserialize_state, _sync_quantity


PARAMS_SHEET = "params_region_new"
OBJECTIVE_MODE = run_gs.OBJECTIVE_MODE_WITHOUT_MU_AND_PENALTIES
EXPECTED_STAGE1_INPUT_SHA256 = (
    "F426E480A20565286F6CD6678A5EEC0DE6E7376382826983E7FC3D39C2DEA5E0"
)
DEFAULT_INPUT = (
    ROOT
    / "_MOVE"
    / "demand_calibration"
    / "correction_20260915_131409"
    / "input_data_intertemporal_corrected_20260915_131409.xlsx"
)
DEFAULT_STAGE1_ROOT = ROOT / "outputs" / "new_equilibria" / "penalized_profiles"

# The Africa-first and EU-first histories continued for ten locally numbered
# sweeps after their initial sweep 30.  Replaying both files therefore recovers
# the exact global-sweep-40 anchors used by the existing Stage-2 search.
STAGE1_PROFILES: dict[str, dict[str, Any]] = {
    "ch-af-apac-eu-row-us": {
        "order": ["ch", "af", "apac", "eu", "row", "us"],
        "global_sweep": 26,
        "replay": [("penalized_sweeps_001_026.xlsx", 26)],
    },
    "af-eu-us-apac-row-ch": {
        "order": ["af", "eu", "us", "apac", "row", "ch"],
        "global_sweep": 40,
        "replay": [
            ("penalized_sweeps_001_030.xlsx", 30),
            ("penalized_sweeps_031_040.xlsx", 10),
        ],
    },
    "eu-us-af-row-apac-ch": {
        "order": ["eu", "us", "af", "row", "apac", "ch"],
        "global_sweep": 40,
        "replay": [
            ("penalized_sweeps_001_030.xlsx", 30),
            ("penalized_sweeps_031_040.xlsx", 10),
        ],
    },
}

DEFAULT_PRICE_FACTORS = (0.8, 1.0, 1.2)
DEFAULT_CAPACITY_WEIGHTS = (0.5, 1.0)
DEFAULT_ALPHAS = (0.3, 0.4)

_ANCHOR_CACHE: dict[tuple[Any, ...], tuple[Any, dict[str, dict], dict[str, Any]]] = {}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 * (attempt + 1))


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
    *,
    fix_offers_to_cost: bool = False,
) -> dict[str, Any]:
    offer_description = (
        "bilateral offers fixed at exporter-period manufacturing cost"
        if fix_offers_to_cost else
        f"bilateral export offers initialized at {price_factor:.2f} times period-specific manufacturing cost"
    )
    return {
        "description": (
            f"retained Stage-1 endpoint; {offer_description}; "
            f"Stage-1 net capacity changes scaled by {capacity_weight:.2f}; "
            f"fixed all-player Gauss--Seidel damping {alpha:.2f}"
        ),
        "kind": (
            "clean_objective_stage2_capacity_only_factorial"
            if fix_offers_to_cost else "clean_objective_stage2_price_capacity_factorial"
        ),
        "fix_offers_to_cost": fix_offers_to_cost,
        "price_offer_factor_to_manufacturing_cost": price_factor,
        "capacity_change_path_weight": capacity_weight,
        "capacity_reference_path": "retained Stage-1 endpoint",
        "capacity_zero_weight_endpoint": "observed initial capacity with zero net changes",
        "domestic_offer_initialization": "period-specific manufacturing cost",
        "bilateral_export_offer_initialization": (
            "fixed at exporter-period manufacturing cost"
            if fix_offers_to_cost else
            "price factor times period-specific manufacturing cost, clipped to model bounds"
        ),
        "alpha": alpha,
        "objective_mode": OBJECTIVE_MODE,
        "subtract_mu_offer_from_producer_margin": False,
        "economic_quadratic_penalties": 0.0,
        "algorithmic_proximal_penalties": 0.0,
        "move_cap": None,
        "gain_filter": None,
        "players_frozen": False,
    }


def build_tasks(
    *,
    sequences: list[str],
    price_factors: list[float],
    capacity_weights: list[float],
    alphas: list[float],
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for sequence in sequences:
        for price_factor in price_factors:
            for capacity_weight in capacity_weights:
                for alpha in alphas:
                    tasks.append(
                        {
                            **common,
                            "sequence": sequence,
                            "branch": branch_name(price_factor, capacity_weight, alpha),
                            "price_factor": price_factor,
                            "capacity_weight": capacity_weight,
                            "alpha": alpha,
                            "branch_specification": branch_specification(
                                price_factor, capacity_weight, alpha,
                                fix_offers_to_cost=bool(common.get("fix_offers_to_cost", False)),
                            ),
                        }
                    )
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        task["run_index"] = index
        task["run_total"] = total
    return tasks


def clean_configuration(
    input_path: Path, terminal_salvage_fraction: float, *, fix_offers_to_cost: bool = False
) -> run_gs.RunConfig:
    return run_gs._effective_run_config(
        run_gs.RunConfig(
            excel_path=str(input_path.resolve()),
            params_region_sheet=PARAMS_SHEET,
            solver="ipopt",
            feastol=1e-4,
            opttol=1e-4,
            eps_x=1e-3,
            eps_comp=1e-3,
            objective_mode=OBJECTIVE_MODE,
            cap_keep_reward=0.0,
            capex_subsidy=0.0,
            terminal_salvage_fraction=terminal_salvage_fraction,
            terminal_capacity_state_only=True,
            decommission_penalty=0.0,
            fix_q_offer_to_kcap=True,
            fix_p_offer_to_c_man_t=fix_offers_to_cost,
            force_mu_offer_zero=False,
            fix_a_bid_to_true_dem=True,
            discount_rate=0.02,
            base_year=2025,
        )
    )


def assert_clean_objective(data: Any) -> None:
    settings = data.settings or {}
    expected_zero = (
        "c_pen_q",
        "c_pen_p",
        "c_pen_a",
        "c_pen_dk",
        "c_quad_q",
        "c_quad_p",
        "c_quad_a",
    )
    nonzero = {name: settings.get(name) for name in expected_zero if float(settings.get(name, 0.0)) != 0.0}
    if settings.get("objective_mode") != OBJECTIVE_MODE:
        raise RuntimeError(f"Unexpected objective mode: {settings.get('objective_mode')!r}")
    if bool(settings.get("subtract_mu_offer_from_producer_margin", True)):
        raise RuntimeError("Clean objective still subtracts mu_offer from the producer margin")
    if bool(settings.get("economic_quadratic_penalties_enabled", True)):
        raise RuntimeError("Clean objective still enables economic quadratic penalties")
    if bool(settings.get("algorithmic_proximal_penalties_enabled", True)):
        raise RuntimeError("Clean objective still enables algorithmic proximal penalties")
    if nonzero:
        raise RuntimeError(f"Clean objective contains nonzero penalties: {nonzero}")


def load_stage1_anchor(
    input_path: Path,
    stage1_root: Path,
    sequence: str,
    terminal_salvage_fraction: float,
    *,
    specification: dict[str, Any] | None = None,
    fix_offers_to_cost: bool = False,
) -> tuple[Any, dict[str, dict], dict[str, Any]]:
    cache_key = (
        str(input_path.resolve()),
        str(stage1_root.resolve()),
        sequence,
        terminal_salvage_fraction,
        fix_offers_to_cost,
        json.dumps(specification, sort_keys=True) if specification is not None else None,
    )
    cached = _ANCHOR_CACHE.get(cache_key)
    if cached is not None:
        data, source, metadata = cached
        return data, clone_state(source), dict(metadata)

    specification = specification or STAGE1_PROFILES[sequence]
    cfg = clean_configuration(
        input_path, terminal_salvage_fraction, fix_offers_to_cost=fix_offers_to_cost
    )
    # Keep subprocess output limited to labelled Stage-2 progress lines.  The
    # loader's calibration diagnostics remain reproducible in the input hash.
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_data_from_excel(
            str(input_path.resolve()), params_region_sheet=cfg.params_region_sheet
        )
        run_gs._apply_data_overrides(data, cfg)
    assert_clean_objective(data)

    state = _build_fresh_state(
        data, period_specific_cost_offers=fix_offers_to_cost
    )
    replay_records: list[dict[str, Any]] = []
    maximum_replay_error = 0.0
    for filename, through_iteration in specification["replay"]:
        workbook = (stage1_root / sequence / filename).resolve()
        if not workbook.is_file():
            raise FileNotFoundError(workbook)
        state, replay_error = replay_accepted_state(
            workbook,
            state,
            data,
            through_iteration=int(through_iteration),
            expected_player_order=list(specification["order"]),
        )
        maximum_replay_error = max(maximum_replay_error, float(replay_error))
        replay_records.append(
            {
                "workbook": relative(workbook),
                "workbook_sha256": sha256(workbook),
                "through_local_iteration": int(through_iteration),
                "replay_error": float(replay_error),
            }
        )
    if maximum_replay_error > 1e-10:
        raise RuntimeError(
            f"{sequence}: Stage-1 endpoint replay residual is {maximum_replay_error:.3g}"
        )
    _sync_quantity(data, state)
    metadata = {
        "sequence": sequence,
        "player_order": list(specification["order"]),
        "source_global_sweep": int(specification["global_sweep"]),
        "source_replay": replay_records,
        "maximum_replay_error": maximum_replay_error,
    }
    _ANCHOR_CACHE[cache_key] = (data, clone_state(state), dict(metadata))
    return data, state, metadata


def make_factorial_initial_state(
    data: Any,
    source: dict[str, dict],
    *,
    price_factor: float,
    capacity_weight: float,
) -> dict[str, dict]:
    state = clone_state(source)
    for key, value in source["dK_net"].items():
        state["dK_net"][key] = capacity_weight * float(value)
    _sync_quantity(data, state)

    operating_times = set(mm._operating_times(data))
    for exporter in data.regions:
        for importer in data.regions:
            upper = float(data.p_offer_ub[(exporter, importer)])
            for period in data.times or []:
                if period not in operating_times:
                    target = 0.0
                else:
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


def update_player(
    data: Any,
    state: dict[str, dict],
    response: dict[str, dict],
    player: str,
    alpha: float,
) -> None:
    for period in mm._move_times(list(data.times or [])):
        key = (player, period)
        state["dK_net"][key] = (
            (1.0 - alpha) * float(state["dK_net"][key])
            + alpha * float(response["dK_net"][key])
        )
    if not bool((data.settings or {}).get("fix_p_offer_to_c_man_t", False)):
        for importer in data.regions:
            if importer == player:
                continue
            for period in mm._operating_times(data):
                key = (player, importer, period)
                state["p_offer"][key] = (
                    (1.0 - alpha) * float(state["p_offer"][key])
                    + alpha * float(response["p_offer"][key])
                )
    _sync_quantity(data, state)


def load_saved_state(path: Path, data: Any, template: dict[str, dict]) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "ending_profile" in payload:
        serialized = payload["ending_profile"]["strategy"]
    elif "profile" in payload:
        serialized = payload["profile"]["strategy"]
    else:
        raise ValueError(f"No serialized strategy in {path}")
    return _deserialize_state(serialized, data, template)


def progress_prefix(task: dict[str, Any]) -> str:
    return (
        f"[RUN {int(task['run_index']):02d}/{int(task['run_total']):02d} "
        f"{task['sequence']}/{task['branch']} pid={os.getpid()}]"
    )


def audit_profile(
    data: Any,
    state: dict[str, dict],
    *,
    order: list[str],
    maxiter: int,
    tolerance: float,
    task: dict[str, Any],
    audit_label: str,
) -> dict[str, Any]:
    market, market_diagnostics = solve_nested_market(data, state)
    rows: list[dict[str, Any]] = []
    for player_index, player in enumerate(order, start=1):
        player_started = time.perf_counter()
        reference = float(nested_economic_objective(data, state, market, player))
        best, response, diagnostics = nested_best_response(
            data,
            state,
            market,
            player,
            maxiter=maxiter,
            starts=1,
        )
        gain = max(float(best) - reference, 0.0) / max(abs(reference), 1.0)
        move, coordinate, absolute = _strategy_distance(data, state, response, player)
        attempts = diagnostics.get("attempts", [])
        all_attempts_successful = bool(attempts) and all(
            bool(attempt.get("success")) for attempt in attempts
        )
        row = {
            "player": player,
            "reference_objective": reference,
            "best_response_objective": float(best),
            "relative_gain": gain,
            "relative_gain_percent": 100.0 * gain,
            "strategy_move": float(move),
            "largest_move_coordinate": coordinate,
            "largest_move_absolute": float(absolute),
            "optimizer_success": bool(diagnostics["success"]),
            "all_attempts_successful": all_attempts_successful,
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - player_started,
        }
        rows.append(row)
        if task["progress"] == "player":
            print(
                f"{progress_prefix(task)} {audit_label} audit-player "
                f"{player_index}/6 {player}: gain={gain:.4%} "
                f"success={bool(diagnostics['success'])} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    limiting = max(rows, key=lambda row: float(row["relative_gain"]))
    all_successful = all(bool(row["all_attempts_successful"]) for row in rows)
    return {
        "created": now(),
        "common_profile_frozen": True,
        "starts_per_best_response": 1,
        "maxiter": maxiter,
        "objective_mode": OBJECTIVE_MODE,
        "subtract_mu_offer_from_producer_margin": False,
        "economic_quadratic_penalties": 0.0,
        "algorithmic_proximal_penalties": 0.0,
        "relative_gain_tolerance": tolerance,
        "reference_market_diagnostics": market_diagnostics,
        "players": rows,
        "max_relative_gain": float(limiting["relative_gain"]),
        "max_gain_player": str(limiting["player"]),
        "all_attempts_successful": all_successful,
        "equilibrium_verified": bool(
            all_successful and float(limiting["relative_gain"]) <= tolerance
        ),
    }


def run_or_load_audit(
    path: Path,
    data: Any,
    state: dict[str, dict],
    *,
    order: list[str],
    task: dict[str, Any],
    audit_label: str,
) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    audit = audit_profile(
        data,
        state,
        order=order,
        maxiter=int(task["maxiter"]),
        tolerance=float(task["relative_gain_tolerance"]),
        task=task,
        audit_label=audit_label,
    )
    write_json(path, audit)
    return audit


def audit_record(
    audit: dict[str, Any],
    *,
    profile_path: Path,
    audit_path: Path,
    sweep: int,
) -> dict[str, Any]:
    return {
        "sweep": sweep,
        "profile_path": relative(profile_path),
        "audit_path": relative(audit_path),
        "max_relative_gain": float(audit["max_relative_gain"]),
        "max_gain_player": str(audit["max_gain_player"]),
        "all_six_solves_successful": bool(audit["all_attempts_successful"]),
        "local_one_percent_equilibrium": bool(audit["equilibrium_verified"]),
    }


def run_branch(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    sequence = str(task["sequence"])
    branch = str(task["branch"])
    output_root = Path(task["output_root"]).resolve()
    branch_root = output_root / sequence / branch
    status_path = branch_root / "status.json"
    prefix = progress_prefix(task)
    try:
        branch_root.mkdir(parents=True, exist_ok=True)
        if task["progress"] != "quiet":
            print(
                f"{prefix} START alpha={float(task['alpha']):.2f} "
                f"price_factor={float(task['price_factor']):.2f} "
                f"capacity_weight={float(task['capacity_weight']):.2f}",
                flush=True,
            )
        write_json(
            status_path,
            {
                "created": now(),
                "status": "initializing",
                "pid": os.getpid(),
                "sequence": sequence,
                "branch": branch,
                "objective_mode": OBJECTIVE_MODE,
            },
        )

        data, source, source_metadata = load_stage1_anchor(
            Path(task["input_path"]),
            Path(task["stage1_root"]),
            sequence,
            float(task["terminal_salvage_fraction"]),
            specification=task.get("stage1_spec"),
            fix_offers_to_cost=bool(task.get("fix_offers_to_cost", False)),
        )
        assert_clean_objective(data)
        order = list((task.get("stage1_spec") or STAGE1_PROFILES[sequence])["order"])
        initialization_path = branch_root / "initialization.json"
        if initialization_path.exists():
            initial_state = load_saved_state(initialization_path, data, source)
        else:
            initial_state = make_factorial_initial_state(
                data,
                source,
                price_factor=float(task["price_factor"]),
                capacity_weight=float(task["capacity_weight"]),
            )
            initial_market, initial_market_diagnostics = solve_nested_market(
                data, initial_state
            )
            write_json(
                initialization_path,
                {
                    "created": now(),
                    "sequence": sequence,
                    "branch": branch,
                    "branch_specification": task["branch_specification"],
                    "player_order": order,
                    "stage1_source": source_metadata,
                    "input": relative(Path(task["input_path"])),
                    "input_sha256": sha256(Path(task["input_path"])),
                    "objective_mode": OBJECTIVE_MODE,
                    "subtract_mu_offer_from_producer_margin": False,
                    "economic_quadratic_penalties": 0.0,
                    "algorithmic_proximal_penalties": 0.0,
                    "terminal_salvage_fraction": float(
                        task["terminal_salvage_fraction"]
                    ),
                    "terminal_capacity_state_only": True,
                    "fix_q_offer_to_kcap": True,
                    "fix_a_bid_to_true_dem": True,
                    "force_mu_offer_zero": False,
                    "move_cap": None,
                    "gain_filter": None,
                    "players_frozen": False,
                    "profile": full_state_payload(data, initial_state, initial_market),
                    "market_diagnostics": initial_market_diagnostics,
                },
            )

        completed = sorted(branch_root.glob("sweep_*.json"))
        completed_numbers = [int(path.stem.split("_")[-1]) for path in completed]
        if completed_numbers and completed_numbers != list(
            range(1, max(completed_numbers) + 1)
        ):
            raise RuntimeError(f"Non-contiguous saved sweeps: {completed_numbers}")
        if completed_numbers and max(completed_numbers) > int(task["max_sweeps"]):
            raise RuntimeError(
                "Saved checkpoint count exceeds requested max-sweeps; use the original "
                "schedule or a larger value when resuming"
            )
        state = (
            load_saved_state(completed[-1], data, source)
            if completed
            else clone_state(initial_state)
        )

        initial_audit_path = branch_root / "audits" / "audit_initial_one_start.json"
        initial_audit = run_or_load_audit(
            initial_audit_path,
            data,
            initial_state,
            order=order,
            task=task,
            audit_label="initial",
        )
        best = audit_record(
            initial_audit,
            profile_path=initialization_path,
            audit_path=initial_audit_path,
            sweep=0,
        )
        selected: dict[str, Any] | None = (
            best if best["local_one_percent_equilibrium"] else None
        )
        if task["progress"] in ("sweep", "player"):
            print(
                f"{prefix} initial-audit gain={best['max_relative_gain']:.4%} "
                f"player={best['max_gain_player']} "
                f"pass={best['local_one_percent_equilibrium']}",
                flush=True,
            )

        for checkpoint, sweep in zip(completed, completed_numbers):
            checkpoint_state = load_saved_state(checkpoint, data, source)
            audit_path = (
                branch_root / "audits" / f"audit_sweep_{sweep:03d}_one_start.json"
            )
            audit = run_or_load_audit(
                audit_path,
                data,
                checkpoint_state,
                order=order,
                task=task,
                audit_label=f"sweep-{sweep:03d}",
            )
            record = audit_record(
                audit,
                profile_path=checkpoint,
                audit_path=audit_path,
                sweep=sweep,
            )
            if record["max_relative_gain"] < best["max_relative_gain"]:
                best = record
            if selected is None and record["local_one_percent_equilibrium"]:
                selected = record

        first_new_sweep = (completed_numbers[-1] + 1) if completed_numbers else 1
        if selected is None:
            for sweep in range(first_new_sweep, int(task["max_sweeps"]) + 1):
                sweep_started = time.perf_counter()
                before_sweep = clone_state(state)
                starting_market, starting_market_diagnostics = solve_nested_market(
                    data, state
                )
                starting_objectives = {
                    player: float(
                        nested_economic_objective(
                            data, state, starting_market, player
                        )
                    )
                    for player in order
                }
                player_rows: list[dict[str, Any]] = []
                for player_index, player in enumerate(order, start=1):
                    player_started = time.perf_counter()
                    before_player = clone_state(state)
                    market, market_diagnostics = solve_nested_market(data, state)
                    reference = float(
                        nested_economic_objective(data, state, market, player)
                    )
                    best_value, response, diagnostics = nested_best_response(
                        data,
                        state,
                        market,
                        player,
                        maxiter=int(task["maxiter"]),
                        starts=1,
                    )
                    retry_used = False
                    if not diagnostics["success"]:
                        retry_used = True
                        best_value, response, diagnostics = nested_best_response(
                            data,
                            state,
                            market,
                            player,
                            maxiter=max(2 * int(task["maxiter"]), 1000),
                            starts=1,
                        )
                    if not diagnostics["success"]:
                        raise RuntimeError(
                            f"sweep {sweep} {player}: clean-objective best response failed: "
                            f"{diagnostics.get('message')}"
                        )
                    gain = max(float(best_value) - reference, 0.0) / max(
                        abs(reference), 1.0
                    )
                    raw_move = float(
                        _strategy_distance(data, before_player, response, player)[0]
                    )
                    update_player(
                        data,
                        state,
                        response,
                        player,
                        float(task["alpha"]),
                    )
                    applied_move = float(
                        _strategy_distance(data, before_player, state, player)[0]
                    )
                    row = {
                        "player": player,
                        "reference_objective": reference,
                        "best_response_objective": float(best_value),
                        "relative_gain": gain,
                        "relative_gain_percent": 100.0 * gain,
                        "raw_strategy_move": raw_move,
                        "nominal_damping_weight": float(task["alpha"]),
                        "damping_weight": float(task["alpha"]),
                        "update_applied": True,
                        "move_cap": None,
                        "gain_filter": None,
                        "applied_strategy_move": applied_move,
                        "optimizer_success": bool(diagnostics["success"]),
                        "retry_used": retry_used,
                        "starts": 1,
                        "chosen_start_index": diagnostics.get("chosen_start_index"),
                        "attempts": diagnostics.get("attempts", []),
                        "reference_market_diagnostics": market_diagnostics,
                        "elapsed_seconds": time.perf_counter() - player_started,
                    }
                    player_rows.append(row)
                    if task["progress"] == "player":
                        print(
                            f"{prefix} sweep {sweep:02d}/{int(task['max_sweeps']):02d} "
                            f"player {player_index}/6 {player}: seq_gain={gain:.4%} "
                            f"raw_move={raw_move:.4%} applied={applied_move:.4%} "
                            f"retry={retry_used} elapsed={row['elapsed_seconds']:.1f}s",
                            flush=True,
                        )

                ending_market, ending_market_diagnostics = solve_nested_market(data, state)
                ending_objectives = {
                    player: float(
                        nested_economic_objective(data, state, ending_market, player)
                    )
                    for player in order
                }
                strategy_change = max(
                    float(_strategy_distance(data, before_sweep, state, player)[0])
                    for player in order
                )
                objective_change = max(
                    abs(ending_objectives[player] - starting_objectives[player])
                    / max(abs(starting_objectives[player]), 1.0)
                    for player in order
                )
                max_sequential_gain_row = max(
                    player_rows, key=lambda row: float(row["relative_gain"])
                )
                checkpoint = branch_root / f"sweep_{sweep:03d}.json"
                write_json(
                    checkpoint,
                    {
                        "created": now(),
                        "sequence": sequence,
                        "branch": branch,
                        "branch_specification": task["branch_specification"],
                        "sweep": sweep,
                        "player_order": order,
                        "alpha": float(task["alpha"]),
                        "objective_mode": OBJECTIVE_MODE,
                        "subtract_mu_offer_from_producer_margin": False,
                        "economic_quadratic_penalties": 0.0,
                        "algorithmic_proximal_penalties": 0.0,
                        "move_cap": None,
                        "gain_filter": None,
                        "players_frozen": False,
                        "terminal_salvage_fraction": float(
                            task["terminal_salvage_fraction"]
                        ),
                        "players": player_rows,
                        "updated_players": order,
                        "max_sequential_relative_gain": float(
                            max_sequential_gain_row["relative_gain"]
                        ),
                        "max_sequential_gain_player": str(
                            max_sequential_gain_row["player"]
                        ),
                        "strategy_change_metric": strategy_change,
                        "objective_change_metric": objective_change,
                        "starting_common_objectives": starting_objectives,
                        "ending_common_objectives": ending_objectives,
                        "starting_market_diagnostics": starting_market_diagnostics,
                        "ending_market_diagnostics": ending_market_diagnostics,
                        "ending_profile": full_state_payload(
                            data, state, ending_market
                        ),
                    },
                )

                audit_path = (
                    branch_root
                    / "audits"
                    / f"audit_sweep_{sweep:03d}_one_start.json"
                )
                frozen = run_or_load_audit(
                    audit_path,
                    data,
                    state,
                    order=order,
                    task=task,
                    audit_label=f"sweep-{sweep:03d}",
                )
                current = audit_record(
                    frozen,
                    profile_path=checkpoint,
                    audit_path=audit_path,
                    sweep=sweep,
                )
                if current["max_relative_gain"] < best["max_relative_gain"]:
                    best = current
                elapsed = time.perf_counter() - sweep_started
                if task["progress"] in ("sweep", "player"):
                    print(
                        f"{prefix} sweep {sweep:02d}/{int(task['max_sweeps']):02d} "
                        f"move={strategy_change:.4%} "
                        f"seq_gain={float(max_sequential_gain_row['relative_gain']):.4%}"
                        f"({max_sequential_gain_row['player']}) "
                        f"frozen_gain={current['max_relative_gain']:.4%}"
                        f"({current['max_gain_player']}) "
                        f"pass={current['local_one_percent_equilibrium']} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                write_json(
                    status_path,
                    {
                        "updated": now(),
                        "status": "running",
                        "pid": os.getpid(),
                        "sequence": sequence,
                        "branch": branch,
                        "sweep": sweep,
                        "max_sweeps": int(task["max_sweeps"]),
                        "objective_mode": OBJECTIVE_MODE,
                        "latest_strategy_change_metric": strategy_change,
                        "latest_max_sequential_relative_gain": float(
                            max_sequential_gain_row["relative_gain"]
                        ),
                        "latest_frozen_max_relative_gain": current[
                            "max_relative_gain"
                        ],
                        "latest_frozen_max_gain_player": current[
                            "max_gain_player"
                        ],
                        "best_frozen_max_relative_gain": best["max_relative_gain"],
                        "best_sweep": best["sweep"],
                    },
                )
                if current["local_one_percent_equilibrium"]:
                    selected = current
                    break

        chosen = selected or best
        result = {
            "created": now(),
            "status": "accepted" if selected is not None else "no_pass_within_schedule",
            "pid": os.getpid(),
            "run_index": int(task["run_index"]),
            "run_total": int(task["run_total"]),
            "sequence": sequence,
            "branch": branch,
            "branch_specification": task["branch_specification"],
            "player_order": order,
            "stage1_source": source_metadata,
            "input": relative(Path(task["input_path"])),
            "input_sha256": sha256(Path(task["input_path"])),
            "objective_mode": OBJECTIVE_MODE,
            "subtract_mu_offer_from_producer_margin": False,
            "economic_quadratic_penalties": 0.0,
            "algorithmic_proximal_penalties": 0.0,
            "terminal_salvage_fraction": float(task["terminal_salvage_fraction"]),
            "alpha": float(task["alpha"]),
            "price_factor": float(task["price_factor"]),
            "capacity_weight": float(task["capacity_weight"]),
            "max_sweeps": int(task["max_sweeps"]),
            "selected_profile": chosen["profile_path"],
            "selected_sweep": chosen["sweep"],
            "one_start_audit": chosen["audit_path"],
            "one_start_max_relative_gain": chosen["max_relative_gain"],
            "one_start_max_gain_player": chosen["max_gain_player"],
            "all_six_solves_successful": chosen["all_six_solves_successful"],
            "local_one_percent_equilibrium": chosen[
                "local_one_percent_equilibrium"
            ],
            "criterion": (
                "one-start common frozen-profile clean-objective maximum relative "
                f"gain <= {float(task['relative_gain_tolerance']):.2%}, all six solves successful"
            ),
            "claim_scope": "local computational criterion; no multistart or global claim",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, result)
        if task["progress"] != "quiet":
            print(
                f"{prefix} DONE status={result['status']} "
                f"selected_sweep={result['selected_sweep']} "
                f"frozen_gain={result['one_start_max_relative_gain']:.4%} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
        return result
    except Exception as exc:
        failure = {
            "created": now(),
            "status": "failed",
            "pid": os.getpid(),
            "run_index": int(task["run_index"]),
            "run_total": int(task["run_total"]),
            "sequence": sequence,
            "branch": branch,
            "objective_mode": OBJECTIVE_MODE,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(status_path, failure)
        print(f"{prefix} FAILED {failure['error']}", flush=True)
        return failure


def validate_sources(
    *,
    input_path: Path,
    stage1_root: Path,
    sequences: list[str],
    terminal_salvage_fraction: float,
    price_factor: float,
    capacity_weight: float,
    stage1_profiles: dict[str, dict[str, Any]] | None = None,
    fix_offers_to_cost: bool = False,
) -> None:
    print(
        f"[VALIDATE] input={relative(input_path)} sha256={sha256(input_path)}",
        flush=True,
    )
    for sequence in sequences:
        data, source, metadata = load_stage1_anchor(
            input_path,
            stage1_root,
            sequence,
            terminal_salvage_fraction,
            specification=(stage1_profiles or STAGE1_PROFILES)[sequence],
            fix_offers_to_cost=fix_offers_to_cost,
        )
        initial = make_factorial_initial_state(
            data,
            source,
            price_factor=price_factor,
            capacity_weight=capacity_weight,
        )
        _, market_diagnostics = solve_nested_market(data, initial)
        print(
            f"[VALIDATE] {sequence}: global_sweep={metadata['source_global_sweep']} "
            f"replay_error={metadata['maximum_replay_error']:.3g} "
            f"objective={data.settings['objective_mode']} "
            f"subtract_mu={data.settings['subtract_mu_offer_from_producer_margin']} "
            f"c_quad={data.settings['c_quad_q']}/{data.settings['c_quad_p']}/"
            f"{data.settings['c_quad_a']} "
            f"c_pen={data.settings['c_pen_q']}/{data.settings['c_pen_p']}/"
            f"{data.settings['c_pen_a']}/{data.settings['c_pen_dk']} "
            f"market_stationarity={float(market_diagnostics['max_positive_flow_stationarity']):.3g}",
            flush=True,
        )


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "run_index",
        "sequence",
        "branch",
        "status",
        "price_factor",
        "capacity_weight",
        "alpha",
        "selected_sweep",
        "one_start_max_relative_gain",
        "one_start_max_gain_player",
        "all_six_solves_successful",
        "local_one_percent_equilibrium",
        "elapsed_seconds",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument(
        "--stage1-specs", type=Path,
        help="JSON mapping of sequence names to Stage-1 replay specifications.",
    )
    parser.add_argument(
        "--fix-offers-to-cost", action="store_true",
        help="Capacity-only game with every bilateral offer fixed at exporter-period cost.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
    )
    parser.add_argument(
        "--price-factors", nargs="+", type=float, default=list(DEFAULT_PRICE_FACTORS)
    )
    parser.add_argument(
        "--capacity-weights",
        nargs="+",
        type=float,
        default=list(DEFAULT_CAPACITY_WEIGHTS),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-sweeps", type=int, default=15)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--terminal-salvage-fraction", type=float, default=0.5)
    parser.add_argument("--relative-gain-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--progress",
        choices=("quiet", "run", "sweep", "player"),
        default="sweep",
        help=(
            "Terminal detail: run start/end only, one line per sweep (default), "
            "or per-player lines in addition to sweep lines."
        ),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing explicit --output-root from saved branch checkpoints.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help="Allow an input other than the exact workbook used for the retained Stage-1 profiles.",
    )
    args = parser.parse_args()

    stage1_profiles = (
        json.loads(args.stage1_specs.read_text(encoding="utf-8"))
        if args.stage1_specs else STAGE1_PROFILES
    )
    sequences = list(args.sequences or stage1_profiles)
    unknown = set(sequences) - set(stage1_profiles)
    if unknown:
        raise ValueError(f"Unknown Stage-1 sequences: {sorted(unknown)}")
    price_factors = [1.0] if args.fix_offers_to_cost else list(args.price_factors)
    if args.fix_offers_to_cost and args.price_factors != list(DEFAULT_PRICE_FACTORS) and args.price_factors != [1.0]:
        raise ValueError("Fixed cost offers require --price-factors 1.0")

    if args.workers < 1 or args.max_sweeps < 1 or args.maxiter < 1:
        raise ValueError("workers, max-sweeps, and maxiter must be positive")
    if not 0.0 <= args.relative_gain_tolerance < 1.0:
        raise ValueError("relative-gain-tolerance must be in [0, 1)")
    if not 0.0 <= args.terminal_salvage_fraction <= 1.0:
        raise ValueError("terminal-salvage-fraction must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in args.capacity_weights):
        raise ValueError("capacity-weights must lie in [0, 1]")
    if any(not 0.0 < value <= 1.0 for value in args.alphas):
        raise ValueError("alphas must lie in (0, 1]")
    if any(value < 0.0 for value in price_factors):
        raise ValueError("price-factors must be nonnegative")

    input_path = args.input.resolve()
    stage1_root = args.stage1_root.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not stage1_root.is_dir():
        raise FileNotFoundError(stage1_root)
    input_sha256 = sha256(input_path)
    if (
        input_sha256 != EXPECTED_STAGE1_INPUT_SHA256
        and not args.allow_input_mismatch
    ):
        raise ValueError(
            "Input does not match the workbook used to create the retained Stage-1 "
            f"profiles. Expected SHA-256 {EXPECTED_STAGE1_INPUT_SHA256}, got "
            f"{input_sha256}. Use --allow-input-mismatch only for an intentional "
            "cross-input experiment."
        )

    common = {
        "input_path": str(input_path),
        "stage1_root": str(stage1_root),
        "output_root": "",
        "max_sweeps": args.max_sweeps,
        "maxiter": args.maxiter,
        "terminal_salvage_fraction": args.terminal_salvage_fraction,
        "relative_gain_tolerance": args.relative_gain_tolerance,
        "progress": args.progress,
        "fix_offers_to_cost": args.fix_offers_to_cost,
    }
    tasks = build_tasks(
        sequences=sequences,
        price_factors=price_factors,
        capacity_weights=list(args.capacity_weights),
        alphas=list(args.alphas),
        common=common,
    )
    for task in tasks:
        task["stage1_spec"] = stage1_profiles[task["sequence"]]
    if args.plan_only:
        print(
            f"[PLAN] {len(tasks)} runs; workers={min(args.workers, len(tasks))}; "
            f"objective={OBJECTIVE_MODE}; input={relative(input_path)}"
        )
        for task in tasks:
            print(
                f"[PLAN {task['run_index']:02d}/{task['run_total']:02d}] "
                f"{task['sequence']}/{task['branch']}"
            )
        return
    if args.validate_only:
        validate_sources(
            input_path=input_path,
            stage1_root=stage1_root,
            sequences=sequences,
            terminal_salvage_fraction=args.terminal_salvage_fraction,
            price_factor=float(price_factors[0]),
            capacity_weight=float(args.capacity_weights[0]),
            stage1_profiles=stage1_profiles,
            fix_offers_to_cost=args.fix_offers_to_cost,
        )
        print(f"[VALIDATE] OK: {len(tasks)} planned Stage-2 runs", flush=True)
        return

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT
        / "outputs"
        / "new_equilibria"
        / f"clean_stage2_factorial_{stamp}"
    )
    if args.resume:
        if args.output_root is None:
            raise ValueError("--resume requires an explicit --output-root")
        if not output_root.is_dir():
            raise FileNotFoundError(output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    for task in tasks:
        task["output_root"] = str(output_root)

    protocol = {
        "objective_mode": OBJECTIVE_MODE,
        "subtract_mu_offer_from_producer_margin": False,
        "economic_quadratic_penalties": 0.0,
        "algorithmic_proximal_penalties": 0.0,
        "terminal_salvage_fraction": args.terminal_salvage_fraction,
        "terminal_capacity_state_only": True,
        "fix_q_offer_to_kcap": True,
        "fix_a_bid_to_true_dem": True,
        "force_mu_offer_zero": False,
        "move_cap": None,
        "gain_filter": None,
        "players_frozen": False,
        "sequences": sequences,
        "price_factors": price_factors,
        "fix_offers_to_cost": args.fix_offers_to_cost,
        "capacity_weights": list(args.capacity_weights),
        "alphas": list(args.alphas),
        "max_sweeps": args.max_sweeps,
        "maxiter": args.maxiter,
        "relative_gain_tolerance": args.relative_gain_tolerance,
        "input": relative(input_path),
        "input_sha256": input_sha256,
        "stage1_root": relative(stage1_root),
        "stage1_profiles": stage1_profiles,
    }
    manifest_path = output_root / "manifest.json"
    if args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("protocol") != protocol:
            raise ValueError(
                "Resume protocol differs from the existing manifest; use matching "
                "arguments or start a fresh output root"
            )
        manifest = existing
        manifest["status"] = "running"
        manifest["resumed"] = now()
        manifest["workers"] = min(args.workers, len(tasks))
        manifest["progress"] = args.progress
    else:
        manifest = {
            "created": now(),
            "status": "running",
            "pid": os.getpid(),
            "output_root": relative(output_root),
            "workers": min(args.workers, len(tasks)),
            "progress": args.progress,
            "run_count": len(tasks),
            "protocol": protocol,
            "results": [],
        }
    write_json(manifest_path, manifest)

    worker_count = min(args.workers, len(tasks))
    print(
        f"[STAGE2] START {len(tasks)} runs with {worker_count} parallel workers; "
        f"progress={args.progress}; output={relative(output_root)}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        future_tasks = {pool.submit(run_branch, task): task for task in tasks}
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "created": now(),
                    "status": "failed",
                    "run_index": task["run_index"],
                    "run_total": task["run_total"],
                    "sequence": task["sequence"],
                    "branch": task["branch"],
                    "error": f"worker failure: {type(exc).__name__}: {exc}",
                }
            results.append(result)
            results.sort(key=lambda row: int(row["run_index"]))
            manifest["results"] = results
            manifest["completed_runs"] = len(results)
            manifest["failed_runs"] = sum(
                row["status"] == "failed" for row in results
            )
            write_json(manifest_path, manifest)
            if args.progress != "quiet":
                print(
                    f"[STAGE2] completed {len(results):02d}/{len(tasks):02d}: "
                    f"{result['sequence']}/{result['branch']} -> {result['status']}",
                    flush=True,
                )

    results.sort(key=lambda row: int(row["run_index"]))
    manifest["results"] = results
    manifest["completed"] = now()
    manifest["status"] = (
        "completed"
        if all(row["status"] != "failed" for row in results)
        else "failed"
    )
    manifest["accepted_runs"] = sum(
        row["status"] == "accepted" for row in results
    )
    manifest["failed_runs"] = sum(row["status"] == "failed" for row in results)
    write_json(manifest_path, manifest)
    write_summary(output_root / "summary.csv", results)
    print(
        f"[STAGE2] {manifest['status'].upper()}: "
        f"accepted={manifest['accepted_runs']}/{len(results)} "
        f"failed={manifest['failed_runs']} output={relative(output_root)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
