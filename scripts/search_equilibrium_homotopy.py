from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from model import run_gs
from scripts.continue_selected_equilibrium import (
    BASE_PENALTIES,
    PLAYER_ORDER,
    SOURCE_ITERATION,
    SOURCE_WORKBOOK,
    _initial_model_data,
    replay_accepted_state,
)


OUTPUT_ROOT = ROOT / "outputs" / "equilibrium_homotopy" / "ch-row-apac-us-eu-af"
MANIFEST_PATH = ROOT / "workflow" / "equilibrium_homotopy_manifest.json"

PENALTY_SCALES = [1.00, 0.80, 0.60, 0.45, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.00]
DEFAULT_OMEGA = 0.30
OMEGA_RETRIES = [0.30, 0.25, 0.20]
BLOCK_SWEEPS = 5
RAW_RESPONSE_TOLERANCE = 1e-2
STABLE_SWEEPS = 5
SPIKE_MULTIPLIER = 1.5
SOLVER = "conopt"


def _scale_id(scale: float) -> str:
    return f"rho_{scale:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _stage(scale: float) -> dict[str, object]:
    return {
        "id": _scale_id(scale),
        "penalty_scale": float(scale),
        "status": "pending",
        "omega": DEFAULT_OMEGA,
    }


def _new_manifest() -> dict[str, object]:
    return {
        "selected_source": str(SOURCE_WORKBOOK.relative_to(ROOT)),
        "selected_iteration": SOURCE_ITERATION,
        "player_order": PLAYER_ORDER,
        "solver": SOLVER,
        "algorithm": "raw-best-response proximal homotopy",
        "base_algorithmic_proximal_penalties": BASE_PENALTIES,
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "penalty_scales": PENALTY_SCALES,
        "default_omega": DEFAULT_OMEGA,
        "omega_retries": OMEGA_RETRIES,
        "block_sweeps": BLOCK_SWEEPS,
        "raw_response_tolerance": RAW_RESPONSE_TOLERANCE,
        "stable_sweeps": STABLE_SWEEPS,
        "spike_multiplier": SPIKE_MULTIPLIER,
        "stages": [_stage(scale) for scale in PENALTY_SCALES],
        # Only accepted checkpoints appear in runs and are replayed.  Rejected
        # workbooks remain available as diagnostics but never alter the branch.
        "runs": [],
        "rejected_runs": [],
    }


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return _new_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("selected_iteration") != SOURCE_ITERATION:
        raise ValueError("Homotopy manifest does not use the selected iteration-21 source")
    # The solver is a property of the search implementation, while every run
    # also records the solver actually used.  This permits a clean solver
    # change without losing rejected-run diagnostics.
    manifest["solver"] = SOLVER
    return manifest


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Parse Excel booleans without treating the string 'False' as truthy."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}
    )


def _save_manifest(manifest: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _recover_state(
    data: mm.ModelData,
    initial: dict[str, dict],
    manifest: dict[str, object],
) -> tuple[dict[str, dict], list[tuple[str, float]]]:
    state, error = replay_accepted_state(
        SOURCE_WORKBOOK,
        initial,
        data,
        through_iteration=SOURCE_ITERATION,
    )
    checks = [(str(SOURCE_WORKBOOK), error)]
    for entry in manifest.get("runs", []):
        workbook = ROOT / str(entry["workbook"])
        state, error = replay_accepted_state(workbook, state, data)
        checks.append((str(workbook), error))
    return state, checks


def _current_stage(manifest: dict[str, object]) -> dict[str, object] | None:
    for stage in manifest.get("stages", []):
        if stage.get("status") != "complete":
            return stage
    return None


def _stage_runs(manifest: dict[str, object], stage_id: str) -> list[dict[str, object]]:
    return [entry for entry in manifest.get("runs", []) if entry["stage_id"] == stage_id]


def _trailing_stable_count(values: list[float], tolerance: float) -> int:
    count = 0
    for value in reversed(values):
        if value > tolerance:
            break
        count += 1
    return count


def _next_lower_omega(omega: float) -> float | None:
    for candidate in OMEGA_RETRIES:
        if candidate < omega - 1e-12:
            return candidate
    return None


def _insert_midpoint_stage(
    manifest: dict[str, object],
    target: dict[str, object],
) -> dict[str, object] | None:
    stages = manifest["stages"]
    target_index = stages.index(target)
    if target_index == 0:
        return None
    previous_scale = float(stages[target_index - 1]["penalty_scale"])
    target_scale = float(target["penalty_scale"])
    midpoint = 0.5 * (previous_scale + target_scale)
    if abs(midpoint - previous_scale) < 1e-4 or abs(midpoint - target_scale) < 1e-4:
        return None
    midpoint_id = _scale_id(midpoint)
    for stage in stages:
        if stage["id"] == midpoint_id:
            return None
    inserted = _stage(midpoint)
    inserted["inserted_for"] = target["id"]
    stages.insert(target_index, inserted)
    manifest["penalty_scales"] = [float(stage["penalty_scale"]) for stage in stages]
    return inserted


def _run_config(
    base_cfg: run_gs.RunConfig,
    state: dict[str, dict],
    stage: dict[str, object],
) -> run_gs.RunConfig:
    scale = float(stage["penalty_scale"])
    omega = float(stage["omega"])
    out_dir = OUTPUT_ROOT / str(stage["id"])
    return run_gs.RunConfig(
        excel_path=base_cfg.excel_path,
        out_dir=str(out_dir),
        plots_dir=str(OUTPUT_ROOT / "plots"),
        params_region_sheet="params_region_new",
        solver=SOLVER,
        feastol=1e-4,
        opttol=1e-4,
        iters=BLOCK_SWEEPS,
        omega=omega,
        adaptive_omega=False,
        omega_min=omega,
        omega_aggressive_sweeps=0,
        omega_ramp_iters=1,
        # The block runner owns cross-workbook stability counting, so prevent
        # the inner solver from stopping early within a five-sweep checkpoint.
        tol_strat=RAW_RESPONSE_TOLERANCE,
        tol_raw_br=RAW_RESPONSE_TOLERANCE,
        stable_iters=1_000_000,
        eps_x=1e-3,
        eps_comp=1e-3,
        keep_workdir=False,
        c_pen_q=BASE_PENALTIES["q"] * scale,
        c_pen_p=BASE_PENALTIES["p"] * scale,
        c_pen_a=BASE_PENALTIES["a"] * scale,
        c_pen_dk=BASE_PENALTIES["dk"] * scale,
        c_pen_q_mid=None,
        c_pen_p_mid=None,
        c_pen_a_mid=None,
        c_pen_dk_mid=None,
        c_pen_q_final=None,
        c_pen_p_final=None,
        c_pen_a_final=None,
        c_pen_dk_final=None,
        c_pen_ramp_iters=1,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        cap_keep_reward=0.0,
        capex_subsidy=0.0,
        terminal_salvage_fraction=0.0,
        decommission_penalty=0.0,
        fix_q_offer_to_kcap=True,
        force_mu_offer_zero=False,
        fix_a_bid_to_true_dem=True,
        discount_rate=0.02,
        base_year=2025,
        initial_state_override=state,
        player_order=PLAYER_ORDER,
    )


def _audit_checkpoint(manifest: dict[str, object], entry: dict[str, object]) -> None:
    try:
        from scripts.audit_selected_equilibrium import audit

        json_path, xlsx_path = audit(
            "search",
            0.01,
            "conopt",
            PLAYER_ORDER,
            MANIFEST_PATH,
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        valid = [record for record in payload["records"] if record.get("audit_valid")]
        entry["audit"] = {
            "json": str(json_path.relative_to(ROOT)),
            "xlsx": str(xlsx_path.relative_to(ROOT)),
            "all_players_valid": len(valid) == len(PLAYER_ORDER),
            "all_players_pass_1pct": bool(payload.get("all_players_pass")),
            "max_valid_relative_gain": max(
                (float(record["relative_gain"]) for record in valid),
                default=None,
            ),
        }
    except Exception as exc:
        entry["audit"] = {"error": f"{type(exc).__name__}: {exc}"}
    manifest["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _save_manifest(manifest)


def run_blocks(blocks: int, *, audit_after_block: bool) -> None:
    data, base_cfg, excel_initial = _initial_model_data()
    manifest = _load_manifest()
    state, checks = _recover_state(data, excel_initial, manifest)
    worst = max(error for _, error in checks)
    if worst > 1e-10:
        raise RuntimeError(f"State replay check failed: max r_strat error={worst:.3g}")
    print(f"[HOMOTOPY] replay verified; max residual error={worst:.3g}")

    for _ in range(blocks):
        stage = _current_stage(manifest)
        if stage is None:
            print("[HOMOTOPY] all penalty stages are complete")
            return
        stage_id = str(stage["id"])
        scale = float(stage["penalty_scale"])
        omega = float(stage["omega"])
        block_index = len(_stage_runs(manifest, stage_id)) + 1
        print(
            f"[HOMOTOPY] stage={stage_id} block={block_index} "
            f"scale={scale:g} omega={omega:g}",
            flush=True,
        )

        run_gs.write_default_plots = None
        started = datetime.now().astimezone()
        output_path = Path(run_gs.run(_run_config(base_cfg, state, stage)))
        finished = datetime.now().astimezone()
        frame = pd.read_excel(output_path, sheet_name="iters")
        if "r_raw_br" not in frame:
            raise RuntimeError("Checkpoint does not contain the required r_raw_br metric")
        raw_path = [float(value) for value in frame["r_raw_br"]]
        damped_path = [float(value) for value in frame["r_strat"]]
        solves_ok = bool(_as_bool_series(frame["all_solves_acceptable"]).all())

        previous = manifest.get("runs", [])[-1] if manifest.get("runs") else None
        prior_baseline = None if previous is None else float(previous["ending_raw_response_residual"])
        block_baseline = raw_path[0]
        spike_baselines = [block_baseline]
        if prior_baseline is not None:
            spike_baselines.append(prior_baseline)
        baseline = min(spike_baselines)
        # Guard the very first checkpoint as well as later ones: a block whose
        # ending response residual materially exceeds either its own starting
        # value or the last accepted checkpoint is rolled back.
        spike = raw_path[-1] > SPIKE_MULTIPLIER * max(baseline, 1e-12)
        accepted = solves_ok and not spike
        common = {
            "stage_id": stage_id,
            "penalty_scale": scale,
            "block_index": block_index,
            "omega": omega,
            "solver": SOLVER,
            "workbook": str(output_path.relative_to(ROOT)),
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "sweeps_completed": int(len(frame)),
            "raw_response_path": raw_path,
            "damped_step_path": damped_path,
            "ending_raw_response_residual": raw_path[-1],
            "ending_damped_step": damped_path[-1],
            "raw_response_spike_baseline": baseline,
            "all_solves_acceptable": solves_ok,
            "largest_final_raw_response_player": str(frame.iloc[-1]["raw_br_player"]),
            "largest_final_raw_response_coordinate": str(frame.iloc[-1]["raw_br_coordinate"]),
        }

        if not accepted:
            reasons = []
            if not solves_ok:
                reasons.append("one or more player solves were not acceptable")
            if spike:
                reasons.append(
                    f"ending raw residual exceeded {SPIKE_MULTIPLIER:g}x baseline {baseline:.6g}"
                )
            common["rejection_reasons"] = reasons
            manifest.setdefault("rejected_runs", []).append(common)
            lower_omega = _next_lower_omega(omega)
            if not solves_ok:
                # Damping does not repair an indeterminate NLP solve.  Keep
                # the stage and state unchanged for a solver-level retry.
                stage["status"] = "solver_failed"
                common["next_action"] = f"retry {stage_id} after resolving {SOLVER} solve status"
            elif lower_omega is not None:
                stage["omega"] = lower_omega
                stage["status"] = "retry"
                common["next_action"] = f"retry {stage_id} at omega={lower_omega:g}"
            else:
                inserted = _insert_midpoint_stage(manifest, stage)
                if inserted is not None:
                    stage["omega"] = DEFAULT_OMEGA
                    stage["status"] = "pending"
                    common["next_action"] = f"inserted {inserted['id']} before {stage_id}"
                else:
                    stage["status"] = "blocked"
                    common["next_action"] = "manual review required"
            manifest["updated"] = finished.isoformat(timespec="seconds")
            _save_manifest(manifest)
            print("[HOMOTOPY] rejected checkpoint", json.dumps(common, sort_keys=True))
            continue

        new_state, replay_error = replay_accepted_state(output_path, state, data)
        if replay_error > 1e-10:
            raise RuntimeError(
                f"Checkpoint replay failed for {output_path}: r_strat error={replay_error:.3g}"
            )
        state = new_state
        common["replay_error"] = replay_error
        prior_raw = [
            value
            for entry in _stage_runs(manifest, stage_id)
            for value in entry["raw_response_path"]
        ]
        trailing = _trailing_stable_count(
            prior_raw + raw_path,
            RAW_RESPONSE_TOLERANCE,
        )
        common["ending_stable_raw_sweeps"] = trailing
        common["stage_converged"] = trailing >= STABLE_SWEEPS
        manifest.setdefault("runs", []).append(common)
        if common["stage_converged"]:
            stage["status"] = "complete"
            stage["completed"] = finished.isoformat(timespec="seconds")
        else:
            stage["status"] = "active"
        manifest["updated"] = finished.isoformat(timespec="seconds")
        _save_manifest(manifest)
        print("[HOMOTOPY] accepted checkpoint", json.dumps(common, sort_keys=True))

        if audit_after_block:
            _audit_checkpoint(manifest, common)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search from selected iteration 21 using moderate damping, raw-response "
            "stopping, safeguarded rollback, and gradual proximal-penalty homotopy."
        )
    )
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--audit", action="store_true", help="Run a zero-proximal audit after each accepted block.")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = _load_manifest()
    data, _, initial = _initial_model_data()
    _, checks = _recover_state(data, initial, manifest)
    worst = max(error for _, error in checks)
    print(f"[HOMOTOPY] replay checks={checks}")
    if worst > 1e-10:
        raise RuntimeError(f"State replay check failed: max r_strat error={worst:.3g}")
    if not args.check_only:
        run_blocks(max(int(args.blocks), 1), audit_after_block=bool(args.audit))


if __name__ == "__main__":
    main()
