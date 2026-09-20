from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from model import run_gs
from model.data_prep import load_data_from_excel


SOURCE_WORKBOOK = ROOT / "outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx"
SOURCE_ITERATION = 21
PLAYER_ORDER = ["ch", "row", "apac", "us", "eu", "af"]
OUTPUT_ROOT = ROOT / "outputs/continuation/ch-row-apac-us-eu-af"
MANIFEST_PATH = ROOT / "workflow/continuation_manifest.json"


@dataclass(frozen=True)
class Stage:
    name: str
    penalty_scale: float
    omega: float
    block_sweeps: int


STAGES = [
    Stage("half_penalty", 0.50, 0.20, 10),
    Stage("quarter_penalty", 0.25, 0.15, 12),
    Stage("ten_percent_penalty", 0.10, 0.12, 15),
    Stage("two_percent_penalty", 0.02, 0.10, 20),
    Stage("zero_penalty", 0.00, 0.08, 30),
]

BASE_PENALTIES = {"q": 2.0, "p": 3.0, "a": 2.0, "dk": 2.0}
TARGET_RESIDUAL = 1e-3
STABLE_SWEEPS = 3


def _meta(path: Path) -> dict[str, object]:
    frame = pd.read_excel(path, sheet_name="meta")
    return dict(zip(frame.iloc[:, 0].astype(str), frame.iloc[:, 1]))


def _initial_model_data() -> tuple[mm.ModelData, run_gs.RunConfig, dict[str, dict]]:
    cfg = run_gs.RunConfig(
        params_region_sheet="params_region_new",
        discount_rate=0.02,
        base_year=2025,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        fix_q_offer_to_kcap=True,
        fix_a_bid_to_true_dem=True,
        force_mu_offer_zero=False,
    )
    data = load_data_from_excel(cfg.excel_path, params_region_sheet=cfg.params_region_sheet)
    run_gs._apply_data_overrides(data, cfg)
    initial = run_gs._build_initial_state(data, cfg, cfg.excel_path)
    return data, cfg, initial


def _strategy_residual(
    data: mm.ModelData,
    before: dict[str, dict],
    after: dict[str, dict],
) -> float:
    times = list(data.times or [])
    move_times = mm._move_times(times)
    if mm._terminal_capacity_state_only(data):
        conv_times = mm._operating_times(data)
        conv_move_times = move_times
    else:
        conv_times = times[:-1]
        conv_move_times = move_times[:-1]
    initial_capacity = mm._initial_capacity_by_region(data)
    residual = 0.0

    for player in data.players:
        exp_scale = float((data.g_exp_ub or {}).get(player, 0.0))
        if not bool(getattr(data, "g_exp_ub_is_absolute", False)):
            exp_scale *= float(initial_capacity[player])
        dec_scale = float((data.g_dec_ub or {}).get(player, 0.0)) * float(initial_capacity[player])
        dk_scale = max(exp_scale, dec_scale, 1.0)
        for tp in conv_move_times:
            key = (player, tp)
            residual = max(
                residual,
                abs(float(after["dK_net"][key]) - float(before["dK_net"][key])) / dk_scale,
            )
        for tp in conv_times:
            key = (player, tp)
            residual = max(
                residual,
                abs(float(after["Q_offer"][key]) - float(before["Q_offer"][key]))
                / max(float(initial_capacity[player]), 1.0),
            )

    for exporter in data.regions:
        for importer in data.regions:
            scale = max(float(data.p_offer_ub[(exporter, importer)]), 1e-3)
            for tp in conv_times:
                key = (exporter, importer, tp)
                residual = max(
                    residual,
                    abs(float(after["p_offer"][key]) - float(before["p_offer"][key])) / scale,
                )
    return residual


def replay_accepted_state(
    workbook: Path,
    initial_state: dict[str, dict],
    data: mm.ModelData,
    through_iteration: int | None = None,
    expected_player_order: list[str] | None = None,
) -> tuple[dict[str, dict], float]:
    """Recover internal damped theta values from a saved GS workbook.

    The final player's rows contain its raw best response. Other players' rows
    contain their accepted damped strategies. Q_offer and domestic offer-price
    theta values are replayed because fixed-player rows do not expose them.
    """
    detail = pd.read_excel(workbook, sheet_name="detailed_iters")
    iterations = pd.read_excel(workbook, sheet_name="iters").set_index("iter")
    meta = _meta(workbook)
    order = [str(v) for v in ast.literal_eval(str(meta["player_order"]))]
    expected_order = PLAYER_ORDER if expected_player_order is None else list(expected_player_order)
    if order != expected_order:
        raise ValueError(f"Unexpected player order in {workbook}: {order}")
    last_player = order[-1]
    max_iteration = int(iterations.index.max())
    target = max_iteration if through_iteration is None else int(through_iteration)
    if target > max_iteration:
        raise ValueError(f"Requested iteration {target}, but {workbook} ends at {max_iteration}")

    times = list(data.times or [])
    move_times = mm._move_times(times)
    initial_capacity = mm._initial_capacity_by_region(data)

    d_k = {
        (r, tp): float(initial_state.get("dK_net", {}).get((r, tp), 0.0))
        for r in data.players
        for tp in move_times
    }
    implied = mm._implied_capacity_path(data, times, d_k)
    q_offer = {
        (r, tp): mm._clip_value(
            float(initial_state.get("Q_offer", {}).get((r, tp), 0.8 * float(initial_capacity[r]))),
            0.0,
            max(float(implied[(r, tp)]), 0.0),
        )
        for r in data.players
        for tp in times
    }
    p_offer = {
        (ex, im, tp): float(initial_state.get("p_offer", {}).get((ex, im, tp), 0.0))
        for ex in data.regions
        for im in data.regions
        for tp in times
    }
    a_bid = {
        (r, tp): mm._true_demand_intercept(data, r, tp)
        for r in data.players
        for tp in times
    }

    max_replay_error = 0.0
    for iteration in range(1, target + 1):
        before = {
            "dK_net": dict(d_k),
            "Q_offer": dict(q_offer),
            "p_offer": dict(p_offer),
            "a_bid": dict(a_bid),
        }
        omega = float(iterations.loc[iteration, "omega"])
        block = detail[detail["iter"] == iteration].copy()
        block["t"] = block["t"].astype(str)

        for player in order:
            rows = block[block["r"] == player]
            for tp in move_times:
                key = (player, tp)
                recorded = float(rows.loc[rows["t"] == tp, "net_cap_change"].iloc[0])
                d_k[key] = (
                    (1.0 - omega) * d_k[key] + omega * recorded
                    if player == last_player
                    else recorded
                )

            implied = mm._implied_capacity_path(data, times, d_k)
            for tp in times:
                q_key = (player, tp)
                q_offer[q_key] = min(
                    (1.0 - omega) * q_offer[q_key] + omega * float(implied[q_key]),
                    max(float(implied[q_key]), 0.0),
                )
                row = rows.loc[rows["t"] == tp].iloc[0]
                for importer in data.regions:
                    p_key = (player, importer, tp)
                    if importer == player:
                        raw = float((data.c_man_t or {}).get((player, tp), data.c_man[player]))
                        p_offer[p_key] = (1.0 - omega) * p_offer[p_key] + omega * raw
                    else:
                        recorded = float(row[f"p_offer_to_{importer}"])
                        p_offer[p_key] = (
                            (1.0 - omega) * p_offer[p_key] + omega * recorded
                            if player == last_player
                            else recorded
                        )

        after = {
            "dK_net": dict(d_k),
            "Q_offer": dict(q_offer),
            "p_offer": dict(p_offer),
            "a_bid": dict(a_bid),
        }
        replayed = _strategy_residual(data, before, after)
        recorded = float(iterations.loc[iteration, "r_strat"])
        max_replay_error = max(max_replay_error, abs(replayed - recorded))

    return {
        "dK_net": d_k,
        "Q_offer": q_offer,
        "p_offer": p_offer,
        "a_bid": a_bid,
    }, max_replay_error


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {
            "selected_source": str(SOURCE_WORKBOOK.relative_to(ROOT)),
            "selected_iteration": SOURCE_ITERATION,
            "player_order": PLAYER_ORDER,
            "target_strategy_residual": TARGET_RESIDUAL,
            "stable_sweeps": STABLE_SWEEPS,
            "stages": [asdict(stage) for stage in STAGES],
            "runs": [],
        }
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict[str, object]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def recover_current_state(data: mm.ModelData, initial: dict[str, dict], manifest: dict[str, object]):
    state, source_error = replay_accepted_state(
        SOURCE_WORKBOOK, initial, data, through_iteration=SOURCE_ITERATION
    )
    errors = [(str(SOURCE_WORKBOOK), source_error)]
    for entry in manifest.get("runs", []):
        workbook = ROOT / str(entry["workbook"])
        state, error = replay_accepted_state(workbook, state, data)
        errors.append((str(workbook), error))
    return state, errors


def current_stage_index(manifest: dict[str, object]) -> int:
    completed = 0
    for entry in manifest.get("runs", []):
        if bool(entry.get("stage_converged", False)):
            completed = max(completed, int(entry["stage_index"]) + 1)
    return completed


def run_blocks(blocks: int) -> None:
    data, base_cfg, excel_initial = _initial_model_data()
    manifest = _load_manifest()
    state, replay_checks = recover_current_state(data, excel_initial, manifest)
    worst_replay_error = max(error for _, error in replay_checks)
    if worst_replay_error > 1e-10:
        raise RuntimeError(f"State replay check failed: max r_strat error={worst_replay_error:.3g}")
    print(f"[CONTINUATION] State replay verified; max residual error={worst_replay_error:.3g}")

    for _ in range(blocks):
        stage_index = current_stage_index(manifest)
        if stage_index >= len(STAGES):
            print("[CONTINUATION] All stages are complete.")
            return
        stage = STAGES[stage_index]
        stage_runs = [r for r in manifest.get("runs", []) if int(r["stage_index"]) == stage_index]
        block_index = len(stage_runs) + 1
        scale = stage.penalty_scale
        out_dir = OUTPUT_ROOT / f"stage_{stage_index + 1:02d}_{stage.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[CONTINUATION] stage={stage_index + 1}/{len(STAGES)} {stage.name} "
            f"block={block_index} scale={scale:g} omega={stage.omega:g}"
        )
        cfg = run_gs.RunConfig(
            excel_path=base_cfg.excel_path,
            out_dir=str(out_dir),
            plots_dir=str(OUTPUT_ROOT / "plots"),
            params_region_sheet="params_region_new",
            solver="ipopt",
            feastol=1e-4,
            opttol=1e-4,
            iters=stage.block_sweeps,
            omega=stage.omega,
            adaptive_omega=False,
            omega_min=stage.omega,
            omega_aggressive_sweeps=0,
            omega_ramp_iters=1,
            tol_strat=TARGET_RESIDUAL,
            stable_iters=STABLE_SWEEPS,
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

        # Continuation checkpoints do not need a full plot set after every block.
        run_gs.write_default_plots = None
        started = datetime.now().astimezone()
        output_path = Path(run_gs.run(cfg))
        finished = datetime.now().astimezone()
        iter_frame = pd.read_excel(output_path, sheet_name="iters")
        final = iter_frame.iloc[-1]
        stage_converged = int(final["stable_count"]) >= STABLE_SWEEPS
        state, replay_error = replay_accepted_state(output_path, state, data)
        if replay_error > 1e-10:
            raise RuntimeError(
                f"Checkpoint replay failed for {output_path}: r_strat error={replay_error:.3g}"
            )
        entry = {
            "stage_index": stage_index,
            "stage_name": stage.name,
            "block_index": block_index,
            "penalty_scale": scale,
            "omega": stage.omega,
            "workbook": str(output_path.relative_to(ROOT)),
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "sweeps_completed": int(len(iter_frame)),
            "ending_strategy_residual": float(final["r_strat"]),
            "ending_stable_count": int(final["stable_count"]),
            "stage_converged": bool(stage_converged),
            "replay_error": replay_error,
        }
        manifest.setdefault("runs", []).append(entry)
        manifest["updated"] = finished.isoformat(timespec="seconds")
        _save_manifest(manifest)
        print("[CONTINUATION] checkpoint", json.dumps(entry, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the selected equilibrium through staged proximal-penalty removal."
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=1,
        help="Number of checkpoint blocks to run. A stage repeats until it converges.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Reconstruct all saved checkpoints and verify their residual histories without solving.",
    )
    args = parser.parse_args()
    data, _, initial = _initial_model_data()
    manifest = _load_manifest()
    _, checks = recover_current_state(data, initial, manifest)
    worst = max(error for _, error in checks)
    print(f"[CONTINUATION] replay checks={checks}")
    if worst > 1e-10:
        raise RuntimeError(f"State replay check failed: max r_strat error={worst:.3g}")
    if not args.check_only:
        run_blocks(max(args.blocks, 1))


if __name__ == "__main__":
    main()
