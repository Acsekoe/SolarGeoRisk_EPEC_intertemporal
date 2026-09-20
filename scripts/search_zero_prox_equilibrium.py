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
    PLAYER_ORDER,
    SOURCE_ITERATION,
    SOURCE_WORKBOOK,
    _initial_model_data,
    replay_accepted_state,
)


OUTPUT_ROOT = ROOT / "outputs" / "equilibrium_search" / "ch-row-apac-us-eu-af" / "zero_prox_conopt_omega_010"
MANIFEST_PATH = ROOT / "workflow" / "zero_prox_conopt_search_manifest.json"
SOLVER = "conopt"
OMEGA = 0.10
BLOCK_SWEEPS = 10
TARGET_DAMPED_RESIDUAL = 1e-3
STABLE_SWEEPS = 3


def _load_manifest() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {
            "selected_source": str(SOURCE_WORKBOOK.relative_to(ROOT)),
            "selected_iteration": SOURCE_ITERATION,
            "player_order": PLAYER_ORDER,
            "algorithmic_proximal_penalties": {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0},
            "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
            "omega": OMEGA,
            "block_sweeps": BLOCK_SWEEPS,
            "target_damped_strategy_residual": TARGET_DAMPED_RESIDUAL,
            "stable_sweeps": STABLE_SWEEPS,
            "runs": [],
        }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if float(manifest["omega"]) != OMEGA:
        raise ValueError(f"Manifest omega {manifest['omega']} does not match configured omega {OMEGA}")
    return manifest


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


def run_blocks(blocks: int) -> None:
    data, base_cfg, excel_initial = _initial_model_data()
    manifest = _load_manifest()
    state, checks = _recover_state(data, excel_initial, manifest)
    worst = max(error for _, error in checks)
    if worst > 1e-10:
        raise RuntimeError(f"State replay check failed: max r_strat error={worst:.3g}")
    print(f"[ZERO-PROX SEARCH] replay verified; max residual error={worst:.3g}")

    for _ in range(blocks):
        block_index = len(manifest.get("runs", [])) + 1
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cfg = run_gs.RunConfig(
            excel_path=base_cfg.excel_path,
            out_dir=str(OUTPUT_ROOT),
            plots_dir=str(OUTPUT_ROOT / "plots"),
            params_region_sheet="params_region_new",
            solver=SOLVER,
            feastol=1e-4,
            opttol=1e-4,
            iters=BLOCK_SWEEPS,
            omega=OMEGA,
            adaptive_omega=False,
            omega_min=OMEGA,
            omega_aggressive_sweeps=0,
            omega_ramp_iters=1,
            tol_strat=TARGET_DAMPED_RESIDUAL,
            stable_iters=STABLE_SWEEPS,
            eps_x=1e-3,
            eps_comp=1e-3,
            keep_workdir=False,
            c_pen_q=0.0,
            c_pen_p=0.0,
            c_pen_a=0.0,
            c_pen_dk=0.0,
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

        run_gs.write_default_plots = None
        print(
            f"[ZERO-PROX SEARCH] block={block_index} sweeps={BLOCK_SWEEPS} "
            f"omega={OMEGA:g} c_pen=(0,0,0,0)",
            flush=True,
        )
        started = datetime.now().astimezone()
        output_path = Path(run_gs.run(cfg))
        finished = datetime.now().astimezone()
        iter_frame = pd.read_excel(output_path, sheet_name="iters")
        final = iter_frame.iloc[-1]
        state, replay_error = replay_accepted_state(output_path, state, data)
        if replay_error > 1e-10:
            raise RuntimeError(
                f"Checkpoint replay failed for {output_path}: r_strat error={replay_error:.3g}"
            )
        entry = {
            "block_index": block_index,
            "omega": OMEGA,
            "solver": SOLVER,
            "workbook": str(output_path.relative_to(ROOT)),
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "sweeps_completed": int(len(iter_frame)),
            "ending_damped_strategy_residual": float(final["r_strat"]),
            "approximate_raw_strategy_gap": float(final["r_strat"]) / OMEGA,
            "ending_stable_count": int(final["stable_count"]),
            "replay_error": replay_error,
        }
        manifest.setdefault("runs", []).append(entry)
        manifest["updated"] = finished.isoformat(timespec="seconds")
        _save_manifest(manifest)
        print("[ZERO-PROX SEARCH] checkpoint", json.dumps(entry, sort_keys=True), flush=True)


def main() -> None:
    global OMEGA, OUTPUT_ROOT, MANIFEST_PATH
    parser = argparse.ArgumentParser(
        description="Search from the selected paper profile with zero proximal penalties and low damping."
    )
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--omega", type=float, default=OMEGA)
    parser.add_argument(
        "--branch",
        default="zero_prox_conopt_omega_010",
        help="Single directory-name label for an independent search branch.",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        help="For a new branch, copy checkpoint entries from this manifest before solving.",
    )
    parser.add_argument(
        "--seed-runs",
        type=int,
        default=0,
        help="Number of checkpoint entries to copy from --seed-manifest.",
    )
    args = parser.parse_args()
    if not 0.0 < float(args.omega) <= 1.0:
        raise ValueError("--omega must be in (0, 1]")
    if Path(args.branch).name != args.branch:
        raise ValueError("--branch must be a single directory name")
    OMEGA = float(args.omega)
    OUTPUT_ROOT = (
        ROOT / "outputs" / "equilibrium_search" / "ch-row-apac-us-eu-af" / args.branch
    )
    MANIFEST_PATH = ROOT / "workflow" / f"{args.branch}_manifest.json"

    if args.seed_manifest is not None and not MANIFEST_PATH.exists():
        seed_path = args.seed_manifest
        if not seed_path.is_absolute():
            seed_path = ROOT / seed_path
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        take = int(args.seed_runs)
        if take < 0 or take > len(seed.get("runs", [])):
            raise ValueError("--seed-runs is outside the source manifest's run range")
        manifest = _load_manifest()
        manifest["runs"] = list(seed.get("runs", []))[:take]
        manifest["seed_manifest"] = str(seed_path.relative_to(ROOT))
        manifest["seed_runs"] = take
        _save_manifest(manifest)

    data, _, initial = _initial_model_data()
    manifest = _load_manifest()
    _, checks = _recover_state(data, initial, manifest)
    worst = max(error for _, error in checks)
    print(f"[ZERO-PROX SEARCH] replay checks={checks}")
    if worst > 1e-10:
        raise RuntimeError(f"State replay check failed: max r_strat error={worst:.3g}")
    if not args.check_only:
        run_blocks(max(int(args.blocks), 1))


if __name__ == "__main__":
    main()
