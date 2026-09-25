"""Run one corrected-input EPEC calculation from a fresh primitive state.

This runner deliberately does not read any historical result, checkpoint,
certified profile, or workbook ``initial_state`` sheet.  It constructs a
deterministic cold state from the corrected model inputs only:

* offered quantity equals existing capacity in every period;
* net capacity change is zero;
* bilateral offer prices equal the exporter's 2025 manufacturing cost; and
* demand bids equal the calibrated true-demand intercept.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import model_main as model_it  # noqa: E402
from model.data_prep import load_data_from_excel  # noqa: E402
from model.run_gs import RunConfig, run  # noqa: E402


PARAMS_SHEET = "params_region_new"
DEFAULT_ORDER = ["ch", "row", "apac", "us", "eu", "af"]
MODEL_FILES = [
    "model/data_prep.py",
    "model/model_main.py",
    "model/gauss_seidel.py",
    "model/run_gs.py",
    "model/results_writer.py",
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _matrix(data: Any, values: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    return {
        region: {period: float(values[(region, period)]) for period in data.times}
        for region in data.regions
    }


def _build_fresh_state(
    data: Any, *, period_specific_cost_offers: bool = False
) -> dict[str, dict[tuple[str, ...], float]]:
    times = list(data.times)
    operating_times = set(model_it._operating_times(data))
    move_times = model_it._move_times(times)
    initial_capacity = model_it._initial_capacity_by_region(data)
    return {
        "Q_offer": {
            (region, period): (
                float(initial_capacity[region]) if period in operating_times else 0.0
            )
            for region in data.players
            for period in times
        },
        "dK_net": {
            (region, period): 0.0
            for region in data.players
            for period in move_times
        },
        "p_offer": {
            (exporter, importer, period): (
                float(
                    (data.c_man_t or {}).get((exporter, period), data.c_man[exporter])
                    if period_specific_cost_offers
                    else data.c_man[exporter]
                ) if period in operating_times else 0.0
            )
            for exporter in data.regions
            for importer in data.regions
            for period in times
        },
        "a_bid": {
            (region, period): (
                float(data.a_dem_t[(region, period)])
                if period in operating_times
                else 0.0
            )
            for region in data.regions
            for period in times
        },
    }


def _state_sha256(state: dict[str, dict[tuple[str, ...], float]]) -> str:
    rows: list[list[Any]] = []
    for name in sorted(state):
        for key, value in sorted(state[name].items()):
            rows.append([name, list(key), float(value)])
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def _config_payload(cfg: RunConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in dataclasses.fields(cfg):
        if field.name == "initial_state_override":
            continue
        value = getattr(cfg, field.name)
        result[field.name] = str(value) if isinstance(value, Path) else value
    result["initial_state_override"] = "fresh_cost_capacity_zero_change"
    return result


def _verify_result_parameters(result_path: Path, data: Any) -> dict[str, Any]:
    frame = pd.read_excel(result_path, sheet_name="regions")
    frame["r"] = frame["r"].astype(str).str.strip().str.lower()
    frame["t"] = frame["t"].astype(str).str.replace(r"\.0$", "", regex=True)
    actual_a = {
        (row.r, row.t): float(row.a_dem_used)
        for row in frame.itertuples(index=False)
    }
    actual_b = {
        (row.r, row.t): float(row.b_dem_used)
        for row in frame.itertuples(index=False)
    }
    a_diffs = {
        f"{region}|{period}": abs(actual_a[(region, period)] - float(data.a_dem_t[(region, period)]))
        for region in data.regions
        for period in data.times
    }
    b_diffs = {
        f"{region}|{period}": abs(actual_b[(region, period)] - float(data.b_dem_t[(region, period)]))
        for region in data.regions
        for period in data.times
    }
    max_a = max(a_diffs.values(), default=0.0)
    max_b = max(b_diffs.values(), default=0.0)
    tolerance = 1e-9
    return {
        "verified": max_a <= tolerance and max_b <= tolerance,
        "absolute_tolerance": tolerance,
        "max_abs_a_difference": max_a,
        "max_abs_b_difference": max_b,
        "a_differences": a_diffs,
        "b_differences": b_diffs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", default=",".join(DEFAULT_ORDER))
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--terminal-salvage-fraction", type=float, default=0.5)
    parser.add_argument(
        "--fix-offers-to-cost", action="store_true",
        help="Fix every bilateral offer to exporter-period marginal manufacturing cost.",
    )
    parser.add_argument("--log-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    order = [item.strip().lower() for item in args.order.split(",") if item.strip()]
    if len(order) != len(set(order)):
        raise ValueError(f"Player order contains duplicates: {order}")
    if args.terminal_salvage_fraction < 0.0:
        raise ValueError("terminal-salvage-fraction must be non-negative")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    provenance_path = output_dir / "provenance.json"
    start_clock = time.perf_counter()

    data = load_data_from_excel(str(input_path), params_region_sheet=PARAMS_SHEET)
    data.settings = dict(data.settings or {})
    data.settings["terminal_capacity_state_only"] = True
    if order != list(dict.fromkeys(order)) or set(order) != set(data.players):
        raise ValueError(f"Order must contain every player exactly once. Got {order}; players={data.players}")
    initial_state = _build_fresh_state(
        data, period_specific_cost_offers=args.fix_offers_to_cost
    )

    cfg = RunConfig(
        excel_path=str(input_path),
        out_dir=str(output_dir / "results"),
        plots_dir=str(output_dir / "plots"),
        params_region_sheet=PARAMS_SHEET,
        solver="ipopt",
        feastol=1e-4,
        opttol=1e-4,
        iters=int(args.iters),
        omega=0.8,
        adaptive_omega=True,
        omega_min=0.4,
        omega_aggressive_sweeps=5,
        omega_ramp_iters=10,
        tol_strat=1e-2,
        tol_raw_br=None,
        stable_iters=3,
        eps_x=1e-3,
        eps_comp=1e-3,
        keep_workdir=False,
        c_pen_q=0.5,
        c_pen_p=1.0,
        c_pen_a=0.5,
        c_pen_dk=0.5,
        c_pen_q_mid=1.0,
        c_pen_p_mid=2.0,
        c_pen_a_mid=1.0,
        c_pen_dk_mid=1.0,
        c_pen_q_final=2.0,
        c_pen_p_final=3.0,
        c_pen_a_final=2.0,
        c_pen_dk_final=2.0,
        c_pen_ramp_iters=10,
        c_quad_q=0.1,
        c_quad_p=0.1,
        c_quad_a=0.1,
        cap_keep_reward=0.0,
        capex_subsidy=0.0,
        terminal_salvage_fraction=float(args.terminal_salvage_fraction),
        terminal_capacity_state_only=True,
        decommission_penalty=0.0,
        fix_q_offer_to_kcap=True,
        fix_p_offer_to_c_man_t=args.fix_offers_to_cost,
        force_mu_offer_zero=False,
        fix_a_bid_to_true_dem=True,
        discount_rate=0.02,
        base_year=2025,
        initial_state_override=initial_state,
        player_order=order,
    )

    provenance: dict[str, Any] = {
        "classification": "corrected cold-start algorithm-development run; not paper-ready until equilibrium audit",
        "status": "running",
        "start_time": _now(),
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "log_path": str(args.log_path.resolve()) if args.log_path else None,
        "output_dir": str(output_dir),
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "parameter_sheet": PARAMS_SHEET,
        },
        "loaded_parameters": {
            "regions": list(data.regions),
            "periods": list(data.times),
            "Dmax": _matrix(data, data.Dmax_t),
            "a_dem": _matrix(data, data.a_dem_t),
            "b_dem": _matrix(data, data.b_dem_t),
            "validation": "passed in model.data_prep.load_data_from_excel before model construction",
            "operating_periods": model_it._operating_times(data),
            "terminal_capacity_state": list(data.times)[-1],
            "terminal_market": "inactive; no terminal demand, production, trade, prices, or operating payoff",
        },
        "player_order": order,
        "algorithm": _config_payload(cfg),
        "initialization": {
            "method": "fresh_cost_capacity_zero_change",
            "source": "constructed only from corrected input primitives; workbook initial_state sheets not read",
            "Q_offer": "existing capacity in every period",
            "dK_net": "zero in every capacity-move period",
            "p_offer": (
                "exporter-period marginal manufacturing cost for every importer"
                if args.fix_offers_to_cost
                else "exporter 2025 manufacturing cost for every importer and period"
            ),
            "a_bid": "calibrated true-demand intercept",
            "canonical_state_sha256": _state_sha256(initial_state),
        },
        "software": {
            "python": sys.version,
            "gamspy": importlib.metadata.version("gamspy"),
            "git_head": _git_value("rev-parse", "HEAD"),
            "git_status_porcelain": _git_value("status", "--porcelain"),
            "code_sha256": {
                relative: _sha256(ROOT / relative)
                for relative in MODEL_FILES
            },
        },
    }
    _write_json(provenance_path, provenance)

    try:
        result_path = Path(run(cfg)).resolve()
        result_parameter_check = _verify_result_parameters(result_path, data)
        input_hash_after = _sha256(input_path)
        provenance.update(
            {
                "status": "complete",
                "end_time": _now(),
                "elapsed_seconds": time.perf_counter() - start_clock,
                "result": {
                    "path": str(result_path),
                    "sha256": _sha256(result_path),
                    "loaded_parameter_check": result_parameter_check,
                },
                "input_sha256_after_run": input_hash_after,
                "input_unchanged_during_run": input_hash_after == provenance["input"]["sha256"],
            }
        )
        if not result_parameter_check["verified"]:
            provenance["status"] = "failed_parameter_verification"
            _write_json(provenance_path, provenance)
            raise RuntimeError("Result workbook demand parameters do not match the corrected loaded matrices")
        _write_json(provenance_path, provenance)
        print(f"[PROVENANCE] {provenance_path}", flush=True)
        return 0
    except Exception as exc:
        provenance.update(
            {
                "status": "failed",
                "end_time": _now(),
                "elapsed_seconds": time.perf_counter() - start_clock,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "input_sha256_after_run": _sha256(input_path),
            }
        )
        _write_json(provenance_path, provenance)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
