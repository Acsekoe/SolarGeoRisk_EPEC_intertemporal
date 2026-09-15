from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import model_main as mm
from scripts.run_local_paper_equilibrium_experiment import (
    CERTIFIED_AUDIT,
    CERTIFIED_CHECKPOINT,
    OUTPUT_ROOT,
    PLAYER_ORDER,
    normalized_l2,
    paper_state,
)
from scripts.search_nested_equilibrium import _deserialize_state


BEST_AUDIT = OUTPUT_ROOT / "branch_F_alpha_0p50" / "audit_sweep_006.json"


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def state_from_profile(data, paper, profile):
    return _deserialize_state(profile["strategy"], data, paper)


def distance_components(data, paper, state):
    times = list(data.times or [])
    cap0 = mm._implied_capacity_path(data, times, paper["dK_net"])
    cap1 = mm._implied_capacity_path(data, times, state["dK_net"])
    cap_values = np.array([cap1[(p, t)] - cap0[(p, t)] for p in PLAYER_ORDER for t in times])
    cap_scales = np.array([max(abs(cap0[(p, t)]), 1.0) for p in PLAYER_ORDER for t in times])
    price_values, price_scales = [], []
    for e in PLAYER_ORDER:
        for i in data.regions:
            if i == e: continue
            for t in times:
                price_values.append(state["p_offer"][(e, i, t)] - paper["p_offer"][(e, i, t)])
                price_scales.append(max(float(data.p_offer_ub[(e, i)]), 1.0))
    return {
        "combined_normalized_l2": normalized_l2(data, paper, state),
        "capacity_relative_rms": float(np.sqrt(np.mean((cap_values / cap_scales) ** 2))),
        "offer_price_bound_normalized_rms": float(np.sqrt(np.mean((np.asarray(price_values) / np.asarray(price_scales)) ** 2))),
        "capacity_mean_absolute_change_gw": float(np.mean(np.abs(cap_values))),
        "offer_price_mean_absolute_change_usd_per_kw": float(np.mean(np.abs(price_values))),
    }


def main():
    data, paper, _ = paper_state()
    paper_audit = json.loads((OUTPUT_ROOT / "audit_paper_profile.json").read_text(encoding="utf-8"))
    best_audit = json.loads(BEST_AUDIT.read_text(encoding="utf-8"))
    certified_audit = json.loads(CERTIFIED_AUDIT.read_text(encoding="utf-8"))
    certified_checkpoint = json.loads(CERTIFIED_CHECKPOINT.read_text(encoding="utf-8"))
    states = {
        "paper_profile": paper,
        "best_local_candidate": state_from_profile(data, paper, best_audit["profile"]),
        "previous_cost_price_candidate": _deserialize_state(certified_checkpoint["state"], data, paper),
    }
    audits = {
        "paper_profile": paper_audit,
        "best_local_candidate": best_audit,
        "previous_cost_price_candidate": certified_audit,
    }
    metadata = {
        "paper_profile": {"initialization": "reported paper profile, iteration 21", "alpha": None, "sweeps": 0},
        "best_local_candidate": {"initialization": "Branch F: 50% paper-price / 50% manufacturing-cost interpolation; paper capacities", "alpha": 0.50, "sweeps": 6},
        "previous_cost_price_candidate": {"initialization": "manufacturing-cost offer-price restart (separate prior robustness result)", "alpha": "mixed capped search", "sweeps": 5},
    }
    caps, offers, prices, objs, summary = [], [], [], [], []
    full = {}
    for name, state in states.items():
        audit = audits[name]
        market = audit["profile"]["market"] if "profile" in audit else None
        if market is None:
            # Older certified audit stores diagnostics only; the pinned comparison
            # has already validated the state, and this report reconstructs values.
            from scripts.nested_market_audit import solve_nested_market
            market_state, market_diag = solve_nested_market(data, state)
            market = {
                "clearing_prices": [{"region": r, "time": t, "value": market_state["lam"][(r, t)]}
                    for r in data.regions for t in list(data.times or [])]
            }
        else:
            market_diag = audit["reference_market_diagnostics"]
        capacity = mm._implied_capacity_path(data, list(data.times or []), state["dK_net"])
        audit_rows = {r["player"]: r for r in audit["players"]}
        for p in PLAYER_ORDER:
            for t in list(data.times or []):
                caps.append({"profile": name, "player": p, "time": t, "capacity_gw": capacity[(p, t)]})
            for importer in data.regions:
                if importer == p: continue
                for t in list(data.times or []):
                    offers.append({"profile": name, "exporter": p, "importer": importer, "time": t,
                        "offer_price_usd_per_kw": state["p_offer"][(p, importer, t)]})
            objs.append({"profile": name, "player": p,
                "economic_objective": audit_rows[p]["reference_objective"],
                "relative_unilateral_gain": audit_rows[p]["relative_gain"]})
        for row in market["clearing_prices"]:
            prices.append({"profile": name, "region": row["region"], "time": row["time"],
                "market_price_usd_per_kw": row["value"]})
        distance = distance_components(data, paper, state)
        diag = audit.get("reference_market_diagnostics", audit.get("reference_market", {}))
        summary.append({"profile": name, **metadata[name], "max_relative_gain": audit["max_relative_gain"],
            "max_gain_player": max(audit["players"], key=lambda r: r["relative_gain"])["player"],
            "all_attempts_successful": audit["all_attempts_successful"], "equilibrium_verified": audit["equilibrium_verified"],
            **distance, "max_balance_residual": diag["max_balance_residual"],
            "max_capacity_violation": diag["max_capacity_violation"],
            "max_stationarity_residual": diag["max_positive_flow_stationarity"]})
        full[name] = {"metadata": metadata[name], "distance_from_paper": distance,
            "audit": {"max_relative_gain": audit["max_relative_gain"], "players": audit["players"],
                "start_definitions": audit.get("start_definitions"),
                "all_attempts_successful": audit["all_attempts_successful"], "equilibrium_verified": audit["equilibrium_verified"]},
            "market_diagnostics": diag}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cross_profile_distances = {
        "paper_to_best_local": normalized_l2(data, states["paper_profile"], states["best_local_candidate"]),
        "paper_to_previous_cost_price": normalized_l2(data, states["paper_profile"], states["previous_cost_price_candidate"]),
        "best_local_to_previous_cost_price": normalized_l2(data, states["best_local_candidate"], states["previous_cost_price_candidate"]),
    }
    write_csv(OUTPUT_ROOT / "comparison_capacities.csv", caps)
    write_csv(OUTPUT_ROOT / "comparison_offer_prices.csv", offers)
    write_csv(OUTPUT_ROOT / "comparison_market_prices.csv", prices)
    write_csv(OUTPUT_ROOT / "comparison_objectives.csv", objs)
    write_csv(OUTPUT_ROOT / "comparison_summary.csv", summary)
    (OUTPUT_ROOT / "final_comparison.json").write_text(json.dumps({"profiles": full,
        "cross_profile_distances": cross_profile_distances, "tables": {
        "capacities": caps, "offer_prices": offers, "market_prices": prices, "objectives": objs,
        "summary": summary}}, indent=2) + "\n", encoding="utf-8")

    run_manifest = json.loads((ROOT / "workflow/local_paper_profile_zero_prox_manifest.json").read_text(encoding="utf-8"))
    lines = ["# Local search around the reported paper profile", "", "## Outcome", "",
        "No new branch produced a solver-verified computational relative 1%-equilibrium. The best new frozen candidate is Branch F at alpha=0.50 after sweep 6; its maximum gain is 3.8441% (CH). The separately preserved cost-price candidate remains verified at 0.6615% (APAC).", "",
        "## Profile comparison", "",
        "| Profile | Max gain | Player | Distance from paper | Mean abs capacity change | Mean abs price change | Sweeps |", "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['profile']} | {row['max_relative_gain']:.4%} | {row['max_gain_player'].upper()} | {row['combined_normalized_l2']:.5f} | {row['capacity_mean_absolute_change_gw']:.2f} GW | {row['offer_price_mean_absolute_change_usd_per_kw']:.2f} USD/kW | {row['sweeps']} |")
    lines += ["", "## Branch results", "", "| Run | Sweeps | Ending max raw gain | Ending strategy change | Cycle | Best frozen audit |", "|---|---:|---:|---:|---|---:|"]
    for row in run_manifest["runs"]:
        best = row.get("best_audit"); best_text = "--" if not best else f"{best['max_relative_gain']:.4%} ({best['max_gain_player'].upper()})"
        lines.append(f"| {row['run_name']} | {row['sweeps_completed']} | {row['ending_max_raw_gain']:.4%} | {row['ending_strategy_change']:.4%} | {'yes' if row['cycle_detected'] else 'no'} | {best_text} |")
    lines += ["", "## Direct answers", "",
        "1. The paper profile's maximum frozen unilateral gain is 47.5777%, from ROW.",
        "2. Exact-start alpha=0.65 GS does not approach a verified 1%-equilibrium; it remains non-convergent after 15 sweeps.",
        "3. Neither the seeded +/-5% nor +/-10% full-strategy perturbation reaches one.",
        "4. Branch F performs best, with the alpha=0.50 sweep-6 profile giving the lowest audited maximum gain (3.8441%).",
        "5. No new branch qualifies; the best local candidate occurs after 6 sweeps.",
        "6. CH is hardest at the best local frozen candidate; APAC and ROW drive much of the trajectory instability.",
        "7. The best local candidate is only modestly closer to the paper profile in the bound-scaled strategy metric (0.0622 versus 0.0656 for the old candidate), and is not economically close: its mean absolute capacity change is 63.1 GW and mean absolute offer-price change is 88.3 USD/kW.",
        f"8. It moves toward the old candidate in offer-price space and is only {cross_profile_distances['best_local_to_previous_cost_price']:.4f} away from it in the combined metric, but it does not converge to the same capacity profile (for example, 2040 ROW capacity is 2.98 GW versus 66.33 GW).",
        "9. Alpha=0.50 smooths the transient but does not produce a verified equilibrium; alpha=0.30 also destabilizes after a promising transient.",
        "10. Material player-order dependence of a verified local candidate is not assessed, because no new local candidate passes the 1% audit and the prespecified trigger for order tests is therefore not met.",
        "11. No new local candidate can be defended under the requested criterion. The preserved cost-price candidate can still be described as a solver-verified computational relative 1%-equilibrium, with its disclosed initialization caveat.", "",
        "## Method and reproducibility", "",
        "Best responses solve the original unpenalized economic game in reduced form: each strategic problem embeds an exact solve of the convex lower market rather than asking a complementarity solver to choose among known locally inferior KKT branches. Damping is applied only after the unrestricted response, and all six players are updated even when their gain is below 1%. Frozen audits use three starts: candidate, standard zero-capacity-change/manufacturing-cost-price, and a feasible 0.95x price perturbation.", "",
        "Primary command pattern:", "", "```powershell", "$env:PYTHONDONTWRITEBYTECODE='1'", "python scripts/run_local_paper_equilibrium_experiment.py --branch A --alpha 0.65 --max-sweeps 50 --starts 1 --maxiter 500", "```", "",
        "Replace `A` by `B` through `F`; the runner resumes from its last complete per-sweep checkpoint. Generate the comparison with:", "", "```powershell", "python scripts/report_local_paper_equilibrium_experiment.py", "```", ""]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_ROOT / "report.md")


if __name__ == "__main__":
    main()
