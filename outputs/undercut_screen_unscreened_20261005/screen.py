"""Apply the undercutting screen of eq1_deviation_check_20260928_104717 to the two
locally accepted Stage-2 profiles that the screen skipped.

Flags and tests are identical to check.screen_profiles; idle capacity is taken
from the selected-sweep profile because one profile is absent from the metrics CSVs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
spec = importlib.util.spec_from_file_location(
    "check", ROOT / "outputs" / "eq1_deviation_check_20260928_104717" / "check.py")
check = importlib.util.module_from_spec(spec)
sys.modules["check"] = check
spec.loader.exec_module(check)

PROFILES = (
    "ch-af-apac-eu-row-us/pf100_k100_a040",  # excluded earlier for a 641 USD/kW EU price
    "eu-us-af-row-apac-ch/pf120_k050_a040",  # removed earlier, test script not preserved
)


def main():
    data, _, _, _ = check.load_data()
    rows, summary = [], []
    for profile_id in PROFILES:
        state, _, ending = check.load_profile(profile_id, data)
        flows = check.rows_to_map(ending["market"]["trade_flows"], ("exporter", "importer", "time"))
        prices = check.rows_to_map(ending["market"]["clearing_prices"], ("region", "time"))
        caps = check.rows_to_map(ending["capacities"], ("player", "time"))
        market, _ = check.solve_nested_market(data, state)
        best_gain, best_case, flagged = float("-inf"), "", 0
        for exporter in data.regions:
            for period in check.PERIODS:
                idle = caps[(exporter, period)] - sum(flows[(exporter, i, period)] for i in data.regions)
                if idle <= 1.0:
                    continue
                cost = float(data.c_man_t[(exporter, period)])
                for importer in data.regions:
                    if importer == exporter:
                        continue
                    flow = max(flows[(exporter, importer, period)], 0.0)
                    margin = prices[(importer, period)] - float(data.c_ship[(exporter, importer)]) - cost
                    if flow > 1.0 or margin <= 50.0:
                        continue
                    flagged += 1
                    objective = check._economic_objective(data, state, market, exporter)
                    current = float(state["p_offer"][(exporter, importer, period)])
                    tests = [(f"screen_undercut_eps_{e:g}", check.offer_target(
                        data, state, market, exporter, importer, period, e)) for e in check.EPSILONS]
                    tests += [(f"screen_grid_{s:02d}_of_10", cost + (current - cost) * s / 10.0)
                              for s in range(11)]
                    for label, offer in tests:
                        trial = check.changed_state(state, exporter, {(importer, period): offer}, data)
                        row = check.record(rows, data=data, profile_id=profile_id, player=exporter,
                                           importer=importer, period=period, deviation_type=label,
                                           new_offer=offer, reference=state, market_ref=market,
                                           objective_ref=objective, candidate=trial,
                                           notes=f"idle={idle:.3f}; screen_margin={margin:.3f}")
                        if row["relative_gain_percent"] > best_gain:
                            best_gain = row["relative_gain_percent"]
                            best_case = f"{exporter}->{importer}/{period}/{label}"
        verdict = "not an equilibrium" if best_gain > 1.0 else "equilibrium at 1% tolerance"
        summary.append({"profile": profile_id, "flagged_routes": flagged,
                        "best_gain_percent": best_gain if flagged else "",
                        "best_case": best_case, "verdict": verdict})
        print(f"{profile_id}: flagged={flagged} best={best_gain:.3f}% ({best_case}) -> {verdict}", flush=True)
    check.write_csv(OUT / "results.csv", rows, check.FIELDS)
    check.write_csv(OUT / "summary.csv", summary,
                    ("profile", "flagged_routes", "best_gain_percent", "best_case", "verdict"))


if __name__ == "__main__":
    main()
