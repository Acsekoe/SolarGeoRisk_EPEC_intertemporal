"""Unilateral capacity-expansion deviations for the EU and the US in the 14 reported profiles.

For each profile, the player's net capacity change is moved towards its expansion
bound g_exp in 2025-2035, all other strategies (including the player's own offers)
stay fixed, the market is re-cleared, and the change in the player's objective is
recorded. Uses the same market clearing and objective as the undercutting screen.
"""

from __future__ import annotations

import csv
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

RETAINED = ROOT / "outputs" / "paper_plots" / "14_equilibria" / "retained_profiles.txt"
PLAYERS = ("eu", "us")
MOVE_PERIODS = ("2025", "2030", "2035")
SHARES = (0.25, 0.5, 0.75, 1.0)


def main():
    data, _, _, _ = check.load_data()
    profiles = [l.strip() for l in RETAINED.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")]
    rows, summary = [], []
    for profile_id in profiles:
        state, _, _ = check.load_profile(profile_id, data)
        market, _ = check.solve_nested_market(data, state)
        for player in PLAYERS:
            objective = check._economic_objective(data, state, market, player)
            bound = float(data.g_exp_ub[player])
            best = None
            for share in SHARES:
                moves = {t: state["dK_net"][(player, t)]
                         + share * max(bound - state["dK_net"][(player, t)], 0.0)
                         for t in MOVE_PERIODS}
                trial = check.changed_state(state, player, {}, data, capacity_changes=moves)
                row = check.record(rows, data=data, profile_id=profile_id, player=player,
                                   importer="all", period="2040",
                                   deviation_type=f"expand_share_{share:g}", new_offer="",
                                   reference=state, market_ref=market, objective_ref=objective,
                                   candidate=trial)
                gain_bn = (row["deviation_objective"] - objective) / 1000.0
                if best is None or row["relative_gain_percent"] > best[0]:
                    best = (row["relative_gain_percent"], gain_bn, share)
            summary.append({"profile": profile_id, "player": player,
                            "objective_bn": objective / 1000.0,
                            "best_gain_percent": best[0], "best_gain_bn": best[1],
                            "best_share_of_gap": best[2]})
            print(f"{profile_id} {player}: best {best[0]:.3f}% ({best[1]:.2f} bn) at share {best[2]}",
                  flush=True)
    check.write_csv(OUT / "results.csv", rows, check.FIELDS)
    check.write_csv(OUT / "summary.csv", summary,
                    ("profile", "player", "objective_bn", "best_gain_percent", "best_gain_bn",
                     "best_share_of_gap"))


if __name__ == "__main__":
    main()
