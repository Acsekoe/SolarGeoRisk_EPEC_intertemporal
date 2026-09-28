# Greedy multi-route offer deviation (capacity fixed) for every player in every retained profile.
import sys
ROOTP = r"d:/Alexander/Studium/EEG/Complementarity Modelling/SolarGeoRisk_EPEC_intertemporal"
sys.path.insert(0, ROOTP); sys.path.insert(0, ROOTP + "/outputs/eq1_deviation_check_20260928_104717")
import check
from scripts.nested_market_audit import solve_nested_market
from scripts.audit_selected_equilibrium import _economic_objective
data, *_ = check.load_data()
ids = [s.strip() for s in open(ROOTP + "/outputs/paper_plots/15_equilibria/retained_profiles.txt") if s.strip()]
periods = [str(t) for t in data.times if str(t) in ("2025", "2030", "2035", "2040")]
only = sys.argv[1] if len(sys.argv) > 1 else None   # optional: restrict routes to one period
print("profile,player,gain_percent")
for pid in ids:
    ref, _, _ = check.load_profile(pid, data)
    mref, _ = solve_nested_market(data, ref)
    for player in data.regions:
        obj0 = _economic_objective(data, ref, mref, player); best = obj0; changes = {}
        for sweep in range(2):
            for per in periods:
                if only and per != only: continue
                for imp in data.regions:
                    if imp == player: continue
                    cost = float(data.c_man_t[(player, per)]); ship = float(data.c_ship[(player, imp)])
                    hi = max(float(mref["lam"][(imp, per)]) - ship, cost)
                    keep = changes.get((imp, per))
                    for g in [cost + (hi - cost) * k / 10 for k in range(11)]:
                        trial = dict(changes); trial[(imp, per)] = g
                        st = check.changed_state(ref, player, trial, data)
                        m, _ = solve_nested_market(data, st, mref)
                        v = _economic_objective(data, st, m, player)
                        if v > best + 1e-6: best, keep = v, g
                    if keep is not None: changes[(imp, per)] = keep
        print(f"{pid},{player},{100*(best-obj0)/abs(obj0):.2f}", flush=True)
