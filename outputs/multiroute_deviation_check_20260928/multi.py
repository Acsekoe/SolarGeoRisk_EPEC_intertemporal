# Greedy multi-route offer deviation for Eq 1 (capacity fixed), reusing check.py helpers.
import sys, time
ROOTP = r"d:/Alexander/Studium/EEG/Complementarity Modelling/SolarGeoRisk_EPEC_intertemporal"
sys.path.insert(0, ROOTP); sys.path.insert(0, ROOTP + "/outputs/eq1_deviation_check_20260928_104717")
import check
from scripts.nested_market_audit import solve_nested_market
from scripts.audit_selected_equilibrium import _economic_objective
data, *_ = check.load_data()
EQ1 = "ch-af-apac-eu-row-us/pf080_k100_a030"
ref, _, _ = check.load_profile(EQ1, data)
mref, _ = solve_nested_market(data, ref)
periods = [str(t) for t in data.times if str(t) in ("2025", "2030", "2035", "2040")]
for player in sys.argv[1:]:
    obj0 = _economic_objective(data, ref, mref, player)
    changes = {}
    best = obj0
    t0 = time.time()
    for sweep in range(2):
        for per in periods:
            for imp in data.regions:
                if imp == player:
                    continue
                cost = float(data.c_man_t[(player, per)]); ship = float(data.c_ship[(player, imp)])
                lam = float(mref["lam"][(imp, per)])
                lo, hi = cost, max(lam - ship, cost)
                grid = [lo + (hi - lo) * k / 10 for k in range(11)] + [float(ref["p_offer"][(player, imp, per)])]
                keep = changes.get((imp, per))
                for g in grid:
                    trial = dict(changes); trial[(imp, per)] = g
                    st = check.changed_state(ref, player, trial, data)
                    m, _ = solve_nested_market(data, st, mref)
                    v = _economic_objective(data, st, m, player)
                    if v > best + 1e-6:
                        best, keep = v, g
                if keep is not None:
                    changes[(imp, per)] = keep
        print(f"{player} sweep {sweep+1}: gain {100*(best-obj0)/abs(obj0):.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    st = check.changed_state(ref, player, changes, data); m, _ = solve_nested_market(data, st, mref)
    print(player, "changed routes:", {k: round(v, 1) for k, v in sorted(changes.items(), key=lambda kv: kv[0][1])})
    for per in periods:
        ex0 = sum(float(mref["x"][(player, i, per)]) for i in data.regions if i != player)
        ex1 = sum(float(m["x"][(player, i, per)]) for i in data.regions if i != player)
        print(f"  {per}: exports {ex0:.1f} -> {ex1:.1f} GW; prices",
              {i: f"{float(mref['lam'][(i, per)]):.0f}->{float(m['lam'][(i, per)]):.0f}" for i in ("eu", "us", "af", "row")})
