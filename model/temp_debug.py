import sys
sys.path.append("model")
import model_main as _it
from model_main import build_model, apply_player_fixings, extract_state
from data_prep import load_data_from_excel

path = "c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/inputs/input_data_intertemporal.xlsx"
data = load_data_from_excel(path)
data.settings["c_pen_q"] = 0.1
data.settings["c_pen_p"] = 0.1
data.settings["c_pen_a"] = 0.1
data.settings["c_quad_q"] = 0.1
data.settings["c_quad_p"] = 0.1
data.settings["c_quad_a"] = 0.1

ctx = build_model(data)

times = data.times
move_times = _it._move_times(times)
init_kcap = _it._initial_capacity_by_region(data)

theta_dK_net = {(r, tp): 0.0 for r in data.players for tp in move_times}
implied_kcap = _it._implied_capacity_path(data, times, theta_dK_net)
theta_Q = {(r, tp): 0.8 * float(implied_kcap[(r, tp)]) for r in data.players for tp in times}
theta_p_offer = {
    (ex, im, tp): 0.5 * float(data.p_offer_ub[(ex, im)])
    for ex in data.regions for im in data.regions for tp in times
}
theta_a_bid = {
    (r, tp): _it._true_demand_intercept(data, r, tp)
    for r in data.players for tp in times
}

omega = 0.7
sweep_order = list(data.players)
sweep_order.remove("ch")
sweep_order.append("ch")
print("sweep_order:", sweep_order)

for p in sweep_order:
    apply_player_fixings(ctx, data, theta_Q, theta_dK_net, theta_p_offer, theta_a_bid, player=p)
    ctx.models[p].solve(solver="ipopt")
    state = extract_state(ctx)

    dK_net_sol = state.get("dK_net", {})

    # Print EU's Icap_pos LEVEL from the GAMS records directly
    icap_recs = ctx.vars["Icap_pos"].records
    eu_icap = icap_recs[(icap_recs["R"] == "eu") & (icap_recs["T"] == "2025")]
    eu_icap_level = eu_icap["level"].values[0] if len(eu_icap) > 0 else "NOT FOUND"
    print(f"After solving {p}: EU Icap_pos level 2025 = {eu_icap_level}")

    kcap_recs = ctx.vars["Kcap"].records
    eu_kcap30 = kcap_recs[(kcap_recs["R"] == "eu") & (kcap_recs["T"] == "2030")]
    eu_kcap30_level = eu_kcap30["level"].values[0] if len(eu_kcap30) > 0 else "NOT FOUND"
    print(f"After solving {p}: EU Kcap 2030 = {eu_kcap30_level}")

    # Also print the dK_net_sol for EU
    eu_dk = dK_net_sol.get(("eu", "2025"), "NOT FOUND")
    print(f"After solving {p}: EU dK_net_sol 2025 = {eu_dk}")

    # Update dK_net
    for tp in move_times:
        key = (p, tp)
        if key in dK_net_sol:
            br = float(dK_net_sol[key])
            theta_dK_net[key] = (1.0 - omega) * theta_dK_net[key] + omega * br

    implied_kcap = _it._implied_capacity_path(data, times, theta_dK_net)

    # Update Q_offer
    Q_sol = state.get("Q_offer", {})
    for tp in times:
        key = (p, tp)
        if key in Q_sol:
            br = _it._clip_value(float(Q_sol[key]), 0.0, max(float(implied_kcap[key]), 0.0))
            theta_Q[key] = (1.0 - omega) * theta_Q[key] + omega * br

    # Update p_offer
    poffer_sol = state.get("p_offer", {})
    for im in data.regions:
        for tp in times:
            key = (p, im, tp)
            if key in poffer_sol:
                br = float(poffer_sol[key])
                theta_p_offer[key] = (1.0 - omega) * theta_p_offer[key] + omega * br

print("\nFinal theta_dK_net for eu:")
for k, v in theta_dK_net.items():
    if k[0] == "eu":
        print(f"  {k}: {v}")
