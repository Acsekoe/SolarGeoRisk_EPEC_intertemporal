import sys
import os
import gamspy as gp

sys.path.insert(0, "./model")
import data_prep
import gauss_seidel
import model_main
from run_gs import RunConfig, _build_initial_state

def main():
    data = data_prep.load_data_from_excel("./inputs/input_data_intertemporal.xlsx")
    if data.settings is None:
        data.settings = {}
    data.settings["use_quad"] = True
    data.settings["fix_a_bid_to_true_dem"] = True
    
    cfg = RunConfig(iters=1, scenario="llp_planner", eps_comp=1.0)
    init_state = _build_initial_state(data, cfg)

    ctx = model_main.build_model(data)
    model_main.apply_player_fixings(
        ctx, data,
        theta_Q=init_state["Q_offer"],
        theta_dK_net=init_state["dK_net"],
        theta_p_offer=init_state["p_offer"],
        theta_a_bid=init_state["a_bid"],
        player="ch"
    )

    print("Solving CH MPEC...")
    ctx.models["ch"].solve(solver="ipopt", solver_options={"tol": 1e-4, "print_level": 5, "print_info_string": "yes"}, output=sys.stdout)
    
    ctx.models["ch"].computeInfeasibilities()
    print("Infeasibilities:")
    for name, eq in ctx.equations.items():
        if hasattr(eq, "infeasibility"):
            val = eq.infeasibility
            if val is not None and val.shape[0] > 0 and val.max().max() > 1e-4:
                print(f"{name}:\n{val[val > 1e-4].dropna()}")

    print("\n--- Variables for CH ---")
    print("Kcap:")
    print(ctx.vars["Kcap"].records.query("R == 'ch'"))
    print("\ndK_net:")
    print(ctx.vars["dK_net"].records.query("R == 'ch'"))
    print("\nIcap_pos:")
    print(ctx.vars["Icap_pos"].records.query("R == 'ch'"))
    print("\nQ_offer:")
    print(ctx.vars["Q_offer"].records.query("R == 'ch'"))

if __name__ == "__main__":
    main()
