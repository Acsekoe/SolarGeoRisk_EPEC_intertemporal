import re

def go():
    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/model_main.py', 'r', encoding='utf-8') as f:
        text = f.read()

    def normalize(s):
        return s.replace('\\r\\n', '\\n')

    text = normalize(text)

    # 1. Delete rho_p_p
    target1 = normalize("""    rho_p_p = Parameter(
        m, "rho_p", domain=[R],
        records=[(r, data.rho_p[r]) for r in data.regions],
    )""")
    text = text.replace(target1, '    # Deleted rho_p_p parameter here')

    # 2. Re-wire central penalisation scalars
    target2 = normalize("""    rho_prox_val = float(settings.get("rho_prox", 0.0))
    rho_prox = gp.Number(rho_prox_val)

    # c_pen_* proximal penalty scalars (controllable from run_gs.py via settings).
    # Each penalises the squared deviation of the strategic variable from its
    # previous Gauss-Seidel iterate: -0.5 * c_pen * (X - X_last)^2
    c_pen_q   = gp.Number(float(settings.get("c_pen_q",   0.0)))  # Q_offer
    c_pen_cap = gp.Number(float(settings.get("c_pen_cap", 0.0)))  # Icap_pos & Dcap_neg
    c_pen_p   = gp.Number(float(settings.get("c_pen_p",   0.0)))  # p_offer (supplements rho_prox)
    c_pen_a   = gp.Number(float(settings.get("c_pen_a",   0.0)))  # a_bid

    kappa_map = getattr(data, "kappa_Q", None) or {}
    kappa_by_r = {k: float(kappa_map.get(k, 0.0)) for k in data.regions}
    kappa_Q = Parameter(
        m, "kappa_Q", domain=[R],
        records=[(k, kappa_by_r[k]) for k in data.regions],
    )""")
    rep2 = normalize("""    # Algorithmic proximal penalties (solver stabilization)
    c_pen_q = gp.Number(float(settings.get("c_pen_q", 0.0)))
    c_pen_p = gp.Number(float(settings.get("c_pen_p", 0.0)))
    c_pen_a = gp.Number(float(settings.get("c_pen_a", 0.0)))
    
    # Economic quadratic penalties (convex costs / disutility)
    c_quad_q = gp.Number(float(settings.get("c_quad_q", 0.0)))
    c_quad_p = gp.Number(float(settings.get("c_quad_p", 0.0)))
    c_quad_a = gp.Number(float(settings.get("c_quad_a", 0.0)))""")
    
    if target2 not in text: print("Target 2 not found")
    text = text.replace(target2, rep2)

    # 3. Refactor the entire objective function
    target3_regex = re.compile(
        r'# ---- Penalties \(replicated per period\) ----.*?'
        r'obj_welfare = \(\n[^\)]*\)', 
        re.DOTALL
    )
    
    rep3 = normalize("""# ---- Penalties (replicated per period) ----
        # Economic quadratic penalties (convex costs / disutility)
        pen_quad_q = Sum(
            T,
            -gp.Number(0.5) * beta_p[T] * ytn_p[T] * c_quad_q * Q_offer[r, T] * Q_offer[r, T],
        )

        pen_quad_p = Sum(
            T,
            -gp.Number(0.5) * beta_p[T] * ytn_p[T] * c_quad_p * Sum(j, p_offer[r, j, T] * p_offer[r, j, T]),
        )

        pen_quad_a = Sum(
            T,
            -gp.Number(0.5) * beta_p[T] * ytn_p[T] * c_quad_a * (a_dem_t_p[r, T] - a_bid[r, T]) * (a_dem_t_p[r, T] - a_bid[r, T]),
        )

        # Algorithmic proximal penalties (solver stabilization)
        pen_prox_poffer = Sum(
            T,
            -gp.Number(0.5) * beta_p[T] * ytn_p[T] * c_pen_p * Sum(
                j,
                (p_offer[r, j, T] - p_offer_last[r, j, T])
                * (p_offer[r, j, T] - p_offer_last[r, j, T]),
            ),
        )

        pen_prox_q = Sum(
            T,
            -gp.Number(0.5) * beta_p[T] * ytn_p[T] * c_pen_q
            * (Q_offer[r, T] - Q_offer_last[r, T])
            * (Q_offer[r, T] - Q_offer_last[r, T]),
        )

        pen_prox_a = Sum(
            T,
            -gp.Number(0.5) * beta_p[T] * ytn_p[T] * c_pen_a
            * (a_bid[r, T] - a_bid_last[r, T])
            * (a_bid[r, T] - a_bid_last[r, T]),
        )

        # ---- Assemble objective ----
        obj_welfare = (
            d_surplus_t
            + producer_term_t
            + capacity_cost_t
            + pen_quad_q
            + pen_quad_p
            + pen_quad_a
            + pen_prox_poffer
            + pen_prox_q
            + pen_prox_a
        )""")
    
    match = target3_regex.search(text)
    if not match: 
        print("Target 3 not found")
    else:
        text = text[:match.start()] + rep3 + text[match.end():]

    # 4. Remove passed parameters block
    target4 = normalize("""            "p_offer_ub": p_offer_ub_p,
            "rho_p": rho_p_p,
            "kappa_Q": kappa_Q,
            "g_exp": g_exp_p,""")
    rep4 = normalize("""            "p_offer_ub": p_offer_ub_p,
            "g_exp": g_exp_p,""")
    if target4 not in text: print("Target 4 not found")
    text = text.replace(target4, rep4)

    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/model_main.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Replacements in model_main.py done!")

if __name__ == '__main__':
    go()
