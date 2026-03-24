import sys

def go():
    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/run_gs.py', 'r', encoding='utf-8') as f:
        text = f.read()

    target1 = """    # Scalers
    kappa_q: float | None = 0.1
    rho_prox: float | None = 0.00
    use_quad: bool = True

    # Proximal penalty scalars: -0.5 * c_pen * (X - X_last)^2 added to each player's ULP.
    # Set to 0.0 to disable. Larger values shrink step sizes and improve GS stability.
    c_pen_q:   float = 0.1   # Q_offer
    c_pen_cap: float = 0.0   # Icap_pos and Dcap_neg
    c_pen_p:   float = 0.1   # p_offer (additive on top of rho_prox)
    c_pen_a:   float = 0.1   # a_bid"""

    rep1 = """    # Algorithmic proximal penalties: -0.5 * c_pen * (X - X_last)^2 added to ULP objective.
    # Set to 0.0 to disable. Larger values shrink step sizes and improve GS stability.
    c_pen_q: float = 0.1   # For Q_offer
    c_pen_p: float = 0.1   # For p_offer
    c_pen_a: float = 0.1   # For a_bid

    # Economic quadratic penalties: -0.5 * c_quad * X^2
    # Represents convex costs or disutility.
    c_quad_q: float = 0.1  # For Q_offer (production cost)
    c_quad_p: float = 0.1  # For p_offer (offer deviation)
    c_quad_a: float = 0.1  # For a_bid (demand withholding cost)"""

    target2 = """    if cfg.kappa_q is not None and data.kappa_Q is not None:
        for r in data.regions:
            data.kappa_Q[r] = float(cfg.kappa_q)

    if data.settings is None:
        data.settings = {}
    if cfg.rho_prox is not None:
        data.settings["rho_prox"] = float(cfg.rho_prox)
    data.settings["use_quad"] = bool(cfg.use_quad)
    # fix_a_bid_to_true_dem=True clamps declared demand to true demand (no strategic withholding).
    # This override always takes effect; it cannot be disabled via Excel settings.
    data.settings["fix_a_bid_to_true_dem"] = True

    # Proximal penalty scalars — passed through to build_model via data.settings.
    data.settings["c_pen_q"]   = float(cfg.c_pen_q)
    data.settings["c_pen_cap"] = float(cfg.c_pen_cap)
    data.settings["c_pen_p"]   = float(cfg.c_pen_p)
    data.settings["c_pen_a"]   = float(cfg.c_pen_a)"""

    rep2 = """    if data.settings is None:
        data.settings = {}

    # fix_a_bid_to_true_dem=True clamps declared demand to true demand (no strategic withholding).
    # This override always takes effect; it cannot be disabled via Excel settings.
    data.settings["fix_a_bid_to_true_dem"] = True

    # Proximal penalty scalars — passed through to build_model via data.settings.
    data.settings["c_pen_q"]   = float(cfg.c_pen_q)
    data.settings["c_pen_p"]   = float(cfg.c_pen_p)
    data.settings["c_pen_a"]   = float(cfg.c_pen_a)
    
    # Economic quadratic scalars
    data.settings["c_quad_q"]  = float(cfg.c_quad_q)
    data.settings["c_quad_p"]  = float(cfg.c_quad_p)
    data.settings["c_quad_a"]  = float(cfg.c_quad_a)"""

    target3 = """            # --- Capacity scalers ---
            "kappa_q":               float(cfg.kappa_q) if cfg.kappa_q is not None else 0.0,
            "rho_prox":              float(cfg.rho_prox) if cfg.rho_prox is not None else 0.0,
            "use_quad":              bool(cfg.use_quad),"""

    rep3 = """            # --- Penalties and Scalers ---
            "c_pen_q":               float(cfg.c_pen_q),
            "c_pen_p":               float(cfg.c_pen_p),
            "c_pen_a":               float(cfg.c_pen_a),
            "c_quad_q":              float(cfg.c_quad_q),
            "c_quad_p":              float(cfg.c_quad_p),
            "c_quad_a":              float(cfg.c_quad_a),"""

    # normalize newlines to match reliably
    import re
    def normalize(s):
        return s.replace('\\r\\n', '\\n')
    
    t1_norm = normalize(target1)
    t2_norm = normalize(target2)
    t3_norm = normalize(target3)
    text_norm = normalize(text)

    if t1_norm not in text_norm: print("t1 not found")
    if t2_norm not in text_norm: print("t2 not found")
    if t3_norm not in text_norm: print("t3 not found")

    text_norm = text_norm.replace(t1_norm, normalize(rep1))
    text_norm = text_norm.replace(t2_norm, normalize(rep2))
    text_norm = text_norm.replace(t3_norm, normalize(rep3))

    with open('c:/EEG/EPEC/EPEC_VS_code/SolarGeoRisk_EPEC_intertemporal/model/run_gs.py', 'w', encoding='utf-8') as f:
        f.write(text_norm)
    print("Replacements done successfully!")

if __name__ == '__main__':
    go()
