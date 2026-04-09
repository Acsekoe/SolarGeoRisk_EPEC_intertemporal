"""
run_convergence.py
==================
Single targeted run for the ch->row->apac->us->eu->af player order
with increased iterations (100) and higher omega_min (0.6) to improve convergence.
Output goes directly to outputs/sens/.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from model.run_gs import RunConfig, run

ORDER     = ["ch", "row", "apac", "us", "eu", "af"]
OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs", "sens")

cfg = RunConfig(
    player_order           = ORDER,
    out_dir                = OUT_DIR,
    iters                  = 100,
    omega                  = 0.8,
    omega_min              = 0.6,          # raised from 0.4
    omega_aggressive_sweeps= 5,
    omega_ramp_iters       = 10,
)

if __name__ == "__main__":
    label = "->".join(ORDER)
    print(f"Running: {label}")
    print(f"  iters={cfg.iters}, omega_min={cfg.omega_min}")
    out = run(cfg)
    print(f"\nSaved: {out}")
