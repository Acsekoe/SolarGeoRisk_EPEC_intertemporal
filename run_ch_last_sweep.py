"""3 additional CH-last runs with more iterations to test convergence."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from model.run_gs import RunConfig, run

# Existing CH-last orders (for reference, not re-run):
#   ['af', 'eu', 'us', 'apac', 'row', 'ch']  r_strat=0.0370
#   ['eu', 'us', 'af', 'row', 'apac', 'ch']  r_strat=0.0586
#   ['us', 'apac', 'af', 'row', 'eu', 'ch']  r_strat=0.0256
#   ['us', 'row', 'eu', 'apac', 'af', 'ch']  r_strat=0.0135  <- best so far

ORDERS = [
    # 1. Big exporters first (APAC, ROW before importers), CH last
    ["apac", "row", "us", "eu", "af", "ch"],
    # 2. Importers grouped, then small exporter, then big exporter, CH last
    ["eu", "af", "us", "apac", "row", "ch"],
    # 3. Rerun of best previous CH-last order with more iters
    ["us", "row", "eu", "apac", "af", "ch"],
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "sens")

if __name__ == "__main__":
    for i, order in enumerate(ORDERS, 1):
        label = ",".join(order)
        print(f"\n{'='*60}")
        print(f"  RUN {i}/3 — player_order = [{label}]")
        print(f"{'='*60}\n")
        try:
            out = run(RunConfig(
                player_order=order,
                iters=50,
                out_dir=OUT_DIR,
            ))
            print(f"\n  -> Saved: {out}\n")
        except Exception as e:
            print(f"\n  -> FAILED: {e}\n")
