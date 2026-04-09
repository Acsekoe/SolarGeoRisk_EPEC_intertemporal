"""Run the model with 6 different player orders to test equilibrium sensitivity."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from model.run_gs import RunConfig, run

ORDERS = [
    # 1. By capacity descending (big first → small last = small-player advantage)
    ["ch", "row", "apac", "us", "eu", "af"],
    # 2. By capacity ascending (small first → big last = big-player advantage)
    ["af", "eu", "us", "apac", "row", "ch"],
    # 3. CH isolated last, rest mixed
    ["us", "apac", "af", "row", "eu", "ch"],
    # 4. CH isolated first, rest mixed
    ["ch", "af", "eu", "us", "row", "apac"],
    # 5. Alternating big/small
    ["ch", "af", "apac", "eu", "row", "us"],
    # 6. Reverse alternating
    ["us", "row", "eu", "apac", "af", "ch"],
]

if __name__ == "__main__":
    for i, order in enumerate(ORDERS, 1):
        label = ",".join(order)
        print(f"\n{'='*60}")
        print(f"  RUN {i}/6 — player_order = [{label}]")
        print(f"{'='*60}\n")
        try:
            out = run(RunConfig(player_order=order))
            print(f"\n  -> Saved: {out}\n")
        except Exception as e:
            print(f"\n  -> FAILED: {e}\n")
