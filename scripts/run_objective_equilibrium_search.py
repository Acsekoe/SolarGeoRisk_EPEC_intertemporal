from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.search_nested_equilibrium import MANIFEST_PATH, audit, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run checkpointed objective-gain equilibrium-search blocks."
    )
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--sweeps-per-block", type=int, default=5)
    parser.add_argument("--omega", type=float, default=0.10)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--max-damped-move", type=float, default=0.01)
    parser.add_argument("--gain-tol", type=float, default=0.01)
    parser.add_argument("--stable-sweeps", type=int, default=3)
    args = parser.parse_args()

    if args.blocks < 1 or args.sweeps_per_block < 1:
        raise ValueError("blocks and sweeps-per-block must be positive")

    for block in range(1, args.blocks + 1):
        print(f"[OBJECTIVE SEARCH] block {block}/{args.blocks}", flush=True)
        run(
            args.sweeps_per_block,
            args.omega,
            args.maxiter,
            args.starts,
            args.max_damped_move,
            args.gain_tol,
            args.gain_tol,
            args.stable_sweeps,
        )
        audit(args.maxiter, args.starts, args.gain_tol)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        latest = manifest["audits"][-1]
        print(
            f"[OBJECTIVE SEARCH] block {block} audited max gain="
            f"{float(latest['max_relative_gain']):.3%}",
            flush=True,
        )
        if latest.get("equilibrium_verified"):
            print("[OBJECTIVE SEARCH] equilibrium criterion verified", flush=True)
            return

    print("[OBJECTIVE SEARCH] sweep budget exhausted without verification", flush=True)


if __name__ == "__main__":
    main()
