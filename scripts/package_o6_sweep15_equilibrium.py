from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = (
    ROOT
    / "outputs/equilibrium_search/ch-row-apac-us-eu-af/overnight"
    / "overnight_20260914_224500"
)
O6 = RUN / "branches/O6"
LOCAL = ROOT / "outputs/equilibrium_search/ch-row-apac-us-eu-af/local_paper_profile_zero_prox"
BRANCH_F = LOCAL / "branch_F_alpha_0p50"
DEST = (
    ROOT
    / "outputs/equilibria/ch-row-apac-us-eu-af"
    / "o6_sweep_015_20260915"
)

SOURCE_SWEEP = O6 / "sweep_015.json"
ONE_START_AUDIT = RUN / "audits/lightweight/audit_O6_s015.json"
THREE_START_AUDIT = RUN / "audits/final/audit_O6_s015.json"
TOLERANCE = 0.01


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def gains(audit: dict) -> dict[str, float]:
    return {
        str(row["player"]): float(row["best"]["relative_gain"])
        for row in audit["players"]
    }


def validate(sweep: dict, one_start: dict, three_start: dict) -> None:
    if sweep.get("branch") != "O6" or int(sweep.get("sweep", -1)) != 15:
        raise RuntimeError("The selected checkpoint is not O6 sweep 15.")
    if not one_start.get("common_profile_frozen"):
        raise RuntimeError("The acceptance audit did not freeze a common profile.")
    if float(one_start.get("algorithmic_proximal_penalties", 1.0)) != 0.0:
        raise RuntimeError("The acceptance audit used nonzero algorithmic proximal penalties.")
    if int(one_start.get("starts_per_player", 0)) != 1:
        raise RuntimeError("The acceptance audit is not the agreed one-start audit.")
    if not one_start.get("complete") or not one_start.get("all_attempts_successful"):
        raise RuntimeError("The acceptance audit is incomplete or contains failed solves.")
    if float(one_start["max_relative_gain"]) > TOLERANCE:
        raise RuntimeError("O6 sweep 15 does not pass the agreed 1% criterion.")
    if int(three_start.get("starts_per_player", 0)) != 3:
        raise RuntimeError("The diagnostic audit is not the recorded three-start audit.")


def make_provenance_copies() -> None:
    copy(LOCAL / "paper_profile.json", DEST / "provenance/paper_profile.json")
    copy(
        BRANCH_F / "initialization.json",
        DEST / "provenance/branch_F_alpha_0p50/initialization.json",
    )
    for sweep in range(1, 7):
        copy(
            BRANCH_F / f"sweep_{sweep:03d}.json",
            DEST / f"provenance/branch_F_alpha_0p50/sweep_{sweep:03d}.json",
        )
    copy(
        BRANCH_F / "audit_sweep_006.json",
        DEST / "provenance/branch_F_alpha_0p50/audit_sweep_006.json",
    )
    copy(O6 / "initialization.json", DEST / "provenance/O6/initialization.json")
    for sweep in range(1, 16):
        copy(
            O6 / f"sweep_{sweep:03d}.json",
            DEST / f"provenance/O6/sweep_{sweep:03d}.json",
        )


def write_search_path() -> None:
    rows = []
    for stage, directory, count in (
        ("Branch F", BRANCH_F, 6),
        ("O6", O6, 15),
    ):
        for index in range(1, count + 1):
            payload = load_json(directory / f"sweep_{index:03d}.json")
            rows.append(
                {
                    "stage": stage,
                    "sweep": index,
                    "alpha": float(payload["alpha"]),
                    "max_raw_unilateral_gain": float(payload["max_raw_unilateral_gain"]),
                    "strategy_change_metric": float(payload["strategy_change_metric"]),
                    "distance_from_paper": float(payload["distance_from_paper"]),
                }
            )

    csv_path = DEST / "provenance/search_path.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    o6_rows = [row for row in rows if row["stage"] == "O6"]
    x = [row["sweep"] for row in o6_rows]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True)
    axes[0].plot(x, [100 * row["max_raw_unilateral_gain"] for row in o6_rows], marker="o")
    axes[0].axhline(1.0, color="#B22222", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Raw gain [%]")
    axes[1].plot(x, [row["strategy_change_metric"] for row in o6_rows], marker="o")
    axes[1].set_ylabel("Strategy change")
    axes[1].set_xlabel("O6 sweep")
    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DEST / "provenance/o6_sweeps_001_015.png", dpi=200)
    plt.close(fig)


def write_manifest() -> None:
    files = []
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(DEST)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    write_json(
        DEST / "MANIFEST.json",
        {
            "created": now(),
            "package": "O6 sweep 15 accepted one-start local 1% equilibrium",
            "files": files,
        },
    )


def package() -> None:
    sweep = load_json(SOURCE_SWEEP)
    one_start = load_json(ONE_START_AUDIT)
    three_start = load_json(THREE_START_AUDIT)
    validate(sweep, one_start, three_start)
    DEST.mkdir(parents=True, exist_ok=True)

    profile = {
        "created": now(),
        "label": "O6 sweep 15",
        "status": "accepted_one_start_local_1pct_equilibrium",
        "source_checkpoint": relative(SOURCE_SWEEP),
        "state": sweep["ending_profile"]["strategy"],
        "search": {
            "branch": "O6",
            "sweep": 15,
            "player_order": sweep["player_order"],
            "alpha": sweep["alpha"],
            "next_alpha": sweep["next_alpha"],
            "algorithmic_proximal_penalties": sweep["proximal_coefficients"],
            "distance_from_paper": sweep["distance_from_paper"],
        },
    }
    write_json(DEST / "profile.json", profile)
    copy(ONE_START_AUDIT, DEST / "frozen_audit_one_start.json")
    copy(THREE_START_AUDIT, DEST / "frozen_audit_three_start_diagnostic.json")

    one_gains = gains(one_start)
    three_gains = gains(three_start)
    certification = {
        "created": now(),
        "status": "accepted_one_start_local_1pct_equilibrium",
        "claim_scope": "computational local equilibrium criterion; not a global Nash-equilibrium certificate",
        "source_workbook": "outputs\\sens\\converged\\sens_ch-row-apac-us-eu-af.xlsx",
        "source_iteration": 21,
        "player_order": sweep["player_order"],
        "candidate_checkpoint": relative(SOURCE_SWEEP),
        "packaged_profile": relative(DEST / "profile.json"),
        "candidate_audit": relative(ONE_START_AUDIT),
        "algorithmic_proximal_penalties": {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0},
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "common_profile_frozen": True,
        "relative_gain_tolerance": TOLERANCE,
        "multistart_count": 1,
        "all_audit_attempts_successful": bool(one_start["all_attempts_successful"]),
        "maximum_frozen_profile_relative_gain": float(one_start["max_relative_gain"]),
        "maximum_gain_player": one_start["max_gain_player"],
        "frozen_profile_relative_gains": one_gains,
        "market_diagnostics": sweep["ending_market_diagnostics"],
        "three_start_diagnostic": {
            "file": "frozen_audit_three_start_diagnostic.json",
            "maximum_relative_gain": float(three_start["max_relative_gain"]),
            "maximum_gain_player": three_start["max_gain_player"],
            "relative_gains": three_gains,
            "passes_1pct": bool(three_start["equilibrium_verified"]),
            "role": "retained diagnostic; not the agreed acceptance criterion",
        },
    }
    write_json(DEST / "certification.json", certification)

    comparison = load_json(RUN / "comparison.json")
    selected = [
        row
        for row in comparison["profiles"]
        if row.get("name") in {"paper_profile", "best_overnight_candidate"}
    ]
    write_json(
        DEST / "comparison_to_paper.json",
        {
            "created": now(),
            "source": relative(RUN / "comparison.json"),
            "note": "The O6 regret values in this comparison come from the retained three-start diagnostic. The package acceptance decision uses frozen_audit_one_start.json.",
            "profiles": selected,
        },
    )

    make_provenance_copies()
    write_search_path()

    gain_lines = "\n".join(
        f"| {player.upper()} | {100 * one_gains[player]:.4f}% | {100 * three_gains[player]:.4f}% |"
        for player in ("ch", "row", "apac", "us", "eu", "af")
    )
    provenance = f"""# Provenance of the O6 sweep 15 profile

## Accepted profile

This folder preserves the ending strategy profile from O6 sweep 15. Under the agreed acceptance rule, it satisfies a computational local 1% equilibrium criterion: each player's unregularized unilateral deviation problem was solved once from that player's candidate strategy while all other players were fixed at the same profile. The maximum identified relative gain was {100 * float(one_start['max_relative_gain']):.4f}% for {str(one_start['max_gain_player']).upper()}.

This is a local computational criterion, not a global Nash-equilibrium certificate.

## How the profile was obtained

1. The starting reference was the reported iteration-21 paper profile reconstructed from `outputs/sens/converged/sens_ch-row-apac-us-eu-af.xlsx` with player order CH, ROW, APAC, US, EU, AF.
2. Local Branch F retained the paper capacity strategy and initialized every bilateral export offer halfway between the paper offer and the exporter's time-specific manufacturing cost.
3. Branch F then ran six zero-algorithmic-proximal Gauss--Seidel sweeps with fixed damping `alpha = 0.50`. Its sweep-6 ending profile became the O6 initialization.
4. O6 used zero algorithmic proximal penalties and adaptive damping, starting at `alpha = 0.30`, with sequential order CH, ROW, APAC, US, EU, AF. Sweep 15 used `alpha = {float(sweep['alpha']):.3f}`. Its ending profile is the saved profile in this folder.
5. The overnight branch continued beyond sweep 15 to its configured limit. Sweep 15 was selected retrospectively because it had the smallest one-start frozen-profile regret among the saved overnight checkpoints.

## Frozen-profile audits

| Player | One start (acceptance) | Three starts (diagnostic) |
|---|---:|---:|
{gain_lines}

The one-start audit passes the 1% threshold. The separately retained three-start diagnostic finds a {100 * float(three_start['max_relative_gain']):.4f}% deviation for {str(three_start['max_gain_player']).upper()} and therefore does not pass 1%. It is included so that the numerical record remains complete; it was not used as the acceptance rule requested for the revision.

## Folder contents

- `profile.json`: normalized strategy profile used by the exporter.
- `certification.json`: machine-readable acceptance rule, gains, and caveat.
- `frozen_audit_one_start.json`: the agreed frozen-profile acceptance audit.
- `frozen_audit_three_start_diagnostic.json`: the stronger diagnostic audit.
- `comparison_to_paper.json`: detailed paper-profile and O6 outcome comparison.
- `accepted_equilibrium_results.xlsx`: reconstructed market outcome used by the paper plots.
- `paper_figures/`: price, capacity, and welfare figures in PNG and PDF formats.
- `provenance/`: the paper replay, Branch-F sweeps 1--6, O6 sweeps 1--15, audit evidence, and a compact search-path plot/CSV.
- `MANIFEST.json`: file sizes and SHA-256 hashes for the self-contained package.
"""
    (DEST / "PROVENANCE.md").write_text(provenance, encoding="utf-8")
    write_manifest()
    print(f"Packaged core files in {DEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the accepted O6 sweep-15 profile.")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        if not DEST.is_dir():
            raise FileNotFoundError(DEST)
        write_manifest()
        print(f"Refreshed {DEST / 'MANIFEST.json'}")
    else:
        package()


if __name__ == "__main__":
    main()
