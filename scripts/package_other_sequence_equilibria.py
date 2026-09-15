from __future__ import annotations

"""Package the accepted Equilibrium-1 and Equilibrium-2 local 1% profiles."""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SPECS: list[dict[str, Any]] = [
    {
        "sequence": "ch-af-apac-eu-row-us",
        "equilibrium_number": 1,
        "source_iteration": 24,
        "order": ["ch", "af", "apac", "eu", "row", "us"],
        "label": "Branch-F sweep 6",
        "package_name": "branch_f_sweep_006_20260915",
        "checkpoint": "outputs/equilibrium_search/ch-af-apac-eu-row-us/local_paper_profile_zero_prox/branch_F_alpha_0p50/sweep_006.json",
        "one_start": "outputs/equilibrium_search/ch-af-apac-eu-row-us/local_paper_profile_zero_prox/branch_F_alpha_0p50/audit_sweep_006_one_start.json",
        "three_start": "outputs/equilibrium_search/ch-af-apac-eu-row-us/local_paper_profile_zero_prox/branch_F_alpha_0p50/audit_sweep_006.json",
        "provenance": [
            "outputs/equilibrium_search/ch-af-apac-eu-row-us/local_paper_profile_zero_prox/paper_profile.json",
            "outputs/equilibrium_search/ch-af-apac-eu-row-us/local_paper_profile_zero_prox/branch_F_alpha_0p50/initialization.json",
            *[
                f"outputs/equilibrium_search/ch-af-apac-eu-row-us/local_paper_profile_zero_prox/branch_F_alpha_0p50/sweep_{sweep:03d}.json"
                for sweep in range(1, 7)
            ],
        ],
        "method": "50% interpolation from the manuscript offer prices toward manufacturing cost, followed by six zero-proximal Gauss--Seidel sweeps at alpha=0.50",
    },
    {
        "sequence": "ch-af-eu-us-row-apac",
        "equilibrium_number": 2,
        "source_iteration": 15,
        "order": ["ch", "af", "eu", "us", "row", "apac"],
        "label": "O6 sweep 10 plus selective refinement sweep 3",
        "package_name": "o6_s010_selective_s003_20260915",
        "checkpoint": "outputs/equilibrium_search/ch-af-eu-us-row-apac/selective_refinement/selective_eq2_20260915_123400/selective_a010_cap010/sweep_003.json",
        "one_start": "outputs/equilibrium_search/ch-af-eu-us-row-apac/selective_refinement/selective_eq2_20260915_123400/selective_a010_cap010/audits/audit_sweep_003_one_start.json",
        "three_start": "outputs/equilibrium_search/ch-af-eu-us-row-apac/selective_refinement/selective_eq2_20260915_123400/selective_a010_cap010/audits/audit_sweep_003_three_start.json",
        "provenance": [
            "outputs/equilibrium_search/ch-af-eu-us-row-apac/local_paper_profile_zero_prox/paper_profile.json",
            "outputs/equilibrium_search/ch-af-eu-us-row-apac/local_paper_profile_zero_prox/branch_F_alpha_0p50/initialization.json",
            *[
                f"outputs/equilibrium_search/ch-af-eu-us-row-apac/local_paper_profile_zero_prox/branch_F_alpha_0p50/sweep_{sweep:03d}.json"
                for sweep in range(1, 7)
            ],
            "outputs/equilibrium_search/ch-af-eu-us-row-apac/o6_analogue/other_sequences_20260915_114939/branches/O6/initialization.json",
            *[
                f"outputs/equilibrium_search/ch-af-eu-us-row-apac/o6_analogue/other_sequences_20260915_114939/branches/O6/sweep_{sweep:03d}.json"
                for sweep in range(1, 11)
            ],
            *[
                f"outputs/equilibrium_search/ch-af-eu-us-row-apac/selective_refinement/selective_eq2_20260915_123400/selective_a010_cap010/sweep_{sweep:03d}.json"
                for sweep in range(1, 4)
            ],
        ],
        "method": "Branch-F sweep 6 followed by the adaptive O6 path through sweep 10, then three zero-proximal selective sweeps at alpha=0.10 with normalized applied moves capped at 0.01 and players already within 1% frozen",
    },
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def relative_gains(audit: dict[str, Any]) -> dict[str, float]:
    return {str(row["player"]): float(row["relative_gain"]) for row in audit["players"]}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def package(spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint = load(spec["checkpoint"])
    one = load(spec["one_start"])
    three = load(spec["three_start"])
    if not one["equilibrium_verified"] or float(one["max_relative_gain"]) > 0.01:
        raise RuntimeError(f"{spec['sequence']} selected one-start audit does not pass")
    if not one["all_attempts_successful"] or int(one["starts"]) != 1:
        raise RuntimeError(f"{spec['sequence']} selected audit does not meet the agreed one-start rule")
    destination = ROOT / "outputs" / "equilibria" / spec["sequence"] / spec["package_name"]
    destination.mkdir(parents=True, exist_ok=True)
    provenance_dir = destination / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    ending_profile = checkpoint["ending_profile"]
    write(
        destination / "profile.json",
        {
            "created": now(),
            "label": spec["label"],
            "status": "accepted_one_start_local_1pct_equilibrium",
            "source_checkpoint": spec["checkpoint"],
            "state": ending_profile["strategy"],
            "capacities": ending_profile.get("capacities"),
            "market": ending_profile.get("market"),
            "search": {
                "method": spec["method"],
                "player_order": spec["order"],
                "algorithmic_proximal_penalties": {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0},
            },
        },
    )
    shutil.copy2(ROOT / spec["one_start"], destination / "frozen_audit_one_start.json")
    shutil.copy2(ROOT / spec["three_start"], destination / "frozen_audit_three_start_diagnostic.json")

    copied = []
    for index, source_string in enumerate(spec["provenance"], start=1):
        source = ROOT / source_string
        if not source.exists():
            raise FileNotFoundError(source)
        target = provenance_dir / f"{index:02d}_{source.parent.name}_{source.name}"
        shutil.copy2(source, target)
        copied.append({"step": index, "source": source_string, "packaged_copy": str(target.relative_to(ROOT))})
    write(destination / "search_path.json", {"created": now(), "sequence": spec["sequence"], "method": spec["method"], "steps": copied})

    certification = {
        "created": now(),
        "status": "accepted_one_start_local_1pct_equilibrium",
        "claim_scope": "computational local equilibrium criterion; not a global Nash-equilibrium certificate",
        "equilibrium_number_in_original_manuscript": spec["equilibrium_number"],
        "source_workbook": f"outputs/sens/converged/sens_{spec['sequence']}.xlsx",
        "source_iteration": spec["source_iteration"],
        "source_profile_selection_rule": "first endpoint of three consecutive reported strategy-change residuals below 1%",
        "player_order": spec["order"],
        "candidate_generation": spec["method"],
        "candidate_checkpoint": spec["checkpoint"],
        "packaged_profile": str((destination / "profile.json").relative_to(ROOT)),
        "algorithmic_proximal_penalties": {"q": 0.0, "p": 0.0, "a": 0.0, "dk": 0.0},
        "economic_quadratic_penalties": {"q": 0.1, "p": 0.1, "a": 0.1},
        "common_profile_frozen": True,
        "relative_gain_tolerance": 0.01,
        "multistart_count": 1,
        "all_audit_attempts_successful": bool(one["all_attempts_successful"]),
        "maximum_frozen_profile_relative_gain": float(one["max_relative_gain"]),
        "maximum_gain_player": one["max_gain_player"],
        "frozen_profile_relative_gains": relative_gains(one),
        "market_diagnostics": one["reference_market_diagnostics"],
        "three_start_diagnostic": {
            "file": "frozen_audit_three_start_diagnostic.json",
            "all_attempts_successful": bool(three["all_attempts_successful"]),
            "maximum_relative_gain": float(three["max_relative_gain"]),
            "maximum_gain_player": three["max_gain_player"],
            "relative_gains": relative_gains(three),
            "passes_1pct": bool(three["equilibrium_verified"]),
            "role": "retained diagnostic; not the agreed acceptance criterion",
        },
    }
    write(destination / "certification.json", certification)
    readme = f"""# {spec['label']}

This directory preserves the accepted profile for manuscript Equilibrium {spec['equilibrium_number']} ({' -> '.join(v.upper() for v in spec['order'])}).

Under the agreed computational local 1% equilibrium criterion, one unregularized unilateral deviation problem was solved for every player from the candidate strategy while all rival strategies were frozen at one common profile. All six solves succeeded; the maximum identified relative gain was {100.0 * float(one['max_relative_gain']):.4f}% for {str(one['max_gain_player']).upper()}.

The separate three-start diagnostic found a maximum gain of {100.0 * float(three['max_relative_gain']):.4f}% for {str(three['max_gain_player']).upper()}. It is retained for transparency but is not the agreed acceptance criterion.

This is a local computational criterion, not a global Nash-equilibrium certificate. See `certification.json` and `search_path.json` for the exact scope and provenance.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    manifest_entries = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file() and item.name != "MANIFEST.json"):
        manifest_entries.append(
            {
                "path": str(path.relative_to(destination)),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    write(
        destination / "MANIFEST.json",
        {
            "created": now(),
            "package": str(destination.relative_to(ROOT)),
            "files": manifest_entries,
        },
    )
    return {
        "sequence": spec["sequence"],
        "package": str(destination.relative_to(ROOT)),
        "maximum_relative_gain": float(one["max_relative_gain"]),
        "maximum_gain_player": one["max_gain_player"],
        "three_start_maximum_relative_gain": float(three["max_relative_gain"]),
        "three_start_maximum_gain_player": three["max_gain_player"],
    }


def close_workflow_manifests(results: list[dict[str, Any]]) -> None:
    other_path = ROOT / "workflow" / "other_sequences_20260915_114939_manifest.json"
    other = json.loads(other_path.read_text(encoding="utf-8"))
    other["status"] = "complete_with_selective_refinement"
    other["updated"] = now()
    other["extended_o6_stopped_after_sweep"] = 30
    other["stopping_reason"] = "pure O6 continuation moved away from its best sweep-10 profile; refinement continued in workflow/selective_eq2_20260915_123400_manifest.json"
    other["accepted_packages"] = results
    write(other_path, other)

    selective_path = ROOT / "workflow" / "selective_eq2_20260915_123400_manifest.json"
    selective = json.loads(selective_path.read_text(encoding="utf-8"))
    selective["status"] = "complete_with_cancelled_inferior_branch"
    selective["updated"] = now()
    selective["cancelled_branch"] = {
        "branch": "selective_a005_cap005",
        "reason": "stopped after the independent selective_a010_cap010 branch passed; its first three audits were inferior",
        "sweeps_preserved": 3,
    }
    write(selective_path, selective)


def main() -> None:
    results = [package(spec) for spec in SPECS]
    close_workflow_manifests(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
