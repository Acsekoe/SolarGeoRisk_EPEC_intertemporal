"""Selection of accepted Stage-2 branches for reported economic results."""

from __future__ import annotations

import json
from pathlib import Path


def excluded_candidate_ids(run_root: Path) -> set[str]:
    selection_path = run_root / "results_selection.json"
    if not selection_path.is_file():
        return set()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = selection.get("excluded_candidates", [])
    ids = [str(entry["candidate"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate reporting exclusions in {selection_path}")
    return set(ids)
