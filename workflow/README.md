# Workflow records

This directory contains stable run manifests and short timestamped research summaries.

## Naming convention

- New conversational summaries: `summary_YYYYMMDD_HHMMSS.md`, using Europe/Vienna local time.
- Active machine-readable run records: `*_manifest.json`. These remain in the workflow root because project scripts reference them there.
- Superseded non-timestamped narrative summaries: `archive/`.

Summaries are immutable snapshots. Create a new timestamped file for each requested update rather than appending to or replacing an earlier summary.

