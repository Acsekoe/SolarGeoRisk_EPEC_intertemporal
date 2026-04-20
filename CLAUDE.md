# Project Instructions

## Paper Notes

Whenever you identify a finding that is highly relevant for the paper — in terms of **methodology**, **results**, **convergence behaviour**, or **equilibrium properties** — write it to:

```
notes/notes_YYYY-MM-DD.txt
```

where `YYYY-MM-DD` is today's actual date. Append to the file if it already exists for that date; create it if not.

Each entry should be a short, self-contained note — 2–5 sentences max — that could be directly useful when writing the paper. Focus on non-obvious insights, not restatements of what was just done.

## Overleaf Git Sync

If the user asks to "push IEEE Paper to master" or "sync overleaf" or anything similar, it means the AI should automatically execute the following sequence of commands (with `run_command` replacing the cd command with `Cwd: 'c:\EEG\EPEC\EPEC_VS_code\SolarGeoRisk_EPEC_intertemporal\IEEE Paper'`):

```bash
git add .
git commit -m "Updated paper from AI assistant"
git push origin master
```
