# Antigravity Troubleshooting Notes

## Git Worktree Bug

**Symptom**: Antigravity freezes, stops responding to messages, or silently crashes in the background.

**Cause**: Other CLI AI tools (like Claude Code) sometimes create temporary Git worktrees and modify the repository's configuration by setting `extensions.worktreeConfig = true` in the `.git/config` file. Antigravity's internal Git parser doesn't currently support this specific extension, causing a failure when parsing the workspace.

**Solution**:
Run the following commands in the terminal (or execute the `scripts/fix_git_worktree_bug.ps1` script):
```bash
git config --local --unset extensions.worktreeConfig
git worktree prune
```
After running these commands, restart the IDE to restore the connection.
