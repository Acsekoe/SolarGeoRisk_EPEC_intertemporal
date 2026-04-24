# This script removes unsupported Git worktree configuration that causes silent
# crashes in Antigravity's internal Git parser (e.g., when Claude Code leaves them behind).

Write-Host "Fixing Antigravity Git worktree bug..."
git config --local --unset extensions.worktreeConfig
git worktree prune
Write-Host "Done! Please restart your IDE if Antigravity is still unresponsive."
