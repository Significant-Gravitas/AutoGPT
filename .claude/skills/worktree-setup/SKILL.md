---
name: worktree-setup
description: Set up a new git worktree for parallel development. Creates the worktree, copies .env files, installs dependencies, generates Prisma client, and optionally starts the app (with port conflict resolution) or runs tests.
user-invokable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Worktree Setup

## Steps

1. **Create worktree**: Use `branchlet create` (npm: `branchlet`) if available — it auto-copies config files. Otherwise: `git worktree add ../<RepoName><N> <branch-name>`
2. **Copy .env files** (gitignored — branchlet handles this automatically, otherwise copy manually):
   - `autogpt_platform/backend/.env`
   - `autogpt_platform/frontend/.env`
   - `autogpt_platform/.env` (if exists)
3. **Install deps + generate Prisma**:
   - `cd <worktree>/autogpt_platform/backend && poetry install && poetry run prisma generate`
   - `cd <worktree>/autogpt_platform/frontend && pnpm install`

## Running the App

Free ports before starting — backend services use: 8001, 8002, 8003, 8005, 8006, 8007, 8008.

```bash
for port in 8001 8002 8003 8005 8006 8007 8008; do
  lsof -ti :$port | xargs kill -9 2>/dev/null || true
done
cd <worktree>/autogpt_platform/backend && poetry run app
```

## CoPilot Testing Gotcha

SDK mode (`use_claude_agent_sdk=True`) spawns a Claude subprocess — **won't work inside Claude Code**. Set `CHAT_USE_CLAUDE_AGENT_SDK=false` in `backend/.env` to use baseline mode instead.
