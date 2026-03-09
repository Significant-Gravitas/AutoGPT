---
name: worktree-setup
description: Set up a new git worktree for parallel development. Creates the worktree, copies .env files, installs dependencies, generates Prisma client, and optionally starts the app (with port conflict resolution) or runs tests. TRIGGER when user asks to set up a worktree, work on a branch in isolation, or needs a separate environment for a branch or PR.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Worktree Setup

## Preferred: Use Branchlet

The repo has a `.branchlet.json` config — it handles env file copying, dependency installation, and Prisma generation automatically.

```bash
npm install -g branchlet                                      # install once
branchlet create -n <name> -s <source-branch> -b <new-branch>
branchlet list --json   # list all worktrees
```

## Manual Fallback

If branchlet isn't available:

1. `git worktree add ../<RepoName><N> <branch-name>`
2. Copy `.env` files: `backend/.env`, `frontend/.env`, `autogpt_platform/.env`, `db/docker/.env`
3. Install deps:
   - `cd autogpt_platform/backend && poetry install && poetry run prisma generate`
   - `cd autogpt_platform/frontend && pnpm install`

## Running the App

Free ports first — backend uses: 8001, 8002, 8003, 8005, 8006, 8007, 8008.

```bash
for port in 8001 8002 8003 8005 8006 8007 8008; do
  lsof -ti :$port | xargs kill -9 2>/dev/null || true
done
cd <worktree>/autogpt_platform/backend && poetry run app
```

## CoPilot Testing Gotcha

SDK mode spawns a Claude subprocess — **won't work inside Claude Code**. Set `CHAT_USE_CLAUDE_AGENT_SDK=false` in `backend/.env` to use baseline mode.
