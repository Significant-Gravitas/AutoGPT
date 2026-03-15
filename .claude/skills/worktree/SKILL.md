---
name: worktree
description: Set up a new git worktree for parallel development. Creates the worktree, copies .env files, installs dependencies, and generates Prisma client. TRIGGER when user asks to set up a worktree, work on a branch in isolation, or needs a separate environment for a branch or PR.
user-invocable: true
args: "[name] — optional worktree name (e.g., 'AutoGPT7'). If omitted, uses next available AutoGPT<N>."
metadata:
  author: autogpt-team
  version: "3.0.0"
---

# Worktree Setup

## Create the worktree

If a name is provided as argument, use it. Otherwise, check `git worktree list` and pick the next `AutoGPT<N>`.

```bash
# From an existing branch
git worktree add /Users/majdyz/Code/<NAME> <branch-name>

# From a new branch off dev
git worktree add -b <new-branch> /Users/majdyz/Code/<NAME> dev
```

## Copy environment files

```bash
ROOT=/Users/majdyz/Code/AutoGPT
TARGET=/Users/majdyz/Code/<NAME>

cp "$ROOT/autogpt_platform/backend/.env" "$TARGET/autogpt_platform/backend/.env"
cp "$ROOT/autogpt_platform/frontend/.env" "$TARGET/autogpt_platform/frontend/.env"
cp "$ROOT/autogpt_platform/.env" "$TARGET/autogpt_platform/.env"
```

## Install dependencies

```bash
cd "$TARGET/autogpt_platform/backend" && poetry install && poetry run prisma generate
cd "$TARGET/autogpt_platform/frontend" && pnpm install
```

## Running the app (optional)

Backend uses ports: 8001, 8002, 8003, 8005, 8006, 8007, 8008. Free them first if needed:

```bash
for port in 8001 8002 8003 8005 8006 8007 8008; do
  lsof -ti :$port | xargs kill -9 2>/dev/null || true
done
cd "$TARGET/autogpt_platform/backend" && poetry run app
```

## CoPilot testing

SDK mode spawns a Claude subprocess — won't work inside Claude Code. Set `CHAT_USE_CLAUDE_AGENT_SDK=false` in `backend/.env` to use baseline mode.

## Cleanup

```bash
git worktree remove /Users/majdyz/Code/<NAME>
```

## Alternative: Branchlet (optional)

If [branchlet](https://www.npmjs.com/package/branchlet) is installed:

```bash
branchlet create -n <name> -s <source-branch> -b <new-branch>
```
