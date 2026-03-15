---
name: worktree-setup
description: Set up a new git worktree for parallel development. Creates the worktree, copies .env files, installs dependencies, and generates Prisma client. TRIGGER when user asks to set up a worktree, work on a branch in isolation, or needs a separate environment for a branch or PR.
user-invocable: true
metadata:
  author: autogpt-team
  version: "2.0.0"
---

# Worktree Setup

## Create the worktree

Convention: `AutoGPT<N>` where N is the next available number.

```bash
# From an existing branch
git worktree add /Users/majdyz/Code/AutoGPT<N> <branch-name>

# From a new branch off dev
git worktree add -b <new-branch> /Users/majdyz/Code/AutoGPT<N> dev
```

Check existing worktrees first: `git worktree list`

## Copy environment files

```bash
ROOT=/Users/majdyz/Code/AutoGPT
TARGET=/Users/majdyz/Code/AutoGPT<N>

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
git worktree remove /Users/majdyz/Code/AutoGPT<N>
```

## Alternative: Branchlet (optional)

If [branchlet](https://www.npmjs.com/package/branchlet) is installed and `.branchlet.json` is configured, it automates env copying and dependency installation:

```bash
npm install -g branchlet
branchlet create -n <name> -s <source-branch> -b <new-branch>
branchlet list --json
```
