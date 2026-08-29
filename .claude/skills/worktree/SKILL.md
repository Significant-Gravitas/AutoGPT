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

Derive paths from the git toplevel. If a name is provided as argument, use it. Otherwise, check `git worktree list` and pick the next `AutoGPT<N>`.

```bash
ROOT=$(git rev-parse --show-toplevel)
PARENT=$(dirname "$ROOT")

# From an existing branch
git worktree add "$PARENT/<NAME>" <branch-name>

# From a new branch off dev
git worktree add -b <new-branch> "$PARENT/<NAME>" dev
```

## Copy environment files

Copy `.env` from the root worktree. Falls back to `.env.default` if `.env` doesn't exist.

```bash
ROOT=$(git rev-parse --show-toplevel)
TARGET="$(dirname "$ROOT")/<NAME>"

for envpath in autogpt_platform/backend autogpt_platform/frontend autogpt_platform; do
  if [ -f "$ROOT/$envpath/.env" ]; then
    cp "$ROOT/$envpath/.env" "$TARGET/$envpath/.env"
  elif [ -f "$ROOT/$envpath/.env.default" ]; then
    cp "$ROOT/$envpath/.env.default" "$TARGET/$envpath/.env"
  fi
done
```

## Install dependencies

```bash
TARGET="$(dirname "$(git rev-parse --show-toplevel)")/<NAME>"
cd "$TARGET/autogpt_platform/autogpt_libs" && poetry install
cd "$TARGET/autogpt_platform/backend" && poetry install && poetry run prisma generate
cd "$TARGET/autogpt_platform/frontend" && pnpm install
```

Replace `<NAME>` with the actual worktree name (e.g., `AutoGPT7`).

## Running the app (optional)

Backend uses ports: 8001, 8002, 8003, 8005, 8006, 8007, 8008. Free them first if needed:

```bash
TARGET="$(dirname "$(git rev-parse --show-toplevel)")/<NAME>"
for port in 8001 8002 8003 8005 8006 8007 8008; do
  lsof -ti :$port | xargs kill -9 2>/dev/null || true
done
cd "$TARGET/autogpt_platform/backend" && poetry run app
```

## CoPilot testing

SDK mode spawns a Claude subprocess — won't work inside Claude Code. Set `CHAT_USE_CLAUDE_AGENT_SDK=false` in `backend/.env` to use baseline mode.

## Cleanup

```bash
# Replace <NAME> with the actual worktree name (e.g., AutoGPT7)
git worktree remove "$(dirname "$(git rev-parse --show-toplevel)")/<NAME>"
```

## Alternative: Branchlet (optional)

If [branchlet](https://www.npmjs.com/package/branchlet) is installed:

```bash
branchlet create -n <name> -s <source-branch> -b <new-branch>
```
