---
name: setup-repo
description: Initialize a worktree-based repo layout for parallel development. Creates a main worktree, a reviews worktree for PR reviews, and N numbered work branches. Handles .env creation, dependency installation, and branchlet config. TRIGGER when user asks to set up the repo from scratch, initialize worktrees, bootstrap their dev environment, "setup repo", "setup worktrees", "initialize dev environment", "set up branches", or when a freshly cloned repo has no sibling worktrees.
user-invocable: true
args: "No arguments — interactive setup via prompts."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Repository Setup

This skill sets up a worktree-based development layout from a freshly cloned repo. It creates:
- A **main** worktree (the primary checkout)
- A **reviews** worktree (for PR reviews)
- **N work branches** (branch1..branchN) for parallel development

## Step 1: Identify the repo

Determine the repo root and parent directory:

```bash
ROOT=$(git rev-parse --show-toplevel)
REPO_NAME=$(basename "$ROOT")
PARENT=$(dirname "$ROOT")
```

If the repo is already inside a worktree layout (i.e. `$ROOT` is already named `main` and siblings exist), detect that and use the existing structure. Otherwise, proceed with fresh setup.

## Step 2: Ask the user questions

Use AskUserQuestion to gather setup preferences:

1. **How many parallel work branches do you need?** (Options: 4, 8, 16, or custom)
   - These become `branch1` through `branchN`
2. **Which branch should be the base?** (Options: origin/master, origin/dev, or custom)
   - All work branches and reviews will start from this

## Step 3: Fetch and set up branches

```bash
cd "$ROOT"
git fetch origin

# Create the reviews branch from base (skip if already exists)
if git show-ref --verify --quiet refs/heads/reviews; then
  echo "INFO: Branch 'reviews' already exists, skipping"
else
  git branch reviews <base-branch>
fi

# Create numbered work branches from base (skip if already exists)
for i in $(seq 1 "$COUNT"); do
  if git show-ref --verify --quiet "refs/heads/branch$i"; then
    echo "INFO: Branch 'branch$i' already exists, skipping"
  else
    git branch "branch$i" <base-branch>
  fi
done
```

## Step 4: Create worktrees

Create worktrees as siblings to the main checkout:

```bash
git worktree add "$PARENT/reviews" reviews
for i in $(seq 1 "$COUNT"); do
  git worktree add "$PARENT/branch$i" "branch$i"
done
```

## Step 5: Set up environment files

**Do NOT assume .env files exist.** For each worktree (including main if needed):

1. Check if `.env` exists in the source worktree for each path
2. If `.env` exists, copy it
3. If only `.env.default` or `.env.example` exists, copy that as `.env`
4. If neither exists, warn the user and list which env files are missing

Env file locations to check:
- `autogpt_platform/.env`
- `autogpt_platform/backend/.env`
- `autogpt_platform/frontend/.env`
- `autogpt_platform/db/docker/.env`

```bash
SOURCE="$ROOT"
WORKTREES="reviews"
for i in $(seq 1 "$COUNT"); do WORKTREES="$WORKTREES branch$i"; done

for wt in $WORKTREES; do
  TARGET="$PARENT/$wt"
  for envpath in autogpt_platform autogpt_platform/backend autogpt_platform/frontend autogpt_platform/db/docker; do
    if [ -f "$SOURCE/$envpath/.env" ]; then
      cp "$SOURCE/$envpath/.env" "$TARGET/$envpath/.env"
    elif [ -f "$SOURCE/$envpath/.env.default" ]; then
      cp "$SOURCE/$envpath/.env.default" "$TARGET/$envpath/.env"
      echo "NOTE: $wt/$envpath/.env was created from .env.default — you may need to edit it"
    elif [ -f "$SOURCE/$envpath/.env.example" ]; then
      cp "$SOURCE/$envpath/.env.example" "$TARGET/$envpath/.env"
      echo "NOTE: $wt/$envpath/.env was created from .env.example — you may need to edit it"
    else
      echo "WARNING: No .env, .env.default, or .env.example found at $SOURCE/$envpath/"
    fi
  done
done
```

If no .env files exist at all in the source, ask the user if they want to continue without them or set them up first.

## Step 6: Copy branchlet config

Copy `.branchlet.json` from main to each worktree so branchlet can manage sub-worktrees:

```bash
if [ -f "$ROOT/.branchlet.json" ]; then
  for wt in $WORKTREES; do
    cp "$ROOT/.branchlet.json" "$PARENT/$wt/.branchlet.json"
  done
fi
```

## Step 7: Install dependencies

Install deps in all worktrees. Run these sequentially per worktree:

```bash
for wt in $WORKTREES; do
  TARGET="$PARENT/$wt"
  echo "=== Installing deps for $wt ==="
  (cd "$TARGET/autogpt_platform/autogpt_libs" && poetry install) &&
  (cd "$TARGET/autogpt_platform/backend" && poetry install && poetry run prisma generate) &&
  (cd "$TARGET/autogpt_platform/frontend" && pnpm install) &&
  echo "=== Done: $wt ===" ||
  echo "=== FAILED: $wt ==="
done
```

This is slow. Run in background if possible and notify when complete.

## Step 8: Verify and report

After setup, verify and report to the user:

```bash
git worktree list
```

Summarize:
- Number of worktrees created
- Which env files were copied vs created from defaults vs missing
- Any warnings or errors encountered

## Final directory layout

```
parent/
  main/              # Primary checkout (already exists)
  reviews/           # PR review worktree
  branch1/           # Work branch 1
  branch2/           # Work branch 2
  ...
  branchN/           # Work branch N
```
