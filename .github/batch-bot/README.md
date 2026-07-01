# Batch-deploy bot

Batch several approved PRs onto one **rollup branch** so the preview environment
deploys and is tested **once** for all of them — instead of waiting on N serial
preview deploys to verify N unrelated features. The rollup branch is both the
deploy artifact and the merge artifact: `/batch-merge` enqueues it and its
members land together.

This is deterministic GitHub Actions ChatOps — no AI, nothing needs Claude to run.

## Commands

Comment on any PR (requires **write** access; the listener does a real
repo-permission lookup, not the unreliable `author_association`):

| Command | Effect |
|---|---|
| `/batch` | Add this PR to the current batch; rebuild + redeploy the shared preview |
| `/batch-remove` | Remove this PR from the batch; rebuild + redeploy |
| `/batch-merge` | Land every PR in the batch together (enqueues the rollup) |
| `/batch help` | Post the command list |

## How it works

- **Membership** = the `batch` label. No database; the label is the source of truth.
- On every change the bot **rebuilds `batch/rollup`** from a fresh `dev` by
  sequentially `git merge`-ing each member. A member whose files conflict with
  the group is **ejected** (label removed, author told to rebase) so one
  un-mergeable PR never stalls the batch.
- A **rollup PR** (`batch/rollup` → `dev`) is the sticky status surface, the
  thing the preview env deploys, and the thing `/batch-merge` enqueues.
- `/batch-merge` verifies every member is **approved + green**, then enqueues the
  rollup with a **merge commit** (see "Merge method" below).

## Migrations batch on purpose

Migrations are the highest-value thing to test in a combined deploy — you cannot
verify that two migrations coexist and apply in order without deploying them
together. During assembly `schema.prisma` is merged with git's **union driver**
(set in `.git/info/attributes`, assembly-scoped, never committed) so additive
migrations combine. A genuine same-object clash yields an invalid schema that the
preview's `prisma migrate` fails — which is exactly the migration-vs-migration
conflict you want caught before prod. The preview env must give each batch build a
**fresh database** so the batched migrations apply from scratch.

## Setup

### 1. Bot token — `BATCH_BOT_TOKEN` (repo secret)

Needed (not just `GITHUB_TOKEN`) for two reasons: a `repository_dispatch` made
with `GITHUB_TOKEN` will not trigger the handler, and pushes/merges by
`GITHUB_TOKEN` do not trigger the preview-deploy/CI workflows — and the whole
payoff is the rollup push kicking off one preview.

Use a **fine-grained PAT on a bot account** (or a GitHub App installation token —
preferred long-term: it rotates and triggers workflows). Scope it to this repo
with:

| Permission | Level | Why |
|---|---|---|
| Contents | Read/write | create + force-update the ephemeral `batch/rollup` branch |
| Pull requests | Read/write | label, comment, open/update the rollup PR, enqueue |
| Issues | Read/write | label CRUD + PR conversation comments |
| Actions | Read | read member CI status before batch/merge |
| Metadata | Read | required baseline |

Also: add the bot account as a **write collaborator**, and make sure branch
protection lets it **enqueue the rollup PR** (the rollup carries its members'
human approvals; either auto-approve the rollup PR or exempt the bot on that one
branch). No admin/bypass of review is needed — humans still approve each member.

Optional repo **variables**: `BATCH_BASE_BRANCH` (default `dev`),
`BATCH_BOT_NAME`, `BATCH_BOT_EMAIL`.

### 2. Merge method (attribution)

For members to show **"Merged"** (not "Closed") when the rollup lands, the rollup
must merge as a **merge commit** so member commit SHAs stay in `dev`'s history.
Ensure the merge queue for `dev` is configured for a merge commit. If your queue
is squash-only, members will show "Closed" with their commits landed; add a
post-land step to mark them (or merge each member via its own ref) — flagged as
the one open fallback.

### 3. Preview environment

The preview tooling must deploy on push to `batch/rollup` and provision a fresh
DB per deploy. Put the preview URL into the rollup PR body (the bot leaves a slot).

## Notes / policy call

The `batch/rollup` branch is **force-updated** on every rebuild (via the git refs
API, `force=true`). It is ephemeral and bot-owned — nobody branches off it — so
this is not a force-push of a shared branch. If your policy forbids even that,
switch to a per-run branch name (`batch/rollup-<run_id>`) and repoint the rollup
PR; that trades the tidy single branch for zero force-updates.

Opt a PR out of batching entirely with the `batch:never` label.
