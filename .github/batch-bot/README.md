# Batch-deploy bot

Batch several approved PRs onto one **rollup branch** so the preview environment
deploys and is tested **once** for all of them — instead of waiting on N serial
preview deploys to verify N unrelated features. The rollup branch is both the
deploy artifact and the merge artifact: `/batch-merge` lands it and its members
go together.

Deterministic GitHub Actions ChatOps — no AI, nothing needs Claude to run.

## Commands

Comment on any PR (requires **write** access; the listener does a real
repo-permission lookup, not the unreliable `author_association`):

| Command | Effect |
|---|---|
| `/batch` | Add this PR to the current batch; rebuild + redeploy the shared preview |
| `/batch-remove` | Remove this PR from the batch; rebuild + redeploy |
| `/batch-merge` | Land every PR in the batch together (enables auto-merge on the rollup) |
| `/batch help` | Post the command list |

## How it works

- **Membership** = the `batch` label. No database; the label is the source of truth.
- On every change the bot **rebuilds `batch/rollup`** from a fresh `dev` by
  sequentially `git merge`-ing each member. A member that is a fork branch, has an
  unsafe ref name, or conflicts with the group is **ejected** (label removed,
  author told to rebase) so one bad PR never stalls the batch.
- A **rollup PR** (`batch/rollup` → `dev`) is the sticky status surface, the thing
  the preview env deploys, and the thing `/batch-merge` lands.
- `/batch-merge` requires every member **`reviewDecision === APPROVED` + green**,
  then enables **auto-merge (squash)** on the rollup. It lands once the rollup PR's
  own required human approval + checks pass.
- After the rollup squash-merges, `batch-reconcile.yml` **closes each member** with
  a comment crediting the rollup (squash rewrites SHAs, so members would otherwise
  show neither "Merged" nor closed) and deletes the rollup branch.

`dev` is squash-only with linear history and no merge queue, so this uses
auto-merge, not a queue. If you later add a native merge queue you get the
always-green-merge-result guarantee; swap the `--auto` enable for an enqueue.

## Migrations batch on purpose

Migrations are the highest-value thing to test in a combined deploy — you cannot
verify two migrations coexist and apply in order without deploying them together.
During assembly `schema.prisma` is merged with git's **union driver**
(`.git/info/attributes`, assembly-scoped, never committed) so additive migrations
combine. The **preview's `prisma migrate` is the backstop**: union can concatenate
two edits to the same model into a valid-but-wrong schema, so the combined preview
run is what actually proves the migrations are compatible. Give each batch build a
**fresh database**. (Cross-PR migration ordering is timestamp-driven — review the
rollup if ordering matters.)

## Security requirements (must hold before enabling)

This bot was security-reviewed. The following are **hard requirements**, not
suggestions:

1. **The rollup PR must get a genuine human approval — never auto-approved or
   bypassed.** The rollup is the *union* of everyone's code plus a union-merged
   schema; per-member approvals do not cover the merged result. The bot authors the
   rollup PR, so a maintainer approving it is never a self-approval. Keep `dev`'s
   required-review rule; do **not** add a bot self-approve or a review bypass.
2. **Enable "dismiss stale reviews on push"** on `dev`, and **protect
   `batch/rollup`** so only the bot identity can push to it. Otherwise a member
   could get the rollup approved, then push, and land on a stale approval.
3. **Preview isolation.** `/batch` deploys code that is not yet code-reviewed (that
   is the point — test early), and it co-locates multiple authors' code + arbitrary
   migration SQL in one environment. The preview MUST use **non-prod, per-preview,
   network-isolated secrets** and a **fresh DB that shares no server/creds** with
   other environments. Optionally set `BATCH_REQUIRE_APPROVAL=1` to require an
   approval before a PR can be added to the batch.
4. **Pinned actions.** All actions are pinned to commit SHAs (`slash-command-dispatch`,
   `checkout`, `setup-node`); keep them pinned when bumping.
5. **No shell.** `batch.mjs` uses array-arg `execFileSync` only — attacker-controlled
   PR titles / branch names are passed as single argv entries and cannot inject.
   Do not reintroduce string-form `exec`.

## Setup

### 1. Bot token — `BATCH_BOT_TOKEN` (repo secret)

Needed (not just `GITHUB_TOKEN`) because a `repository_dispatch` made with
`GITHUB_TOKEN` won't trigger the handler, and pushes by `GITHUB_TOKEN` won't
trigger the preview-deploy — and the payoff is the rollup push kicking off one
preview.

**Prefer a GitHub App installation token** (short-lived, auto-rotating,
repo-scoped) over a PAT. If you must use a PAT, make it **fine-grained,
single-repo, with an expiry** — never a classic PAT. Either way grant only:

| Permission | Level | Why |
|---|---|---|
| Contents | Read/write | create + update the ephemeral `batch/rollup` branch |
| Pull requests | Read/write | label, comment, open/update/merge the rollup PR |
| Issues | Read/write | label CRUD + PR conversation comments |
| Actions | Read | read member CI status before batch/merge |
| Metadata | Read | required baseline |

Add the identity as a **write collaborator**. Do **not** grant admin or a
review-bypass — that would turn the token into a review-bypass primitive.

Optional repo **variables**: `BATCH_BASE_BRANCH` (default `dev`), `BATCH_BOT_NAME`,
`BATCH_BOT_EMAIL`.

### 2. Preview environment

Deploy on push to `batch/rollup` with a fresh, isolated DB per build (see security
requirement 3). Put the preview URL into the rollup PR body.

## Opt-out

Opt a PR out of batching with the `batch:never` label.
