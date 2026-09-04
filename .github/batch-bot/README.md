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
| `/batch` | Add this PR to the `default` batch; rebuild + redeploy its preview |
| `/batch <name>` | Add this PR to the named batch `<name>` (e.g. `/batch hotfix`) |
| `/batch-remove` | Remove this PR from its batch; rebuild + redeploy |
| `/batch-merge` | Land every PR in this PR's batch together (auto-merge on its rollup) |
| `/batch help` | Post the command list |

Up to **`BATCH_MAX` (default 4)** named batches run concurrently, each with its own
label, rollup branch, rollup PR, and isolated preview. A PR belongs to **one** batch
at a time — re-batching it into another moves it. `never`, `help`, `remove` and
`merge` are reserved and can't be batch names; names are lowercased and must match
`[a-z0-9][a-z0-9-]*`.

## How it works

- **Membership** = the `batch:<key>` label (e.g. `batch:default`, `batch:hotfix`). No
  database; the labels are the source of truth. `batch:never` is the opt-out. Labels
  are created on demand.
- On every change the bot **rebuilds `batch/rollup-<key>`** from a fresh `dev` by
  sequentially `git merge`-ing each member. A member that is a fork branch, has an
  unsafe ref name, or conflicts with the group is **ejected** (label removed,
  author told to rebase) so one bad PR never stalls the batch.
- A per-batch **rollup PR** (`batch/rollup-<key>` → `dev`) is the sticky status
  surface, the thing the preview env deploys, and the thing `/batch-merge` lands.
  Each rollup PR gets its own preview namespace, so multiple batches deploy at once.
- **Preview:** the bot reuses the repo's existing per-PR preview system. On each
  batch change it posts a bare **`!deploy`** comment on the rollup PR, which the
  `platform-dev-deploy-event-dispatcher.yml` → `AutoGPT_cloud_infrastructure`
  pipeline turns into ONE **isolated per-PR environment** keyed by the rollup PR's
  number (namespace `pr-<n>`, Postgres schema `pr_<n>`, URLs `pr-<n>-server.agpt.co`
  / `autogpt-pr-<n>.vercel.app`) that **runs all the batched migrations against its
  own schema**. That isolation is provided by the preview system, not this bot.
- `/batch-merge` requires every member **`reviewDecision === APPROVED` + green**,
  then enables **auto-merge (squash)** on the rollup. It lands once the rollup PR's
  own required human approval + checks pass.
- After the rollup squash-merges, `batch-reconcile.yml` posts **`!undeploy`** to
  tear the preview down and **closes each member** with a comment crediting the
  rollup (squash rewrites SHAs, so members would otherwise show neither "Merged"
  nor closed), then deletes the rollup branch. (The deploy dispatcher also
  auto-undeploys on the rollup PR's close; the overlap is idempotent.)

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
   `batch/rollup-*`** — a ruleset targeting `refs/heads/batch/rollup-*` that
   restricts create/update/delete to the bot App (bypass actor) — so only the bot
   can push to any rollup branch. Otherwise a member could get the rollup approved,
   then push, and land on a stale approval.
3. **Preview isolation.** `/batch` deploys code that is not yet code-reviewed (that
   is the point — test early), co-locating multiple authors' code + migration SQL in
   one environment. This is handled by the existing per-PR preview system
   (`AutoGPT_cloud_infrastructure`): each rollup gets its own namespace `pr-<n>`,
   Postgres schema `pr_<n>`, and URLs — non-prod and per-PR isolated. Confirm those
   previews use non-prod secrets (they do by design). Nuance: isolation is
   schema-level within a shared preview Postgres cluster, not separate DB servers —
   fine for non-prod previews. Optionally set the repository variable
   `BATCH_REQUIRE_APPROVAL=1` (wired into `batch-command-handler.yml`) to require
   an approval before a PR can be added to the batch.
4. **Pinned actions.** All actions are pinned to commit SHAs (`slash-command-dispatch`,
   `checkout`, `setup-node`); keep them pinned when bumping.
5. **No shell.** `batch.mjs` uses array-arg `execFileSync` only — attacker-controlled
   PR titles / branch names are passed as single argv entries and cannot inject.
   Do not reintroduce string-form `exec`.

## Setup

### 1. Bot identity — GitHub App (`BATCH_BOT_APP_ID` variable + `BATCH_BOT_PRIVATE_KEY` secret)

Both workflows mint a short-lived installation token per run via
`actions/create-github-app-token`. A real identity (not `GITHUB_TOKEN`) is required
because a `repository_dispatch` made with `GITHUB_TOKEN` won't trigger the handler,
and pushes by `GITHUB_TOKEN` won't trigger the preview-deploy — the payoff is the
rollup push kicking off one preview.

Create a **GitHub App** (org-owned preferred) and grant only these **repository**
permissions:

| Permission | Level | Why |
|---|---|---|
| Contents | Read/write | create + update the ephemeral `batch/rollup-*` branches |
| Pull requests | Read/write | label, comment, open/update/merge the rollup PR |
| Issues | Read/write | label CRUD + PR conversation comments |
| Actions | Read | read member CI status before batch/merge |
| Metadata | Read | required baseline |

Store the App's **App ID** as the repo variable `BATCH_BOT_APP_ID` and its **private
key** (`.pem` contents) as the secret `BATCH_BOT_PRIVATE_KEY`, then **install the App
on this repo**. All three workflows (listener, handler, reconcile) mint their token
from these — there is no PAT fallback. Do **not** grant admin or a review-bypass —
that would turn the token into a review-bypass primitive.

Optional repo **variables**: `BATCH_BASE_BRANCH` (default `dev`), `BATCH_MAX`
(default `4`, max concurrent batches), `BATCH_BOT_NAME`, `BATCH_BOT_EMAIL`.

### 2. Preview environment

No new preview infra needed — the bot reuses the repo's existing per-PR preview
system by posting `!deploy` on the rollup PR (handled by
`platform-dev-deploy-event-dispatcher.yml` → `AutoGPT_cloud_infrastructure`'s
`autogpt-platform-preview-env-cd.yml`). Two prerequisites: that per-PR preview
system is enabled, and the App bot's `!deploy`/`!undeploy` comments clear the deploy
dispatcher's gate on the commenter's `author_association` — verify this end-to-end,
since an App bot comment can read as `NONE`; if it doesn't qualify, add the bot to
the dispatcher's allowlist (or keep the write-collaborator PAT fallback for deploys).

## Opt-out

Opt a PR out of batching with the `batch:never` label.
