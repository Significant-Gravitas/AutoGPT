#!/usr/bin/env node
// Deterministic PR-batching bot for preview-deploy testing.
//
// Purpose: batch several PRs onto ONE unified branch so the preview environment
// deploys and is tested ONCE for the whole group, instead of N serial previews.
// The unified branch is both the deploy artifact AND the merge artifact —
// `/batch-merge` lands it and its members go together.
//
// MULTIPLE BATCHES: batches are keyed by name. `/batch` uses the `default` batch;
// `/batch <name>` uses a named batch. Up to BATCH_MAX (default 4) run at once, each
// with its own label `batch:<key>`, rollup branch `batch/rollup-<key>`, rollup PR,
// and isolated preview. A PR belongs to at most one batch (re-batching moves it).
//
// Commands (from repository_dispatch action, minus the `-command` suffix), plus
// `reconcile` (fired by batch-reconcile.yml when a rollup PR merges):
//   batch [name]  — add the commenting PR to batch <name> (default: `default`)
//   batch-remove  — remove the commenting PR from its batch
//   batch-merge   — enable auto-merge on the rollup of the PR's batch
//   batch help    — post the command list (ARG1 === "help")
//   reconcile     — after a rollup squash-merges, close its members + clean up
//
// State lives in GitHub, not a DB: `batch:<key>` labels are the source of truth;
// each rollup branch is rebuilt from scratch on every change; a per-key "rollup PR"
// (batch/rollup-<key> -> BASE) is the sticky status + deploy + merge target.
//
// SECURITY: every git/gh call goes through execFileSync with ARRAY args — no shell,
// so attacker-controlled PR titles / branch names / batch names can never break out
// (single argv entries). Batch names are additionally slug-validated. Do not
// reintroduce string commands.
//
// Migrations batch on purpose: a combined preview deploy is the only way to test
// that two migrations coexist and apply in order. `schema.prisma` is union-merged
// during assembly; the preview's `prisma migrate` is the backstop that catches a
// genuine clash (union can otherwise concatenate into a valid-but-wrong schema).

import { execFileSync } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const REPO = req("REPO");
const BASE = process.env.BASE || "dev";
const LABEL_PREFIX = process.env.BATCH_LABEL_PREFIX || "batch:";
const NEVER_KEY = "never"; // batch:never = opt-out escape hatch
const NEVER = LABEL_PREFIX + NEVER_KEY;
const DEFAULT_KEY = "default";
// Positive-integer cap; a non-numeric BATCH_MAX must not silently disable the limit.
const MAX_BATCHES = (() => {
  const n = Number(process.env.BATCH_MAX);
  return Number.isInteger(n) && n > 0 ? n : 4;
})();
const RESERVED = new Set([NEVER_KEY, "help", "remove", "merge"]); // not usable as batch names
const REQUIRE_APPROVAL_TO_ADD = process.env.BATCH_REQUIRE_APPROVAL === "1";
const MARKER = "<!-- batch-bot:rollup -->";
const command = req("COMMAND").replace(/-command$/, ""); // batch | batch-remove | batch-merge | reconcile
const prNumber = process.env.PR_NUMBER ? Number(process.env.PR_NUMBER) : null;
const arg1 = (process.env.ARG1 || "").trim();
const rollupUrl = process.env.ROLLUP_URL || "";
const rollupPr = process.env.ROLLUP_PR ? Number(process.env.ROLLUP_PR) : null;
const rollupBranchEnv = process.env.ROLLUP_BRANCH || ""; // reconcile: the merged rollup head ref
const SAFE_REF = /^[\w./][\w./-]*$/; // allowed git ref chars, no leading dash (arg-injection guard)

function req(name) {
  const v = process.env[name];
  if (!v) throw new Error(`missing required env ${name}`);
  return v;
}

// --- exec: array args only, never a shell string -------------------------
function run(file, args, opts = {}) {
  return execFileSync(file, args, { encoding: "utf8", stdio: ["pipe", "pipe", "pipe"], ...opts }).trim();
}
function tryRun(file, args) {
  try {
    return { ok: true, out: run(file, args) };
  } catch (e) {
    return { ok: false, out: (e.stdout || "") + (e.stderr || "") };
  }
}
const git = (args) => run("git", args);
const tryGit = (args) => tryRun("git", args);
const gh = (args) => run("gh", args);
const tryGh = (args) => tryRun("gh", args);
const ghJSON = (args) => JSON.parse(gh(args));

// --- batch keys, labels, branches ---------------------------------------

const labelFor = (key) => `${LABEL_PREFIX}${key}`; // batch:default
const rollupBranch = (key) => `batch/rollup-${key}`; // batch/rollup-default

// Validate + normalize a batch name from a command arg. Empty → the default batch.
function batchKey(arg) {
  const raw = (arg || "").trim().toLowerCase();
  if (!raw) return DEFAULT_KEY;
  if (!/^[a-z0-9][a-z0-9-]{0,29}$/.test(raw))
    throw new Error(`invalid batch name "${arg}" — use lowercase letters, digits and hyphens (e.g. \`/batch hotfix\`)`);
  if (RESERVED.has(raw)) throw new Error(`"${raw}" is reserved and can't be a batch name`);
  return raw;
}

// Batch keys a PR currently belongs to (from its labels; excludes the opt-out label).
function batchesOf(prJson) {
  return prJson.labels
    .map((l) => l.name)
    .filter((n) => n.startsWith(LABEL_PREFIX) && n.slice(LABEL_PREFIX.length) !== NEVER_KEY)
    .map((n) => n.slice(LABEL_PREFIX.length));
}

// Derive the batch key from a rollup branch name (reconcile).
function keyFromBranch(branch) {
  const m = /^batch\/rollup-(.+)$/.exec(branch || "");
  return m ? m[1] : null;
}

// Distinct batch keys currently in use across open PRs (for the concurrency cap).
// `gh pr list --limit` is a hard fetch cap, not pagination — fetch a large page and
// FAIL LOUDLY if it saturates rather than silently undercount batches (which could
// let a new batch exceed the cap).
function activeKeys() {
  const LIMIT = 1000;
  const rows = ghJSON(["pr", "list", "--repo", REPO, "--state", "open", "--limit", String(LIMIT), "--json", "labels"]);
  if (rows.length >= LIMIT)
    throw new Error(`activeKeys hit the ${LIMIT}-open-PR list cap — needs true pagination to count batches safely`);
  const keys = new Set();
  for (const p of rows)
    for (const l of p.labels)
      if (l.name.startsWith(LABEL_PREFIX)) {
        const k = l.name.slice(LABEL_PREFIX.length);
        if (k && k !== NEVER_KEY) keys.add(k);
      }
  return keys;
}

// --- membership ----------------------------------------------------------

const PR_FIELDS = "number,title,headRefName,headRefOid,url,labels,reviewDecision,mergeable,isDraft";

// `gh pr list --limit` caps results (no pagination); fail loudly if a single batch
// ever saturates it rather than silently drop members from the rollup.
function members(key) {
  const LIMIT = 300;
  const rows = ghJSON([
    "pr", "list", "--repo", REPO, "--state", "open", "--label", labelFor(key), "--limit", String(LIMIT),
    "--json", PR_FIELDS,
  ]);
  if (rows.length >= LIMIT)
    throw new Error(`batch \`${key}\` hit the ${LIMIT}-member list cap — needs true pagination`);
  return rows.filter((p) => !p.labels.some((l) => l.name === NEVER));
}

// `members()` lists via GitHub's label search index, which lags a just-edited label
// by a few seconds in BOTH directions — a just-added PR can be missing and a
// just-removed one can still appear. Reconcile the caller's known edit with a direct
// (consistent) read so the first /batch and /batch-remove aren't no-ops.
function membersFor(key, { ensure = null, exclude = null } = {}) {
  let list = members(key);
  if (exclude != null && list.some((p) => p.number === exclude)) {
    const p = ghJSON(["pr", "view", String(exclude), "--repo", REPO, "--json", "number,labels"]);
    if (!p.labels.some((l) => l.name === labelFor(key))) list = list.filter((q) => q.number !== exclude);
  }
  if (ensure != null && !list.some((p) => p.number === ensure)) {
    const p = ghJSON(["pr", "view", String(ensure), "--repo", REPO, "--json", PR_FIELDS]);
    if (p.labels.some((l) => l.name === labelFor(key)) && !p.labels.some((l) => l.name === NEVER)) list.push(p);
  }
  return list;
}

function comment(pr, body) {
  const f = `/tmp/batch-comment-${Number(pr)}.md`;
  writeFileSync(f, body);
  gh(["pr", "comment", String(Number(pr)), "--repo", REPO, "--body-file", f]);
}

// Named-batch labels are created on demand; `--add-label` fails on a missing label.
function ensureLabel(name) {
  tryGh(["label", "create", name, "--repo", REPO, "--color", "5319e7", "--description", "batch-bot batch membership"]);
}
function addLabel(pr, key) {
  ensureLabel(labelFor(key));
  gh(["pr", "edit", String(Number(pr)), "--repo", REPO, "--add-label", labelFor(key)]);
}
function removeLabel(pr, key) {
  tryGh(["pr", "edit", String(Number(pr)), "--repo", REPO, "--remove-label", labelFor(key)]);
}

// --- rollup branch assembly ---------------------------------------------

// Rebuild batch/rollup-<key> from a fresh BASE by sequentially merging each member.
// A member with an unsafe branch name, an unfetchable branch (e.g. a fork), or a
// non-schema conflict is EJECTED (label removed) so one bad PR never stalls a batch.
function buildRollup(key, list) {
  const branch = rollupBranch(key);
  git(["fetch", "origin", BASE, "--quiet"]);
  git(["checkout", "-B", branch, `origin/${BASE}`]);

  // Local-only union driver for schema.prisma (never committed, assembly-scoped).
  mkdirSync(".git/info", { recursive: true });
  writeFileSync(".git/info/attributes", "**/schema.prisma merge=union\n");

  const merged = [];
  const ejected = [];
  for (const p of list) {
    if (!SAFE_REF.test(p.headRefName)) {
      ejected.push({ ...p, files: "unsafe branch name" });
      removeLabel(p.number, key);
      comment(p.number, `${MARKER}\n🤖 Removed from batch \`${key}\` — branch name is not a plain git ref.`);
      continue;
    }
    const fetched = tryGit(["fetch", "origin", p.headRefName, "--quiet"]);
    if (!fetched.ok) {
      ejected.push({ ...p, files: "branch not on origin (fork?)" });
      removeLabel(p.number, key);
      comment(
        p.number,
        `${MARKER}\n🤖 Removed from batch \`${key}\` — the head branch could not be fetched from \`origin\` ` +
          `(cross-fork PRs cannot be batched). Push the branch to the main repo to batch it.`,
      );
      continue;
    }
    const res = tryGit(["merge", "--no-ff", "FETCH_HEAD", "-m", `batch: ${p.title} (#${p.number})`]);
    if (res.ok) {
      merged.push(p);
      continue;
    }
    tryGit(["merge", "--abort"]);
    const files = tryGit(["diff", "--name-only", "--diff-filter=U"]).out || "conflicting files";
    ejected.push({ ...p, files });
    removeLabel(p.number, key);
    comment(
      p.number,
      `${MARKER}\n🤖 Removed from batch \`${key}\` — this PR conflicts with the rest of the group and could ` +
        `not be merged onto \`${branch}\` (${files}). Rebase onto \`${BASE}\` (or resolve against the other ` +
        `batched PRs) and re-add with \`/batch ${key}\`.`,
    );
  }

  // Force-push the ephemeral, bot-owned rollup branch. The refs API can't be used
  // here: the merge commits exist only in this runner's clone and the API refuses to
  // point a ref at objects the server has never received (422 "Object does not
  // exist"). The push uploads the objects and creates/force-moves the branch in one
  // step, as the bot (checkout token) — the batch/rollup-* ruleset keeps these
  // branches bot-only (see README).
  git(["push", "--force", "origin", `HEAD:refs/heads/${branch}`]);
  return { merged, ejected };
}

// --- rollup PR (sticky status + deploy + merge target) -------------------

function findRollupPR(key) {
  // Plain branch name only: `gh pr list --head` does NOT understand the
  // owner-qualified `owner:branch` form (that's a `pr create` convention) — it
  // silently matches nothing, which makes every rebuild after the first try to
  // create a duplicate rollup PR and die on "already exists".
  const rows = ghJSON([
    "pr", "list", "--repo", REPO, "--head", rollupBranch(key), "--base", BASE, "--state", "open",
    "--json", "number,url",
  ]);
  return rows[0] || null;
}

function rollupBody(key, merged, ejected) {
  const lines = [MARKER, "", `### Batch rollup \`${key}\` — ${merged.length} PR(s)`, ""];
  lines.push("Deploying the union of these PRs to a single preview so they are tested together.", "");
  for (const p of merged) lines.push(`- [ ] #${p.number} — ${p.title}`);
  if (ejected.length) {
    lines.push("", "**Ejected (conflicted with the group, rebase to re-add):**");
    for (const p of ejected) lines.push(`- #${p.number} — ${p.title}`);
  }
  lines.push(
    "",
    "---",
    `Commands (comment on any member PR): \`/batch ${key}\` add · \`/batch-remove\` drop · \`/batch-merge\` land all · \`/batch help\``,
    "Preview: an isolated per-PR full-stack env (namespace `pr-<n>`, its own DB schema running the",
    "batched migrations) is (re)deployed automatically via `!deploy` on each batch change, and torn",
    "down when this PR merges or closes.",
    "",
    "> This PR is the union of everyone's code. Its approval must be a real human review of the",
    "> merged result — do not auto-approve or bypass; per-member approvals do not cover the union.",
  );
  return lines.join("\n");
}

function upsertRollupPR(key, merged, ejected) {
  const f = "/tmp/rollup-body.md";
  writeFileSync(f, rollupBody(key, merged, ejected));
  const title = `Batch rollup \`${key}\`: ${merged.length} PR(s)`;
  let pr = findRollupPR(key);
  if (!pr) {
    if (merged.length === 0) return null;
    const url = gh([
      "pr", "create", "--repo", REPO, "--head", rollupBranch(key), "--base", BASE, "--draft",
      "--title", title, "--body-file", f,
    ]);
    return { number: Number(url.split("/").pop()), url };
  }
  gh(["pr", "edit", String(pr.number), "--repo", REPO, "--title", title, "--body-file", f]);
  if (merged.length === 0) {
    // Closing the rollup PR triggers the deploy dispatcher's PR-close auto-undeploy.
    tryGh(["pr", "close", String(pr.number), "--repo", REPO, "--delete-branch"]);
    return null;
  }
  return pr;
}

// The infra preview system deploys a per-PR isolated env on an exact `!deploy`
// comment (autogpt-platform-preview-env-cd.yml, keyed by PR number). Posting it on a
// rollup PR (re)deploys ONE env whose DB runs all that batch's migrations. Each
// rollup PR gets its own namespace, so multiple batches preview concurrently.
function deployRollup(pr) {
  tryGh(["pr", "comment", String(pr.number), "--repo", REPO, "--body", "!deploy"]);
}

// Clear any armed auto-merge on a batch's rollup PR before we rewrite the branch.
// The bot has write access, so GitHub does NOT auto-cancel auto-merge on our own
// push — without this, a rebuilt (content-changed) rollup could still land on the
// prior approval/checks. /batch-merge re-arms it only after the fresh set passes.
function disarmRollup(key) {
  const pr = findRollupPR(key);
  if (pr) tryGh(["pr", "merge", String(pr.number), "--repo", REPO, "--disable-auto"]);
}

// --- commands ------------------------------------------------------------

function assertBatchable(pr) {
  const p = ghJSON([
    "pr", "view", String(Number(pr)), "--repo", REPO,
    "--json", "number,isDraft,state,labels,reviewDecision",
  ]);
  if (p.state !== "OPEN") throw new Error(`PR #${pr} is not open`);
  if (p.labels.some((l) => l.name === NEVER))
    throw new Error(`PR #${pr} is labeled ${NEVER} (opted out of batching)`);
  if (REQUIRE_APPROVAL_TO_ADD && p.reviewDecision !== "APPROVED")
    throw new Error(`PR #${pr} is not approved (${p.reviewDecision || "no reviews"}) and BATCH_REQUIRE_APPROVAL is set`);
  return p;
}

// Rebuild one batch's rollup branch + PR + preview. Disarms any armed auto-merge
// first (branch is about to change), reconciles label-index lag for a just-added
// (ensurePr) or just-removed (excludePr) member, and optionally reports on a PR.
function rebuildBatch(key, { ensurePr = null, excludePr = null, note = null, reportTo = null } = {}) {
  disarmRollup(key);
  const list = membersFor(key, { ensure: ensurePr, exclude: excludePr });
  const { merged, ejected } = buildRollup(key, list);
  const pr = upsertRollupPR(key, merged, ejected);
  if (pr) deployRollup(pr); // (re)deploy the combined preview to reflect the new batch
  if (note && reportTo) {
    const names = merged.map((p) => `#${p.number}`).join(", ") || "none";
    comment(
      reportTo,
      `${MARKER}\n🤖 ${note} Batch \`${key}\` (${merged.length}): ${names}.` +
        (ejected.length ? ` Ejected: ${ejected.map((p) => "#" + p.number).join(", ")}.` : "") +
        (pr ? `\n\nDeploying the combined preview (${pr.url}); \`/batch-merge\` lands them together.` : ""),
    );
  }
  console.log(`batch[${key}]: ${merged.length} member(s)${pr ? ` → ${pr.url}` : " (empty)"}`);
  return { merged, ejected, pr };
}

function cmdBatch() {
  if (arg1.toLowerCase() === "help") return cmdHelp();
  if (!prNumber) throw new Error("no PR number");
  const key = batchKey(arg1);
  const p = assertBatchable(prNumber);

  // Concurrency cap: a brand-new batch key can't push us past BATCH_MAX active
  // batches. This deliberately does NOT try to credit a batch this PR might vacate by
  // moving: the label search index lags, so "is this PR the sole member of that
  // batch?" can't be answered reliably, and crediting an only-seemingly-empty batch
  // would let a 5th batch slip past the cap. Erring strict never exceeds the cap; the
  // workaround for a move at the limit is `/batch-remove` first (pointed to below).
  const active = activeKeys();
  if (!active.has(key) && active.size >= MAX_BATCHES) {
    comment(
      prNumber,
      `${MARKER}\n🤖 Can't start batch \`${key}\` — the limit of ${MAX_BATCHES} concurrent batches is reached ` +
        `(active: ${[...active].sort().map((k) => "`" + k + "`").join(", ")}). Add this PR to one of those ` +
        `(\`/batch <name>\`), land one with \`/batch-merge\`, or \`/batch-remove\` here then \`/batch ${key}\`.`,
    );
    return;
  }

  // One batch per PR: move it out of any other batch it's in, and rebuild those
  // (excluding this PR, since the label-search index lags the removal).
  const others = batchesOf(p).filter((k) => k !== key);
  for (const other of others) removeLabel(prNumber, other);
  addLabel(prNumber, key);
  for (const other of others) rebuildBatch(other, { excludePr: prNumber });
  rebuildBatch(key, { ensurePr: prNumber, note: `Added #${prNumber} to batch \`${key}\`.`, reportTo: prNumber });
}

function cmdBatchRemove() {
  if (!prNumber) throw new Error("no PR number");
  const p = ghJSON(["pr", "view", String(prNumber), "--repo", REPO, "--json", "number,labels"]);
  const keys = batchesOf(p);
  if (keys.length === 0) {
    comment(prNumber, `${MARKER}\n🤖 #${prNumber} isn't in any batch.`);
    return;
  }
  for (const key of keys) removeLabel(prNumber, key);
  for (const key of keys)
    rebuildBatch(key, { excludePr: prNumber, note: `Removed #${prNumber} from batch \`${key}\`.`, reportTo: prNumber });
}

// Returns merge-blocker strings for a member's CI. `gh pr checks --json` exits
// non-zero when checks are failing (1) or still pending (8) but STILL prints the
// JSON to stdout, so we must read the output regardless of exit code — gating on
// the exit code alone silently skips CI for that member. Only `pass`/`skipping`
// count as green; fail, cancel, pending, or an unreadable/empty response all block.
// Fail closed: a member must be provably green to land.
function checkBlockers(p) {
  const raw = tryGh(["pr", "checks", String(p.number), "--repo", REPO, "--json", "bucket"]).out || "";
  // tryGh folds stderr into `out` on non-zero exit, so extract just the array.
  const start = raw.indexOf("[");
  const end = raw.lastIndexOf("]");
  let rows = null;
  if (start !== -1 && end > start) {
    try {
      rows = JSON.parse(raw.slice(start, end + 1));
    } catch {
      rows = null;
    }
  }
  if (rows === null) return [`#${p.number} check status could not be determined`];
  // Zero reported checks can't be confirmed green — block (fail closed) rather than
  // let a PR with no CI slip through the gate on an empty array.
  if (rows.length === 0) return [`#${p.number} has no reported checks — cannot confirm green`];
  const bad = rows.filter((c) => c.bucket !== "pass" && c.bucket !== "skipping");
  if (!bad.length) return [];
  const buckets = [...new Set(bad.map((c) => c.bucket))].sort().join(", ");
  return [`#${p.number} not green (${buckets})`];
}

function cmdBatchMerge() {
  if (!prNumber) throw new Error("no PR number");
  const meta = ghJSON(["pr", "view", String(prNumber), "--repo", REPO, "--json", "number,labels"]);
  const keys = batchesOf(meta);
  if (keys.length === 0) {
    comment(prNumber, `${MARKER}\n🤖 #${prNumber} isn't in any batch — nothing to merge.`);
    return;
  }
  if (keys.length > 1) {
    // A PR should only ever be in one batch (re-batching moves it). If it somehow
    // carries multiple batch:* labels, refuse rather than guess which one to land.
    comment(
      prNumber,
      `${MARKER}\n🤖 #${prNumber} is in multiple batches (${keys.map((k) => "`" + k + "`").join(", ")}) — ` +
        `remove it from all but one with \`/batch-remove\` before \`/batch-merge\`.`,
    );
    return;
  }
  const key = keys[0];
  // Use the lag-compensated list (ensuring the commenting PR, which we just confirmed
  // is a member): a member added right before this in the serialized handler queue
  // could otherwise be missed — a false "empty" reply or a partial rollup.
  const list = membersFor(key, { ensure: prNumber });
  if (list.length === 0) {
    comment(prNumber, `${MARKER}\n🤖 Batch \`${key}\` is empty — nothing to merge.`);
    return;
  }
  disarmRollup(key); // clear any prior auto-merge before the branch is rewritten
  const { merged, ejected } = buildRollup(key, list);
  const pr = upsertRollupPR(key, merged, ejected);
  if (!pr) {
    // Every member was ejected during assembly (conflict / fork / unsafe ref), so
    // upsertRollupPR closed the rollup PR. Report cleanly instead of throwing.
    comment(prNumber, `${MARKER}\n🤖 Every PR in batch \`${key}\` was ejected during assembly — nothing left to merge.`);
    return;
  }
  // Refresh the combined preview so a maintainer approves what will actually land.
  deployRollup(pr);
  // Gate on the PRs that actually made it into the rollup — assembly may have ejected
  // members, and an ejected PR's CI must not block a rollup it's no longer part of.
  // Each remaining member must be explicitly approved and green: `reviewDecision`
  // must be exactly APPROVED (null / REVIEW_REQUIRED / CHANGES_REQUESTED all block).
  const blockers = [];
  for (const p of merged) {
    if (p.reviewDecision !== "APPROVED")
      blockers.push(`#${p.number} not approved (${p.reviewDecision || "no reviews"})`);
    blockers.push(...checkBlockers(p));
  }
  if (blockers.length) {
    comment(prNumber, `${MARKER}\n🤖 Not merging batch \`${key}\` — resolve first:\n- ${blockers.join("\n- ")}`);
    return;
  }
  // No merge queue on dev + squash-only: use auto-merge (squash). It lands the rollup
  // once its OWN required human approval + green checks are satisfied — the bot never
  // approves or bypasses. batch-reconcile.yml closes members afterward.
  gh(["pr", "ready", String(pr.number), "--repo", REPO]);
  const enq = tryGh(["pr", "merge", String(pr.number), "--repo", REPO, "--squash", "--auto"]);
  const note = enq.ok
    ? `Auto-merge enabled on the \`${key}\` rollup (${merged.length} PR(s)). It lands once a maintainer ` +
      `approves ${pr.url} and checks are green — then all members merge together.`
    : `Could not enable auto-merge on the \`${key}\` rollup: ${enq.out}. Check branch protection / that a ` +
      `reviewer can approve ${pr.url}.`;
  comment(prNumber, `${MARKER}\n🤖 ${note}`);
}

// Fired by batch-reconcile.yml after a rollup PR squash-merges. Squash rewrites SHAs,
// so members won't auto-flip to "Merged" — close them explicitly with credit. The
// batch key is derived from the merged rollup branch (ROLLUP_BRANCH).
function cmdReconcile() {
  const key = keyFromBranch(rollupBranchEnv);
  if (!key) throw new Error(`reconcile: could not derive batch key from ROLLUP_BRANCH="${rollupBranchEnv}"`);
  // Tear the combined preview down explicitly (the deploy dispatcher also
  // auto-undeploys on the rollup PR's close; this guaranteed, idempotent path is safe).
  if (rollupPr) tryGh(["pr", "comment", String(rollupPr), "--repo", REPO, "--body", "!undeploy"]);
  const list = members(key);
  for (const p of list) {
    comment(
      p.number,
      `${MARKER}\n🤖 Landed on \`${BASE}\` via batch \`${key}\` rollup${rollupUrl ? ` ${rollupUrl}` : ""}. ` +
        `Closing — your change is merged.`,
    );
    removeLabel(p.number, key);
    tryGh(["pr", "close", String(p.number), "--repo", REPO]);
  }
  tryGit(["push", "origin", "--delete", rollupBranch(key)]);
  console.log(`reconciled batch[${key}]: ${list.length} member(s)`);
}

function cmdHelp() {
  if (!prNumber) return;
  comment(
    prNumber,
    `${MARKER}\n🤖 **Batch commands** (comment on any PR):\n` +
      "- `/batch` — add this PR to the `default` batch\n" +
      "- `/batch <name>` — add this PR to a named batch (e.g. `/batch hotfix`)\n" +
      "- `/batch-remove` — remove this PR from its batch\n" +
      "- `/batch-merge` — land every PR in this PR's batch together (auto-merge on the rollup)\n" +
      "- `/batch help` — show this message\n\n" +
      `Up to ${MAX_BATCHES} batches run at once, each with its own preview. A PR belongs to one batch ` +
      `(re-batching moves it). Requires write access. Opt a PR out with the \`${NEVER}\` label.`,
  );
}

// --- dispatch ------------------------------------------------------------

try {
  if (command === "batch") cmdBatch();
  else if (command === "batch-remove") cmdBatchRemove();
  else if (command === "batch-merge") cmdBatchMerge();
  else if (command === "reconcile") cmdReconcile();
  else throw new Error(`unknown command: ${command}`);
} catch (e) {
  const msg = e && e.message ? e.message : String(e);
  if (prNumber) {
    try {
      comment(prNumber, `${MARKER}\n🤖 Batch command failed: ${msg}`);
    } catch {}
  }
  console.error(msg);
  process.exit(1);
}
