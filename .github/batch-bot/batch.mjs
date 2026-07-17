#!/usr/bin/env node
// Deterministic PR-batching bot for preview-deploy testing.
//
// Purpose: batch several approved PRs onto ONE unified branch so the preview
// environment deploys and is tested ONCE for all of them, instead of N serial
// preview deploys. The unified branch is both the deploy artifact AND the merge
// artifact — `/batch-merge` lands it and its members go together.
//
// Commands (from repository_dispatch action, minus the `-command` suffix), plus
// `reconcile` (fired by batch-reconcile.yml when the rollup PR merges):
//   batch         — add the commenting PR to the batch, rebuild + redeploy the group
//   batch-remove  — remove the commenting PR from the batch, rebuild + redeploy
//   batch-merge   — enable auto-merge on the rollup; members land once it's approved+green
//   batch help    — post the command list (ARG1 === "help")
//   reconcile     — after the rollup squash-merges, close members with credit + clean up
//
// State lives in GitHub, not a DB: the `batch` label is the source of truth; the
// ROLLUP branch is rebuilt from scratch on every change; a "rollup PR"
// (ROLLUP -> BASE) is the sticky status surface + deploy + merge target.
//
// SECURITY: every `git`/`gh` call goes through execFileSync with ARRAY args — no
// shell, so attacker-controlled PR titles / branch names can never break out
// (they are passed as single argv entries). Do not reintroduce string commands.
//
// Migrations batch on purpose: a combined preview deploy is the only way to test
// that two migrations coexist and apply in order. `schema.prisma` is union-merged
// during assembly; the preview's `prisma migrate` is the backstop that catches a
// genuine clash (union can otherwise concatenate into a valid-but-wrong schema).

import { execFileSync } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const REPO = req("REPO");
const BASE = process.env.BASE || "dev";
const ROLLUP = process.env.ROLLUP_BRANCH || "batch/rollup";
const LABEL = process.env.BATCH_LABEL || "batch";
const NEVER = process.env.NEVER_BATCH_LABEL || "batch:never"; // opt-out escape hatch
const REQUIRE_APPROVAL_TO_ADD = process.env.BATCH_REQUIRE_APPROVAL === "1";
const MARKER = "<!-- batch-bot:rollup -->";
const command = req("COMMAND").replace(/-command$/, ""); // batch | batch-remove | batch-merge | reconcile
const prNumber = process.env.PR_NUMBER ? Number(process.env.PR_NUMBER) : null;
const arg1 = (process.env.ARG1 || "").trim();
const rollupUrl = process.env.ROLLUP_URL || "";
const rollupPr = process.env.ROLLUP_PR ? Number(process.env.ROLLUP_PR) : null;
const SAFE_REF = /^[\w./-]+$/; // git ref chars we allow through, defense-in-depth

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

// --- membership ----------------------------------------------------------

function members() {
  const rows = ghJSON([
    "pr", "list", "--repo", REPO, "--state", "open", "--label", LABEL, "--limit", "100",
    "--json", "number,title,headRefName,headRefOid,url,labels,reviewDecision,mergeable,isDraft",
  ]);
  return rows.filter((p) => !p.labels.some((l) => l.name === NEVER));
}

function comment(pr, body) {
  const f = `/tmp/batch-comment-${Number(pr)}.md`;
  writeFileSync(f, body);
  gh(["pr", "comment", String(Number(pr)), "--repo", REPO, "--body-file", f]);
}

function unlabel(pr) {
  tryGh(["pr", "edit", String(Number(pr)), "--repo", REPO, "--remove-label", LABEL]);
}

// --- rollup branch assembly ---------------------------------------------

// Rebuild ROLLUP from a fresh BASE by sequentially merging each member. A member
// with an unsafe branch name, an unfetchable branch (e.g. a fork), or a non-schema
// conflict is EJECTED (label removed) so one bad PR never stalls the batch.
function buildRollup(list) {
  git(["fetch", "origin", BASE, "--quiet"]);
  git(["checkout", "-B", ROLLUP, `origin/${BASE}`]);

  // Local-only union driver for schema.prisma (never committed, assembly-scoped).
  mkdirSync(".git/info", { recursive: true });
  writeFileSync(".git/info/attributes", "**/schema.prisma merge=union\n");

  const merged = [];
  const ejected = [];
  for (const p of list) {
    if (!SAFE_REF.test(p.headRefName)) {
      ejected.push({ ...p, files: "unsafe branch name" });
      unlabel(p.number);
      comment(p.number, `${MARKER}\n🤖 Removed from the batch — branch name is not a plain git ref.`);
      continue;
    }
    const fetched = tryGit(["fetch", "origin", p.headRefName, "--quiet"]);
    if (!fetched.ok) {
      ejected.push({ ...p, files: "branch not on origin (fork?)" });
      unlabel(p.number);
      comment(
        p.number,
        `${MARKER}\n🤖 Removed from the batch — the head branch could not be fetched from \`origin\` ` +
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
    unlabel(p.number);
    comment(
      p.number,
      `${MARKER}\n🤖 Removed from the batch — this PR conflicts with the rest of the current ` +
        `group and could not be merged onto \`${ROLLUP}\` (${files}). Rebase onto \`${BASE}\` ` +
        `(or resolve against the other batched PRs) and re-add with \`/batch\`.`,
    );
  }

  // Force-push the ephemeral, bot-owned rollup branch. The refs API cannot be
  // used here: the merge commits exist only in this runner's clone, and the
  // API refuses to point a ref at objects the server has never received
  // (422 "Object does not exist"). The push uploads the objects and creates
  // or force-moves the branch in one step, authenticated as the bot via the
  // checkout token — keep batch/rollup protected to bot-only pushes (README).
  git(["push", "--force", "origin", `HEAD:refs/heads/${ROLLUP}`]);
  return { merged, ejected };
}

// --- rollup PR (sticky status + deploy + merge target) -------------------

function findRollupPR() {
  // Plain branch name only: `gh pr list --head` does NOT understand the
  // owner-qualified `owner:branch` form (that's a `pr create` convention) —
  // it silently matches nothing, which made every rebuild after the first
  // try to create a duplicate rollup PR and die on "already exists".
  const rows = ghJSON([
    "pr", "list", "--repo", REPO, "--head", ROLLUP, "--base", BASE, "--state", "open",
    "--json", "number,url",
  ]);
  return rows[0] || null;
}

function rollupBody(merged, ejected) {
  const lines = [MARKER, "", `### Batch rollup — ${merged.length} PR(s)`, ""];
  lines.push("Deploying the union of these PRs to a single preview so they are tested together.", "");
  for (const p of merged) lines.push(`- [ ] #${p.number} — ${p.title}`);
  if (ejected.length) {
    lines.push("", "**Ejected (conflicted with the group, rebase to re-add):**");
    for (const p of ejected) lines.push(`- #${p.number} — ${p.title}`);
  }
  lines.push(
    "",
    "---",
    "Commands (comment on any PR): `/batch` add · `/batch-remove` drop · `/batch-merge` land all · `/batch help`",
    "Preview: an isolated per-PR full-stack env (namespace `pr-<n>`, its own DB schema running the",
    "batched migrations) is (re)deployed automatically via `!deploy` on each batch change, and torn",
    "down when this PR merges or closes.",
    "",
    "> This PR is the union of everyone's code. Its approval must be a real human review of the",
    "> merged result — do not auto-approve or bypass; per-member approvals do not cover the union.",
  );
  return lines.join("\n");
}

function upsertRollupPR(merged, ejected) {
  const f = "/tmp/rollup-body.md";
  writeFileSync(f, rollupBody(merged, ejected));
  const title = `Batch rollup: ${merged.length} PR(s)`;
  let pr = findRollupPR();
  if (!pr) {
    if (merged.length === 0) return null;
    const url = gh([
      "pr", "create", "--repo", REPO, "--head", ROLLUP, "--base", BASE, "--draft",
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
// comment (autogpt-platform-preview-env-cd.yml, keyed by PR number). Posting it on
// the rollup PR (re)deploys ONE env whose DB runs all the batched migrations. The
// bot must be a write collaborator for the deploy dispatcher to honor the comment.
function deployRollup(pr) {
  tryGh(["pr", "comment", String(pr.number), "--repo", REPO, "--body", "!deploy"]);
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

function rebuildAndReport(actionNote, ensureNumber = null) {
  // GitHub's label index lags behind `pr edit --add-label`, so the very first
  // /batch on a PR can list an empty batch and build nothing. When the caller
  // just labeled a PR, fetch it directly and union it into the member list.
  let list = members();
  if (ensureNumber && !list.some((p) => p.number === ensureNumber)) {
    const row = ghJSON([
      "pr", "view", String(ensureNumber), "--repo", REPO,
      "--json", "number,title,headRefName,headRefOid,url,labels,reviewDecision,mergeable,isDraft",
    ]);
    if (!row.labels.some((l) => l.name === NEVER)) list.push(row);
  }
  const { merged, ejected } = buildRollup(list);
  const pr = upsertRollupPR(merged, ejected);
  if (pr) deployRollup(pr); // (re)deploy the combined preview to reflect the new batch
  if (prNumber) {
    const names = merged.map((p) => `#${p.number}`).join(", ") || "none";
    comment(
      prNumber,
      `${MARKER}\n🤖 ${actionNote} Current batch (${merged.length}): ${names}.` +
        (ejected.length ? ` Ejected: ${ejected.map((p) => "#" + p.number).join(", ")}.` : "") +
        (pr ? `\n\nDeploying the combined preview (${pr.url}); \`/batch-merge\` lands them together.` : ""),
    );
  }
  console.log(`batch: ${merged.length} member(s)${pr ? ` → ${pr.url}` : " (empty)"}`);
}

function cmdBatch() {
  if (arg1.toLowerCase() === "help") return cmdHelp();
  if (!prNumber) throw new Error("no PR number");
  assertBatchable(prNumber);
  gh(["pr", "edit", String(prNumber), "--repo", REPO, "--add-label", LABEL]);
  rebuildAndReport(`Added #${prNumber} to the batch.`, prNumber);
}

function cmdBatchRemove() {
  if (!prNumber) throw new Error("no PR number");
  unlabel(prNumber);
  rebuildAndReport(`Removed #${prNumber} from the batch.`);
}

function cmdBatchMerge() {
  const list = members();
  if (list.length === 0) {
    if (prNumber) comment(prNumber, `${MARKER}\n🤖 The batch is empty — nothing to merge.`);
    return;
  }
  // Every member must be explicitly approved and green. `reviewDecision` must be
  // exactly APPROVED — null / REVIEW_REQUIRED / CHANGES_REQUESTED all block.
  const blockers = [];
  for (const p of list) {
    if (p.reviewDecision !== "APPROVED")
      blockers.push(`#${p.number} not approved (${p.reviewDecision || "no reviews"})`);
    const checks = tryGh(["pr", "checks", String(p.number), "--repo", REPO, "--json", "bucket"]);
    if (checks.ok) {
      const bad = JSON.parse(checks.out).filter((c) => c.bucket === "fail" || c.bucket === "cancel");
      if (bad.length) blockers.push(`#${p.number} has failing checks`);
    }
  }
  const { merged, ejected } = buildRollup(list);
  const pr = upsertRollupPR(merged, ejected);
  if (!pr) throw new Error("rollup PR missing after build");
  if (blockers.length) {
    comment(prNumber || pr.number, `${MARKER}\n🤖 Not merging — resolve first:\n- ${blockers.join("\n- ")}`);
    return;
  }
  // No merge queue on dev + squash-only: use auto-merge (squash). It lands the
  // rollup once its OWN required human approval + green checks are satisfied — the
  // bot never approves or bypasses. batch-reconcile.yml closes members afterward.
  gh(["pr", "ready", String(pr.number), "--repo", REPO]);
  const enq = tryGh(["pr", "merge", String(pr.number), "--repo", REPO, "--squash", "--auto"]);
  const note = enq.ok
    ? `Auto-merge enabled on the rollup (${merged.length} PR(s)). It lands once a maintainer ` +
      `approves ${pr.url} and checks are green — then all members merge together.`
    : `Could not enable auto-merge on the rollup: ${enq.out}. Check branch protection / that a ` +
      `reviewer can approve ${pr.url}.`;
  comment(prNumber || pr.number, `${MARKER}\n🤖 ${note}`);
}

// Fired by batch-reconcile.yml after the rollup PR squash-merges. Squash rewrites
// SHAs, so members won't auto-flip to "Merged" — close them explicitly with credit.
function cmdReconcile() {
  // Tear the combined preview down explicitly (the deploy dispatcher also
  // auto-undeploys on the rollup PR's close; this is the guaranteed path and
  // idempotent teardown makes the overlap harmless).
  if (rollupPr) tryGh(["pr", "comment", String(rollupPr), "--repo", REPO, "--body", "!undeploy"]);
  const list = members();
  for (const p of list) {
    comment(p.number, `${MARKER}\n🤖 Landed on \`${BASE}\` via batch rollup${rollupUrl ? ` ${rollupUrl}` : ""}. Closing — your change is merged.`);
    unlabel(p.number);
    tryGh(["pr", "close", String(p.number), "--repo", REPO]);
  }
  tryGit(["push", "origin", "--delete", ROLLUP]);
  console.log(`reconciled ${list.length} member(s)`);
}

function cmdHelp() {
  const target = prNumber || (findRollupPR() || {}).number;
  if (!target) return;
  comment(
    target,
    `${MARKER}\n🤖 **Batch commands** (comment on any PR):\n` +
      "- `/batch` — add this PR to the current batch (rebuilds the shared preview branch)\n" +
      "- `/batch-remove` — remove this PR from the batch\n" +
      "- `/batch-merge` — land every PR in the batch together (enables auto-merge on the rollup)\n" +
      "- `/batch help` — show this message\n\n" +
      `Requires write access. Opt a PR out with the \`${NEVER}\` label.`,
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
