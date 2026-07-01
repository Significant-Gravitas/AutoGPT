#!/usr/bin/env node
// Deterministic PR-batching bot for preview-deploy testing.
//
// Purpose: batch several approved PRs onto ONE unified branch so the preview
// environment deploys and is tested ONCE for all of them, instead of N serial
// preview deploys. The unified branch is both the deploy artifact AND the merge
// artifact — `/batch-merge` enqueues it and its members land together.
//
// Commands (from repository_dispatch action, minus the `-command` suffix):
//   batch         — add the commenting PR to the batch, rebuild + redeploy the group
//   batch-remove  — remove the commenting PR from the batch, rebuild + redeploy
//   batch-merge   — enqueue the group branch; members land together
//   batch help    — post the command list (ARG1 === "help")
//
// State lives in GitHub, not a DB:
//   - the `batch` label is the source of truth for membership
//   - the ROLLUP branch is rebuilt from scratch on every change
//   - a "rollup PR" (ROLLUP -> BASE) is the sticky status surface + deploy + enqueue target
//
// Migrations are DELIBERATELY batchable: a combined preview deploy is the only
// way to test that two migrations coexist and apply in order. During assembly
// `schema.prisma` is merged with git's union driver so additive migrations
// combine; a genuine same-object clash produces an invalid schema that the
// preview's `prisma migrate` fails — which is exactly the signal you want.

import { execSync } from "node:child_process";
import { writeFileSync, mkdirSync } from "node:fs";

const REPO = req("REPO");
const BASE = process.env.BASE || "dev";
const ROLLUP = process.env.ROLLUP_BRANCH || "batch/rollup";
const LABEL = process.env.BATCH_LABEL || "batch";
const NEVER = process.env.NEVER_BATCH_LABEL || "batch:never"; // opt-out escape hatch
const MARKER = "<!-- batch-bot:rollup -->";
const command = req("COMMAND").replace(/-command$/, ""); // batch | batch-remove | batch-merge
const prNumber = process.env.PR_NUMBER ? Number(process.env.PR_NUMBER) : null;
const arg1 = (process.env.ARG1 || "").trim();

function req(name) {
  const v = process.env[name];
  if (!v) throw new Error(`missing required env ${name}`);
  return v;
}
function sh(cmd, opts = {}) {
  return execSync(cmd, { encoding: "utf8", stdio: ["pipe", "pipe", "pipe"], ...opts }).trim();
}
function trySh(cmd) {
  try {
    return { ok: true, out: sh(cmd) };
  } catch (e) {
    return { ok: false, out: (e.stdout || "") + (e.stderr || "") };
  }
}
const gh = (args, opts) => sh(`gh ${args}`, opts);
const ghJSON = (args) => JSON.parse(gh(args));

// ---- membership -----------------------------------------------------------

function members() {
  // Open PRs carrying the batch label, minus anything explicitly opted out.
  const rows = ghJSON(
    `pr list --repo ${REPO} --state open --label ${LABEL} --limit 100 ` +
      `--json number,title,headRefName,headRefOid,url,labels,reviewDecision,mergeable,isDraft`,
  );
  return rows.filter((p) => !p.labels.some((l) => l.name === NEVER));
}

function comment(pr, body) {
  const f = `/tmp/batch-comment-${pr}.md`;
  writeFileSync(f, body);
  gh(`pr comment ${pr} --repo ${REPO} --body-file ${f}`);
}

// ---- rollup branch assembly ----------------------------------------------

// Rebuild ROLLUP from a fresh BASE by sequentially merging each member. A member
// whose non-schema files conflict with the group is EJECTED (its label removed)
// so one un-mergeable PR can never stall the batch — the same "don't stall the
// batch" rule the contribute-block workflow uses. schema.prisma is union-merged
// so additive migrations combine instead of conflicting.
function buildRollup(list) {
  sh(`git fetch origin ${BASE} --quiet`);
  sh(`git checkout -B ${ROLLUP} origin/${BASE}`);

  // Local-only union driver for schema.prisma (never committed, assembly-scoped).
  mkdirSync(".git/info", { recursive: true });
  writeFileSync(".git/info/attributes", "**/schema.prisma merge=union\n");

  const merged = [];
  const ejected = [];
  for (const p of list) {
    sh(`git fetch origin ${p.headRefName} --quiet`);
    const res = trySh(`git merge --no-ff FETCH_HEAD -m "batch: ${p.title} (#${p.number})"`);
    if (res.ok) {
      merged.push(p);
      continue;
    }
    // Conflict: back out cleanly, drop from the batch, tell the author why.
    trySh("git merge --abort");
    const files = trySh("git diff --name-only --diff-filter=U").out || "conflicting files";
    ejected.push({ ...p, files });
    trySh(`gh pr edit ${p.number} --repo ${REPO} --remove-label ${LABEL}`);
    comment(
      p.number,
      `${MARKER}\n🤖 Removed from the batch — this PR conflicts with the rest of the ` +
        `current group and could not be merged onto \`${ROLLUP}\` (${files}). ` +
        `Rebase onto \`${BASE}\` (or resolve against the other batched PRs) and re-add with \`/batch\`.`,
    );
  }

  // Force-update the ephemeral, bot-owned rollup branch via the refs API.
  // NOTE: this is a non-fast-forward update of a throwaway branch nobody else
  // builds on — it is NOT a force-push of a real/shared branch. If policy
  // forbids even this, switch to a per-run branch name (see README).
  const sha = sh("git rev-parse HEAD");
  const ref = `refs/heads/${ROLLUP}`;
  const exists = trySh(`gh api repos/${REPO}/git/${ref}`).ok;
  if (exists) {
    gh(`api -X PATCH repos/${REPO}/git/${ref} -f sha=${sha} -F force=true`);
  } else {
    gh(`api -X POST repos/${REPO}/git/refs -f ref=${ref} -f sha=${sha}`);
  }
  return { merged, ejected };
}

// ---- rollup PR (sticky status + deploy + enqueue target) ------------------

function findRollupPR() {
  const owner = REPO.split("/")[0];
  const rows = ghJSON(
    `pr list --repo ${REPO} --head ${owner}:${ROLLUP} --base ${BASE} --state open --json number,url`,
  );
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
    "Preview: deployed from this branch by the preview-environment workflow.",
  );
  return lines.join("\n");
}

function upsertRollupPR(merged, ejected) {
  const body = rollupBody(merged, ejected);
  const f = "/tmp/rollup-body.md";
  writeFileSync(f, body);
  let pr = findRollupPR();
  if (!pr) {
    if (merged.length === 0) return null; // nothing to open a PR for
    const url = gh(
      `pr create --repo ${REPO} --head ${ROLLUP} --base ${BASE} --draft ` +
        `--title "Batch rollup: ${merged.length} PR(s)" --body-file ${f}`,
    );
    const number = Number(url.split("/").pop());
    return { number, url };
  }
  gh(`pr edit ${pr.number} --repo ${REPO} --title "Batch rollup: ${merged.length} PR(s)" --body-file ${f}`);
  if (merged.length === 0) {
    gh(`pr close ${pr.number} --repo ${REPO} --delete-branch`);
    return null;
  }
  return pr;
}

// ---- commands -------------------------------------------------------------

function assertBatchable(pr) {
  // `gh pr view --json` returns a single object.
  const p = ghJSON(`pr view ${pr} --repo ${REPO} --json number,isDraft,state,labels,mergeable`);
  if (p.state !== "OPEN") throw new Error(`PR #${pr} is not open`);
  if (p.labels.some((l) => l.name === NEVER))
    throw new Error(`PR #${pr} is labeled ${NEVER} (opted out of batching)`);
  return p;
}

function rebuildAndReport(actionNote) {
  const list = members();
  const { merged, ejected } = buildRollup(list);
  const pr = upsertRollupPR(merged, ejected);
  const where = pr ? ` → rollup ${pr.url}` : " (batch now empty)";
  if (prNumber) {
    const names = merged.map((p) => `#${p.number}`).join(", ") || "none";
    comment(
      prNumber,
      `${MARKER}\n🤖 ${actionNote} Current batch (${merged.length}): ${names}.` +
        (ejected.length ? ` Ejected: ${ejected.map((p) => "#" + p.number).join(", ")}.` : "") +
        (pr ? `\n\nOne preview will deploy from the rollup branch; \`/batch-merge\` lands them together.` : ""),
    );
  }
  console.log(`batch: ${merged.length} member(s)${where}`);
}

function cmdBatch() {
  if (arg1.toLowerCase() === "help") return cmdHelp();
  if (!prNumber) throw new Error("no PR number");
  assertBatchable(prNumber);
  gh(`pr edit ${prNumber} --repo ${REPO} --add-label ${LABEL}`);
  rebuildAndReport(`Added #${prNumber} to the batch.`);
}

function cmdBatchRemove() {
  if (!prNumber) throw new Error("no PR number");
  gh(`pr edit ${prNumber} --repo ${REPO} --remove-label ${LABEL}`);
  rebuildAndReport(`Removed #${prNumber} from the batch.`);
}

function cmdBatchMerge() {
  const list = members();
  if (list.length === 0) {
    if (prNumber) comment(prNumber, `${MARKER}\n🤖 The batch is empty — nothing to merge.`);
    return;
  }
  // Every member must be product-approved and green before the group can land.
  const blockers = [];
  for (const p of list) {
    if (p.reviewDecision && p.reviewDecision !== "APPROVED")
      blockers.push(`#${p.number} not approved (${p.reviewDecision})`);
    const checks = trySh(`gh pr checks ${p.number} --repo ${REPO} --json bucket`);
    if (checks.ok) {
      const bad = JSON.parse(checks.out).filter((c) => c.bucket === "fail" || c.bucket === "cancel");
      if (bad.length) blockers.push(`#${p.number} has failing checks`);
    }
  }
  const { merged, ejected } = buildRollup(list);
  const pr = upsertRollupPR(merged, ejected);
  if (!pr) throw new Error("rollup PR missing after build");
  if (blockers.length) {
    comment(
      prNumber || pr.number,
      `${MARKER}\n🤖 Not merging — resolve first:\n- ${blockers.join("\n- ")}`,
    );
    return;
  }
  // Enqueue the rollup into the merge queue. `--merge` (not squash) preserves the
  // members' commit SHAs so, once the queue lands the merge commit, each member PR
  // flips to "Merged" by ancestry. If your queue squashes, see README for the
  // post-land reconciliation fallback.
  gh(`pr ready ${pr.number} --repo ${REPO}`);
  const enq = trySh(`gh pr merge ${pr.number} --repo ${REPO} --merge --auto`);
  const note = enq.ok
    ? `Enqueued the rollup (${merged.length} PR(s)) into the merge queue; members land together as a unit.`
    : `Tried to enqueue the rollup but the merge queue rejected it: ${enq.out}. Check branch protection / queue config.`;
  comment(prNumber || pr.number, `${MARKER}\n🤖 ${note}`);
  for (const p of merged) comment(p.number, `${MARKER}\n🤖 Queued to land via batch rollup ${pr.url}.`);
}

function cmdHelp() {
  const target = prNumber || (findRollupPR() || {}).number;
  if (!target) return;
  comment(
    target,
    `${MARKER}\n🤖 **Batch commands** (comment on any PR):\n` +
      "- `/batch` — add this PR to the current batch (rebuilds the shared preview branch)\n" +
      "- `/batch-remove` — remove this PR from the batch\n" +
      "- `/batch-merge` — land every PR in the batch together (enqueues the rollup)\n" +
      "- `/batch help` — show this message\n\n" +
      `Requires write access. Opt a PR out with the \`${NEVER}\` label.`,
  );
}

// ---- dispatch -------------------------------------------------------------

try {
  if (command === "batch") cmdBatch();
  else if (command === "batch-remove") cmdBatchRemove();
  else if (command === "batch-merge") cmdBatchMerge();
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
