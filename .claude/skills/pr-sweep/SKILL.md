---
name: pr-sweep
description: Mass PR triage for the open queue — review, cluster duplicates, consolidate overlapping PRs onto a winner, auto-close unambiguous junk, and recommend the rest. TRIGGER when user asks to sweep/triage/clean up the PR queue, mass-review PRs, find duplicate PRs, or combine overlapping PRs.
user-invocable: true
args: "[scope] — optional: a label, author, path, or 'all' (default: all open PRs against dev)."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR Sweep (Maintainer Queue Triage)

Queue-level triage: classify every open PR, cluster duplicates/overlaps, consolidate complementary work onto a winner, close unambiguous junk with a comment, and produce a recommendation report for everything that needs human judgment.

This skill is maintainer-facing. It acts on other people's work — every mutation gets a comment explaining the reason, and judgment calls are **recommended, not executed**.

## Authority model

| Action | Authority |
|---|---|
| Comment (triage status, duplicate links, consolidation notes) | Always allowed |
| Close: spam/abuse, blank/untouched template with no concrete problem, **superseded with proof** (the same change already on `dev` — cite the merged PR/commit AND verify the code is actually there) | Auto, with comment |
| Close: docs-only churn, refactor-only drive-bys, test-only without linked bug, stale/abandoned, dirty branches mixing unrelated changes | **Recommend-only** — report with a ready-to-run command |
| Consolidate a duplicate cluster onto a winner | Allowed after the cluster verdict survives an adversarial check (see below) |
| Merge a PR | Never. Out of scope. |

**Hard gate:** if a sweep would mutate (close/push/consolidate) more than **5** PRs, stop and present the exact list with counts and evidence for confirmation before executing any of them. **Never close for red CI alone** — CI status is context, not a category. Never close PRs that are assigned, actively discussed (author response within 14 days), or maintainer-authored.

## Step 1 — Snapshot and classify the queue

For queues over ~20 PRs, launch the `sweep-prs` workflow (`.claude/workflows/sweep-prs.js`) — it fans out read-only classification, clusters duplicates at a barrier, adversarially verifies every actionable verdict, and returns the action plan. The workflow never mutates; this skill executes the plan afterward.

For small scopes, classify inline. Per PR gather: title/body vs the template (untouched template + no problem statement = blank), diff stats and files, author identity and recent activity (`gh api users/<login>` — a signal for caution, never proof), linked issues, CI buckets, last author activity, and overlap comments already posted by `pr-overlap-check.yml`.

Categories: `auto-close` (the three above), `recommend-close`, `duplicate-cluster`, `needs-deep-review` (plausible focused fix worth a real review), `keep` (active/green/assigned).

## Step 2 — Verify before acting

Every `auto-close` and every cluster verdict gets an adversarial second look whose job is to **refute** it (default to keep when uncertain):

- Superseded: does the cited merged PR/commit actually ship the same behavior? `git log --oneline origin/dev -- <paths>` + read the code on `dev`. Similar title is not proof.
- Blank template: is there a concrete problem statement anywhere — body, commits, linked issue? A non-English but substantive description is not blank.
- Duplicate cluster: do the PRs actually solve the same problem at the same surface, or just touch the same files?

For `needs-deep-review` candidates, run `/pr-review` per PR (it posts inline findings with the evidence-map discipline). For bug-fix PRs include provenance: `git blame` the implicated lines, report "introduced by @author in #N (confidence: clear/likely/unknown)" — separate the code author from the PR merger; don't guess.

## Step 3 — Consolidate duplicate clusters

When a cluster survives verification, combine onto a **winner**, carrying the implications of all members:

1. **Pick the winner on evidence:** most complete implementation, best tests, green CI, responsive author — state the reasons. Prefer the earliest PR when otherwise equal.
2. **Check writability:** `gh pr view <N> --json maintainerCanModify,headRepositoryOwner,headRefName`. Maintainer-editable fork branches and in-repo branches can be pushed to; otherwise fall back to a recommendation comment on the winner describing exactly what to port.
3. **Port the complement:** `gh pr checkout <winner>`, then bring over what the losers have that the winner lacks (cherry-pick when clean, otherwise re-apply by hand). Every ported commit carries `Co-authored-by:` trailers for the source PR's author(s). Never rebase or force-push the winner branch.
4. **Run the relevant verification** for the touched surface (serialized — one pytest at a time) before pushing.
5. **Comment on the winner:** what was ported, from which PRs, crediting authors.
6. **Close the losers** with a comment linking the winner, naming what of theirs was carried over, and inviting them to review it there. A closed-as-duplicate author should be able to see their work credited in the winner.

## Step 4 — Comment templates

Always heredoc bodies (`gh pr comment <N> --body-file` with a quoted heredoc) — never `-b` with backticks. Every close comment states: the category, the specific evidence (link the merged PR/commit for superseded), and "if this is wrong, comment here or reopen — this was an automated triage pass."

## Step 5 — Report

End with the queue ledger:

1. Actions executed: closes (with category + evidence), consolidations (winner, ported-from list, push result).
2. Recommendations: each with its ready-to-run command and one-line evidence.
3. Clusters found, winners chosen, and why.
4. Deep-review results (delegated `/pr-review` outcomes).
5. Kept: count by reason.
6. Anything that hit the >5 gate and is awaiting confirmation.
