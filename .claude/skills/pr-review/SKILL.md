---
name: pr-review
description: Address all open PR review comments systematically. Fetches comments, addresses each one, reacts +1/-1, and replies when clarification is needed. Keeps iterating until all comments are addressed and CI is green. TRIGGER when user shares a PR URL, asks to address review comments, fix PR feedback, or respond to reviewer comments.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR Review Comment Workflow

## Steps

1. **Find PR**: `gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT`
2. **Fetch comments** (all three sources):
   - `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews` (top-level reviews)
   - `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments` (inline review comments)
   - `gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments` (PR conversation comments)
3. **Skip** comments already reacted to by PR author
4. **For each unreacted comment**:
   - Read referenced code, make the fix (or reply if you disagree/need info)
   - **Inline review comments** (`pulls/{N}/comments`):
     - React: `gh api repos/.../pulls/comments/{ID}/reactions -f content="+1"` (or `-1`)
     - Reply: `gh api repos/.../pulls/{N}/comments/{ID}/replies -f body="..."`
   - **PR conversation comments** (`issues/{N}/comments`):
     - React: `gh api repos/.../issues/comments/{ID}/reactions -f content="+1"` (or `-1`)
     - No threaded replies — post a new issue comment if needed
   - **Top-level reviews**: no reaction API — address in code, reply via issue comment if needed
5. **Include autogpt-reviewer bot fixes** too
6. **Format**: `cd autogpt_platform/backend && poetry run format`, `cd autogpt_platform/frontend && pnpm format`
7. **Commit & push**
8. **Re-fetch comments** immediately — address any new unreacted ones before waiting on CI
9. **Stay productive while CI runs** — don't idle. In priority order:
   - Run any pending local tests (`poetry run pytest`, e2e, etc.) and fix failures
   - Address any remaining comments
   - Only poll `gh pr checks {N}` as the last resort when there's truly nothing left to do
10. **If CI fails** — fix, go back to step 6
11. **Re-fetch comments again** after CI is green — address anything that appeared while CI was running
12. **Done** only when: all comments reacted AND CI is green.

## CRITICAL: Do Not Stop

**Loop is: address → format → commit → push → re-check comments → run local tests → wait CI → re-check comments → repeat.**

Never idle. If CI is running and you have nothing to address, run local tests. Waiting on CI is the last resort.

## Rules

- One todo per comment
- For inline review comments: reply on existing threads. For PR conversation comments: post a new issue comment (API doesn't support threaded replies)
- React to every comment: +1 addressed, -1 disagreed (with explanation)
