---
name: pr-review
description: Address all open PR review comments systematically. Fetches comments, addresses each one, reacts +1/-1, and replies when clarification is needed. Keeps iterating until all comments are addressed and CI is green.
user-invokable: true
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
6. **Format**: `poetry run format` (backend), `pnpm format` (frontend)
7. **Commit & push**
8. **Wait for CI**: `gh pr checks {N} --repo Significant-Gravitas/AutoGPT` — poll until all green. If any fail, fix and go back to step 6.
9. **Re-fetch all comments** — new ones may appear after each push. If any are unreacted, go back to step 4.
10. **Done** only when: all comments reacted AND CI is green.

## CRITICAL: Do Not Stop

**Loop is: address comments → format → commit → push → wait CI → re-check comments → repeat.**

Never stop early. New comments can appear after every push. CI must be fully green before declaring done.

## Rules

- One todo per comment
- Reply on existing threads, never new top-level comments
- React to every comment: +1 addressed, -1 disagreed (with explanation)
