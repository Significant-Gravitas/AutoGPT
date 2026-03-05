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
7. **Commit & push**, then re-fetch — new comments may appear. Repeat until all reacted.
8. **Verify CI**: `gh pr checks {N} --repo Significant-Gravitas/AutoGPT` — wait until green.

## CRITICAL: Do Not Stop

**Keep looping until ALL of these are true:**
- Every comment has a +1 or -1 reaction
- No new unaddressed comments remain (new ones appear after each push)
- `gh pr checks` shows all CI checks green

If CI fails, investigate and fix. If new comments appear, address them. Do not stop early.

## Rules

- One todo per comment
- Reply on existing threads, never new top-level comments
- React to every comment: +1 addressed, -1 disagreed (with explanation)
