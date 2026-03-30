---
name: open-pr
description: Open a pull request with proper PR template, test coverage, and review workflow. Guides agents through creating a PR that follows repo conventions, ensures existing behaviors aren't broken, covers new behaviors with tests, and handles review via bot when local testing isn't possible. TRIGGER when user asks to "open a PR", "create a PR", "make a PR", "submit a PR", "open pull request", "push and create PR", or any variation of opening/submitting a pull request.
user-invocable: true
args: "[base-branch] — optional target branch (defaults to dev)."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Open a Pull Request

## Step 1: Pre-flight checks

Before opening the PR:

1. Ensure all changes are committed
2. Ensure the branch is pushed to the remote (`git push -u origin <branch>`)
3. Run linters/formatters across the whole repo (not just changed files) and commit any fixes

## Step 2: Test coverage

**This is critical.** Before opening the PR, verify:

### Existing behavior is not broken
- Identify which modules/components your changes touch
- Run the existing test suites for those areas
- If tests fail, fix them before opening the PR — do not open a PR with known regressions

### New behavior has test coverage
- Every new feature, endpoint, or behavior change needs tests
- If you added a new block, add tests for that block
- If you changed API behavior, add or update API tests
- If you changed frontend behavior, verify it doesn't break existing flows

If you cannot run the full test suite locally, note which tests you ran and which you couldn't in the test plan.

## Step 3: Create the PR using the repo template

Read the canonical PR template at `.github/PULL_REQUEST_TEMPLATE.md` and use it **verbatim** as your PR body:

1. Read the template: `cat .github/PULL_REQUEST_TEMPLATE.md`
2. Preserve the exact section titles and formatting, including:
   - `### Why / What / How`
   - `### Changes 🏗️`
   - `### Checklist 📋`
3. Replace HTML comment prompts (`<!-- ... -->`) with actual content; do not leave them in
4. **Do not pre-check boxes** — leave all checkboxes as `- [ ]` until each step is actually completed
5. Do not alter the template structure, rename sections, or remove any checklist items

**PR title must use conventional commit format** (e.g., `feat(backend): add new block`, `fix(frontend): resolve routing bug`, `dx(skills): update PR workflow`). See CLAUDE.md for the full list of scopes.

Use `gh pr create` with the base branch (defaults to `dev` if no `[base-branch]` was provided). Use `--body-file` to avoid shell interpretation of backticks and special characters:

```bash
BASE_BRANCH="${BASE_BRANCH:-dev}"
PR_BODY=$(mktemp)
cat > "$PR_BODY" << 'PREOF'
<filled-in template from .github/PULL_REQUEST_TEMPLATE.md>
PREOF
gh pr create --base "$BASE_BRANCH" --title "<type>(scope): short description" --body-file "$PR_BODY"
rm "$PR_BODY"
```

## Step 4: Review workflow

### If you have a workspace that allows testing (docker, running backend, etc.)
- Run `/pr-test` to do E2E manual testing of the PR using docker compose, agent-browser, and API calls. This is the most thorough way to validate your changes before review.
- After testing, run `/pr-review` to self-review the PR for correctness, security, code quality, and testing gaps before requesting human review.

### If you do NOT have a workspace that allows testing
This is common for agents running in worktrees without a full stack. In this case:

1. Run `/pr-review` locally to catch obvious issues before pushing
2. **Comment `/review` on the PR** after creating it to trigger the review bot
3. **Poll for the review** rather than blindly waiting — check for new review comments every 30 seconds using `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews --paginate` and the GraphQL inline threads query. The bot typically responds within 30 minutes, but polling lets the agent react as soon as it arrives.
4. Do NOT proceed or merge until the bot review comes back
5. Address any issues the bot raises — use `/pr-address` which has a full polling loop with CI + comment tracking

```bash
# After creating the PR:
PR_NUMBER=$(gh pr view --json number -q .number)
gh pr comment "$PR_NUMBER" --body "/review"
# Then use /pr-address to poll for and address the review when it arrives
```

## Step 5: Address review feedback

Once the review bot or human reviewers leave comments:
- Run `/pr-address` to address review comments. It will loop until CI is green and all comments are resolved.
- Do not merge without human approval.

## Related skills

| Skill | When to use |
|---|---|
| `/pr-test` | E2E testing with docker compose, agent-browser, API calls — use when you have a running workspace |
| `/pr-review` | Review for correctness, security, code quality — use before requesting human review |
| `/pr-address` | Address reviewer comments and loop until CI green — use after reviews come in |

## Step 6: Post-creation

After the PR is created and review is triggered:
- Share the PR URL with the user
- If waiting on the review bot, let the user know the expected wait time (~30 min)
- Do not merge without human approval
