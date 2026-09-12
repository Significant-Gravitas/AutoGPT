---
name: pr-testability-review
description: Judge whether a pull request needs to exist, can be tested, and what it breaks. Use when asked to review a PR, assess whether a change is ready, check whether a diff is covered by tests, or decide if a PR is worth merging.
triggers:
  - review this PR
  - is this PR ready
  - does this need tests
  - what does this break
version: "1"
---

# Reviewing a pull request for testability and necessity

Three questions, in this order. Stop at the first one that fails — a change
that should not exist does not need a test-coverage discussion.

## 1. Does this need to exist?

Find the failing case behind the change. A PR should be traceable to a bug
report, an error in production, a stated requirement, or a measured problem.

- If the description asserts a problem, check it. "This is slow" is a claim;
  find the measurement or say it is unverified.
- Search the codebase for an existing helper that already does this. Duplicated
  logic is the most common avoidable PR.
- If it is a refactor with no behaviour change, say so explicitly — that is
  fine, but it changes what review means (no new tests expected, but the
  existing suite must cover the refactored paths, or the refactor is unverified).

Report: the failing case in one sentence, or "no failing case stated, and I
could not find one" with what you searched.

## 2. Can it be tested?

For each behaviour the diff adds or changes, find the test that would fail if
it were reverted. This is the core of the review.

- Read the test files in the diff. Do the assertions actually pin the new
  behaviour, or do they assert something that was already true?
- The decisive check: **mentally delete the new code and ask which test goes
  red.** If none, the behaviour is undefended, no matter how many tests the PR
  adds. Say exactly that, naming the behaviour.
- Watch for tests that pass inert values. An escaping or sanitising path
  asserted with `"updated"` proves nothing; the test needs the character that
  would break it.
- Check the error paths, not just the happy path. Most regressions live in the
  branch nobody wrote a case for.

Report per behaviour: covered (name the test), or undefended (name the
behaviour and suggest the cheapest case that would cover it — an existing
fixture is better than a new file).

## 3. What does it break?

- **Signature and shape changes.** Grep every call site of a changed function
  or field. A default argument or a new optional field is not automatically
  safe: check whether an existing caller relied on the old shape.
- **Persisted and in-flight data.** If the change alters what is written or
  read, ask what happens to rows, files, cached values, or tokens created
  before the deploy. A change with no backfill and no fallback silently breaks
  everything that already exists — this is the single most commonly missed
  defect in review.
- **Concurrency.** If two callers can reach the changed code at once, say what
  happens. Read-modify-write over an await is the usual shape.
- **Failure blast radius.** If a new dependency (a cache, a queue, a network
  call) fails, does the feature degrade or does the whole path die? Guarded or
  unguarded — state which.

## Writing the review

Anchor every point to `file:line`. Give the concrete failure: inputs, then
wrong outcome. "This could be a problem" is not a finding; "two users linked
to the same server both pass this check, so Alice can edit Bob's message" is.

Rank by consequence, not by how easy the fix is. Separate:

- **Blocker** — data loss, a security boundary, or existing users breaking on
  deploy.
- **Should fix** — a real defect with a bounded blast radius.
- **Nice to have** — clarity, or a latent issue nothing reaches today.

Say what you did not check. A review that implies whole-diff coverage it did
not do is worse than a short honest one.

## Never

Do not approve, merge, request changes, or post a comment unless you were
asked for that exact action. Produce the review; a human sends it.
