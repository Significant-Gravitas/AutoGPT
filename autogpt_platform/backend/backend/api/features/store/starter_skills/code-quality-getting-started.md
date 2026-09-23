---
name: "code-quality-getting-started"
description: "Set up review and QA work from the repository, test setup, CI, release process, and merge rules."
triggers: ["QA setup", "code review setup", "start testing", "quality process", "engineering onboarding"]
version: "1"
---

# Code quality getting started

Use this before reviewing a pull request or planning tests.

## Map the codebase and checks

Ask for the repository, main branches, languages, how to run the tests
locally, the CI jobs and what each one checks, known flaky tests, the release
process, where logs and incidents are recorded, and the issue tracker. Note
which of these can be read directly and which the user must paste.

## Set the working rules

Agree who approves and merges a pull request, who can deploy, which checks
must pass before merge, and how review comments are shared. Agree how findings
are ranked, what counts as a blocker, and what evidence is needed to call a
bug fixed or a test passing.

## Return the setup brief

Give the user:

1. Repositories and services in scope.
2. Test and CI map, with gaps and known flaky tests.
3. Merge, deploy, and rollback rules, with owners.
4. Access still needed, such as logs or the tracker.
5. The first three reviews or test plans, with reasons.

Never guess how a check works or claim access you do not have. Do not merge,
deploy, push, skip a check, or disable a test; people with the rights to do so
decide.
