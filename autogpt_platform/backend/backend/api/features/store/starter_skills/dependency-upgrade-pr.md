---
name: "dependency-upgrade-pr"
description: "Draft a narrow dependency upgrade pull request with evidence, code changes, tests, risk notes, and rollback instructions."
triggers: ["upgrade PR", "dependency pull request", "draft update PR", "bump package", "security upgrade patch"]
version: "1"
---

# Dependency upgrade PR

Use this after an upgrade plan has a reviewed target.

## Keep the change narrow

Update the manifest and lockfile with the package manager. Make only the code
or configuration edits required by the verified migration notes. Do not mix
formatting, refactors, unrelated package bumps, or silent alert suppressions
into the change.

## Validate

Run the smallest relevant tests first, then the project checks required by its
contribution guide. Inspect the resolved dependency tree and built artifact.
For a security fix, rerun the scanner or version check that found it. Record
commands, results, and anything you could not run.

## Draft the pull request

State the old and new versions, reason, advisory or release-note sources,
behaviour changed, files changed, test evidence, known risk, rollout check, and
rollback. Mark it draft when approval or a check remains.

## Stop point

Never merge, deploy, bypass a failed check, or claim production is fixed. Hand
the draft and evidence to the named reviewer.
