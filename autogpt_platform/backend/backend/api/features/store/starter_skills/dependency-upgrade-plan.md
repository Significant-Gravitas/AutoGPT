---
name: "dependency-upgrade-plan"
description: "Plan a small, testable dependency upgrade with compatibility checks, owners, rollout evidence, and rollback steps."
triggers: ["upgrade plan", "dependency migration", "package upgrade", "update library", "upgrade safely"]
version: "1"
---

# Dependency upgrade plan

Use this once the current and target versions are known.

## Read before planning

Check release notes, migration guides, support policy, known regressions, and
the project's use of changed APIs or configuration. Name every source and the
versions it covers. Search the repository for removed or changed features.

## Write the plan

Include:

1. Reason and evidence for the target version.
2. Files and services likely to change.
3. Required code, config, schema, or build edits.
4. Tests to add or run, tied to the affected behaviour.
5. Build and scanner checks.
6. Rollout owner, observation window, and stop conditions.
7. Exact rollback route.

Split major, framework, runtime, and unrelated security upgrades when they can
be tested alone. Mark assumptions and missing test coverage.

## Approval line

The plan may propose a branch or pull request. It does not authorise a merge,
deploy, data migration, risk acceptance, or production change.
