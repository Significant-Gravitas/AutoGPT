---
name: "dependency-change-risk-review"
description: "Review a dependency change for breaking APIs, runtime shifts, supply-chain risk, test gaps, rollout risk, and rollback quality."
triggers: ["review dependency change", "upgrade risk", "dependency PR review", "breaking change review", "package bump risk"]
version: "1"
---

# Dependency change risk review

Use this on a proposed plan, patch, or pull request.

## Review the evidence

Confirm the old and new resolved versions, source registry, checksums or lock
data, release and migration notes, maintainer status, and reason for change.
Flag renamed or transferred packages, install scripts, new binary downloads,
new permissions, and large transitive-tree changes.

## Review the application impact

Search for changed APIs, defaults, configuration, data formats, network calls,
and runtime requirements. Check production and development paths separately.
Map each risk to a test, manual check, rollout signal, or unresolved gap.

## Return a decision brief

Use **ready for review**, **needs changes**, or **blocked on evidence**. List
findings by impact, with file or source evidence, owner, and required action.
State whether rollback restores the old lock state and whether a data or config
change makes rollback unsafe.

Do not approve, merge, deploy, or waive a check. A clean diff does not prove a
safe runtime change.
