---
name: "dependency-security-getting-started"
description: "Set up dependency and security hygiene from repository evidence, with a clear scope, source list, risk queue, and approval line."
triggers: ["dependency security", "security hygiene", "dependency review", "start dependency audit", "software supply chain"]
version: "1"
---

# Dependency security getting started

Use this at the start of dependency or vulnerability work.

## Set the scope

Record the repository, branch or commit, supported runtimes, deployed services,
package managers, and environments in scope. Ask which systems face the public,
handle sensitive data, or have strict uptime needs. List anything you cannot
inspect.

## Build the evidence set

Collect manifests, lockfiles, container definitions, software bills of
materials, scanner exports, and the commands used to produce them. Date every
external advisory and release note. Keep requested versions separate from
installed versions.

## Return the first brief

Give the user:

1. Scope and evidence checked.
2. Gaps that could change the result.
3. Confirmed risks, each with package, installed version, source, and exposure.
4. The next three checks or upgrades, in order.
5. Actions that need owner approval.

## Approval line

You may inspect, rank, plan, and draft a patch or pull request. Do not merge,
deploy, suppress a finding, change production, or claim a fix without verified
advisory, stack, version, and test evidence.
