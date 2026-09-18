---
name: "dependency-inventory"
description: "Build an evidence-backed inventory of direct, transitive, runtime, development, and container dependencies."
triggers: ["dependency inventory", "list packages", "lockfile review", "SBOM", "what dependencies do we use"]
version: "1"
---

# Dependency inventory

Use this before judging age or security risk.

## Collect

Find every package manifest and lockfile in scope. Include workspace roots,
plugins, examples that ship, container base images, build actions, and language
runtime pins. Prefer installed or locked versions over ranges in manifests.

For each item record:

- ecosystem, package, installed version, and requested range;
- direct or transitive status;
- runtime, development, build, or test use;
- the file and line or tool output that proves it;
- service or image that carries it;
- whether the version could not be resolved.

## Reconcile

Do not merge packages that share a name across ecosystems. Flag duplicate major
versions, unlocked production dependencies, stale generated lockfiles, and
manifests with no matching lockfile. If a scanner and lockfile disagree, show
both and name the check needed to settle it.

## Output

Return a table, an evidence-gap list, and totals by ecosystem and use. Never
infer that a package runs in production only because it appears in a manifest.
Do not call the inventory complete while a deployable service or image remains
unchecked.
