---
name: "outdated-dependency-review"
description: "Review outdated packages by support, change size, exposure, and value instead of treating every newer version as urgent."
triggers: ["outdated dependencies", "package updates", "dependency freshness", "upgrade backlog", "version review"]
version: "1"
---

# Outdated dependency review

Use this to turn an update report into an ordered backlog.

## Verify each gap

For each direct dependency, record the installed version, newest compatible
version, newest stable version, release date, support status, and source checked.
Read the release notes and migration guide for candidate versions. Do not use a
registry's latest tag as proof that an upgrade is safe.

## Rank the work

Place each package in one group:

- **Act now** — unsupported, exposed security fix, broken compatibility, or a
  needed fix with a small change path.
- **Plan** — useful change with migration or test work.
- **Watch** — current line remains supported and the update has little value.
- **Unknown** — installed version, support policy, or impact is not proven.

State the reason, likely change size, affected services, test owner, and next
review date. Keep security findings linked to their advisory rather than
copying severity into this review.

## Guardrails

Do not recommend a major upgrade without reading its migration notes. Do not
bundle unrelated upgrades merely to clear a dashboard. Never change version
ranges or lockfiles until the user approves the plan.
