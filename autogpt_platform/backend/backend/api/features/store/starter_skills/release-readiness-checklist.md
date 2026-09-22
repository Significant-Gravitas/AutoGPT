---
name: "release-readiness-checklist"
description: "Check a release for what changed, what was tested, what was not, and how to roll back."
triggers: ["release checklist", "ready to ship", "go/no-go", "deploy checklist", "pre-release check"]
version: "1"
---

# Release readiness checklist

Use this before a release, deploy, or go/no-go meeting.

## What is in the release

List every change in the release: pull request, owner, and a one-line
summary. Flag database migrations, config changes, new dependencies, feature
flags, and anything that changes a public API or stored data.

## Evidence of testing

For each change record what was tested, where, and the link to the run or
note. Mark each as **tested**, **partly tested**, or **not tested**. List open
bugs that ship with the release and who accepted them. Only count a check as
passed if you saw the result.

## Rollback and sign-off

- How to roll back each risky change, and how long it takes.
- Whether a migration can be reversed, and what happens to new data.
- What to watch after the release: metrics, logs, and alerts.
- Who is on call.
- Who signs off, and the go/no-go call with its reasons.

Return the checklist with every gap listed first.

Never mark a check passed or a release ready without evidence. Do not deploy,
tag, merge, or skip a check; the release owner makes the call.
