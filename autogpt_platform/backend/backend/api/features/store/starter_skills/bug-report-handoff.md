---
name: "bug-report-handoff"
description: "Turn customer reports into an engineering bug report with steps, scope, frequency, and evidence."
triggers: ["bug report", "escalate to engineering", "reproduce issue", "customer bug", "file a bug"]
version: "1"
---

# Bug report handoff

Use this when one or more tickets point to something broken in the product.

## Gather the evidence

From each linked ticket collect the account, plan, browser or app version,
device, time first seen, exact error text, screenshots, and what the customer
did just before. Check the status page and known-issue list for a match.

## Write the report

1. Title: what breaks, where, for whom.
2. Steps to reproduce, numbered, from a clean start.
3. Expected result and actual result.
4. Environment: browser, version, device, plan, region.
5. Frequency: how many tickets and accounts, since when.
6. Affected accounts and ticket ids.
7. Impact on the customer: blocked, slowed, or cosmetic.
8. Workaround, only if a teammate has tested it.
9. Attachments and logs, linked rather than pasted.

## What stays unknown

List what you could not reproduce or confirm, the questions still open for
the customer, and who is waiting on an answer. Keep customer reports separate
from what support reproduced.

Do not file the bug or change its priority yourself; draft it for the owner.
Never guess a root cause or a fix date, or paste one customer's data into
another customer's thread.
