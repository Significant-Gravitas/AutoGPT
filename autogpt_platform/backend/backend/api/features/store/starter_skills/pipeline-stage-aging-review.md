---
name: "pipeline-stage-aging-review"
description: "Review pipeline stage age against team rules and identify stalled deals, weak next steps, stale records, and owner actions."
triggers: ["pipeline aging", "stage aging", "stalled deals", "pipeline review", "old opportunities"]
version: "1"
---

# Pipeline stage aging review

Use this on a CRM export or deal table.

## Establish the rules

Get the team's stage definitions and age limits. Record the report time, stage
entry date, last customer action, next step, next-step date, owner, amount, and
close date source. Flag repeated close-date moves only from close-date change
history; without it, mark that check **insufficient data**. Do not apply generic limits when the team has its own.

## Classify each deal

Use **within limit**, **due for review**, **stalled**, or **insufficient data**.
Flag missing or past next-step dates, repeated close-date moves, stage changes
with no customer evidence, and records whose last activity is internal only.

## Return the review

Rank by age breach and deal impact. For each flagged deal state the dated facts,
why it is flagged, the question that must be answered, owner, and next action.
Show totals by stage and owner, but keep the deal rows available for audit.

Do not change stages, close dates, forecast classes, or owners. Do not treat
record age alone as proof that a deal is lost.
