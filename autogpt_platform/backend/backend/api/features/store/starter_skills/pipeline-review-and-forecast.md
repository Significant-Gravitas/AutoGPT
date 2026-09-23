---
name: "pipeline-review-and-forecast"
description: "Use when the user wants the honest read on their book: coverage, forecast, stuck deals, and hygiene."
triggers: ["pipeline review", "forecast", "coverage", "stuck deals", "commit", "quota pacing", "CRM hygiene"]
version: "1"
---

# Pipeline review and forecast

Run this when the user wants the honest read on their book. Report
from the numbers source only — anything untraceable is UNKNOWN,
never estimated.

## Inputs

The CRM view or export they trust, their quota, their commit style,
and the deal sheets with qualification state.

## Report the book

Counts by stage, movement since last review, coverage as open pipe
over quota — flag under 3x, alarm under 2x — the oldest stuck five
with stall ages, and pacing vs goal. Split the read: the pipeline
review moves deals (what is stuck, what advanced, owner plus date);
the forecast review defends the number (what closes, amount, date,
commit/upside/best-case confidence). Monthly, add segment conversion
trends; quarterly, re-check the coverage and stage assumptions; a
third close-date move or a top-five deal leaving commit triggers an
off-cycle review.

## Grade the forecast

Commit, best case, pipeline, or omitted per deal, with the
qualification evidence behind each call. Flag commit deals with open
must-have letters and best-case deals with no dated next step.

## Name the hygiene breaks

Stale next steps, deals with no activity in 14 days, single-threaded
enterprise deals, and stage-to-qualification mismatches. Each flag
carries the one fix and its owner.

## Stage next steps as paste-ready text

Draft the CRM next-step updates as text the user can paste. Apply
nothing yourself.

## Output

The pipeline report with forecast grades and hygiene flags, plus the
staged next-step drafts.

## Fallbacks

No CRM access: review from their export or pasted numbers and say
which checks you could not run. No quota on file: pacing is UNKNOWN
until they give it.

## Approval gate

Reads and drafts only. CRM writes and commit calls wait for your yes.
