---
name: "funnel-conversion-analysis"
description: "Measure funnel conversion with explicit eligibility, event order, windows, identity rules, drop-off checks, and segment evidence."
triggers: ["funnel analysis", "conversion rate", "drop off", "signup funnel", "checkout funnel"]
version: "1"
---

# Funnel conversion analysis

Use this when the steps form a real ordered path. Do not force unrelated events
into a funnel.

## Define the funnel

For every step record the exact event, eligible population, entity key, required
order, conversion window, timezone, deduplication rule, and whether repeat
attempts count. State whether the funnel is closed to one starting cohort or
open to all events in a period.

## Validate the events

Check missing steps, events out of order, duplicate events, client and server
disagreement, identity changes, late arrival, version changes, and bot or test
traffic rules supplied by the owner. Reconcile starts with an independent source
when available.

## Calculate the view

For each step show entrants, completions, step conversion, overall conversion,
drop count, and time to next step. Keep denominators visible. Show both account
and user grain only if each has a defined business meaning.

Break down the largest drop by approved segments, then report sample size and
whether the segment explains the total change or only has a low rate. A segment
with a sharp fall and little volume may not drive the business result.

## Form hypotheses

Link supplied releases, errors, price changes, or channel shifts by time and
segment. Label them hypotheses unless a valid test supports cause. Give one
instrumentation or product check that could disprove each leading explanation.

## Safety rules

Never change the window, order, or exclusion rule after seeing the result
without saying so. Do not hide tracking gaps, double count repeat events, or
expose individual paths. Do not claim a product change caused conversion to
move from a before-and-after comparison alone.
