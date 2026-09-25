---
name: "hiring-pipeline-analytics"
description: "Use when the user asks how hiring is going, wants funnel numbers per role, or needs a hiring report for the team."
triggers: ["how is hiring going", "hiring funnel numbers", "time to hire report", "recruiting metrics for the team", "which stage loses candidates", "offer acceptance rate", "hiring pipeline review", "are our reqs aging"]
version: "1"
---

# Hiring pipeline analytics

Run this when the user asks how hiring is going, wants funnel numbers, or
needs a hiring report for the team. This is also the pass behind an on-demand
weekly pipeline review.

## Inputs

The tracker — roles, candidates, loops — the shortlist, and the outreach log.
Name the window up front and compare it against the prior window, never
against vibes.

## Report the funnel

Counts per role and stage: sourced, screened, in loop, offer, closed. Add
conversion between stages and flag the stage losing the most candidates, with
the top two reasons in plain words.

## Report pacing with both clocks

Time to fill, from req posted to offer accepted, and time to hire, from
applied or sourced to accepted. Plus time to slate, days in stage per
candidate, and reqs aging past the stalled bar. Name the oldest five stuck
items with owner and days.

## Report quality and mix

Source mix — sourced, inbound, referral — plus source of hire: the channel
that produced each signed hire. Offer acceptance rate. Decline reasons
grouped. Quality of hire as 90-day attrition with exit-survey notes where you
have them. Under 30 candidates in the window, say the sample is small and read
it as directional, not truth.

## Report hygiene

Rows missing stage or owner, scorecards overdue, loops with no next step,
outreach past three touches with no reply. Every item gets the one action that
fixes it.

## Decompose every move

What changed, by how much, and the driver. Label each line FACT — from the
tracker — INFERENCE, with your math and assumptions shown, or UNKNOWN for
missing data, never estimated silently.

## Close on the calls that need a human

List, a line apiece, the calls that need the user or a hiring manager, plus
next week's interview load by day with any day too heavy for the panel.

## Output

The funnel read in chat, under 300 words unless they asked for the full table,
saved with the date to the hiring folder.

## Fallbacks

With a thin or stale tracker, report what is there, name the gaps, and ask for
the one export that fills them. Never invent a stage change, a number, or a
commitment.

## Approval gate

Post it to a channel or mail it to the team only when they ask for that
specific send. Reports carry role-level numbers; candidate details stay out of
anything shared.
