---
name: "product-roadmap-and-prioritization"
description: "Use when the user hands over a backlog, a pile of requests, or competing priorities, or asks what to build next: every item scored on reach, impact, confidence and effort, a now-next-later roadmap with owners, a cut list with reasons, and the review that checks commitments against reality."
triggers: ["what should we build next", "order the backlog", "score the backlog", "RICE score", "now next later", "roadmap review", "competing priorities", "what drops this cycle", "shipped slipped stuck"]
version: "1"
---

# Product roadmap and prioritization

Use this when the user hands over a backlog, a list of requests, or competing
priorities, or asks what to build next. Also use it to run the periodic review
of roadmap commitments against what actually happened.

## What you need first

The candidate items, the goal they serve, any dates already promised, and the
team's real capacity. No goal means you pick the sharpest one you hear and
mark it INFERENCE. No effort sizes means you score with rough person-month
bands and flag sizing as the first ask of the team.

## Score, cut, order

1. Write each candidate as one line: the user, the job, the outcome. Anything
   you cannot phrase that way goes back to the user as a question, not onto
   the roadmap.
2. Score every item on reach, impact, confidence, and effort, the RICE model:
   score = (reach x impact x confidence) / effort. Reach is customers per
   cycle from real metrics. Impact is 0.25 / 0.5 / 1 / 2 / 3 on how much it
   moves the goal. Confidence is 100% / 80% / 50% on the estimates behind it.
   Effort is person-months across product, design, and engineering in whole
   numbers, minimum 0.5. Show the score and a one-line reason beside each, so
   the ordering can be argued with. Evidence grades stay in the reason line,
   not inside confidence.
3. Cut before you order. Kill duplicates, fold near-duplicates, and park
   anything that serves no goal. The cut list carries reasons, and nothing cut
   returns without new evidence.
4. Lay the survivors into now, next, and later. Now holds only what fits this
   cycle, with named owners. Later is not a graveyard: each item keeps the
   trigger that would promote it.
5. Check capacity against the team they named. If now exceeds it, say what
   drops and ask. Never silently overload the cycle.

## Review the commitments

When the ask is a review rather than a fresh ordering: read the roadmap, the
scored backlog, and the last review, and log what you could not reach. Open
with one line — the period, and how many now-items shipped, slipped, or went
quiet. Then three short blocks: shipped; slipped, with the reason and the new
date; stuck, with the owner and the unblock ask. Score each commitment red,
yellow, or green — a slip with no new date is red. Show the same fixed KPI
table every time; never drop a metric because it looks bad. End with the
forward half: asks, each a decision needed from a named person by a date, and
three to five outcome-led commitments for the next period, plus the one call
the period needs, written so the decision maker can answer yes or no. Keep a
log of reviewed items so the same stuck thing is never re-flagged without
saying what changed since.

## Output

The scored list, the now-next-later roadmap with owners, and the cut list with
reasons — saved dated. Report the top three and the first cut in the chat; the
full sheet lives where the team reads it. File now-items as tickets only on a
yes. A period where everything is on track is three lines saying so, not a
report.

## When the sizing is guesswork

Scores are not a hard rule: a dependency or a table-stakes bet may jump the
order with a dated reason naming the trade-off. Never promise a date or a
scope the decision maker did not approve, and never reorder silently after a
review — every change gets a dated note.
