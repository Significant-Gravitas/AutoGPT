---
name: "ticket-triage"
description: "Sort open tickets by impact, urgency, and evidence, group duplicates, and name the next owner."
triggers: ["triage tickets", "sort the queue", "urgent tickets", "prioritise support", "duplicate tickets"]
version: "1"
---

# Ticket triage

Use this for a full queue or a batch of new tickets.

## Read each ticket

For each ticket note the customer, plan, channel, first message date, what
they are trying to do, what went wrong, and what they have already tried.
Check the account record, known-issue list, and status page. Mark anything you
could not verify rather than assuming it.

## Rank and group

Rank by impact and urgency, not tone:

- **urgent**: many users blocked, data at risk, payment failing, or security;
- **high**: one account blocked with no workaround;
- **normal**: a workaround exists or the question is answered in the docs;
- **low**: how-to or feature request with no blocker.

Group tickets only when they share the same symptom and at least one other
piece of evidence. Use browser, plan, and start date as supporting context,
and cite the ticket ids in each group. A loud customer is not automatically
urgent, and a polite one is not automatically low.

## Queue brief

Return the counts by rank, the duplicate groups, tickets that the docs already
answer, tickets that need engineering, finance, or another owner, and what is
still unknown. Name the next owner for each.

Do not close, merge, reassign, or reply to a ticket yourself. Never invent a
cause, a fix, or an account detail; say what you could not check.
