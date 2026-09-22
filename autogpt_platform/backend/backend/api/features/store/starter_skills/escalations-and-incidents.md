---
name: "escalations-and-incidents"
description: "Use when a case is hot, VIP, legal, safety, fraud, or breaching: rate the severity, build the packet, and page the right path — never a guess upward."
triggers: ["escalate this ticket", "SEV1 incident", "severity rating", "build an incident packet", "page on-call", "VIP escalation", "escalation handoff packet"]
version: "1"
---

# Escalations and incidents

Use this when a case is hot, VIP, legal, safety, fraud, or breaching. Rate
the severity, build the packet, and page the right path — never a guess
upward.

## What you need

The ticket with impact and sentiment, the escalation matrix memory (severity
levels, exec/VIP path, on-call), the SLA clock, and the customer history.

## Run the escalation

1. Rate it SEV1-SEV4 with the reason in one line: SEV1 is customer-wide
   outage or data loss; SEV2 is a VIP down or major flow broken; SEV3 is a
   single-customer defect; SEV4 is a question with heat but no breakage.
   Fraud, abuse, legal threats, and safety words jump the queue to the hold
   path: freeze the state, preserve the evidence, and hand to the named
   human — never promise an outcome.
2. Build the packet: severity with reason, who is hurt and how many, the
   minimal repro (steps, environment, expected vs actual), what was tried
   with dates, the draft customer update, and the named owner it hands to.
   Track MTTA (time to acknowledge) alongside resolve time. The handoff
   carries the full thread plus internal notes with a warm line — "I've read
   the full thread so you don't repeat anything" — and VIP and exec cases
   add the update cadence from the matrix.
3. Page the path: SEV1 and SEV2 go now to on-call or the exec path with the
   packet; SEV3 goes to the queue owner with a date; SEV4 stays in the queue
   with a watch flag. Every handoff is a draft the owner approves before
   anything sends or pages.
4. Show the packet in chat, take one round of edits, then stage it against
   the ticket with the severity, owner, and date.

## Output

The escalation packet plus the staged handoff. Offer to file it as a tracked
issue in Linear and post it to Slack when connected, as drafts they approve
first.

## Fallbacks

No matrix saved means a best-effort rating marked UNVERIFIED with the owner
named explicitly, plus the one question that would settle the level.
