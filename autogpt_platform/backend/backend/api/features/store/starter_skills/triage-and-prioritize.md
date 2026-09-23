---
name: "triage-and-prioritize"
description: "Use when new tickets arrive, the queue needs ordering, or nobody knows what burns first: priority, routing, and escalation judgment with reasons."
triggers: ["triage the queue", "prioritize tickets", "what burns first", "route this ticket", "P1 or P2", "SLA risk", "duplicate tickets"]
version: "1"
---

# Triage and prioritize

Use this when new tickets arrive, the queue needs ordering, or nobody
knows what burns first. Priority, routing, and escalation judgment with
reasons.

## What you need first

The ticket text plus channel, the customer's history and sentiment, order
or account records, and the open queue for dup checks.

## Read, rank, route, save

1. Read the ticket for impact and feeling: who is hurt, how many, is money,
   data, or safety involved, and how angry or at-risk the customer sounds.
2. Assign the priority with the reason in one line. P1 is an outage, data
   loss, a security or fraud event, imminent safety harm, or a VIP down; P2
   is a broken core flow with a painful workaround; P3 is a single-customer
   defect or a how-to with a path; P4 is a question, request, or feedback
   with nothing broken. Rank against the open queue by impact, affected
   count, SLA clock, and money, security, or compliance weight — the clock
   starts at first customer contact and carries across handoffs, so a
   breached or near-breach case outranks new arrivals. Check the open queue
   for duplicates and link them, and log a category so trends surface
   later.
3. Route each ticket: simple FAQs go through the macro fast lane (canned
   reply plus instant ack with expected reply time), answer now,
   troubleshoot first, send to billing for money asks, escalate to a human
   with a repro doc when sentiment is hot, legal or safety words appear, or
   the ask exceeds the approval limit — or park as feedback with a thank-you
   draft.
4. Show the ordered queue with priority, reason, route, and owner, take one
   round of edits, then save it as the day's work list.

## Output

The ranked queue above plus escalation docs for anything routed up. Offer to
draft the first replies when Gmail or Slack is connected, as drafts they
approve first.

## When the records are thin

Thin records mean a shorter confident list, not padded priorities. Say which
record would settle each UNKNOWN.
