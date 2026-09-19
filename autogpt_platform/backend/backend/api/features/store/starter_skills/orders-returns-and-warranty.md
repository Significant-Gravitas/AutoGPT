---
name: "orders-returns-and-warranty"
description: "Use when a customer asks where their stuff is or wants it back: track the order, run the return or warranty by the book, and draft the update."
triggers: ["where is my order", "track this order", "start a return", "warranty claim", "send a replacement", "issue a return label", "order status for a customer"]
version: "1"
---

# Orders returns and warranty

Use this when a customer asks where their stuff is or wants it back.
Track the order, run the return or warranty by the book, and draft the
update.

## What you need first

The order record (items, tracking, dates), the return or warranty
policy passage, and inventory or replacement availability.

## Track, qualify, resolve, draft

1. Track the order against the carrier record: current location, stall
   point, and promised date.
2. Qualify the return or warranty ask against the policy passage: window,
   condition, and exclusions.
3. Resolve by the book: return label, replacement, repair, or refund
   path, each framed as pending approval where money moves.
4. Draft the update with next step, date, and owner, staged for owner
   yes.

## Output

The order status plus the qualified return or warranty path with
drafts. No refunds, replacements, or labels without owner yes.

## When the records are thin

Thin records mean a narrower promise, not a guessed ship date. Say
which record would settle each UNKNOWN.
