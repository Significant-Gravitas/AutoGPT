---
name: "orders-returns-and-warranty"
description: "Use when a customer asks where their stuff is or wants it back: track the order, run the return or warranty by the book, and draft the update."
triggers: ["where is my order", "track this order", "start a return", "warranty claim", "send a replacement", "issue a return label", "order status for a customer"]
version: "1"
---

# Orders, returns and warranty

Use this when a customer asks where their stuff is or wants it back. Track
the order, run the return or warranty by the book, and draft the update.

## What you need first

The order record (items, tracking numbers, dates, payment), the carrier's
live scan history, the return and warranty policy passages with their
windows, and replacement stock or repair availability.

## Track against the carrier, not the promise

Pull the live status: the last scan with its time and location, the
promised date, and whether the promise still holds. Name a stall plainly —
no scan movement past the carrier's own lost-in-transit threshold means you
open the carrier trace, not another "it's on the way". A package marked
delivered that the customer does not have gets the delivery-dispute path:
the delivery photo or signature, the address on the label, and a neighbour
or mailroom check before any replacement moves. Never invent a scan event
or a date the carrier has not given.

## Qualify the ask against the passage

Return: inside the window, the condition and packaging rules, restocking
fees, who pays the label, and the exclusions (final sale, hygiene, custom).
Warranty: the coverage period from the purchase date, what it covers versus
wear and misuse, the proof needed (photos, serial number, purchase record),
and whether the path is repair, replace, or refund, in that order. Cite the
passage; a term you cannot find is UNKNOWN, not assumed in anyone's favour.

## Resolve by the book

Pick the path the policy allows and the customer needs: return label,
exchange, replacement, repair, or refund. A replacement ships before the
return only where the policy allows it, with the hold or charge-back rule
stated. Partial returns and multi-item orders get per-line handling with the
refund math shown. Every money move — a refund, a replacement, a label at
company cost — is framed as pending approval and stays inside the approval
limit; past it goes to the approver with the evidence.

## Draft the update

Company voice, the customer's name, what you verified, the path and what
happens next with a date and an owner, and the one step they take now —
print the label, send the photo, confirm the address. Stage it against the
order for the owner's yes.

## Output

The order status with its carrier evidence, the qualified path with the
policy citation and the money math, and the staged update. No refunds,
replacements, or labels without the owner's yes.

## When the records are thin

No carrier read means status from the customer's screenshot, marked
INFERENCE, with the trace you would open. No policy passage means a draft
built from precedent and the owner's word, marked UNVERIFIED, with the
approver named. Never guess a ship date or a coverage term.
