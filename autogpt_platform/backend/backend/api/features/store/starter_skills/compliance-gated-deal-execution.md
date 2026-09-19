---
name: "compliance-gated-deal-execution"
description: "Use when deals must clear compliance gates: licensing, suitability, audit readiness, and regulated-ops checks before close."
triggers: ["compliance gate", "licensing check", "suitability review", "audit-ready deal file", "regulated deal", "gate checklist"]
version: "1"
---

# Compliance gated deal execution

Run this when a deal must clear compliance gates before it can close.
A deal with a failed gate does not advance, and nothing buyer-facing
moves without the owner's explicit yes.

## Inputs

The deal, the gates in play with their owners, the licensing or
suitability rules, the audit and documentation standard, and the
target close date.

## List every gate first

Cover licensing checks, suitability review, privacy and ethics rules,
and any industry code in play. Each gate gets an owner, a checklist,
and a cleared, pending, or failed state.

## Review the opportunity early

Run the compliance review at opportunity stage, not at signature:
suitability of the product for the buyer, licensed sellers on the
deal, and clean documentation of every buyer commitment. Review
misses become rep coaching, tracked as a scored metric.

## Keep the file audit-ready

Every term, concession, and buyer promise gets a date, an owner, and a
source. Audit-grade — the evidence standard a financial-controls audit
applies — means a stranger can re-run the deal from the file; a file
that cannot be re-run is blocked.

## Track regulated-ops steps

Title, registration, or transfer work each carry an owner,
jurisdiction, and date. Multi-state or multi-entity deals get one
master track with dated sub-tracks per filing. For software deals the
same tracking covers security reviews, data processing agreements, and
order-form approvals.

## Report gates in the forecast

Pending gates with dates, failed gates with rescue plans, cleared
gates with evidence. A commit deal with a pending gate is Best Case
until the gate clears.

## Output

Gate checklist with states and owners, opportunity review with
coaching flags, audit-ready deal file, regulated-ops track, and the
gated forecast read.

## Fallbacks

No gate list: draft from the last approved deal's checks and mark
every gate INFERENCE until legal confirms. No licensed owner on the
deal: flag it blocked and ask for assignment before any buyer step.

## Approval gate

Nothing buyer-facing sends and no gated deal advances without your
explicit yes.
