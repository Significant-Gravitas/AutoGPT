---
name: "deal-economics-and-pricing-guardrails"
description: "Use for the money side of a deal: the deal P&L, margin floors and discount bands, approval routing, the concession log, and the mix impact of a price change."
triggers: ["deal p&l", "margin floor", "discount approval math", "price change impact", "cost to serve", "concession log", "rev share bands"]
version: "1"
---

# Deal economics and pricing guardrails

Run this when a deal needs the money checked: what it earns, whether
the ask clears the guardrails, and who has to approve it. You do the
math and the routing. You never approve a discount and never message
the customer.

## Inputs

The deal terms — price, volume, term, discount ask — the guardrail
bands, and the margin floor.

## Build the deal P&L

Revenue, cost to serve, margin in currency and in percent, and payback
where it applies. Put current, proposed, and the ask in one table so
the gap is visible without arithmetic.

## Check the ask against the bands

Discount floors, revenue-share bands, and cap rules. Inside the bands
gets a draft approval note for the owner to send. Outside the bands
gets routed to the named approver with the margin impact attached —
never approved by you, and never rounded into the band.

## Defend the discount with numbers

Counter with cited numbers, trade give-get concessions — term, volume,
payment speed — and name the walk-away line before the call, not during
it. A concession with nothing traded back is a price cut with extra
steps.

## Log every concession

Approver, reason, and margin impact, written to the deal and pricing
log. No silent discounts, and no concession that exists only in a
thread.

## Model a pricing change before it ships

Model the mix impact across tiers, state plainly who wins and who pays
more, and size the grandfathering cost as its own line. A change that
looks flat in aggregate usually is not flat per segment.

## Output

The comparison table, the guardrail check with its routing, the
concession log entry, and the draft note or counter.

## Fallbacks

No guardrail bands saved: ask once for the margin floor and the
discount authority, and hold the verdict until they answer. No
cost-to-serve data: give revenue-only math labeled INFERENCE and flag
the gap that would change the answer.

## Approval gate

Drafts only. Nothing customer-facing goes out, no price, discount, or
term is promised, and no ask outside the bands is approved without the
named owner's yes.
