---
name: "pricing-and-packaging"
description: "Use when the user names pricing, packaging, tiers, or discounts, or hands over a price list, competitor pricing, or deal terms: the pricing recommendation with guardrails and mix impact."
triggers: ["pricing", "packaging", "tiers", "discounting", "price list", "willingness to pay", "should we raise prices"]
version: "1"
---

# Pricing and packaging

Use this when the user names pricing, packaging, tiers, or discounts, or
hands over a price list, competitor pricing, or deal terms for a read.
Start from the current model and tiers, the competitor price pages, the
deal or discount reality, and the margin floor. When Stripe is connected,
read the actual plan mix and revenue by tier before you touch anything —
that is the FACT layer every recommendation sits on.

## Read the current state first

The value metric, the tiers, what each tier gates, and where discounting
actually lands. Say back the model in three lines before you change it.

## Measure willingness to pay before you price

The Van Westendorp four questions (too expensive, getting expensive, a
good deal, too cheap to trust) across current customers, known prospects,
and strangers for the acceptable range; a forced-choice feature survey for
what each tier gates; Gabor-Granger price points for the demand curve, or
conjoint when you need feature-level precision. Revisit pricing quarterly
— static pricing drifts from value.

## The competitor read

Per-plan tables with fetch dates, billing unit, minimums, limits, and free
tiers, in vendor wording. Never estimate a price you did not read;
call-for-pricing stays UNKNOWN.

## Recommend the model

Value metric, tier shape, and what moves up or down a tier. Each change
gets who wins, who pays more, and the expansion path it opens.

## Set the guardrails

Discount floors, tiered approval bands, and the walk-away line. Defaults
to adapt: 0–15% rep or manager same-day; 16–25% deal desk on a 4-hour
turnaround; 26–35% VP Sales plus Finance on a 24-hour turnaround; above
35% executive sign-off on 48 hours. Every band prescribes its give-gets.
Inside bands gets a draft approval note; outside bands routes to the
owner, never approved by you.

## Model the mix impact

One table across tiers: current vs proposed, revenue, margin in currency
and percent, and grandfathering cost. State the denominator behind every
rate. Put the model in a sheet the team can edit when Sheets is connected.

## What you hand back

The recommendation with model, tiers, guardrails, and mix impact, plus the
rollout note: what changes for existing customers, the grandfathering
rule, and the talk track for the first hard call. Log every
recommendation dated with its inputs. Pricing math ships as a draft — you
never announce a price; the owner does, after their yes.

## Fallbacks

No margin floor means you ask once and hold the verdict. No cost-to-serve
data means revenue-only math labeled INFERENCE. Never invent a
willingness-to-pay number or a competitor price.
