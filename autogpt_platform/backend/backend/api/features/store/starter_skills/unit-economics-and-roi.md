---
name: "unit-economics-and-roi"
description: "Use for customer acquisition cost, lifetime value, payback, contribution margin, net present value, internal rate of return, and fund-or-kill calls on a bet."
triggers: ["unit economics", "cac payback", "ltv to cac", "contribution margin", "fund or kill this bet", "npv and irr", "is this vendor worth it"]
version: "1"
---

# Unit economics and ROI

Run this when the question is whether a unit, a cohort, or a bet pays
for itself. Every input is labeled FACT, INFERENCE with its assumption
shown, or UNKNOWN. An UNKNOWN input blocks the verdict, not the draft.

## Inputs

The cost and revenue lines behind the bet, the time horizon, and the
discount rate or hurdle the team uses.

## Build the unit first

Revenue per unit, cost per unit, contribution margin — all from FACT
inputs. If gross margin is unknown, say so before any lifetime-value
math, because every ratio downstream inherits that hole.

## Compute on one cohort definition, formulas shown

Show each formula with the numbers plugged in:

- Fully-loaded customer acquisition cost (CAC) = (sales and marketing
  salaries + marketing spend + pro-rated tooling) / net new logos.
  Expansion deals are excluded from both sides.
- Lifetime value (LTV) = (average revenue per user (ARPU) x gross
  margin) / churn rate.
- LTV-to-CAC = LTV / CAC.
- CAC payback in months = CAC / (ARPU x gross margin).

Label churn as logo churn or revenue churn, never blended. Common
benchmarks, which vary by segment and sales motion: LTV:CAC at or above
3:1, payback under 18 months is healthy and under 12 is elite. Never
present a blended ratio as a cohort truth.

## Run three cases with a break-even line

Base, Adverse, and Opportunity on the two drivers that matter most, and
draw the break-even line in each. Say which case you believe and why,
in one line.

## Investment calls

Net present value (NPV) and internal rate of return (IRR) against the
team's hurdle, plus the payback period and the top three risks with
owners. The verdict is fund, hold, or kill, and it names the single
number that decides it.

## Vendor and cost bets

Score the savings opportunity by mechanism — switch, renegotiate, or
cut waste — the spend it traces to, and why now. A saving with no
traced spend line behind it is INFERENCE, not a saving.

## Output

The unit table, the ratios with their formulas, the three cases with
break-even, and the verdict with its deciding number.

## Fallbacks

Partial costs: give the range the missing line implies instead of one
false-precise number. No hurdle rate on file: run 10% (typical cost of
capital) and 15% (risk-adjusted) side by side and ask which the team
uses.

## Approval gate

Analysis and drafts only. Funding decisions, vendor changes, and price
moves are the owner's call.
