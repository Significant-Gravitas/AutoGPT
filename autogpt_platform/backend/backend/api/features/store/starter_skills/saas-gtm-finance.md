---
name: "saas-gtm-finance"
description: "Use for the ARR bridge, MRR movement, retention and churn math, renewal base, pipeline coverage, and go-to-market efficiency reads; payback questions go to the unit economics skill."
triggers: ["arr bridge", "net revenue retention", "churn math", "mrr movement", "magic number", "rule of 40", "burn multiple", "renewal base by quarter"]
version: "1"
---

# SaaS and GTM finance

Run this for subscription revenue math and go-to-market (GTM)
efficiency. Every term in a bridge carries its number or an UNKNOWN
label — a bridge that balances because you plugged it is worse than one
that doesn't.

## Inputs

The revenue ledger or a CRM pull: ARR or MRR by account, new against
expansion against contraction against churn, pipeline by stage, and
sales capacity.

## Build the revenue bridge

Starting ARR, plus new, plus expansion, minus contraction, minus churn,
equals ending ARR. Say which period and which cohort in the same line.

## Retention, on one cohort and one period

Compute net revenue retention (NRR) and gross revenue retention (GRR)
on the same cohort and period, and state the denominator plainly. Never
blend cohorts silently. Common SaaS benchmarks, which vary by segment
and deal size: NRR at or above 100% is healthy and 120% is elite; GRR at
or above 90% is healthy and 95% is elite. Say which segment a benchmark
was drawn from before grading against it.

Split logo churn from revenue churn, and check the new-logo share of
growth: when nearly all growth is expansion, a strong NRR can be masking
a new-logo problem, so say what share came from new logos and what that
implies for this stage of company rather than applying one fixed ratio.

## Pipeline coverage and the quarter grade

Read coverage against target — under 3x is a flag, under 2x is an
alarm. Grade the quarter Base (commit), Adverse (downside), or
Opportunity (upside), with one evidence line behind each.

## Flag the hygiene breaks that poison a forecast

Stale close dates, amount-and-stage mismatches, and deals with no next
step. Drafts only — you never write to the CRM.

## Renewals

List the renewal base by quarter from the revenue ledger or the CRM
pull, with at-risk accounts named: contraction last period, or a stale
close date or missing next step from the hygiene check. Attach one save
play per at-risk account — owner, action, date. Save plays are drafts
and go to the account owner, never to the customer.

## GTM efficiency, on FACT inputs only

Show the assumption box, then:

- Magic Number = (quarterly net new ARR x 4) / sales and marketing
  spend. Above 0.75 is healthy, above 1.5 is elite.
- Rule of 40 = revenue growth % + EBITDA margin %, at or above 40.
- Burn Multiple = net burn / net new ARR, under 1.5.

Customer acquisition cost and its payback exclude expansion spend and
expansion ARR — new logos only — and they are owned by the unit
economics skill. Hand it the cohort definition and the sales and
marketing spend, and quote its result rather than computing your own.

## Output

The bridge, NRR and GRR with their denominators, the coverage read with
its grade, the hygiene flags, and the renewal or efficiency table that
was asked for.

## Fallbacks

No CRM connection: work from a pipeline export and name its date. No
cohort history: give point-in-time metrics only and say plainly that
NRR and GRR need one more period.

## Approval gate

Reads and drafts only. No CRM writes, no customer contact, and no
commit call on the owner's behalf.
