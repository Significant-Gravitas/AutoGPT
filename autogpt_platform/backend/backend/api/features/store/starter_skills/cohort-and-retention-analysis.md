---
name: "cohort-and-retention-analysis"
description: "Build cohort and retention views with explicit entry events, return events, windows, eligibility, censoring, and sample sizes."
triggers: ["cohort analysis", "retention", "churn", "return rate", "customer cohorts"]
version: "1"
---

# Cohort and retention analysis

Retention has no single default definition. Define it before calculating it.

## Set the cohort contract

Record:

- entity: user, account, subscription, or another supplied unit;
- cohort entry event and cohort period;
- eligible population and exclusions;
- return or retained event;
- observation windows and grace periods;
- timezone;
- treatment of reactivation, upgrades, downgrades, deletion, and late events;
- minimum follow-up needed for each cohort age.

Do not compare a mature cohort with a newer cohort at an age the newer cohort
has not reached. Mark right-censored cells as unavailable, not zero.

## Build and check

Show starting cohort size, retained count, retention rate, and available follow-
up at each age. Reconcile cohort starts to the source population. Check duplicate
entities, identity merges, event gaps, backfills, and definition changes.

Segment only on fields known at the relevant decision time. Keep sample sizes
visible, and suppress or group very small cells when they could expose a person
or produce unstable claims.

## Interpret carefully

Separate changes in acquisition mix from changes within like cohorts. Report
absolute percentage-point differences as well as relative change. Label possible
drivers as hypotheses and name the evidence needed to test them.

## Safety rules

Never redefine churn or retention to improve the result, fill unobserved future
periods, hide small samples, or treat correlation as cause. Do not expose person-
level behaviour. Route contractual churn, health, credit, employment, or other
high-impact interpretations to the named owner.
