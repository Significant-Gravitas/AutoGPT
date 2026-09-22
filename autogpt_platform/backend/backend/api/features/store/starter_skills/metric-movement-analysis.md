---
name: "metric-movement-analysis"
description: "Explain why a metric moved by decomposing its definition, population, segments, timing, and data pipeline into testable hypotheses."
triggers: ["why did this metric move", "metric changed", "KPI decline", "KPI growth", "root cause analysis"]
version: "1"
---

# Metric movement analysis

Use this for an ad hoc `why did X move?` question. Begin with the identity of X,
not a story about the business.

## Reproduce the move

State the metric definition, current and comparison periods, values, absolute
and relative change, population, sample, source, and extract time. Recalculate
the result from the supplied detail when possible. Stop if definitions or data
coverage differ enough to break the comparison.

## Decompose in order

1. Numerator and denominator.
2. Volume, rate, and mix effects.
3. New, retained, reactivated, and lost populations where relevant.
4. Product step, plan, channel, geography, platform, and other approved
   segments.
5. Day, week, season, billing cycle, and event timing.
6. One large account or item.
7. Tracking, schema, join, and pipeline changes.
8. Supplied product, price, campaign, policy, or operating changes.

Quantify how much of the total change each component explains. Avoid summing
overlapping segments as if they were independent.

## Rank explanations

Label direct observations `FACT`, proposed explanations `HYPOTHESIS`, and missing
evidence `OPEN`. For each hypothesis, give supporting evidence, contradicting
evidence, confidence, and the smallest test that could disprove it. If the data
supports no cause, say so.

## Safety rules

Never claim cause from timing or correlation alone. Do not select only the
segment that supports the requested story, alter the comparison after seeing
the result, or invent missing context. Aggregate sensitive data and route
high-impact decisions to the named owner.
