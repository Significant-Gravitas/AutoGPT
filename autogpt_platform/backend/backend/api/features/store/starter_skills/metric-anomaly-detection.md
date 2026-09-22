---
name: "metric-anomaly-detection"
description: "Flag metric anomalies against declared baselines while separating data faults, expected seasonality, and real business changes."
triggers: ["metric anomaly", "spike", "drop", "outlier", "unexpected KPI change"]
version: "1"
---

# Metric anomaly detection

An anomaly is a signal that needs review, not proof of a problem or a cause.

## Set the detection rule first

Record the metric version, grain, eligible population, timezone, expected data
delay, baseline window, known seasonality, and alert threshold. Use a rule the
owner supplied or explain the proposed rule before applying it. Keep absolute
and relative thresholds when low volumes can distort rates.

## Check the data before the business

For every alert, test:

1. extract freshness and missing partitions;
2. schema or event-version changes;
3. duplicate or lost rows;
4. numerator and denominator movement;
5. timezone, calendar, holiday, and billing-cycle effects;
6. one large account, order, or segment dominating the move;
7. known release, campaign, price, or policy changes from supplied records.

Compare against more than one useful baseline when available: prior period,
same weekday or season, and a recent range. Do not use a model or threshold that
you cannot explain in the report.

## Return an anomaly card

Give metric, observed and expected value, size of difference, affected period,
sample, segment concentration, quality-check result, confidence, ranked
hypotheses, and one disproof test per hypothesis. Mark status as data issue,
expected pattern, possible business change, or unresolved.

Track alert status and owner. Do not close an alert because the metric returned
to normal; record whether anyone found a cause.

## Safety rules

Never label fraud, misconduct, or customer harm from an aggregate anomaly alone.
Do not expose individual records unless an approved investigation needs them.
Never hide a false positive or tune the rule after the event without recording
the change.
