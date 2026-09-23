---
name: "customer-health-score"
description: "Build an auditable customer health score while keeping observed signals, judgement, and missing data separate."
triggers: ["customer health score", "account health", "health model", "red yellow green accounts", "customer score"]
version: "1"
---

# Customer health score

Use this to define or apply an account health model.

## Define before scoring

For every signal record its meaning, source, date range, refresh rate, good and
bad thresholds, weight, and missing-data rule. Keep adoption, outcome,
support, relationship, and commercial measures visible as separate parts.

## Score with evidence

For each account show the raw value, baseline or target, trend period, component
score, and source. Label manual judgement and explain it. Do not convert a
missing value to zero or healthy. Do not use logins as proof of value when the
customer's outcome calls for another measure.

## Return

Give the component view, overall result, confidence, missing data, material
change since the last period, and next check. Include the formula or rules so a
person can reproduce the result.

Never invent usage, sentiment, support history, goals, or contract facts. A
health score directs review; it does not prove renewal or churn.
