---
name: "metric-definition-and-data-quality"
description: "Test metric definitions and source data for freshness, coverage, duplicates, missing values, impossible values, and drift."
triggers: ["data quality", "metric definition", "validate data", "analytics audit", "bad data"]
version: "1"
---

# Metric definition and data quality

Run this before trusting a KPI, anomaly, cohort, funnel, or experiment result.

## Pin the expected shape

Record the source, covered period, extraction time, expected grain, primary or
compound key, required fields, field types, allowed ranges, timezone, and known
late-arrival window. State the metric formula and population separately from the
file schema.

## Test in order

1. **Freshness:** latest event and extract time against the expected delay.
2. **Coverage:** first and last dates, missing intervals, source partitions, and
   control totals.
3. **Schema:** missing, renamed, added, or changed-type fields.
4. **Uniqueness:** exact duplicates and duplicate keys.
5. **Completeness:** null and blank rates by required field and period.
6. **Validity:** negative durations, impossible dates, out-of-range rates,
   inconsistent currencies, and broken state order.
7. **Consistency:** totals across independent supplied sources and stable metric
   definitions across periods.
8. **Join quality:** matched, unmatched-left, unmatched-right, and many-to-many
   row multiplication.

## Report the effect

For each issue give the check, expected result, observed result, affected rows
and periods, likely metric impact, severity, and owner. Show results before and
after any proposed exclusion. Do not drop a bad row silently.

Use `blocked` when the issue can change the answer enough to affect the stated
decision. Use `provisional` when the result remains useful with a clear bound.

## Safety rules

Never repair values by guess, backfill from an unrelated period, deduplicate on
name alone, or hide a failed check. Avoid exposing raw personal or sensitive
fields in the report. Treat a pipeline change as a possible cause of a metric
move until an independent source confirms the business change.
