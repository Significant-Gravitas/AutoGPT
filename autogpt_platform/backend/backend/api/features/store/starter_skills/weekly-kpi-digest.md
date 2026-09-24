---
name: "weekly-kpi-digest"
description: "Turn approved weekly metrics into a short digest of material movements, evidence, hypotheses, owners, and next checks."
triggers: ["weekly KPI", "metrics digest", "weekly analytics", "KPI report", "weekly business review"]
version: "1"
---

# Weekly KPI digest

Use one fixed reporting window, comparison rule, and metric definition set. A
digest should help a team decide where to look, not list every number available.

## Validate the packet

Record the current and comparison periods, timezone, extract time, metric
version, target or materiality threshold, and source owner. Run the required
quality checks before interpreting changes. Mark late or incomplete metrics.

## Select the movements

Include a metric when it crosses the agreed absolute, relative, target, or
quality threshold. For each selected metric show:

- current value and period;
- prior or expected value;
- absolute and relative change;
- numerator, denominator, and sample where relevant;
- main segment or component contribution;
- source and refresh date;
- `FACT`, `HYPOTHESIS`, or `OPEN` explanation;
- owner and next check.

Do not rank by percentage change alone. A small base, denominator shift, changed
definition, or late data can make a large percentage misleading.

## Write the digest

Start with data health. Then give three to seven material moves, decisions or
help needed, and an appendix of stable metrics. Use the same order each week so
readers can compare periods. Keep targets, forecasts, and actuals distinct.

If no metric crossed a threshold, say so and name any data-quality issue; do not
create a story to fill the page.

## Safety rules

Never invent a cause, suppress an unfavourable result, change a threshold after
seeing the data, or publish personal row-level detail. Do not claim causation
without a valid design. Route decisions with legal, privacy, clinical, credit,
employment, or material finance impact to the named owner.
