---
name: "kpi-analysis-getting-started"
description: "Frame a KPI question with metric definitions, sources, periods, owners, decision use, and data limits before analysis starts."
triggers: ["KPI setup", "analyze metrics", "analytics intake", "business metrics", "data analysis"]
version: "1"
---

# KPI analysis getting started

Use this before calculating a result. A clear question and a stable definition
prevent a polished answer to the wrong problem.

## Define the decision

Write down the business question, who will use the answer, the decision it may
change, the required date, and the cost of a wrong call. Separate a descriptive
question (`what changed?`) from a causal one (`what caused it?`).

## Define every metric

For each metric record:

- plain business meaning;
- formula with numerator and denominator;
- unit and direction of improvement;
- event or source fields;
- population, eligibility, and exclusions;
- grain: event, user, account, order, or other supplied unit;
- attribution and conversion windows;
- timezone and period boundaries;
- source owner, refresh time, and known breaks.

Do not accept the same label for two formulas. Version a changed definition and
state where comparison stops.

## Inventory the data

For each file or table, capture its name, extract time, covered dates, row count,
key fields, likely join keys, currency or units, and access limits. Keep raw
source columns apart from derived fields. Record the expected control total.

## Plan the analysis

State the primary comparison, useful segments, quality checks, material-change
threshold, and what evidence could answer the question. List what the data
cannot establish. Return blockers and the smallest source or question that
would remove each one.

## Safety rules

Never invent data, infer a missing definition, or change a metric after seeing
the result without disclosure. Use the least personal data needed; aggregate or
redact identifiers. Do not claim causation from a descriptive export. Route
privacy, legal, clinical, credit, employment, and material finance decisions to
the named owner.
