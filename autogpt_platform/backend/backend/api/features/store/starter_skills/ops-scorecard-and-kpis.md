---
name: "ops-scorecard-and-kpis"
description: "Use when the user asks how operations is doing, hands over numbers for a read, or wants operational KPIs defined with formulas, targets, and owners."
triggers: ["build an ops scorecard", "define ops KPIs and targets", "how is ops doing", "score my ops metrics", "ops metrics read", "set ops metric owners"]
version: "1"
---

# Build the ops scorecard

Use this when the user asks how operations is doing, hands over numbers
for a read, or wants KPIs and targets defined.

## What you need first

The exports or connected numbers covering this period and the last, the
metric definitions with targets, and the currency.

No metric definitions means you propose them from the columns you have
and mark every target UNKNOWN. One source only means you build the
scorecard for that one and name what is missing.

## Fix the period first

Say out loud the days you are scoring and the days you compare against.
Never compare a partial period to a full one.

## Check coverage before you compute anything

Name which sources you have and which days they cover. A source with no
numbers gets named, never dropped quietly.

## Define each metric once

Name, formula, source column, target, owner, and type — INPUT
(controllable, leading) or OUTPUT (lagging result). The owner watches
their metrics and knows normal variance from exception.

Ops metrics that recur: error and rework rate, cycle time, forecast
accuracy, utilization, SLA compliance, invoice accuracy, milestone
attainment.

Money rounds to whole units, rates to one decimal.

Quarterly, audit every metric: the owner confirms the definition,
source, and target still hold; dead metrics get retired; and the read
notes who certifies the accuracy.

## Score, then rank the movers

Score every metric against its target with red/yellow/green (R/Y/G),
then rank the movers — biggest miss first, biggest gain next.

Decompose each move with the columns you have and tie it to something
visible: a process that changed, a vendor that slipped, a hire that
landed. When the numbers cannot explain the move, say so in one line
and name the one thing you would need.

## Call a thin sample what it is

Fewer than about 30 observations in the period gets a line saying the
move sits inside the noise. Monthly, roll the weeks into one read
against plan.

## Rules

No claim without a number behind it. No recommendation to spend or cut
without the metric that justifies it — and even then, the plan
proposes, the approver disposes. Label each line FACT, INFERENCE, or
UNKNOWN.

Save the scorecard dated and write the period's figures to the metrics
history.
