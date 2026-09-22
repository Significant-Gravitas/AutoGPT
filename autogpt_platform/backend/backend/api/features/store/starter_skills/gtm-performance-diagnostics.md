---
name: "gtm-performance-diagnostics"
description: "Use when the user asks how GTM is doing, hands over funnel or pipeline numbers, or wants KPIs and targets defined: the scorecard with movers, causes, and the forecast read."
triggers: ["how is GTM doing", "funnel numbers", "pipeline coverage", "win rate", "GTM KPIs", "scorecard", "forecast read"]
version: "1"
---

# GTM performance diagnostics

Use this when the user asks how GTM is doing, hands over funnel or
pipeline numbers, or wants KPIs and targets defined. Start from the
exports or connected numbers covering this period and the last — HubSpot,
Amplitude for activation and adoption, a sheet, or a pasted export — the
metric definitions with targets, and the currency.

## Fix the period first

Say it out loud: the days you are scoring and the days you compare
against. Never compare a partial period to a full one.

## Check coverage before you compute

Which sources you have, and which days. A source with no numbers gets
named, never dropped quietly.

## Define each metric once

Name, formula, source column, target, owner, and type — INPUT
(controllable, leading) or OUTPUT (lagging result). GTM metrics that
recur: pipeline created and coverage (3x flag, 2x alarm), win rate, deal
velocity and age, activation and adoption, expansion and retention,
forecast accuracy, and revenue influenced. Money rounds to whole units,
rates to one decimal. When a rate's denominator is zero, report it as
N/A with the absolute counts instead of a percentage.

## Score and rank the movers

Every metric against its target with R/Y/G, then the movers: biggest miss
first, biggest gain next. Decompose each move with the columns you have
and tie it to something visible — a segment that shifted, a play that
landed, a competitor that moved. When the numbers cannot explain the
move, say that in one line and name the one thing you would need.

## Call a thin sample what it is

Fewer than about 30 observations in the period gets a line saying the
move sits inside the noise. Then close with interventions: the two or
three fixes worth doing, each with the metric that justifies it.

## The forecast read

Beside the scorecard: Commit, Best-Case, and Pipeline grades, hygiene
flags (past close date, stale next step, amount and stage mismatch), and
the backward funnel math to the target.

## What you hand back

The scorecard table, the top movers with causes, what to look at, and the
gaps. Save the scorecard dated and write the period's figures to the
metrics history — a sheet the team edits when Sheets is connected. You
diagnose and recommend; CRM hygiene stays with sales ops and finance
reporting with finance.

## Fallbacks

No metric definitions means you propose them from the columns you have
and mark every target UNKNOWN. One source only means you build it for
that one and name what is missing. Label each line FACT, INFERENCE, or
UNKNOWN.
