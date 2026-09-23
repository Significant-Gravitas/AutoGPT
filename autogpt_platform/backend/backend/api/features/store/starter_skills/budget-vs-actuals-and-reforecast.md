---
name: "budget-vs-actuals-and-reforecast"
description: "Use when the user asks about budgets, pacing against plan, forecasts, re-forecasts, annual plans, or where a period lands at the current run rate."
triggers: ["budget vs actuals", "where does the month land", "reforecast", "budget pace check", "are we on plan", "annual plan refresh", "reallocate budget", "run rate to land on plan"]
version: "1"
---

# Budget vs actuals and reforecast

Run this when the user wants to know where a period lands: pacing
against plan, the forecast behind it, or a re-forecast after something
moved. Read from the ledger only — anything untraceable is UNKNOWN,
never estimated.

## Inputs

The finance ledger, the budget set with one row per line and owner, and
the period calendar.

## Name the drivers before the dollars

Forecast driver-first: name the 5-15 drivers that actually move the
plan — units times price, headcount times rate, pipeline times win rate
— and give each one an owner in the assumption log. Model the drivers,
then let the dollars follow. A plan built straight from last year plus
a percentage is not a forecast.

## Read every line against its plan

Per line: actuals to date, the share of the plan already used, days
elapsed against days in the period, and where the period lands if the
current run rate holds. Show the math, not just the verdict. Keep a
rolling 12-18 month horizon — drop the expired month, add a new one at
the end.

This is also the pace check the user can ask for at any point in the
month: every budget line read at once, with month-to-date actuals and
the landing point per line.

## Flag both directions

Flag any line projected to pass its plan, and give the run rate it
would take to land on plan instead. Flag any line pacing more than 15%
under plan too — budget left on the table is a miss, not a saving.
Every flag carries the gap in currency, the driver behind it, and one
play with a named owner.

## Grade the forecast

Base (commit), Adverse (downside), or Opportunity (upside), with one
evidence line behind each grade. Base needs a named driver, not hope.

## Re-forecast without re-baselining

Re-forecast monthly with a deeper refresh each quarter. On every
re-forecast: restate the prior forecast, name what moved and by how
much, and carry forward only the drivers still true. Never silently
re-baseline. Track accuracy against the prior forecast — bias direction
and miss size — and feed repeat misses back into the driver
assumptions.

## Reallocation asks

Move named money between lines with the total flat, cap any single move
at 30% of the source line, and say which owner approved each move.
Draft only — you never move a budget yourself.

## Name the gaps

Missing periods in the ledger get named, and you ask for that export
rather than projecting across a gap. Never invent a figure to fill a
hole.

## Output

One line per budget line, then the total, then the forecast grade and
any reallocation draft. Under 200 words unless they asked for a table.

## Fallbacks

No budget set: build the read against last period's actuals and label
every plan number INFERENCE. No ledger at all: ask for one export and
say exactly which columns you need.

## Approval gate

Reads and drafts only. Budget moves, plan changes, and anything leaving
the team wait for the owner's yes.
