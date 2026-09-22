---
name: "gtm-performance-diagnostics"
description: "Use when the user asks how GTM is doing, hands over funnel or pipeline numbers, or wants KPIs and targets defined: the scorecard with movers, causes, and the forecast read."
triggers: ["how is GTM doing", "funnel numbers", "pipeline coverage", "win rate", "GTM KPIs", "scorecard", "forecast read"]
version: "1"
---

# GTM performance diagnostics

Use this when the user asks how go-to-market is doing, hands over funnel or
pipeline numbers, or wants GTM metrics and targets defined. This is the
commercial engine read — how leads become pipeline, pipeline becomes revenue,
and revenue retains — not the operations scorecard. Start from this period's
and last period's numbers (HubSpot, Amplitude for activation and adoption, a
sheet, or a pasted export), the targets, and the currency.

## Fix the period and the funnel first

Say the days you are scoring and the days you compare against out loud, and
never compare a partial period to a full one. Then name the funnel stages
the numbers actually use — lead, MQL, SAL, SQL, opportunity, won — and which
source covers which stage. A stage with no source gets named, never dropped
quietly.

## The GTM metric set

Define each once with its formula and denominator, then score it against
target red, yellow, or green:

- Pipeline created in the period, and coverage: open qualified pipeline over
  the remaining target — 3x is the usual flag line, 2x the alarm, adjusted
  to the historical win rate.
- Win rate: won over won plus lost, by segment and by source, on closed
  deals only. Open deals are not a denominator.
- Deal velocity: opportunities times win rate times average deal size, over
  the average cycle in days — and deal age against that cycle.
- Stage conversion: each stage into the next, against last period.
- Activation and adoption for product-led or free-tier motions.
- Expansion and retention: net revenue retention where the cohort supports
  it, and logo churn where it does not.
- Forecast accuracy: last period's commit against what actually closed.
- Revenue influenced by each motion, with the attribution rule stated.

Money rounds to whole units, rates to one decimal. When a rate's
denominator is zero, report N/A with the absolute counts instead of a
percentage.

## Rank the movers and decompose them

Biggest miss first, biggest gain next. Break each move into its funnel
drivers — volume in at the top, conversion between stages, deal size, cycle
time — and tie it to something visible: a segment that shifted, a campaign
that landed, a competitor that moved, a rep who left. A move you cannot
explain gets one line saying so and the one thing you would need.

## Call a thin sample what it is

Fewer than about 30 observations behind a rate means the move sits inside
the noise — say so, and never manufacture a trend from it.

## The forecast read

Beside the scorecard: Commit, Best-Case, and Pipeline grades with the
evidence behind each, the hygiene flags (past close date, stale next step,
amount and stage mismatch, single-threaded enterprise deals), and the
backward funnel math from the target to the pipeline it needs at the current
conversion rates.

## What you hand back

The scorecard table, the top movers with their drivers, the forecast read,
what to look at next, and the two or three interventions worth doing, each
with the metric that justifies it, an owner, and a date. Save the scorecard
dated and write the period's figures to the metrics history — a sheet the
team edits when Sheets is connected. You diagnose and recommend; CRM hygiene
stays with sales ops and finance reporting with finance.

## Fallbacks

No targets means you propose them from history and mark every target
UNKNOWN. One source only means you build the read for that stage and name
what is missing. Label each line FACT, INFERENCE, or UNKNOWN, and never
invent a rate, a count, or a cause.
