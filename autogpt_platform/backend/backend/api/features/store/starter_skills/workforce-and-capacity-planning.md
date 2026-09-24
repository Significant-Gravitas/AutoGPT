---
name: "workforce-and-capacity-planning"
description: "Use when volume meets staffing: forecast contacts, build schedules, watch real-time adherence, and move people intra-day."
triggers: ["forecast contact volume", "support staffing plan", "schedule adherence", "intra-day moves", "shrinkage math", "shift coverage gap", "support headcount plan"]
version: "1"
---

# Workforce and capacity planning

Use this when volume meets staffing. Forecast contacts, build schedules,
watch real-time adherence, and move people intra-day.

## What you need first

Past volume by channel and interval, the workforce-management (WFM) targets
and shrinkage from prefs, the current schedule and absence list, and the
real-time queue state.

## Forecast, staff, schedule, watch, close the loop

1. Forecast volume per channel and interval from history, naming trend,
   seasonality, and known events (launches, promos, outages). State the
   horizon and the miss band plainly; a forecast with no history behind it
   is UNKNOWN, never padded.
2. Turn the forecast into required heads: staffed heads = base requirement /
   (1 - shrinkage/100), where shrinkage = (unavailable hours / logged-in
   time) x 100 for breaks, training, absence, and time offline. Usual range
   10-40%; 30-35% is the planning sanity band for full-service centers.
   This skill folds after-call work into handle time (some shops count it as
   shrinkage — call out the convention in the math block). Erlang-style
   interval math assumes steady arrivals, zero abandonment, and single-skill
   queues, so it tends to overstaff multi-skill or bursty queues — say so
   when it applies. Show the math in one block so the owner can check it.
3. Build the schedule against required heads: shifts, breaks, skill groups,
   and time-off, then assign agents to shifts honoring bids, prefs, and
   business rules, with coverage gaps named, not hidden. Schedule changes go
   to the schedule owner for yes before anyone's hours move.
4. Watch real-time adherence: adherence means agents doing the scheduled
   activity in each interval — logged-in vs scheduled, queue vs forecast,
   SLA risk by interval. When reality beats the plan, stage intra-day
   moves — pull from quiet queues, pause offline work, offer overtime — and
   say what each move costs.
5. Close the loop weekly: forecast vs actual, adherence, SLA hit rate, and
   the one planning fix for next week. Log the forecast, the schedule state,
   and the moves made.

## Output

The forecast with its math, the staged schedule, the intra-day moves, and
the weekly accuracy read.

## When the records are thin

No volume history means a judgment forecast marked INFERENCE with the data
need named, plus the questions that would firm it. Never present a guessed
headcount as modeled.
