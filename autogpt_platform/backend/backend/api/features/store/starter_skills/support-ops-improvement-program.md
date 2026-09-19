---
name: "support-ops-improvement-program"
description: "Use when the numbers should drive the fixes: run the support dashboard, benchmark SLAs, and own the improvement backlog to measured done."
triggers: ["support dashboard read", "SLA benchmark", "support improvement backlog", "CSAT trend", "first contact resolution rate", "cost per contact", "weekly support one-pager"]
version: "1"
---

# Support ops improvement program

Use this when the numbers should drive the fixes. Run the support dashboard,
benchmark SLAs, and own the improvement backlog to measured done.

## What you need

The ticket and call data, the dashboard home and benchmark source from prefs,
the current backlog, and the last review's decisions.

## Run the program

1. Read the dashboard in one pass — one speed, one outcome, one sentiment,
   one cost metric: volume by channel, first-response and resolution times,
   SLA hit rate, CSAT (with response rate), first-contact resolution (FCR),
   reopen rate, abandon rate, occupancy, customer-effort score, net promoter
   score, and cost per contact and per resolution — each with trend and
   period. A metric with no source is cut, never estimated. Read average
   handle time (AHT) alongside FCR, never alone; resolution is not
   deflection.
2. Benchmark SLAs against the agreed source (last quarter, peer team, or
   published standard): where the team leads, where it lags, and the gap in
   plain numbers. Report SLA compliance (answered-within-SLA / eligible
   requests, with start/stop rules and exclusions stated) alongside the
   median and p90 tail — averages hide breach tails. Name the benchmark on
   every comparison; an unsourced "industry standard" is UNKNOWN.
3. Mine the top three drivers behind the worst metric: repeat themes with
   counts and quoted threads, root causes with owning teams. Size each fix:
   effort, expected lift, owner. Rank by lift over effort.
4. Stage the backlog moves: the one fix to launch this week with its measure,
   the one to pilot, the one to kill. Each fix carries a dated numeric target
   (e.g. cut first-response time by 5s by next quarter). Launches go to the
   owner for yes with the rollback line attached. Never declare a fix done
   without its before-and-after numbers.
5. Log decisions, owners, and dates. Offer the weekly one-pager: metrics,
   benchmark deltas, backlog moves, next review date. Refresh internal
   numbers weekly; refresh external benchmarks quarterly.

## Output

The dashboard read, the benchmark comparison, the ranked backlog, and the
staged launch with its measure.

## Fallbacks

No data access means a metrics-UNKNOWN read naming the missing source, with
the backlog ranked by owner judgment only. Never chart guessed numbers.
