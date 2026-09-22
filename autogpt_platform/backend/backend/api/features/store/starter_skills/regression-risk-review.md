---
name: "regression-risk-review"
description: "Map what a change can break beyond the diff: callers, data, config, and integrations."
triggers: ["regression risk", "what could this break", "impact analysis", "blast radius", "risky change"]
version: "1"
---

# Regression risk review

Use this for a change that touches shared code, data, or other services.

## Trace the callers

Find every caller of the changed functions, endpoints, events, and database
fields. Check stored data written by the old code, config and feature flags,
scheduled jobs, public APIs, and third-party integrations. Note what you
searched and what you could not reach.

## Rank the risks

For each risk give:

- what could break and for whom;
- the path from the change to the failure, with file and line;
- likelihood and impact, with the reason;
- whether a test covers it today;
- how it would show up in production.

Rank as **high**, **medium**, **low**, or **unknown**. Keep unknowns visible
rather than folding them into low.

## Checks to run

List the checks that would reduce the top risks: a test to add, a query on
real data, a staged rollout, a flag, or a manual check. Name who should run
each one before release.

Never call a change safe or low risk without tracing it, and do not invent a
caller or a failure. Do not merge, deploy, or change config; the owners of the
change decide.
