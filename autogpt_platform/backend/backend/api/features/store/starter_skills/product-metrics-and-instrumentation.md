---
name: "product-metrics-and-instrumentation"
description: "Use when the user names metrics, a funnel, a dashboard, instrumentation, or asks what moved and why: the North Star metric with its input drivers, the instrumentation spec when the events do not exist, and the read with movers, decomposition, and the call."
triggers: ["north star metric", "instrumentation spec", "read my funnel", "what moved and why", "activation dropped", "which metric matters", "event tracking spec", "name the bottleneck"]
version: "1"
---

# Product metrics and instrumentation

Use this when the user names metrics, a funnel, a dashboard, instrumentation,
or asks what moved and why. The output is either the spec that makes a number
readable or the read that turns it into a decision.

## What you need first

The numbers or the source they come from, the goal they serve, and the period
to read. No source means you write the instrumentation spec first and read
nothing. Their product analytics lives outside the connectors, so work from an
export, a link, or a pasted view.

## Read it

1. Name the one North Star metric — the single metric that best captures value
   delivered to customers — and the two or three inputs that drive it. Check it
   before you trust it: it reflects customer value, it moves with product work,
   and the team can rally around it. A dashboard with twenty KPIs has none.
2. If the events do not exist, write the instrumentation spec: event names,
   properties, and where each one fires. No spec, no dashboard — read nothing
   you cannot trace to an event.
3. Read what moved first, then why. Decompose the headline into its drivers
   before you explain anything, and label every number FACT, INFERENCE, or
   UNKNOWN.
4. Score input and output metrics red, yellow, or green against their targets.
   Carry guardrail and counter-metrics that would catch gaming of the North
   Star, and say the one thing that would move the headline most. No filler
   observations.
5. Close with the call: what changes on the roadmap, the experiment, or the
   launch — or nothing, said plainly. Offer to file the follow-up on a yes.

## Output

The dated read saved next to the last one, so every report compares against the
previous save, or the instrumentation spec when there was nothing to read.

## When the sample is thin

Say it is thin and refuse the verdict; name the period or the volume that would
make the read trustworthy. Never invent a number, a trend, or a cause, and
never read a verdict off a thin sample.
