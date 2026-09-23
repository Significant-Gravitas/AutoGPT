---
name: "product-experiment-design"
description: "Use when the user names an A/B test, an experiment, a growth bet, or a before-and-after read: the plan with one primary metric, guardrails, a computed sample size, precommitted interpretation bands, then the scale-extend-or-kill readout."
triggers: ["design an A/B test", "experiment plan", "minimum detectable effect", "how long should the test run", "scale or kill", "experiment readout", "guardrail metrics", "precommit the bands"]
version: "1"
---

# Product experiment design

Use this when the user names an A/B test, an experiment, a growth bet, or a
before-and-after read. The output is a plan whose verdict is decided before the
data arrives, and then the readout against it.

## What you need first

The belief under test, the decision it informs, and the traffic or users
available. No traffic numbers means you size with bands and name the minimum
sample before launch. No belief stated means you name the bottleneck belief
first and get a yes on it.

## Design it

1. Name the one bottleneck belief the test exists to move. A test that tries to
   prove three things proves none.
2. List two to four candidate tests and compare them on information gain, cost,
   time, and reversibility. Pick one and say in a line why it won.
3. Write the design: control and variant; ONE primary success metric;
   guardrail metrics that catch harm elsewhere; the minimum sample computed
   from the baseline plus the minimum detectable effect at 95% significance
   and 80% power; a duration of at least one to two full weeks; pre-declared
   segments; the business-meaningful threshold stated beside the p-value; and
   the flag configuration. Guardrails are not optional.
4. Precommit the interpretation bands with observable numbers: the upgrade band
   where you scale, the ambiguous band where you extend or rethink, and the
   downgrade band where you kill. No vague words like "several" or "good".
5. State plainly what the test cannot establish, then launch only on a yes.
6. Read it out: the band it landed in, the guardrail check, the sample-ratio
   mismatch check, and the call — scale, extend, or kill.

## Output

The dated experiment plan and, after the run, the dated readout with the call
and the follow-up offered on a yes.

## When the traffic is not there

Say the sample will not reach the minimum and propose the alternative — a
longer run, a bigger effect to look for, a painted-door test, or a qualitative
read instead. Never move the bands after seeing the data, and never call a
test before one to two full weeks and the precomputed sample size.
