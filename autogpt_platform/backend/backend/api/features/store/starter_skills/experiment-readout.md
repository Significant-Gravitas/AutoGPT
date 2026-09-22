---
name: "experiment-readout"
description: "Write an experiment readout with assignment checks, metric definitions, sample sizes, uncertainty, practical effect, and limits."
triggers: ["experiment results", "A/B test", "test readout", "variant analysis", "experiment analysis"]
version: "1"
---

# Experiment readout

Use this for a controlled experiment or a clearly labelled quasi-experiment.
Do not call a before-and-after comparison an A/B test.

## Record the design

Capture the hypothesis written before the test, unit of assignment, eligibility,
variants, allocation, exposure event, start and end times, stopping rule,
primary metric, guardrails, analysis window, exclusions, and planned segments.
Name any field that was chosen after results were visible.

## Validate execution

Check sample-ratio mismatch, duplicate assignment, cross-variant exposure,
missing exposure, unequal data delay, pre-test balance on available baseline
fields, instrumentation changes, and novelty or calendar effects. State how each
fault limits the result.

## Report results

For every primary and guardrail metric show control and variant sample sizes,
raw values, absolute effect, relative effect where useful, uncertainty interval
or other approved uncertainty measure, and missing-data rate. Put the primary
metric first. Keep exploratory metrics and segments in a separate section.

Discuss practical size, not only statistical significance. A result can be
uncertain, too small to matter, or harmful on a guardrail. State whether the
data supports ship, do not ship, continue, or no decision under the owner's
pre-agreed rule; the named owner makes the decision.

## Limits and next test

List deviations from plan, unmeasured outcomes, affected population, follow-up
window, and what the result does not prove. Give the next test only when it
answers a stated open question.

## Safety rules

Never change the primary metric or stopping rule without disclosure, omit an
unfavourable guardrail, invent significance, or generalise past the tested
population. Protect personal data. Route tests with legal, medical, credit,
employment, or safety impact to the responsible expert before action.
