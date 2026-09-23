---
name: "flaky-test-triage"
description: "Triage an intermittent test from run history, environment, and failure pattern before naming a cause."
triggers: ["flaky test", "intermittent failure", "test fails sometimes", "CI flakes", "unstable test"]
version: "1"
---

# Flaky test triage

Use this when a test passes and fails on the same code.

## Collect run history

Gather the last 30 to 100 runs where possible: pass or fail, job, runner,
branch, commit, duration, and the failure message. Say how many runs you
actually saw and over what dates.

## Find the pattern

Group the failures by job, runner, time of day, test order, duration, and
error. Look for common causes: timing and waits, shared state between tests,
test order, real clocks, network calls, random data, and resource limits. Name
each candidate with the evidence for and against it, and how sure you are.
Say when there are too few failures to tell.

## Next check

Propose the smallest check that would confirm or rule out the top candidate,
such as a rerun loop on one job, a fixed seed, or running the test alone.
Then list who owns the test and the options: fix, quarantine with an issue
and a date, or leave it and watch.

Never call a test flaky, or a cause found, without run history to show it.
Do not skip, disable, retry-wrap, or delete a test yourself; the owner decides.
