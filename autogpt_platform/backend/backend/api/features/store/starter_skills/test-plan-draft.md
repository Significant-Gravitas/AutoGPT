---
name: "test-plan-draft"
description: "Plan tests for a change from risk, with cases, data, and what stays uncovered."
triggers: ["test plan", "what should I test", "test cases", "QA plan", "test strategy"]
version: "1"
---

# Test plan draft

Use this for a feature, fix, or refactor before it is merged or released.

## List what can break

From the change and the code around it, list the ways it can fail: wrong
output, bad input, empty or large data, permissions, concurrency, time zones,
retries, and old data. For each, note who it hurts and how it would be
noticed.

## Cases and data

Write the cases in risk order. For each give:

- what it checks and the risk it covers;
- setup and test data;
- steps and expected result;
- level: unit, integration, end-to-end, or manual;
- whether an existing test already covers it.

Prefer a few cases that catch real failures over many that repeat the happy
path.

## Coverage gaps

List what the plan does not cover and why: no test data, no environment, cost,
or low risk. Name who should accept each gap.

Never mark a case as passed or covered without seeing it run. Do not change
the test suite, skip a test, or merge anything; the team decides which cases to
build.
