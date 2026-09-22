---
name: "bug-reproduction-report"
description: "Write a bug report with exact steps, expected and actual results, environment, and frequency."
triggers: ["reproduce bug", "repro steps", "write a bug report", "document a reproduction", "file an issue"]
version: "1"
---

# Bug reproduction report

Use this when someone reports a fault and an engineer needs to act on it.

## Gather the evidence

Collect the user's report in their words, screenshots, error messages, log
lines with timestamps, the version or commit, browser or device, account type,
and when it started. Note which details came from the user and which from logs.
Before copying evidence into the report, remove credentials, tokens, personal
data, account identifiers, and unrelated customer data. Keep only the detail
needed to reproduce the fault.

## Smallest reproduction

Work toward the fewest steps and the least data that still show the fault.
State whether you reproduced it yourself, saw it in logs, or only have the
report. Record how often it happens, such as 3 of 10 tries.

## Report

- Title: what fails, where, and when.
- Steps to reproduce, numbered.
- Expected and actual result.
- Environment and version.
- Frequency and first seen.
- Evidence: log lines, traces, and screenshots, quoted exactly.
- Severity and who is affected.
- Open questions.

Never invent a stack trace, a step, or a reproduction you did not see, and do
not guess at a root cause in the report. Do not file the issue or assign it;
return the draft for the user to file.
