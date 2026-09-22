---
name: "pull-request-review"
description: "Review a change for bugs and risk, with file, line, trigger, and confidence for each finding."
triggers: ["review this PR", "code review", "pull request review", "check my diff", "review my code"]
version: "1"
---

# Pull request review

Use this for one pull request or diff at a time.

## Read the change in context

Read the description and linked issue first, then the diff, then the code the
change calls and the code that calls it. Check the tests that cover it and
whether CI has run. Note what could not be read, such as a truncated diff or a
missing file.

## Findings by severity

Rank each finding as **blocker**, **should fix**, **question**, or **style**.
For each give:

- file and line;
- what goes wrong, in one sentence;
- the input or state that triggers it;
- confidence, and whether it comes from reading or from running the code;
- a suggested fix, if one is clear.

Keep style notes apart and short. Do not pad the list; say so when the change
looks sound.

## Questions and untested paths

List what the review could not settle: behaviour that depends on data or
config you cannot see, paths with no test, and assumptions the author should
confirm.

Never say code is safe or a test passed without seeing the evidence, and do
not invent a failure you cannot trace to a line. Do not approve, merge, or push
to the branch; the reviewer of record decides.
