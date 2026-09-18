---
name: "opportunity-brief"
description: "State one opportunity with the user, problem, evidence, options, risks, and unknowns."
triggers: ["opportunity brief", "problem statement", "should we build this", "product bet", "one-pager"]
version: "1"
---

# Opportunity brief

Use this for one problem the team is deciding whether to work on.

## Problem and evidence

State who has the problem, what they are trying to do, and what goes wrong.
Give the evidence with counts, sources, and dates: tickets, interviews,
usage, and sales notes. Quote users word for word except for redacted spans.
Replace each removed span with `[redacted]` and keep all other words unchanged.
Mark the evidence **strong**, **moderate**, or **thin**, and say what would make
it stronger. Link the problem to one of the team's stated goals. Before quoting
evidence, redact names, contact details, account identifiers, credentials, and
unrelated customer data. Keep only the words needed to show the problem.

## Options and trade-offs

Lay out two to four options, including doing nothing. For each give:

- what changes for the user;
- rough effort, only as supplied by engineering, or marked as a guess;
- risks, dependencies, and what it rules out;
- how you would know it worked.

Keep claims about competitors to what a source shows, with the link and date.

## Decision needed

Close with the exact decision, who makes it, by when, and the open questions
that could change the answer.

Do not add the item to the roadmap or tracker yourself; recommend and draft.
Never invent a quote, metric, estimate, or competitor claim, and never commit
the team to a date.
