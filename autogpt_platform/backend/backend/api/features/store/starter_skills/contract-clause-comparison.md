---
name: "contract-clause-comparison"
description: "Compare clause versions or a clause with supplied playbook text, showing exact additions, removals, and open decisions."
triggers: ["compare clauses", "clause changed", "contract version comparison", "redline comparison", "compare legal text"]
version: "1"
---

# Contract clause comparison

Use source text the user supplies. Name each source and version before comparing.

## Produce a literal comparison

Preserve section numbers and defined terms. Show:

- text unchanged in meaning and wording;
- text added;
- text removed;
- defined terms added, removed, or changed;
- dates, amounts, periods, parties, standards, and exceptions changed;
- cross-references that no longer resolve;
- text that cannot be compared because a schedule or definition is missing.

If a playbook is supplied, show it in a separate column and label the clause `MATCH`, `DEVIATION`, `MISSING`, or `UNCLEAR`. If no playbook is supplied, do not use those labels as a policy judgment; report only the version difference.

## Describe, do not advise

State the operational effect in neutral terms, such as "notice changes from email to courier" or "the period changes from 30 to 60 days." Do not say a change is legal, safer, worse, standard, enforceable, or acceptable.

## Output

Return the side-by-side text, change list, broken references, and questions for counsel. Do not create or send a redline unless counsel supplies or approves the replacement text.
