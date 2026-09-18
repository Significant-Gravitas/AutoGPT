---
name: "procurement-decision-memo"
description: "Draft a procurement decision memo that shows requirements, total cost, evidence gaps, risks, and approvals without selecting or binding."
triggers: ["procurement memo", "vendor decision memo", "vendor recommendation", "purchase approval brief", "supplier decision"]
version: "1"
---

# Procurement decision memo

Draft this after quote comparison and due diligence. Use only evidence in those records.

## Memo structure

1. Decision requested, deadline, request owner, budget owner, and signatory.
2. Business need and cost of the current state.
3. Options considered, including doing nothing where it is real.
4. Must-have coverage and material gaps by option.
5. Total cost on the same term, with assumptions and scenarios.
6. Implementation, service, security, privacy, legal, financial, and exit issues.
7. Open questions and negotiation points.
8. Approval table with `PENDING`, `APPROVED`, or `DECLINED` only from actual owner records.

If a scoring model is used, show weights, raw evidence, and how the result changes when a weight changes. Never hide a must-have gap inside a total score.

## Gate

Vera may summarize which option best matches stated criteria, but cannot select it. End with the exact decisions each owner must make. Do not send the memo, commit spend, accept terms, or instruct a vendor to begin.
