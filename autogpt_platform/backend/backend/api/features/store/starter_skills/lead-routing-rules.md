---
name: "lead-routing-rules"
description: "Write lead routing rules that cover every case once, with a fallback and an owner."
triggers: ["lead routing", "assign leads", "routing rules", "lead assignment", "territory rules"]
version: "1"
---

# Lead routing rules

Use this when leads go unassigned, land with two people, or wait too long.

## List the cases

From the current rules and a dated lead export, list every case the rules must
handle: region, company size, product, source, existing account, existing
owner, partner-sourced, and leads with missing fields. Count how many leads in
the last 90 days fell into each case, and how many had no owner after a day.

## Write the rules

Write the rules in order, first match wins. Each rule has:

- the condition, using fields that exist and are filled;
- the owner or queue it routes to;
- the time the owner has to respond;
- the backup owner when the first is away or at capacity.

End with a fallback rule for anything unmatched, owned by a named person.

## Test against real leads

Run the draft rules on the last 90 days of leads on paper. Report leads that
matched no rule, matched more than one, or would change owner. Show the change
in leads per owner.

Never reassign a lead or edit routing in the CRM yourself, and do not guess a
missing region or size. The sales lead approves the rules before anyone turns
them on.
