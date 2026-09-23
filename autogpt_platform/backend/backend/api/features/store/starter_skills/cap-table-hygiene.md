---
name: "cap-table-hygiene"
description: "Check cap-table records against signed source documents and report ownership, security, date, and total differences for expert review."
triggers: ["cap table", "ownership records", "equity records", "option grants", "cap table cleanup"]
version: "1"
---

# Cap-table hygiene

This skill finds record differences. It does not determine legal ownership,
value securities, or edit the cap table.

## Establish authority

Ask the owner or counsel which signed records and approved cap-table system are
authoritative. Record the covered entity, as-of date, currency where relevant,
fully diluted definition if supplied, and named legal and finance reviewers.

## Build the comparison

For each holder or award, compare only fields present in the supplied records:

- legal name or approved identifier;
- security class and instrument;
- issue, grant, exercise, conversion, cancellation, or transfer date;
- quantity authorised, issued, outstanding, exercised, cancelled, or reserved;
- vesting start, schedule, cliff, and status;
- document name, signature state, approval record, and source location.

Check totals by security class and in aggregate. Keep issued, outstanding,
reserved, and fully diluted totals separate. Do not calculate ownership
percentages until the denominator definition has been supplied.

## Report exceptions

Return source fact A, source fact B, the difference, affected total, likely
record owner, and the document or decision needed. Use `possible duplicate`
rather than merging similar names. Mark unsigned, missing, superseded, and
conflicting records.

## Safety rules

Never create, change, delete, merge, or backdate an ownership record. Never
decide that an unsigned grant is valid, interpret a security, set a valuation,
calculate tax treatment, or advise on a transaction. Limit personal data in the
output. Route every ownership, securities, governance, legal, tax, and valuation
question to qualified counsel, the finance lead, or the approved administrator.
