---
name: "crm-hygiene-report"
description: "Report CRM health for a period with counts, trends, owners, and the next fixes."
triggers: ["CRM hygiene", "data health report", "CRM report", "pipeline hygiene", "stale deals"]
version: "1"
---

# CRM hygiene report

Use this weekly or monthly, from dated exports compared with the last report.

## Headline counts

Open with the export dates and row counts, then give each count against the
last period:

- records with no owner;
- key fields below the agreed fill rate;
- likely duplicate pairs open;
- deals past close date or stale in stage;
- leads with no owner after a day.

Say which change matters most and which report it affects.

## By owner and object

Break the counts down by object and by owner. Show where problems cluster and
whether they are new or old. Keep the tone neutral; the aim is to fix the
data, not to rank people.

## Next fixes

List three to five fixes in order, each with the count it would clear, the
report it helps, the owner who must approve it, and a suggested date. Note
fixes from the last report that are still open.

Never guess a missing value to improve a count, and do not hide records to
make a trend look better. Do not merge, delete, reassign, or edit CRM records
yourself; the owners make each change.
