---
name: "crm-field-audit"
description: "Audit CRM fields by fill rate, format, and use, and propose keep, fix, or retire for each."
triggers: ["field audit", "CRM fields", "empty fields", "data quality CRM", "clean up fields"]
version: "1"
---

# CRM field audit

Use this for one object at a time, or for the fields behind a named report.

## Record the export

State the export date, object, filter, row count, and the list of fields
included. If the export is partial or filtered, say so before any figure.

## Measure each field

For each field record:

- fill rate as a count and a percentage;
- format problems, such as mixed date styles, free text in a picklist, or
  mixed currencies;
- distinct values and the most common ones;
- whether a report, rule, or automation uses it;
- who, if anyone, owns its definition.

Flag fields that look filled but hold placeholders such as "n/a", "tbc", or a
default date.

## Keep, fix, or retire

Label each field **keep**, **fix**, or **retire**, with one line of reason and
the report it affects. For **fix**, name the smallest change: a picklist, a
required rule, or a backfill from a named source. For **retire**, list what
still reads the field.

Return the list for the CRM owner to approve.

Never fill a blank field with a guess, and do not change, hide, or delete a
field or record yourself.
