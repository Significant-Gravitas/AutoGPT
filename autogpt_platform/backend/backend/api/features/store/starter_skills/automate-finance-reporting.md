---
name: "automate-finance-reporting"
description: "Use to connect a number source, map an export into the finance ledger, or QA a sheet: the column mapping, the dedupe key, the load summary, and the checks that must pass before a read ships."
triggers: ["map this export into the ledger", "ledger qa", "connect a number source", "refresh the finance ledger", "is the ledger stale", "reconcile the sheet to the source", "dedupe the ledger rows"]
version: "1"
---

# Automate finance reporting

Run this when numbers have to move from somewhere else into the finance
ledger, or when a sheet needs checking before anyone reads from it. A
number that arrives unchecked is not a fact yet.

## Inputs

The source — a sheet, a saved query export from the business
intelligence (BI) workspace, a report email, or a plain export file —
and the target ledger shape.

## Map the columns before loading anything

Source column to ledger column, one row per mapping, with the transform
written out next to it. Unmapped columns get listed, not dropped
silently. Show the mapping table before the first load, not after.

## Dedupe on a natural key

Replace on the natural key — date plus line plus source — newest wins.
Log how many rows were replaced so the load is auditable afterwards.

## Derive ratios in the open

Write the formula next to the value on first build. Unparseable rows go
to a needs-a-look list with their row numbers; they never land in the
totals.

## QA every write with a live re-read

Row counts match, rollups reconcile to the source total, and dates
cover the period claimed. If any check fails, say which one and hold
the read. A read shipped on a failed check is worse than a late read.

## Name the refresh path

What triggers the refresh, who owns the source, and what stale looks
like. A ledger older than its own refresh promise gets called stale in
every read that uses it.

## Keep the automation boring

One source of truth, one writer per sheet, and a dated snapshot before
any structural change. Clever automation is the thing that breaks
quietly in month four.

## Output

The mapping table, the load summary with its counts and the
needs-a-look list, and the QA result with each check named.

## Fallbacks

A connector fails: take the export path and keep the same mapping and
the same QA. The source has no usable key: build the read in chat only
and do not write the ledger — a load you cannot dedupe will double a
number later.

## Approval gate

Structural changes to a sheet, new writers, and anything that leaves
the team wait for the owner's yes.
