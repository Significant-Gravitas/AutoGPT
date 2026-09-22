---
name: "bookkeeping-getting-started"
description: "Set up a safe bookkeeping intake: reporting period, source records, chart of accounts, owners, approval rules, and open gaps."
triggers: ["start bookkeeping", "bookkeeping setup", "finance records", "books", "accounting intake"]
version: "1"
---

# Bookkeeping getting started

Use this before sorting expenses, reconciling a statement, or drafting a
month-end report. The aim is a complete, traceable input set, not a fast guess.

## Build the intake sheet

Record:

1. Entity name and reporting period.
2. Reporting currency and any foreign currencies in the records.
3. Accounting basis if the owner or accountant has supplied it. Otherwise mark
   it open; do not choose one.
4. Approved chart of accounts and the person who owns changes to it.
5. Source systems and exports supplied: bank, card, billing, payroll, expenses,
   and ledger.
6. Opening balances and the record that supports each one.
7. Who approves invoices, adjustments, customer contact, and final reports.
8. The accountant or finance owner who receives tax, policy, and material
   exceptions.

For each source, state its filename or record name, covered dates, export time,
row count, currency, and control total. Do not merge periods or entities until
the owner confirms that they belong together.

## Check readiness

Return three lists:

- **Ready:** source is present, readable, in period, and tied to a control total.
- **Usable with a warning:** source has a named gap that does not block the task.
- **Blocked:** missing dates, unclear entity, broken file, no opening balance, or
  totals that do not agree.

Ask one question per blocking gap. Prefer a source record over a recollection.

## Safety rules

- Never create a missing amount, date, vendor, customer, account, or currency.
- Never choose tax treatment, accounting policy, or a filing position.
- Never post an entry, change a source record, move money, or contact a third
  party.
- Keep personal and bank data to the least detail needed for the task.
- Route payroll, tax, equity, fraud, and material policy questions to a
  qualified accountant or named finance owner.

End with the work that can start now, the blocked work, and the owner of every
open item.
