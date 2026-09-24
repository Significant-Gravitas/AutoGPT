---
name: "expense-categorization"
description: "Turn receipts and transaction exports into a sourced expense review table without guessing at unclear items or tax treatment."
triggers: ["categorize expenses", "expense coding", "receipts", "card transactions", "expense review"]
version: "1"
---

# Expense categorization

Use this to prepare expenses for review. A proposed category is not a posted
entry and not tax advice.

## Required inputs

Ask for the transaction export, available receipts or invoices, the approved
chart of accounts, the entity and period, and any written coding rules. Preserve
the original transaction description and source row identifier.

## Work each item

For every transaction return:

| Field | Rule |
| --- | --- |
| Source | File or system plus row identifier |
| Date | Use the source date; keep transaction and posting dates separate |
| Vendor | Use the receipt or source text; do not infer a legal name |
| Amount | Keep sign and currency |
| Evidence | Receipt, invoice, contract, or `missing` |
| Proposed category | Exact approved account name, or `unresolved` |
| Reason | One sentence tied to the evidence |
| Confidence | High, medium, or low |
| Review need | The one fact or owner needed next |

Use high confidence only when the evidence and written rules point to one
approved account. Use medium when the purpose is clear but more than one
account could fit. Use low or unresolved when the purpose, entity, split, or
support is missing.

## Checks

- Total source rows, total amount, categorized rows, and unresolved rows.
- Find exact and likely duplicates without deleting either one.
- Keep refunds, reversals, transfers, owner transactions, and payments apart
  from operating expenses.
- Flag foreign-currency items and preserve both source and settled amounts when
  supplied.
- Show any row excluded from the total and why.

## Safety rules

Do not invent a business purpose, split an amount by guess, decide whether an
item is deductible, or choose tax or accounting treatment. Do not post entries
or edit the source. Route payroll, owner distributions, fixed assets, loans,
taxes, gifts, legal settlements, and suspected fraud to the finance owner or a
qualified accountant.
