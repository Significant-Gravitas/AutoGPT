---
name: "invoice-drafting-and-issue"
description: "Draft a customer invoice from approved terms, validate its totals and fields, and hold issue or send for explicit approval."
triggers: ["draft invoice", "create invoice", "invoice customer", "billing draft", "issue invoice"]
version: "1"
---

# Invoice drafting and issue

Draft from approved commercial records. Never turn a chat request into a sent
invoice without a review step.

## Gather the terms

Require the legal seller and customer names, billing addresses, purchase order
or contract reference, approved line items, quantity, rate, currency, service
or delivery period, invoice date, due terms, payment instructions, and any tax
fields supplied by the owner or accountant.

If a term is missing, leave a marked blank. Do not copy a bank detail, tax ID,
or customer address from an unrelated invoice without confirmation.

## Build the draft

Show:

1. Draft invoice number or `to be assigned`.
2. Seller and bill-to details.
3. Contract or purchase-order reference.
4. One line per approved charge with description, period, quantity, rate, and
   line total.
5. Subtotal, supplied discount, supplied tax, and grand total.
6. Currency, due date, payment instructions, and remittance reference.
7. Source for every line and term.

Recalculate line totals and the grand total. State rounding. Compare the draft
against the contract or approved order and list every difference.

## Approval gate

Return a review block with:

- totals checked;
- missing fields;
- term or source conflicts;
- named approver;
- exact action awaiting approval: assign number, issue in the billing system,
  or send to the customer.

Do not claim the invoice exists outside the draft until a supplied record proves
it. After approval, record what the owner says was issued, when, and by whom;
do not perform the action yourself.

## Safety rules

Never invent a charge, tax rate, payment detail, due date, or contract term.
Never change agreed pricing, issue or send an invoice, contact the customer, or
move money without explicit approval. Route tax, withholding, cross-border,
credit-note, and revenue-policy questions to the accountant or finance owner.
