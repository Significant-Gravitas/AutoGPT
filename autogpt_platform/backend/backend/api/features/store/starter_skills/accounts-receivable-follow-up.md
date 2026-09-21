---
name: "accounts-receivable-follow-up"
description: "Review open receivables and draft factual, staged payment follow-ups without inventing status or contacting a customer."
triggers: ["overdue invoice", "chase invoice", "receivables", "payment follow-up", "accounts receivable"]
version: "1"
---

# Accounts receivable follow-up

Use this to prepare a receivables review and message drafts. The owner approves
all customer contact.

## Verify before calling an invoice overdue

For each open invoice, collect the invoice record, customer, currency, issued
date, due date, original amount, credits, payments, open balance, last contact,
dispute status, and account owner. Check payment records through the stated
cutoff time. If the due date or balance cannot be proved, mark the item `status
unconfirmed` rather than overdue.

Calculate days past due from the supplied due date and cutoff date. Keep
part-payments, credits, disputes, and promised payment dates visible.

## Prioritise the queue

Group items as:

- due soon;
- past due with no known dispute;
- disputed or blocked;
- promised payment not yet due;
- status unconfirmed.

Within each group, sort by age and open balance, then note any named customer
or contract rule that changes the handling. Do not invent a credit policy.

## Draft the follow-up

Each draft should state the invoice number, issue and due dates, open balance,
currency, and the requested next step. Keep the tone calm. Ask whether payment
has been made or whether a record is missing. For a dispute, acknowledge the
stated issue and route it to the owner; do not argue the contract.

Prepare a contact log row with draft stage, evidence cutoff, owner, approval
state, and next review date. Drafts must not claim a late fee, service stop,
collection action, or legal consequence unless an approved policy and owner
instruction support it.

## Safety rules

Never send a message, call a customer, apply a fee, stop service, change an
invoice, or threaten collection. Never expose another customer's data. Route
disputes, hardship, insolvency, legal threats, sanctions, and material customer
risk to the finance owner or counsel.
