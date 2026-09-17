---
name: "billing-refunds-and-exceptions"
description: "Use when money is on the table: verify the charge, check the policy, and stage a refund or exception draft that stops at the owner's yes."
triggers: ["customer wants a refund", "charge dispute", "billing exception", "credit request", "waive a fee", "refund policy check", "customer was double charged"]
version: "1"
---

# Billing refunds and exceptions

Use this whenever money is on the table. Verify the charge, check the
policy, and stage a refund or exception draft that stops at the owner's yes.

## What you need first

The order, invoice, or booking record; the governing refund or policy
passage; the approval limit and approver; and the ticket with its sentiment.

## Verify, then check policy, then draft

1. Verify first: the exact charge, date, and what the customer paid for,
   from the records. Name the amount in numbers. A missing record means the
   draft waits; say which record would release it.
2. Check the policy passage and cite it: eligible or not, the exception it
   allows, and who may grant it. Above the approval limit or outside policy
   goes to the approver with the evidence pack, never a promise.
3. Draft the customer message in company voice: what you verified, the
   decision framed as pending approval when approval is needed, the amount
   and timing stated plainly with refunds defaulting to the original payment
   method, and the one step the customer takes now. Stay inside the posted
   policy — it is legally enforceable. A denied ask gets the graceful no:
   "As much as I'd love to help..." plus the resource pointers that come
   closest. Never promise a refund, credit, waiver, or exception before the
   owner's yes in the same conversation.
4. Show the pack in chat — records, policy citation, amount, draft — take
   one round of edits, then stage it against the ticket for approval.

## Output

The evidence pack plus the staged customer draft. Offer to log it to the
ticket when connected, as a draft they approve first.

## When the records are thin

No policy passage means a draft built from precedent and the owner's word,
marked UNVERIFIED, with the approver named explicitly.
