---
name: "statement-reconciliation"
description: "Reconcile a bank or card statement to a ledger with control totals, matched items, timing differences, and owned exceptions."
triggers: ["reconcile statement", "bank reconciliation", "card reconciliation", "statement mismatch", "unmatched transactions"]
version: "1"
---

# Statement reconciliation

A reconciliation explains every difference between two records for one account
and one period. It does not force them to agree.

## Set the controls

Record the entity, account, currency, statement period, statement opening and
closing balances, ledger opening and closing balances, and export cutoff. Keep
pending and posted items separate. Confirm the sign convention before matching.

## Match in passes

1. Match exact amount, date, and reference.
2. Match known timing differences within a stated date window.
3. Match batches only when their components sum exactly and the source gives a
   shared reference.
4. Review fees, interest, refunds, reversals, transfers, and foreign-currency
   settlements separately.
5. Leave every uncertain pair unmatched.

Never match on amount alone when duplicates exist. Keep a link or row identifier
for both sides of every match.

## Return the reconciliation

Show:

- opening balance on both records;
- total debits and credits or charges and payments;
- closing balance on both records;
- matched count and value;
- valid timing differences with dates;
- statement-only and ledger-only items;
- duplicates or possible duplicates;
- unexplained difference.

Use the equation that fits the account and show it in the report. A complete
result has an unexplained difference of zero. If it does not, label it open and
assign each item an owner and next step.

## Safety rules

Do not create, delete, merge, or edit transactions. Do not propose an adjusting
entry unless the source and approved policy support it; present it as a draft
for accountant review. Escalate unknown withdrawals, changed bank details,
duplicate payments, large or repeated differences, and suspected fraud at once.
