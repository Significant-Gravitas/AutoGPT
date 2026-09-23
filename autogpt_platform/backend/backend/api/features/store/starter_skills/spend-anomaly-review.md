---
name: "spend-anomaly-review"
description: "Explain material month-over-month vendor spend changes from invoices, orders, usage, currency, and coding evidence."
triggers: ["spend anomaly", "vendor spend increased", "month over month spend", "invoice variance", "unexpected vendor charge"]
version: "1"
---

# Spend anomaly review

Use source records for two comparable periods: invoices, approved orders, usage or seat reports, credits, taxes, currency rates, and ledger coding.

## Reconcile the change

For each vendor, calculate prior spend, current spend, absolute change, percentage change, and materiality against the threshold the owner supplies. When prior spend is a recorded zero, show percentage change as `N/A` and judge materiality on the absolute change. Decompose the change into:

- price or rate;
- volume, seats, or usage;
- new or ended service;
- one-off fee or credit;
- tax or currency;
- timing or duplicate posting;
- coding change;
- unexplained balance.

Show formulas and source lines. Compare the same currency and period. Never treat missing data as zero.

## Investigate safely

Write a ranked list of checks that would confirm each unexplained amount. Do not allege fraud, misconduct, overbilling, or contract breach without proof. Do not alter a ledger, dispute a charge, stop payment, or contact a vendor.

## Output

Return the variance table, confirmed drivers, unexplained balance, evidence needed, and owner for each next check.
