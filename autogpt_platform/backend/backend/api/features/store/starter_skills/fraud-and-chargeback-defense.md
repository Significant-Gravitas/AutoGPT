---
name: "fraud-and-chargeback-defense"
description: "Use when a fraud alert fires, a payment looks stolen, an account looks taken over, or a chargeback notice lands: verify, hold-or-clear, and defend with evidence."
triggers: ["fraud alert fired", "stolen payment", "account takeover", "chargeback notice", "friendly fraud", "put a hold on funds", "fight this chargeback"]
version: "1"
---

# Fraud and chargeback defense

Use this when a fraud alert fires, a payment looks stolen, an account looks
taken over, or a chargeback notice lands: verify, hold-or-clear, and defend
with evidence.

You need the alert or chargeback notice, the transaction records (amount,
method, date), the customer's history, and the fraud-hold authority from
prefs.

## Verify before acting

Confirm the amount, method, date, and account, each labeled FACT or UNKNOWN.
Pull history for velocity, new device or address, and past friendly-fraud
markers. A case with no transaction record is UNKNOWN, never "confirmed
fraud".

## Build the risk profile

Write one block with signals for and against, each with its source. Check
account-takeover signs (credential-stuffing pattern, email or password
changed just before spend) separately from first-party abuse (the customer
disputes their own real charge).

## Make the hold-or-clear call

Clear with a one-line reason when the evidence is thin, or stage a hold for
owner yes when money is still moving. Never hold funds, lock an account, or
accuse the customer before the owner says yes.

## Defend the chargeback

Look up the reason code first and match compelling evidence to that code: the
order receipt, delivery or usage proof, the customer's own messages, and the
policy quote with its source, plus a short rebuttal letter. File it only
after approval, then diary the representment date — the deadline for
resubmitting evidence to the card network. Track win rate per code;
after a loss, stage the second-chargeback or pre-arbitration path.

## Log it

Log the outcome on the case: cleared, held, or defended, with the evidence
IDs. Offer to draft the customer-facing line — plain, no accusation — when
the owner wants contact.

Deliver the risk profile, the hold-or-clear call, and the staged hold or
defense packet with its evidence list.

## What not to do

No transaction access means an UNKNOWN profile with the access need named,
not a padded verdict. Say which record would settle it.
