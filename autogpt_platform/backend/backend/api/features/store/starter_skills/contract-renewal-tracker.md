---
name: "contract-renewal-tracker"
description: "Extract renewal and notice facts into an owned tracker, with source cites and no automatic renewal or cancellation action."
triggers: ["renewal tracker", "contract renewal dates", "auto renewal", "vendor contract calendar", "notice deadline"]
version: "1"
---

# Contract renewal tracker

Use signed agreements and amendments, not memory or a vendor sales summary.

## Extract one record per agreement

Record legal entities, vendor, service owner, budget owner, contract owner, start, end, initial term, renewal form and length, notice period, notice method and address, current committed spend, price-change rule, termination text, data-return need, source section, source file, and last checked date.

Calculate a notice date only when the source gives a clear end date and period. Show the formula. If amendments conflict or a trigger is unclear, mark `COUNSEL OR CONTRACT OWNER TO CONFIRM`; do not choose an interpretation.

## Prepare the review

Add service performance, open incidents, unused volume, alternative cost, implementation lead time, and decision owners. A review date is a planning prompt, not an automated cancellation or renewal.

## Output and gate

Return the tracker and next-action queue. Never send notice, renew, cancel, accept pricing, or change a calendar without owner approval. Route disputed language to counsel.
