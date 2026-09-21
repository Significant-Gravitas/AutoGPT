---
name: "lost-deal-analysis"
description: "Analyse lost deals by recorded reason, stage, segment, and age, with gaps in the reasons shown."
triggers: ["lost deals", "win loss", "why are we losing", "closed lost", "loss reasons"]
version: "1"
---

# Lost deal analysis

Use this for closed-lost deals over a stated period, from a dated export.

## Check reason quality

Count lost deals and their value. Then count how many have a loss reason, how
many use "other" or a free-text note, and how many reasons were added long
after the close date. If more than a quarter lack a usable reason, say so
first; the breakdown will be weak.

## Break down the losses

Break losses down by:

- recorded reason;
- stage at loss;
- segment, region, and deal size;
- age at loss and days since last activity;
- owner, only where the count is large enough to mean something.

Compare with won deals over the same period where you can. Quote rep notes or
customer words only as written, with the deal id.

## What the data cannot say

State what the records cannot show: reasons picked from a short list, deals
lost before they were entered, and reps who log reasons differently. Suggest
the smallest next check, such as five calls with lost buyers.

Never fill a blank loss reason with a guess or infer one from the amount. Do
not edit deals in the CRM; propose changes to the reason list for approval.
