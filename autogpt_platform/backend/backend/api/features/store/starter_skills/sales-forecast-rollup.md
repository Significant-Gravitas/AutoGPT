---
name: "sales-forecast-rollup"
description: "Build a bottom-up forecast from open deals, with weighted and rep-called figures kept apart."
triggers: ["sales forecast", "forecast rollup", "pipeline forecast", "quarterly forecast", "commit vs best case"]
version: "1"
---

# Sales forecast rollup

Use this for a quarter or month, from a dated export of open deals.

## Check the deal data

Record the export date, period, currency, and deal count. List deals with no
amount, no close date, a close date in the past, no owner, or more than one
currency. Keep them in the count and mark them; do not fill the gaps.

## Build the rollup

For each deal show owner, stage, amount, close date, days in stage, and the
rep's call (commit, best case, pipeline). Then give:

- the weighted figure from stage probabilities, with the probabilities stated;
- the rep-called commit and best case, kept separate;
- totals by owner and by stage;
- the gap between weighted and called, with the deals that drive it.

Every total must trace back to the deals in the list.
Keep totals separate by currency. Combine them only when the owner supplies
or approves the exchange rate, rate date, and source; state all three beside
the converted total.

## Risks and stale deals

List deals past their close date, deals in one stage well past the median, and
large deals with no recent activity. Give their count and value. Leave them in
the forecast and mark them; whether to pull them is the sales lead's call.

Never adjust a number to reach a target, and do not hide stale deals. Do not
change a deal's stage, amount, or date in the CRM yourself.
