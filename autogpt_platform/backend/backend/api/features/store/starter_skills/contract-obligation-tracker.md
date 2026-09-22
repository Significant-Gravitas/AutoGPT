---
name: "contract-obligation-tracker"
description: "Turn signed contract duties into an owned, source-cited tracker while routing unclear obligations to counsel."
triggers: ["contract obligations", "obligation tracker", "post signature duties", "contract deliverables", "notice obligations"]
version: "1"
---

# Contract obligation tracker

Use the final signed agreement and all signed amendments. Drafts do not create tracker duties.

## Extract one action per row

Record the responsible party, action, beneficiary, due date or trigger, recurrence, notice method, required evidence, dependency, business owner, status, document and section, exact source text, and last checked date.

Use these statuses: `NOT STARTED`, `IN PROGRESS`, `EVIDENCE READY`, `OWNER CONFIRMED COMPLETE`, `BLOCKED`, and `UNCLEAR`. Ellis does not mark completion without owner evidence.

Calculate a date only where the trigger and period are clear; show the formula. Keep continuing duties separate from one-time duties. Link amendments to the rows they change rather than overwriting source history.

## Limits

Do not decide who legally bears an unclear duty, whether performance satisfies the contract, whether a breach occurred, or whether notice is valid. Route those questions to counsel with the source text.

## Output

Return the tracker, upcoming owner actions, missing evidence, and unclear items. Do not send notices, certify completion, waive a duty, or change a deadline.
