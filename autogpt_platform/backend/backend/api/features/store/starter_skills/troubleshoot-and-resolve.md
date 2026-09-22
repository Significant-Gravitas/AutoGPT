---
name: "troubleshoot-and-resolve"
description: "Use when something is broken: recreate it, find the root cause, ship relief, and file the defect."
triggers: ["customer reports a bug", "recreate the customer issue", "find the root cause", "ship a workaround", "file a defect", "broken feature in a ticket", "repro steps"]
version: "1"
---

# Troubleshoot and resolve

Use this when something is broken. Recreate it, find the root cause, ship
relief, and file the defect.

## What you need

The ticket with repro details, the account and order records, product docs or
runbooks, and logs or screenshots the customer sent.

## Diagnose and fix

1. Recreate on the exact customer path first: steps, environment, account
   state. Walk it yourself from the records before theorizing. No repro means
   you say which two questions or logs would reveal it and who asks, instead
   of guessing.
2. Set timeline expectations at the start of diagnosis — what you are
   checking and when the next update lands — then diagnose in order: account
   or data setup, recent change on either side, misuse or missed step, then
   product defect. Rank the causes with the one confirm step each and pick
   the one the evidence supports, saying what would prove a different one.
3. Ship relief: the workaround or fix steps the customer runs now, labeled
   workaround with its expiry when it masks a defect, then the permanent fix
   or the defect filed with minimal repro (steps, environment, expected vs
   actual), redacted of secrets, plus the dup check.
4. Show the diagnosis plus relief in chat and take one round of edits. Stage
   the defect and the ticket update for the owner and file them on their yes —
   nothing here writes to the tracker of record on its own.

## Output

The cause with evidence, customer relief steps, and the staged defect waiting
on the owner's yes. Offer
to draft the customer update when Gmail or Slack is connected, as a draft
they approve first.

## Fallbacks

No logs and no repro gets relief built from the most likely setup causes,
each step marked UNVERIFIED, with the two questions that would confirm it.
