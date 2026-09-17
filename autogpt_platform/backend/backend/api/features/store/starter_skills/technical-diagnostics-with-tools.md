---
name: "technical-diagnostics-with-tools"
description: "Use when the breakage needs proof from the systems, not guesses: query the account, check the API and logs, and hand engineering a complete repro."
triggers: ["check the logs for this ticket", "customer hit an API error", "pull the request ID", "query the account record", "prove the defect with data", "engineering repro packet", "tool-backed diagnosis"]
version: "1"
---

# Technical diagnostics with tools

Use this when the breakage needs proof from the systems, not guesses:
query the account, check the API and logs, and hand engineering a
complete repro.

You need the ticket with repro details, read-access notes for the data
store, API docs or runbooks, and any logs, screenshots, or request IDs
the customer sent.

## Repro on the customer path first

Recreate exactly as Troubleshoot and resolve runs it. No repro means
you stop and name the two questions or logs that would reveal it —
queries never replace the repro.

## Check the systems in order

Work through account and data state via the store lookup, recent
changes on either side, the API or integration call with its request
ID and response, auth and SSO state when login fails, and error logs
around the incident time. Quote what each check returned with its
source and time; a check you cannot run is UNKNOWN with the access it
needs.

Rank the causes with the one confirm step each, pick the one the
evidence supports, and say what would prove a different one. Ship
customer relief labeled workaround with its expiry when it masks a
defect.

Show the diagnosis plus the checks table in chat, take one round of
edits, then file it against the ticket with the defect link, request
IDs, and date.

Deliver the cause with tool-backed evidence, customer relief steps,
and the filed defect. Offer to file it as an issue when connected, as
a draft they approve first.

## What not to do

No store or log access means relief from the most likely setup causes,
each step marked UNVERIFIED, with the access grant that would confirm
it named.
