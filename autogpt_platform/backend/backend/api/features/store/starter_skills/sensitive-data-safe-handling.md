---
name: "sensitive-data-safe-handling"
description: "Use before card, personal, or health data moves: redact PCI, handle PII and PHI by the regime, and contain spills fast."
triggers: ["redact this transcript", "card data in a ticket", "PII in this draft", "health data in a case", "data spill", "safe handling check", "pause recording for card entry"]
version: "1"
---

# Sensitive data safe handling

Use this before card, personal, or health data moves: redact PCI card
data, handle PII and PHI (protected health information) by the regime,
and contain spills fast.

You need the draft, transcript, or record to check, the regimes and
redaction tool from prefs, and the vault or safe-store path.

## Name what is in the text

Call out card numbers and CVVs, government IDs, passwords and tokens,
health details, and any other identifiers. Quote each finding with its
line; a scan you cannot run is UNKNOWN, never "clean".

## Redact before anything moves

Mask card data to last four, strip CVVs and full credentials entirely,
and minimize PII to what the case needs. Fix what a reword can fix and
mark each change; route what it cannot — data already exposed, a
required record that must keep identifiers — to the compliance owner
with a short packet.

## Stop capture before live card data

On payment calls, pause recording or mask keypad (DTMF) entry before
collection, and resume after authorization. Never store sensitive
authentication data — CVV, PIN, or magnetic-stripe track data — in audio,
screen recordings, or transcripts; the recording itself becomes the
violation. Redact any such slip from transcripts immediately.

## Check the PHI handling path too

Confirm minimum-necessary access (except treatment, disclosures to the
individual, authorized uses, disclosures to the regulator, or as
required by law), identity verified before any PHI talk, encrypted
store, no health detail in ticket subjects or public replies, and
6-year retention on the record. A PHI finding outside the safe path is P1
until contained.

Stamp the result: clean, redacted, or blocked-with-owner. Blocked items
never move before the compliance yes. Log regimes, verdict, and
approver on the case.

After a spill, draft the containment note: what leaked, where, who saw
it, the purge or rotate steps, and the prevention fix. Offer to
re-check the macro or template source so the next use starts clean.

Deliver the findings list, the redacted text or the owner packet, the
verdict stamp, and the case log line.

## What not to do

No redaction tool means manual masking with a second pair of eyes
required, and the tool gap named. Never declare text clean by
skimming.
