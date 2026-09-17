---
name: "resume-screening"
description: "Use when inbound resumes pile up, the user asks whether a candidate clears the bar, or an advance needs a screen-call plan."
triggers: ["screen these resumes", "does this candidate clear the bar", "review the applicants", "plan a screen call", "resume verdicts for this role", "triage the inbound applications", "who should we advance"]
version: "1"
---

# Resume screening

Run this when inbound resumes pile up, the user wants a screen-call plan, or
asks whether a candidate clears the bar.

## Inputs

The role scorecard, the resumes — pasted, uploaded, in Drive, or exported from
the applicant tracking system (ATS) — and the hiring manager's screen
questions if they have any.

## Score every resume against the same bar

Each must-have gets Met, Partial, or Missing, with the quoted line or the gap
behind it. Nice-to-haves break ties only. Disqualifiers get checked in the
candidate's own evidence, never assumed.

## Red flags are notes, never auto-rejects

Unexplained gaps, shrinking scope, last three roles all under 12 months,
skills claimed with no artifact behind them. One line each, quoted where
possible.

## Screen for bias in your own read

First pass anonymised: cover name, photo, school, and dates, then score the
must-haves. Judge the work artifact, not the logo, the school, or the name.
Never infer age, gender, race, nationality, or family status, and never let a
photo, a gap, or a non-linear path decide anything on its own.

## Give every candidate a verdict

Advance, Hold, or Decline, with the two strongest evidence lines and the one
open question a screen call should close.

## Write the screen plan for every Advance

Thirty minutes: the open question first, then the two must-haves the resume
proves least, then what they want next, then the close with timeline and next
step. Five questions maximum, each with the signal it tests.

## Report the batch honestly

Reviewed, advanced, held, declined, with the top two decline reasons in plain
words. Three declines for the same reason means the bar or the posting is
wrong, so propose the exact edit and apply it only on a yes.

## Output

One verdict block per candidate — verdict, evidence, open question — the
screen plan for each Advance, and the batch counts. Saved to the hiring
folder.

## Fallbacks

With no scorecard, screen against the posting and mark the read a draft
pending calibration. With a resume you cannot parse, put it in a short "needs
a look" list and move on; never guess a verdict.

## Approval gate

Drafts only. Never reject a candidate, send a decline, or move anyone's stage
in the tracker of record without the user asking for that specific action.
