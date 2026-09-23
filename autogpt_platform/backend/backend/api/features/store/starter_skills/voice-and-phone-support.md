---
name: "voice-and-phone-support"
description: "Use when a support case needs a call: place or review calls, run callbacks to kept promises, tune the phone menu, and coach talk tracks from recordings."
triggers: ["call this customer back", "run the callback list", "IVR menu tuning", "phone talk track", "review a call recording", "coach this call", "phone queue tuning"]
version: "1"
---

# Voice and phone support

Use this when a case needs a call: place or review calls, run callbacks to
kept promises, tune the IVR (the automated phone menu), and coach talk
tracks from recordings.

You need the case or callback promise, the voice stack and talk tracks from
prefs, and the recording or transcript when reviewing.

## Stage the call plan

Before any call, stage who, why, the two outcomes that count as resolved,
and the talk track that fits. Never dial before the owner says yes to that
specific call; a scheduled callback keeps its promise time or gets
re-promised before it slips. Route each callback to the specialist who owns
the issue so it resumes where the IVR left off.

## Run the call in three beats

Open with the recording disclosure played before anything else (rule follows
caller location, one-party vs two-party) logged as call metadata, plus
purpose and agenda, and verify identity before account or health-data talk.
Work the issue with the matching skill while narrating next steps. Close
with the receipt (what was done, reference ID) and the confirm line. At the
payment moment, pause recording or mask the keypad tones before card data and resume
after authorization; never read a CVV aloud, and never store sensitive
authentication data — CVV, PIN, or magnetic-stripe track data — in audio,
screen, or transcript. Redact any slip immediately. Keep after-call
work inside the target: log, disposition, and follow-up draft before the
next call.

## Run callbacks to kept promises

Offer the choice with numbers: estimated wait, estimated callback time,
callback or hold — caller's pick. Never offer a callback for urgent calls
(card cancellation, theft, crisis) or when the wait is under ~5 minutes;
those go straight to the next agent. Work oldest promise first: verify the
number and timezone, call only 8a-9p caller-local after a do-not-call list
check, redial
once on no-answer, then leave the short message and re-promise once. A
callback past its time is P2 minimum — say so and move it first.

## Tune IVR and queue paths

Read the path the caller walked: containment vs escape, drop points, repeat
callers, plus abandon rate, task completion rate, and transfer rate. Judge
the menu against the checklist: agent escape always offered, main menu
<=30s, greeting <=8s, customer vocabulary, action-first prompts, back-to-menu
option, all IVR-collected info passed to the agent, selections and
transactions confirmed. Draft menu or routing fixes with the expected effect
stated plainly; publish nothing before owner yes.

## Coach from recordings

Calibrate scoring across reviewers first, then score the call against the
talk track with quoted lines: open, diagnosis, hold handling, close. Run
automated coverage across 100% of calls for disclosure, consent, and
prohibited-language gaps; use sampled human QA for calibration and edge
cases, assessing value created, not activity. One fix per rep per week, with
the example line to use next time. Log the call, the score, and the coach
note.

Deliver the staged call plan or placed-call log, the callback state, the IVR
fix draft, and the coached scorecard.

## What not to do

No recorder or transcript access means coaching from notes only, marked
INFERENCE, with the access need named. Never score a call you cannot hear or
read.
