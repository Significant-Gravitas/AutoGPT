---
name: "nurture-sequence-build-and-readout"
description: "Build a nurture around one segment: the journey and the exit that stops it, the drafts, the experiment that decides whether it worked, the pre-send checks, and the incremental readout afterwards."
triggers: ["build a nurture", "re-engagement series", "journey map", "sequence experiment", "holdout", "sequence readout", "segment this list"]
version: "1"
---

# Nurture sequence build and readout

Use this when someone wants a sequence, a lifecycle journey or a re-engagement
series for a specific segment. Deciding which emails should exist at all is the
lifecycle map's job; this builds the journey around them and judges it
afterwards.

## Map the journey before you write

Name the stage this segment sits in, the one action that moves them out of it,
and the touches that earn it:

- Three to five for a short run at a warm segment.
- Six to eight for a cold one.
- A twelve-week program for ongoing warm nurture.

Then name the exit in one line: what stops the sequence for someone who
converts. A sequence with no exit does not ship.

## Segment tight

One sequence per segment and stage. A blast wearing a nurture costume is still
a blast. Write who is in and who is out, one line each, and say how you know
which is which.

## Draft each email

One goal, one ask, a subject under 50 characters, a first line that does real
work. Every claim carries a number or a source or it gets cut. Missing facts
stay marked in the draft rather than getting filled in.

## Set the experiment before anything sends

- The control against the variant, changing one thing.
- The single metric that decides it.
- A holdout when the list is big enough to afford one.
- The sample you need and how long to wait for it.

Decide this first. An experiment designed after the results is a story.

## Check the build, then the domain

Links, names, dates, the unsubscribe line and the reply-to address. Read every
draft back once looking for the segment's name in the wrong place. Then the
domain side: Sender Policy Framework (SPF), DomainKeys Identified Mail (DKIM)
and Domain-based Message Authentication (DMARC) all passing, spam-score and
rendering previews clean, and bounce and complaint watch switched on before the
first send rather than after it.

## Launch on a yes, then report incrementally

Delivered, bounced, clicked, replied, converted, meetings booked, and the lift
against the holdout or the prior baseline. Opens are a deliverability health
check only — mail privacy features preload images and inflate them, so nothing
gets scored on opens. Judge rates against benchmarks for the send type: cold,
warm and customer sends never share a target. Name the send time and frequency
behind each touch. Label every line FACT, INFERENCE or UNKNOWN.

## What not to do

Never send, schedule or upload a list without a yes. Never invent an open rate,
a click rate or a conversion to fill a readout. Never re-send to an address
that has bounced twice — suppress it.
