---
name: "draft-a-follow-up"
description: "Use when a sent opener got no reply and the row needs its next touch in the sequence."
triggers: ["sales follow-up email", "prospect never replied", "next touch in the sequence", "bump a prospect", "re-engage a cold prospect", "follow up on my cold email", "nurture a no-reply prospect"]
version: "1"
---

# Draft a follow up

Run this when a sent opener got no reply and the row needs its next
touch. Only run on a row the user marked sent — never assume a
message went out.

## Inputs

The target-list row, the earlier touches with dates, and any new
public fact about the person or company since.

## Run the eight-touch cadence

Eight touches, multi-channel, over three to four weeks, front-loaded:
days 1, 3, 5, 7, 10, then spaced a week apart. Never stop before
attempt five. Rotate the channel per touch — email, call, social —
with a new angle on the same pain each time. Touch two lowers the
door with a smaller ask; middle touches add a useful link or a
referral ask; the last touch closes clean with a graceful no-pressure
goodbye, then the row recycles to the nurture pool for a 60-to-90-day
revisit. Any reply at any point ends the sequence.

## Draft rules

Say the send date on each draft. Each touch holds the no-invented-facts
rule and the user's voice. Each touch is shorter than the last; late
touches run two sentences. No emoji, no exclamation points, none of
the banned openers.

## Show and log

Show the draft in chat with one line on which touch it is and why
this angle. Mark the row with last_touch_on and next_step, log the
touch in the outreach log with its date, and save.

## Output

The unsent follow-up draft with its touch number, plus the row update.

## Fallbacks

Nothing new to say: say so and suggest waiting for a trigger instead
of burning the row. A reply that arrived mid-sequence: stop and switch
to Handle a reply.

## Approval gate

Drafts only, nothing sends without your explicit yes.
