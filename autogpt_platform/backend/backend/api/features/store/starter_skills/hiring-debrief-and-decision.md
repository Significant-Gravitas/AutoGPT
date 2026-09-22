---
name: "hiring-debrief-and-decision"
description: "Use after an interview loop finishes and the scorecards need collating, the debrief needs a running order, or the panel's read needs writing up."
triggers: ["run the hiring debrief", "collate the interview scorecards", "where did the panel land", "the panel disagrees on this candidate", "write up the loop outcome", "debrief running order", "missing scorecards after the loop"]
version: "1"
---

# Hiring debrief and decision

Run this after a loop finishes and the user needs the scorecards collated, the
debrief structured, or a written summary of where the panel landed.

## Inputs

The loop row, the scorecards that came in, and anything the panel said in chat
or mail.

## Check what is actually in

List who has filed a scorecard and who has not, and name the competency each
missing one covers. Scorecards are due within 24 hours of the slot with no
edits after. Under 90 percent in, say the debrief is a memory test and name
what cannot be called yet.

## Collate by competency, not by interviewer

So a split shows up as a disagreement about the work rather than a clash of
people. Under each competency: the rating, the evidence behind it in the
interviewer's own words, and one line on whether that evidence supports the
rating. Flag any rating with no evidence and quote what was written instead.
Label each line FACT — quoted evidence — or INFERENCE, with your reasoning
shown.

## Name what is still open

The two or three questions the panel still disagrees on, and what would settle
each one: a follow-up conversation, a work sample, a reference check.

## Give the user the running order

Thirty to forty-five minutes. Ratings written down before discussion so nobody
anchors, junior interviewers first, then the weakest signal, hiring manager
last. Calibrate the bar before the loop opens; a healthy bar decides clearly
four times in five.

## Write the summary afterwards

Where the panel landed per competency, the disagreement and how it resolved,
and the decision the user recorded. Record the decision they tell you, never
one you inferred. Strip anything that is not job-related evidence out of the
summary and say in one line when you removed something. Note the habit behind
any weak scorecard, with the quote that shows it.

## Output

The collated brief before the debrief, the running order, and the written
summary after. Update the candidate row with the stage and the outcome once
they tell you. Saved to the hiring folder with the date and the role.

## Fallbacks

With half the scorecards missing, collate what is in, name the missing
competencies, and say which questions cannot be called yet. Never fill a
missing scorecard with a guess.

## Approval gate

Never score the candidate yourself, never break a tie, and never tell the user
who to hire. You recommend with evidence; the human decides and records it.
Nothing goes to the candidate from this pass.
