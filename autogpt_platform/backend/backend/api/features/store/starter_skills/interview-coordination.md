---
name: "interview-coordination"
description: "Use when a loop needs booking, rebuilding, or rescheduling, when a candidate or interviewer message needs writing, or when the hiring tracker needs updating and reading for stalls."
triggers: ["schedule this interview loop", "reschedule the interview", "who is stuck in the hiring pipeline", "update the candidate tracker", "draft a note to the interviewer", "candidate availability windows", "what is stalled in hiring", "an interviewer dropped a slot"]
version: "1"
---

# Interview coordination

Run this when the user needs to book, rebuild, or reschedule a loop, needs a
candidate or interviewer message written, asks what is stuck, or hands over
roles and candidates to track.

## Inputs

The tracker, the role and stage, the panel with who covers what, candidate
availability with their timezone, interviewer constraints, and either
connected calendars or pasted availability. For tracking: a pasted list, an
export from the applicant tracking system (ATS), a spreadsheet, a CSV
(comma-separated values) file, a shared sheet link, or forwarded candidate
mail.

## Confirm the loop shape before you look for time

Slots, lengths, coverage, gaps. A four-slot onsite needs 10 minutes between
slots and one real break in the middle. Write every window in both timezones
and label which is which. Never show a time without a timezone.

## Offer two or three complete loop options

Best first, each with the date, slot times in both timezones, who takes which
slot, and the single tradeoff. On their pick, produce: the holds as a copyable
table or drafted invites; a short candidate email with the times, what each
conversation covers, and the full process in one line — every stage, who they
meet, the decision timeframe; and a panel note per interviewer with slot and
competency. Confirm that process map at the screen stage. Reschedules and
drop-outs keep the same competency coverage — name a replacement who covers it
or say the slot needs a new owner.

## Write messages in three to five sentences

Open with the answer or the ask, name the next step, give a date to answer by.
The shapes: scheduling; a nudge; a decline that is short and clear with no
invented reason and feedback only when the user gives real feedback; an offer
follow-up with the open question, the agreed deadline, and who to talk to; and
a keep-warm with one honest line on where the role stands. Reply within 24
hours at every stage, 48 at the outside; late-stage declines go by phone.
Match the user's voice when you have it. Times come from the tracker or the
user, never from you.

## Track in three lists

- Roles: role, level, hiring manager, target start, panel, stage, notes.
- Candidates: name, role, stage, source, owner, last contact, next step,
  waiting on, days in stage, notes.
- Loops: candidate, role, date, start and end, interviewers, coverage, status,
  scorecards in or out.

Keep their wording for titles, stages, and notes; normalise only dates and
times. Dedupe candidates on name plus role, loops on candidate plus date; on a
disagreement keep the newer row and say what changed. Unparseable rows go in a
short "needs a look" list. The tracker is the record, chat is not — re-read it
before a run and write it back after.

## Flag what is stalled

Unless the user set their own bar: a candidate waiting on feedback more than
two business days; a scorecard missing more than one day after the slot; an
offer out past the answer-by date; a scheduling request older than two
business days; a candidate quiet after three touches; an interviewer who has
not accepted inside 24 hours of the loop; a decline not drafted within 48
hours of the decision; any thread silent more than three business days with no
update sent.

Group by who holds it up, oldest first, one line each with the single
unblocking action, plus a drafted nudge per item. When the same item stalls
twice, name the person, the days lost, and what it costs the loop.

This is also the pass behind an on-demand morning hiring brief — today's
interviews with the interviewer per slot, who still needs scheduling oldest
first, and who is holding each item up — and behind a sweep for candidate or
interviewer mail that threatens a booked loop, which flags and drafts but
never replies.

## Protected attributes

Drop protected-attribute columns out of any export and say in one line that
you dropped them. Never guess contact details and never go looking for them.

## Output

Loop options, then holds plus candidate mail plus the panel note; drafts ready
to copy, or unsent in Gmail when it is connected; the tracker written back
with rows added and updated, the needs-a-look list, and today's picture in the
same reply. Saved to the hiring folder as roles, candidates, and loops files.

## Fallbacks

With pasted availability only, work from it and say so plainly. With no source
at all, ask for five example rows and build from those.

## Approval gate

Never book, invite, cancel, send, or decline without an explicit yes for that
specific action. Candidate details never go into a group channel.
