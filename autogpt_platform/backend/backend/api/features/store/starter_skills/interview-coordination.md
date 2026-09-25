---
name: "interview-coordination"
description: "Use when an interview loop has to be booked, rebuilt, or moved, when a candidate or interviewer needs a message, or when the hiring tracker needs updating or checking for stalls."
triggers: ["schedule this interview loop", "reschedule the interview", "who is stuck in the hiring pipeline", "update the candidate tracker", "draft a note to the interviewer", "candidate availability windows", "what is stalled in hiring", "an interviewer dropped a slot"]
version: "1"
---

# Interview coordination

Run this when a loop must be booked, rebuilt, or moved; when a candidate
or interviewer needs a message; when the user asks what is stuck; or when
they hand over roles and candidates to keep track of.

## Inputs

The tracker; the role and stage; the panel and the competency each person
owns; the candidate's availability and timezone; interviewer constraints;
and connected calendars or pasted availability. To start or rebuild the
tracker, any one source will do: a pasted list, an export from their
applicant tracking system (ATS), a spreadsheet or CSV (comma-separated
values) file, a link to a shared sheet, or candidate emails they forward.

## Confirm the loop shape before you look for time

Agree the number of slots, their lengths, what each covers, and where the
breathing room goes. A four-slot onsite gets 10 minutes between interviews
plus a proper break near the midpoint. Every window carries two labelled
times, the candidate's and the user's; a time with no zone never appears.

## Offer two or three complete loop options

Strongest first. Each lists the date, every slot in both timezones, who
interviews in each, and the one compromise it makes, such as a tight
turnaround or a stand-in interviewer. Once the user picks, hand over three
things at once: holds laid out in a copyable table (or drafted invites if
Google Calendar is connected); a brief email to the candidate covering
times, what each conversation is about, and the whole process in one line
(every stage, who they meet, when a decision comes; set this map at the
screen stage and keep to it); and a note per interviewer with their slot
and competency.

If an interviewer drops out or the loop moves, the competency still needs
covering: name a substitute who can take it, or tell the user the slot has
no owner yet.

## Write messages in three to five sentences

Lead with the answer or the request, state the next step, and give a reply
date. The shapes:

- Scheduling: the options and what each conversation covers.
- Nudge: what is outstanding, since when, and when it is needed.
- Decline: brief and plain, no made-up reason, feedback only when the user
  hands you real feedback. Late-stage declines happen by phone, so draft
  talking points instead.
- Offer follow-up: what is still unresolved, the agreed deadline, and who
  the candidate can talk to.
- Keep-warm: a truthful line on the role's status and when they will hear
  next.

Candidates hear back within 24 hours at every stage, 48 at most. If the
user has shown you how they write, sound like them. Times come from the
tracker or the user, never from you.

## Track in three lists

These column names are fixed; other skills read them.

- **Roles**, one row per open req: role, level, hiring manager, target
  start, panel (who interviews), stage, notes.
- **Candidates**, one row per person per role: name, role, stage, source,
  owner (who moves them forward), last contact, next step, waiting on (a
  person, not a task), days in stage, notes.
- **Loops**, one row per scheduled interview: candidate, role, date, start
  and end, interviewers, coverage, status, scorecards in or out.

Titles, stage names, and notes stay in the user's wording; only dates and
times get normalised. Candidate rows match on name plus role, loop rows on
candidate plus date; when two copies disagree, the newer wins and you say
what changed. Unparseable rows go on a short "needs a look" list. The
tracker outranks anything said in chat: read it fresh at the start of each
run and save your changes at the end.

## Flag what is stalled

Default thresholds, unless the user set their own:

| What | Stalled once |
|---|---|
| Candidate awaiting feedback after an interview | 2 business days |
| Scorecard not filed after the slot | 1 day |
| Offer still open | past its answer-by date |
| Scheduling request with no loop booked | 2 business days |
| Candidate with no reply | 3 touches |
| Interviewer invite not accepted | inside 24 hours of the loop |
| Decline not drafted after the decision | 48 hours |
| Any thread with no update sent | 3 business days |

Group by whoever holds each item, oldest first, one line each with the
single action that frees it, plus a drafted nudge. An item stalling a
second time gets escalated: who holds it, the days it has cost, and what
that does to the loop.

The same read powers the on-demand morning hiring brief (today's interviews
with each slot's interviewer, who still needs scheduling by longest wait,
and who each open item waits on) and the sweep of candidate and interviewer
mail for anything that endangers a booked loop. The sweep flags and drafts;
it never replies.

## Protected attributes

Leave any protected-characteristic column (age or birth date, gender, race,
nationality, religion, disability, marital or family status) out of an
export and tell the user in one line. Contact details are recorded as
given; never guess an email or phone number, and never go hunting for one.

## What needs their yes

Booking, inviting, cancelling, sending, and declining each wait for the
user to approve that one action. Candidate details stay out of group
channels.

## What you hand back

Ranked loop options, then the holds, candidate email, and panel notes once
one is picked, ready to copy or sitting unsent in Gmail when connected. The
tracker written back, with rows added and updated counted, the needs-a-look
list, and today's snapshot in the same reply. All saved to the hiring
folder as roles, candidates, and loops files.

## Fallbacks

Only pasted availability to go on: use it and tell the user that is what
the options rest on. No list of any kind yet: ask them to type five sample
rows and start the tracker from those.
