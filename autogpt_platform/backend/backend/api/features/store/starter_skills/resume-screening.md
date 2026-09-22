---
name: "resume-screening"
description: "Screen resumes against an approved job rubric, citing evidence and leaving human advance or reject decisions open."
triggers: ["screen resumes", "review applicants", "resume shortlist", "candidate screen", "application review", "screen these resumes", "triage the inbound applications", "does this candidate clear the bar", "evidence matrix for this role"]
version: "1"
---

# Resume screening

Use only when the approved role rubric is present. If it is missing, stop and build it first.

## Inputs

The approved role rubric, the resumes — pasted, uploaded, in Drive, or exported from the applicant tracking system (ATS) — and the hiring manager's screen questions if they have any.

## Prepare the record

Hide names, photos, addresses, graduation dates, and other non-job data when the source lets you. Whether or not it does, never copy them into the record or an `EVIDENCE FOUND` passage; quote only the job-related text. Never infer age, race, ethnicity, nationality, religion, sex, gender, sexual orientation, disability, health, pregnancy, family status, or any other protected trait.

## Screen criterion by criterion

For each rubric criterion, record:

- `EVIDENCE FOUND` with the resume passage and its location;
- `EVIDENCE MISSING` when the document does not state it;
- `CONFIRM IN INTERVIEW` when the claim lacks scope, ownership, or result.

Do not turn a gap into a negative fact. Do not infer skill from school or employer prestige, a name, dates, location, writing style, or time away from work. Do not compare candidates with each other; compare each record with the same rubric.

## Note red flags as evidence, never as auto-rejects

An unexplained gap, shrinking scope, a run of very short tenures, or a skill claimed with no artifact behind it is one quoted line under `CONFIRM IN INTERVIEW`, not a rejection. The named human owner weighs it; the screen never does.

## Fallbacks

A resume you cannot parse goes on a short "needs a look" list and you move on — never guess its evidence. If many records miss the same must-have, that is a signal the bar or the posting is off: say so as a process-quality note and propose the exact edit, applying it only on the owner's yes.

## Output

Return one evidence matrix per candidate, the open verification questions each one raises, and the batch counts (reviewed, evidence-complete, needs interview, needs a look), plus a process-quality note if the rubric yields too many unknowns. Do not rank candidates or recommend advance, reject, hire, or compensation decisions. The named human owner decides and records the reason.

## Approval gate

Drafts only. Never reject a candidate, send a decline, or move anyone's stage in the tracker of record without the owner asking for that specific action.
