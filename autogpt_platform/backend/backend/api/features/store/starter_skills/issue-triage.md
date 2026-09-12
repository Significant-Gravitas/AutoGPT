---
name: issue-triage
description: Triage an incoming issue - decide whether it is actionable, a duplicate, a question, or not a bug, and draft the reply it needs. Use when asked to triage issues, work the issue backlog, check if a report is a duplicate, or judge whether a bug report is actionable.
triggers:
  - triage this issue
  - is this a duplicate
  - work the issue backlog
  - is this actually a bug
version: "1"
---

# Triaging an incoming issue

Sort into one of five outcomes. Pick the first that fits.

## 1. Not reproducible as written

The report describes a failure but not the path to it. Missing: version,
environment, the input, or what was expected versus what happened.

Do not guess and do not close. Ask for the smallest set of missing facts, in
the reporter's own terms — quote the part of their report you are building on
so it reads as engagement, not a form letter. Ask for at most three things.

## 2. Duplicate

Search open **and closed** issues before deciding. Closed matters: a
recurrence of a closed bug is not a duplicate, it is a regression, and that
is a different and more urgent thing. Distinguish them by whether the closing
fix is present in the reporter's version.

Link the original, say which it is, and say what the reporter should watch.

## 3. Question or documentation gap

The behaviour is working as designed and the reporter did not know. Answer the
question, then ask whether the docs should have said so — a question that
reached the issue tracker usually means the documentation failed, and that is
a real finding worth its own issue.

## 4. Not a bug

The behaviour is intentional and correct. Say so plainly, show the reasoning
or link the decision, and acknowledge the expectation that led them here. Do
not soften it into ambiguity — an unclear "not a bug" gets reopened.

## 5. Actionable

Reproducible, novel, and something should change. Then produce:

- **A one-line statement of the defect** — inputs, then wrong outcome.
- **Where it lives**, if you can find it. Search for the symptom, not the
  reporter's guess at the cause; reporters routinely misattribute.
- **Severity by consequence**: data loss or a security boundary, then broken
  for all users, then broken for some, then cosmetic. Not by how annoyed the
  reporter is.
- **What it blocks**, if anything.

## Reading reports fairly

- A terse or frustrated report can still be a real bug. Judge the content.
- A detailed, confident report can still be wrong. Verify the claim rather
  than inheriting its framing.
- If the reporter proposes a fix, evaluate the *problem* first. The proposed
  fix is evidence about the problem, not the thing being triaged.
- A report you cannot verify is not thereby invalid — say what you checked and
  what you could not.

## Never

Do not close, label, assign, or comment on an issue unless asked for that
exact action. Produce the verdict and the draft reply; a human posts it.
