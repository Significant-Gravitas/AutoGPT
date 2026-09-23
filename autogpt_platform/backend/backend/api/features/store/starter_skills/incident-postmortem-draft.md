---
name: "incident-postmortem-draft"
description: "Draft a blameless incident review with an evidence-based timeline, causes, and owned follow-ups."
triggers: ["postmortem", "incident review", "outage report", "root cause analysis", "what went wrong"]
version: "1"
---

# Incident postmortem draft

Use this after an outage, data issue, or failed release.

## Build the timeline

Collect alerts, logs, deploys, chat messages, and status updates with
timestamps. Build a timeline in one time zone: when it started, when it was
noticed, key decisions, when it was fixed, and when it was confirmed fixed.
Cite the source for each entry and mark gaps. Add the impact: who was
affected, for how long, and what data or money was involved.

## Causes and factors

Separate the trigger from the conditions that let it cause harm: missing
tests, alerts, limits, docs, or rollback. For each, give the evidence and how
sure you are. If the cause is not proven, say so and list what would prove it.
Describe what systems and steps did, not what people failed to do.

## Follow-ups with owners

For each follow-up give the action, the problem it prevents, the owner, and a
due date. Prefer fixes to systems over reminders to people. Keep the list
short enough to finish.

Never name a person as the cause, invent a timeline entry, or claim a root
cause without evidence. Do not publish the review or close follow-ups; the
incident owner approves it.
