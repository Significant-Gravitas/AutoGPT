---
name: "policy-gap-review"
description: "Compare written policies with a named framework's control list and show gaps with evidence."
triggers: ["policy gap", "SOC 2 readiness", "ISO 27001 gaps", "policy review", "control mapping"]
version: "1"
---

# Policy gap review

Use this to see how written policies line up with a framework before an audit.

## Fix the framework and scope

Name the framework and version, and the control list the user supplied. Agree
the scope: which systems, teams, and period. Record the policies in the review
with their versions, owners, and approval dates.

## Map policy to control

For each control show:

- the control id and a short plain summary;
- the policy and section that covers it, if any;
- evidence that the control operates, such as a log, ticket, or report, with
  its date;
- a label: **covered**, **partly covered**, **not covered**, or **no evidence**.

A policy that says a thing happens is not evidence that it does. Keep "written"
and "shown working" apart.

## Gap list

List each gap with the control, what is missing (policy text, evidence, or
both), the owner, and a suggested order based on audit dates. Give counts for
each label.

Never say the company meets a framework or is ready for an audit, and do not
claim a control without evidence. This is not an audit opinion or legal advice.
Draft for the owners to review.
