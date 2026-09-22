---
name: "build-the-target-list"
description: "Use when the user first describes who they sell to, hands over companies or an export, or asks to add, drop, or widen names on the list."
triggers: ["target list", "prospect list", "build a list", "add companies", "ideal customer", "icp fit", "find prospects"]
version: "1"
---

# Build the target list

Use this when the user first describes who they sell to, hands over companies
or an export, or asks to add, drop, or widen names. Start from their ideal
customer plus whatever they have: a pasted list, a CRM or sales-tool export,
a link to a sheet, a conference or portfolio page, or nothing but the ideal
customer.

## The row shape

One row per person: prospect_id, company, company_url, person, title,
location, icp_fit, channel, contact, hook, hook_source_url, hook_date,
enriched_on, status, last_touch_on, next_step, notes.

Fixed values: icp_fit is strong, maybe, or weak. channel is email, social,
phone, or other. status runs new, enriched, drafted, approved, sent,
replied, meeting, qualified, no, or on hold.

## Start from companies

With only an ideal customer, find 10 to 20 matching companies on the public
web and say where each one came from. With a paste or an export, read the
columns back before you change anything.

## Score against the ideal customer

Score each company strong, maybe, or weak with one line of reason, naming the
trigger event when there is one — funding, hiring spike, new leader, tool
switch. Leave the weak ones out unless the user wants them. Personalize by
tier: strong rows get individual research, maybe rows get company-level
angles, weak rows get industry-level only.

## Find the people, never guess a contact

Find three or more people per company — buyer, champion, and user at minimum
— who hold the jobs the ideal customer names. Confirm the title on the
company's own site, the person's own profile, or a press page. An unconfirmed
title stays blank and the row stays at new. Never build an email address from
a pattern and never assume a profile URL from a name; a contact goes in a row
only when you found it published and can link to it.

## Deduplicate and show

Deduplicate on person plus company, then on company URL. Drop anyone on the
do-not-contact list and anyone already an inbound lead, open deal, or
customer, and say who you dropped. Show the first ten rows in chat with the
fit reason, take one round of edits, then save the whole list.

## What you hand back

The scored target list saved to file, with fit reasons and the dropped names
named. Stop the first build at 25 rows so they can check your aim before you
go wide, and edit the list in the same turn they add or drop a name.

## Fallbacks

An export that will not parse means you ask them to paste the rows. A company
with no public team page gets a note asking them to name the person.
