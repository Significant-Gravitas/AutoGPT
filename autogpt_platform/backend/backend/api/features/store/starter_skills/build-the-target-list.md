---
name: "build-the-target-list"
description: "Use when the user first says who they sell to, hands you companies or an export, or wants names added to or cut from the prospect list."
triggers: ["target list", "prospect list", "who should I sell to", "add these companies", "icp fit", "find prospects", "widen the list"]
version: "1"
---

# Build the target list

Every prospecting step after this one reads these rows, so build the list
narrow and right before you build it big.

## Where the companies come from

If all you have is an ideal customer, seed 10 to 20 fitting companies from
the public web and cite the page that surfaced each one. A paste, a HubSpot
or Apollo export, or a Google Sheet starts with a readback: name the columns
you see and how each maps to the row below, before you edit anything.

## The row

One person per row. Other skills read these exact names:

- Who: prospect_id (a key you never reuse), company, company_url, person,
  title, location.
- Fit: icp_fit, one of strong, maybe, or weak.
- Reach: channel (email, social, phone, or other) and contact.
- Why now: hook, never without hook_source_url and hook_date; enriched_on
  dates your last check.
- Progress: status, drafted_on (set when a first touch is written),
  last_touch_on, next_step, notes.

Statuses go new, enriched, drafted, approved, sent, replied, meeting,
qualified; no or on hold closes a row. You set only new, or enriched once
the title is confirmed and the hook is sourced and dated.

## Grade the company first

Mark each company strong, maybe, or weak with a one-line reason in notes,
naming the trigger when there is one: funding, a hiring burst, a new buying
leader, a switch off a tool you replace. Example: "strong: Series B in
August (link), six open ops roles, still on spreadsheets." Fit with no
trigger is a maybe; a trigger never makes a poor fit strong. Keep weak
companies off unless they ask.

The grade sets research depth: strong rows earn a hook about the person,
maybe rows a company-level angle, weak rows an industry line at most.

## Three seats per company

Find at least three people per company, each in a role the ideal customer
names: the buyer who signs, a champion who would push for you inside, and a
user who feels the problem. Confirm each title on a page you can link: the
company website, their own profile, or a press release. An unconfirmed
title stays empty and holds the row at new. Name any empty seat rather than
stretch a nearby title.

## Contacts are found, never made

Never build an email from a naming pattern or guess a profile URL from a
name. The contact field holds only what you found published and can link;
otherwise it stays blank.

## Clean, preview, save

Collapse duplicates by person and company first, then by company URL. Strike
anyone the do-not-contact list names, plus anyone already an inbound lead,
open deal, or customer (check HubSpot, or ask), and name each person
you struck. Show ten rows with fit reasons, let them edit once, then save.
Nothing sends or reaches the CRM without their yes. Cap the first pass at 25
rows so they can judge your aim, and go wider when asked. Later adds and
drops land in the same turn.

## What you hand back

The saved, graded list with a reason on every row, who you struck and why,
and the seats or titles still open.

## Fallbacks

An export you cannot read cleanly: ask for a paste. No public team page:
the row stays new and you ask them who holds the seat. Under ten matches:
show them and name the constraint to loosen. No do-not-contact list given:
ask once before saving.
