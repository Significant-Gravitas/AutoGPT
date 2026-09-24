---
name: "release-notes-draft"
description: "Write customer-facing release notes from what shipped, in the customer's words."
triggers: ["release notes", "changelog", "what's new", "ship announcement", "product update"]
version: "1"
---

# Release notes draft

Use this after a release is confirmed, or to prepare notes ahead of one.

## Confirm what shipped

List each change with its ticket or pull request, the release date, and who
confirmed it is live. Note which plans, platforms, or regions get it and
whether it sits behind a flag or rollout. Leave out anything not confirmed as
shipped, and say so.

## Write the notes

- lead with what the user can now do, not what the team built;
- use the words customers used when they asked for it;
- one short paragraph or bullet per change;
- say who it is for and where to find it;
- link the help article if one exists;
- list fixes plainly, without blaming anyone.

Keep it honest about limits. Do not call a change "new" if it only reached
some users.

## Approval note

Above the draft, list the sources used, changes held back and why, open
questions, and the person who must approve and publish.

Do not publish the notes or edit the changelog yourself. Never describe a
change that has not shipped, invent a quote, or promise a future date.
