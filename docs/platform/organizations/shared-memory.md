# Shared memory

Organizations can run agents that build up **shared memory** — facts and context
your agents learn and reuse across runs. Because shared memory is trusted
org-wide, admins get a governance surface for it in organization settings.

## Where to find it

Open **Settings → Organization → General**. Org admins see a **Shared memory**
card. (Members without admin rights don't see it, and personal organizations
don't have it at all.)

## Hold new memories for review

The core control is **Hold new memories for review**. When it's on, new memories
your agents learn are held as **tentative** until an admin reviews them — so a
single run can't quietly change what the whole organization treats as true. When
it's off, memories become usable as soon as they're learned.

Reviewing tentative memories happens in an admin **review queue**: admins see
what's pending and approve or reject each item.

## Current status

The **Shared memory** card ships with the concept and the toggle laid out, but
the interactive parts are **not yet functional** — they're waiting on backend
support that hasn't been built:

- The **Hold new memories for review** toggle is shown **disabled**. There's no
  way to persist an org memory setting yet (the organization update endpoint
  doesn't accept a settings value), so the toggle can't be saved.
- The **review queue** is flagged as unavailable. The endpoints to list
  tentative memories for an org or team, and to approve or reject them, don't
  exist yet. The memory-tiers backend shipped **per-user admin tools**
  (`/api/admin/memory/*`) and deliberately deferred the org-level review flow.

When the backend endpoints land, the toggle will become live and the review
queue will open up. Until then, the card documents the intended behavior and
marks the blocked pieces so there are no surprises.
