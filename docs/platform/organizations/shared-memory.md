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

## Reviewing held memories

The hold buffer is controlled from the org settings API: `PATCH
/api/orgs/{org_id}` accepts `memory_hold_buffer` (true keeps non-admin shared
writes in review; false lets them land active immediately), and `GET
/api/orgs/{org_id}` returns the current value (defaults to true).

Org admins have a review queue: `GET /api/orgs/{org_id}/memory/held` lists the
tentative memories awaiting review across the org tier and all its team tiers,
each labelled with its tier and originating team. `POST
/api/orgs/{org_id}/memory/held/{memory_id}/approve` ratifies a held memory into
active shared memory, and `POST .../reject` retracts it. All three review
endpoints require org-admin (owner/admin) permission and only ever act on the
org's own shared tiers — personal memory is never exposed or modified.

In the org settings **Shared memory** card, the toggle persists through the
setting above; the review-queue page wiring is the remaining UI step and the
card says so where it applies.
