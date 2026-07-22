# Shared memory

Organizations can run agents that build up **shared memory** — facts and context
your agents learn and reuse across runs. Because shared memory is trusted
org-wide, admins get a governance surface for it in organization settings.

## Where to find it

Open **Settings → Organization → General**. Org admins see a **Shared memory**
card. (Members without admin rights don't see it, and personal organizations
don't have it at all.)

## Hold new memories for review

The core control is **Hold new memories for review**. When it's on, organization
and team memories written by non-admin members are held as **tentative** until
an admin reviews them. Admin writes remain active immediately. When the control
is off, every authorized shared-memory write becomes active immediately.

The **review queue** combines tentative organization and team memories. Admins
can approve an item to make it active or reject it to retract it. The setting
defaults to on and fails closed to on if it cannot be read.

## Permissions

Only organization owners and admins can change the hold setting or act on the
review queue. Regular members can use shared memory according to their active
organization and team memberships, but cannot bypass review or govern another
organization's queue.
