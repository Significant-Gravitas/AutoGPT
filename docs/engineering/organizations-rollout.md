# Organization rollout

Organizations (team workspaces, the organization switcher and settings, sharing,
transfers and shared memory) are off for every account until they are turned on
for it. One gate decides, `org_collaboration_enabled` in
`backend/api/features/orgs/rollout.py`, and `GET /api/feature-flags` gives the
browser the same answer as `SHOW_ORG_SETTINGS`. It checks, in order:

1. **Installation override.** `FORCE_FLAG_SHOW_ORG_SETTINGS=true` or `false` on
   the backend decides for every account; build the frontend with the matching
   `NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS`. Leave both unset to decide per
   account.
2. **An admin's grant.** **Admin → Organization Access** turns organizations on
   for one account, found by its sign-in email, with an optional note such as the
   plan. The Users page shows the same switch for the account being looked up.
   The account's team organizations pass access on to their active members and
   to everyone with a pending invitation, so a customer's team can join without
   an admin turning each person on. Personal, archived and deleted organizations
   pass nothing on.
3. **The PostHog flag** `SHOW_ORG_SETTINGS`, for the internal cohort.

Every grant and revoke writes an `AuditLog` row (`entityType` `OrgAccessGrant`,
actor = the admin). Each account's answer is cached in Redis for up to a minute.
A grant or revoke reaches the account itself at once, and the members and
invitees of its organizations within that minute.

## Testing or selling a team plan

- Turn organizations on for the buyer's account on **Admin → Organization Access**
  and note the plan or trial.
- The buyer creates a team organization and invites their team. Each invitee sees
  organizations as soon as the invitation exists, so they can accept it.
- Turning the buyer's access off hides organizations again for the buyer and for
  everyone whose only access came through the buyer's organizations. Memberships,
  invitations and organization data are kept; turning access back on restores
  them.
- A personal organization has a single member and takes no invitations: it bills
  its owner only. Collaboration happens in a team organization.

## Disabled behavior

Organization and team settings, switchers, badges, pickers, invitations, sharing,
received shared workflows, and cross-team Library browsing are hidden.
Organization and team management, grants, transfers, and shared-memory management
APIs are gated as well. Invitation acceptance checks the recipient.

The browser resolves the caller's own personal organization through
`GET /api/orgs/default` and selects its default team before mounting resource
screens. Remembered shared selections cannot be used as a fallback. Lookup
failures show a retry state.

Explicit organization/team headers, scoped API keys and OAuth tokens, write-team
overrides, and new shared execution subscriptions cannot bypass the gate. Normal
resource authorization still runs first. Requested shared writes are rejected,
not silently redirected to personal storage.

Personal and expert memory remain available. Shared memory is excluded from
recall, storage, ingestion, tool arguments, and model instructions. Cleanup
handlers such as leaving, declining, and revoking remain authorized and usable
even though their collaboration UI is hidden. Personal bootstrap, migrations,
persisted execution scope, and authorization checks remain active. Legacy
authorized requests in the caller's own personal organization without a team
retain access to existing resources; turning organizations off is not access
revocation.

## Release checks

- Preserve the database and artifact backups and verify the restore procedure.
- Test migrations against a representative isolated copy before enabling users.
- Deploy API, frontend, executor, scheduler, and other tenant-aware workers
  together; do not run old unscoped workers against tenant-scoped writes.
- With organizations off, verify personal work and cleanup still function and
  shared controls, direct API requests, and shared-memory tools are unavailable.
- Turn organizations off with a shared context selected; ensure cached shared
  content disappears and personal default context is restored, including after
  reload.
- Test owner, member, and outsider authorization, membership revocation,
  credentials, private experts, and persisted executions with organizations on.
- Keep test and image evidence associated with its exact commit. Source tests do
  not establish deployment readiness, and older failures remain historical
  evidence.

Turning organizations off does not undo memberships or tenant-scoped data. Use
normal revocation to remove access. Rolling back application and data requires a
separately reviewed, coordinated restore; do not deploy an old unscoped version
over new tenant-scoped writes.
