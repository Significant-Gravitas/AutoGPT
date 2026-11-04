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

A team plan is sold, not bought in the product: sales invoices the customer,
then an AutoGPT admin sets the plan up.

1. Turn organizations on for the buyer's account on **Admin → Organization
   Access**, with a note such as the invoice. Only an account with its own grant
   (or the installation override forced on, or the PostHog flag for that account)
   can create a team organization or convert its personal one. Access inherited
   through someone else's organization lets an invitee join, not start one;
   `GET /api/orgs/default` says which with `can_create_organization`.
2. The buyer creates the team organization.
3. Find it with `GET /api/admin/org-plans/orgs?email=<buyer>` and set its plan with
   `PUT /api/admin/org-plans/orgs/{org_id}`: `tier` (BASIC, PRO, MAX, BUSINESS or
   ENTERPRISE), `seats`, `ends_at` (optional), `status` (`active` or `canceled`)
   and a `note`.
4. Add the credits the invoice bought with
   `POST /api/admin/org-plans/orgs/{org_id}/credits` (`amount`, `reference`,
   `note`). The reference, such as the invoice number, makes it safe to repeat:
   sending the same reference again adds nothing and answers `granted: false`.
   Runs in the organization spend its wallet, not the members' own.
5. The buyer invites the team.

### What the plan decides

One check decides whether someone may use paid features, and at which tier:
`backend/data/entitlement.py`. Every paywalled route
(`Depends(enforce_payment_paywall)`, which reads the verified request context),
every background job and every Otto tool passes it the organization the work
runs in. `backend/data/entitlement_architecture_test.py` fails when code calls
the per-user checks in `backend/copilot/rate_limit.py` directly.

- **Personal organization, or no organization known:** the user's own plan, as
  before, trials included.
- **Team organization:** the organization's plan and an active membership. The
  plan counts while its status is `active` and `ends_at` has not passed. Members'
  own plans don't count there, so an owner with personal Pro is refused in their
  team organization until it has a plan, and the refusal asks for the
  organization's plan rather than a personal upgrade.
- With `ENABLE_PLATFORM_PAYMENT` off (self-hosted), nobody is refused and seats
  aren't counted.
- Otto's usage limits and workspace storage use the plan's tier for each member:
  every seat gets that tier's whole allowance, nothing is pooled. Each person's
  usage counters are their own in every organization.

### Where the plan lives

`OrganizationSubscription`: `planTier`, `status`, and the `seats`, `endsAt` and
`note` columns (migration `20261103120000_org_plan_seats_and_end`). Every
process reads it through a shared cache for up to a minute; an admin's change,
and converting a personal organization, drop the cached copy at once.

### Seats and roles

- Active members plus pending invitations (unexpired, not accepted, declined or
  revoked) may not exceed the plan's seats. Both creating and accepting an
  invitation are checked, under the organization's membership lock. Without a
  plan that counts, the owner holds the only seat, so nobody can be invited.
- Invitations are rate-limited per organization: 60 creates and resends an
  hour, and 3 resends of one invitation an hour.
- Only the owner makes someone an admin or billing manager, or takes that away,
  on a member or on an invitation. Admins still invite and remove plain members.

### Turning it off

- Setting `status` to `canceled`, or an `ends_at` in the past, refuses paid
  features to everyone in the organization at once. Memberships, data and the
  wallet are kept.
- Turning the buyer's access off hides organizations again for the buyer and
  for everyone whose only access came through the buyer's organizations.
  Memberships, invitations and organization data are kept; turning access back
  on restores them.
- A personal organization has a single member and takes no invitations: it bills
  its owner only. Collaboration happens in a team organization.

Every plan change and credit grant writes an `AuditLog` row (`entityType`
`OrganizationSubscription` or `OrgCreditGrant`, actor = the admin).

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
not silently redirected to personal storage: a run given an organization or team
its user isn't an active member of is refused (`add_graph_execution`), never
moved to the user's own.

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
