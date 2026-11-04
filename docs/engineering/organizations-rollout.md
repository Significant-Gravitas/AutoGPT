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
`note` columns (migration `20261104130000_org_plan_seats_and_end`). Every
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

Explicit organization/team headers, write-team overrides, and new shared
execution subscriptions cannot bypass the gate. API keys and OAuth tokens are
held to the personal organization whether or not the gate is open (see
[API keys and OAuth apps](#api-keys-and-oauth-apps)). Normal resource
authorization still runs first. Requested shared writes are rejected, not
silently redirected to personal storage: a run given an organization or team its
user isn't an active member of is refused (`add_graph_execution`), never moved
to the user's own.

Personal and expert memory remain available. Shared memory is excluded from
recall, storage, ingestion, tool arguments, and model instructions. Cleanup
handlers such as leaving, declining, and revoking remain authorized and usable
even though their collaboration UI is hidden. Personal bootstrap, migrations,
persisted execution scope, and authorization checks remain active. Requests in
the caller's own personal organization without a team act in its default team,
where its existing resources are; turning organizations off is not access
revocation.

## One slot for personal work

A personal organization keeps all of its rows in its default team. The web app
used to save some at the organization's org-home (`teamId` NULL), which reads
restricted to the default team then missed. Migration
`20261104140000_one_personal_team_slot` moves those rows to the default team,
and the trigger `a0_normalize_personal_team`, on every table with both tenant
columns, does the same to every later write that names no team. Code applies
the same rule where a scope is compared exactly: the request context (a
personal organization with no `X-Team-Id` acts in its default team), credential
scoping, the graph, library and store-submission lookups
(`backend/data/personal_scope.py`), and scheduled Otto turns saved at a
personal org-home before the move, which would otherwise find their chat
elsewhere and delete their schedule. Team organizations keep org-home as a
scope of its own. A test that builds a personal `RequestContext` by hand
should give it the default team too, as `rest_api.py`'s `_test_request_context`
does.

The migration leaves a row where it was, and reports it in a `NOTICE`, when its
owner is not an active member of the default team, or when moving it would
collide with a row already there. Prod has no such collisions: those keys were
unique per user until the branch's own migrations.

`backend/data/personal_slot_guard_test.py` fails when a table gains both
tenant columns without the trigger; add it there, or list the table in
`NOT_NORMALIZED` with the reason.

## API keys and OAuth apps

Until credentials get their own design for shared organizations, API keys and
OAuth tokens act only in their owner's personal organization and its default
team, whether or not organizations are on for the account
(`require_personal_scope` in `backend/api/org_rollout.py`):

- A key or OAuth application tagged with a team organization gets a 403 on v1
  and v2 and is refused by the MCP server, as is an `X-Team-Id` naming any
  other team.
- New keys are made in the personal organization's default team only. The API
  keys page lists those keys, whichever organization is selected.
- Keys made in a team organization before this still exist. Revoke one with
  `DELETE /api/api-keys/{key_id}`, sent with that organization's `X-Org-Id`.

## Release checks

- Preserve the database and artifact backups and verify the restore procedure.
- Test migrations against a representative isolated copy before enabling users.
- Before `20261104140000_one_personal_team_slot`, count read-only the rows in a
  personal organization with `teamId` NULL in each of its tables. It moves them
  in one transaction, and creating its triggers holds writes to all sixteen
  tables until it commits; past about a million executions, move
  `AgentGraphExecution` in a batched startup step instead.
- `20260827180000_assert_tenant_integrity` was edited after it shipped to
  branch databases. `prisma migrate deploy` doesn't compare checksums, but
  `prisma migrate dev` asks to reset; update that migration's checksum in
  `_prisma_migrations` instead.
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
