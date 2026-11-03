# Internal organization rollout

Use the existing literal `SHOW_ORG_SETTINGS` key for both frontend and backend,
targeting the same account IDs. The default is disabled. Keep the cohort internal
until the exact release image and a representative database upgrade are tested.

## Disabled behavior

Organization and team settings, switchers, badges, pickers, invitations, sharing,
received shared agents, and cross-team Library browsing are hidden. Organization
and team management, grants, transfers, and shared-memory management APIs are
gated as well. Direct member addition checks both actor and recipient; invitation
acceptance checks the recipient.

The browser resolves the caller's own personal organization through
`GET /api/orgs/default` and selects its default team before mounting resource
screens. Remembered shared selections cannot be used as a fallback. Lookup
failures show a retry state. LaunchDarkly must wrap the organization/team provider
so flag changes reach this boundary.

Explicit organization/team headers, scoped API keys and OAuth tokens, write-team
overrides, and new shared execution subscriptions cannot bypass the flag. Normal
resource authorization still runs first. Requested shared writes are rejected,
not silently redirected to personal storage.

Personal and expert memory remain available. Shared memory is excluded from
recall, storage, ingestion, tool arguments, and model instructions. Cleanup
handlers such as leaving, declining, and revoking remain authorized and usable
even though their collaboration UI is hidden. Personal bootstrap, migrations,
persisted execution scope, and authorization checks remain active. Legacy
authorized requests in the caller's own personal organization without a team
retain access to existing resources; disabling the flag is not access revocation.

## Configuration

For an isolated opt-in installation, set backend
`FORCE_FLAG_SHOW_ORG_SETTINGS=true` and build frontend assets with
`NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS=true`. Single-container images no longer
force organizations on by default. Empty overrides defer to normal flag
targeting; explicit `true` or `false` overrides take precedence.

For a hosted cohort, leave both overrides unset and configure LaunchDarkly on
both sides. The standard and artifact-delta Dockerfiles accept
`NEXT_PUBLIC_LAUNCHDARKLY_ENABLED=true` and `NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID`
as build arguments; Docker Bake also reads matching environment variables.
Public frontend values are build-time settings and require rebuilding assets.
Other expert, memory, and Files flags still apply independently.

## Release checks

- Preserve the database and artifact backups and verify the restore procedure.
- Test migrations against a representative isolated copy before enabling users.
- Deploy API, frontend, executor, scheduler, and other tenant-aware workers
  together; do not run old unscoped workers against tenant-scoped writes.
- With the flag off, verify personal work and cleanup still function and shared
  controls, direct API requests, and shared-memory tools are unavailable.
- Turn the flag off with a shared context selected; ensure cached shared content
  disappears and personal default context is restored, including after reload.
- Test owner, member, and outsider authorization, membership revocation,
  credentials, private experts, and persisted executions with the flag enabled.
- Keep test and image evidence associated with its exact commit. Source tests do
  not establish deployment readiness, and older failures remain historical evidence.

Disabling the flag does not undo memberships or tenant-scoped data. Use normal
revocation to remove access. Rolling back application and data requires a
separately reviewed, coordinated restore; do not deploy an old unscoped version
over new tenant-scoped writes.
