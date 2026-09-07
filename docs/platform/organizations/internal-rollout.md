# Internal organization rollout

The initial release enables organization and team collaboration for a selected
internal cohort. Private experts and credentials retain the limits in the
[access model](access-model.md). Do not expand the cohort until the checks below
pass against the exact release image and a representative database upgrade.

## Feature flag

Use the existing literal flag key `SHOW_ORG_SETTINGS` in both frontend and
backend evaluations. Target the same account IDs on both sides. The backend
defaults to disabled and gates organization creation, personal organization
conversion, direct member addition, invitation creation and resend, and
invitation acceptance. Direct addition also requires the recipient to be in the
cohort; acceptance checks the recipient's flag.

Existing organization reads, leaving, revoking access, and other cleanup remain
available when the flag is disabled. Personal workspace bootstrap and database
migrations run independently of this flag. It controls collaboration entry, not
the underlying authorization checks or data migration.

For an isolated local installation, set backend
`FORCE_FLAG_SHOW_ORG_SETTINGS=true` and build the frontend with
`NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS=true`. The single-container development
Bake target enables both. Public frontend variables are build-time values.
For a hosted internal cohort, leave these global overrides unset and use flag
targeting. Existing expert, memory, and Files flags still apply to their surfaces.

## Before deployment

1. Record the candidate commit, image digests, current database migration state,
   and flag targeting. Preserve a database backup and the corresponding artifact
   storage snapshot. Verify that the restoration procedure is usable.
2. Restore a representative copy into an isolated environment. Include accounts
   with existing agents, schedules, credits, API keys, conversations, private
   experts, nested folders, and files without a provable source.
3. Apply the full migration chain with `poetry run prisma migrate deploy`, then
   start the candidate backend so the personal organization backfill runs. Wait
   for successful completion before opening collaboration.
4. Check ownership, balances, credential references, scheduled execution, and
   file visibility. Run the migration again and verify idempotence. Exercise
   ordinary reads using the configured `platform` schema/search path.
5. Review quarantined and unresolved files and folders with the affected owners.
   Decide which internal accounts can enter the cohort before rollout. A passing
   migration does not mean every historical file is visible.

These read-only inventory queries help identify records requiring review after
the backfill. Run them against the restored database first:

```sql
SELECT 'files' AS kind, "scopeResolved", count(*)
FROM platform."UserWorkspaceFile"
WHERE NOT "isDeleted"
GROUP BY "scopeResolved"
UNION ALL
SELECT 'folders', "scopeResolved", count(*)
FROM platform."UserWorkspaceFolder"
WHERE NOT "isDeleted"
GROUP BY "scopeResolved";

SELECT file.id, workspace."userId", file."sessionId", file."executionId",
       file."organizationId", file."teamId", file."scopeResolved"
FROM platform."UserWorkspaceFile" file
JOIN platform."UserWorkspace" workspace ON workspace.id = file."workspaceId"
WHERE NOT file."isDeleted" AND NOT file."isUserGlobalConfig"
  AND (NOT file."scopeResolved" OR file."organizationId" IS NULL);

SELECT folder.id, workspace."userId", folder."parentId",
       folder."organizationId", folder."teamId", folder."scopeResolved"
FROM platform."UserWorkspaceFolder" folder
JOIN platform."UserWorkspace" workspace ON workspace.id = folder."workspaceId"
WHERE NOT folder."isDeleted"
  AND (NOT folder."scopeResolved" OR folder."organizationId" IS NULL);
```

Keep identifiers in the restricted migration report. Determine recovery from
the original workspace owner and persisted conversation/execution evidence.
Do not bulk-set `scopeResolved` or disable the scope triggers. Recovery for
unproven files requires a separately reviewed operation; this release does not
include an owner-facing recovery screen.

## Enable and test the cohort

Deploy the candidate API, database service, executor, scheduler, notification
service, and frontend together. Old workers must not consume new tenant-scoped
work. Verify health, database connectivity, Redis, RabbitMQ, authentication JWKS,
and frontend CORS configuration before testing workflows. Check the configured
artifact storage and ClamAV when scanning is enabled; a missing scanner must not
be mistaken for a tenancy failure or silently marked clean. Memory surfaces
also require their configured graph store and model providers.

Use three accounts: an organization owner, an ordinary member, and an outsider.
Keep the outsider outside the collaboration cohort for flag-off tests.

| Check | Required result |
| --- | --- |
| Flag off | Direct create/convert/invite/join requests fail; existing cleanup and personal work still function |
| Owner creates an organization and team | Owner can administer both; personal workspace remains available |
| Add or invite a cohort member | Membership is explicit; member sees only permitted management controls |
| Outsider requests a known resource ID | Denied without resource content |
| Team membership removed during work | Subsequent reads and protected ongoing operations lose access |
| Org/team switch | Files, pending approvals, activity, conversations, and cached data follow the new context |
| Team Files | Upload, create folder, preview, download, and move stay within the selected team |
| Shared agent copy | Allowed grant works; revoked or wrong-version/target grants fail; copied credentials are removed |
| API key or OAuth authorization revoked | Blocked work is cancelled promptly and cannot return a successful result |
| Personal expert | Owner can use it; collaborator/shared-org views cannot expose it |
| Existing execution and schedule | Persisted scope reaches workers and result reads |
| Billing | Existing personal balance and payment ownership remain intact |

Check browser errors and failed HTTP responses as well as visible results. Keep
test logs associated with their actual commit and image. Old stack CI failures
remain historical evidence; new checks must not be attributed to an older head.

## Pause or recover

Disable `SHOW_ORG_SETTINGS` for the cohort to stop new collaboration entry. This
does not undo created organizations, memberships, files, or other writes. Keep
the candidate's authorization-aware services running while investigating and
use existing controls to revoke access where necessary.

Do not put the old unscoped application back over a database containing new
tenant-scoped writes. A full rollback requires a coordinated database and
artifact restore, with reconciliation of activity since the snapshot. Review
that operation separately before executing it.
