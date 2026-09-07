from collections.abc import Awaitable, Callable


async def normalize_legacy_workspace_scopes(
    renew_lock: Callable[[], Awaitable[None]] | None = None,
) -> dict[str, int]:
    """Restore source-proven legacy artifacts after their personal scope is assigned.

    Quarantined records and conflicting target paths stay unchanged. Proven
    children of quarantined folders move to root; other folders advance from
    roots to children before files so every parent scope validates.
    """
    from backend.data.org_migration import _assign_team_tenancy_batched

    folders = await _assign_team_tenancy_batched(
        "UserWorkspaceFolder",
        """
        WITH eligible AS (
            SELECT folder.id, folder."workspaceId" AS workspace_id, folder.name,
                   org.id AS org_id, team.id AS team_id,
                   CASE WHEN NOT parent."scopeResolved" THEN NULL ELSE folder."parentId" END AS target_parent_id
            FROM "UserWorkspaceFolder" folder
            JOIN "UserWorkspace" workspace ON workspace.id = folder."workspaceId"
            JOIN "OrgMember" member ON member."userId" = workspace."userId"
              AND member."isOwner" AND member.status = 'ACTIVE'
            JOIN "Organization" org ON org.id = member."orgId"
              AND org."isPersonal" AND org."deletedAt" IS NULL
            JOIN "Team" team ON team."orgId" = org.id
              AND team."isDefault" AND team."archivedAt" IS NULL
            JOIN "TeamMember" team_member ON team_member."teamId" = team.id
              AND team_member."userId" = workspace."userId" AND team_member.status = 'ACTIVE'
            LEFT JOIN "UserWorkspaceFolder" parent ON parent.id = folder."parentId"
            WHERE folder."organizationId" IS NULL AND folder."teamId" IS NULL
              AND folder."scopeResolved" AND NOT folder."isDeleted"
              AND folder."scopeKey" = '__personal__:__org__'
              AND (folder."parentId" IS NULL OR (
                  parent."workspaceId" = folder."workspaceId" AND (
                      NOT parent."scopeResolved" OR (
                          NOT parent."isDeleted" AND parent."organizationId" = org.id
                          AND parent."teamId" = team.id
                      )
                  )
              ))
        ), candidates AS (
            SELECT DISTINCT ON (workspace_id, org_id, team_id, target_parent_id, name)
                   id, org_id, team_id, target_parent_id
            FROM eligible
            WHERE NOT EXISTS (
                SELECT 1 FROM "UserWorkspaceFolder" existing
                WHERE existing."workspaceId" = eligible.workspace_id
                  AND existing."scopeKey" = eligible.org_id || ':' || eligible.team_id
                  AND existing.name = eligible.name
                  AND existing."parentId" IS NOT DISTINCT FROM eligible.target_parent_id
                  AND (eligible.target_parent_id IS NOT NULL OR NOT existing."isDeleted")
            )
            ORDER BY workspace_id, org_id, team_id, target_parent_id, name, id
            LIMIT 1000
        )
        UPDATE "UserWorkspaceFolder" folder
        SET "organizationId" = candidates.org_id, "teamId" = candidates.team_id,
            "parentId" = candidates.target_parent_id
        FROM candidates
        WHERE folder.id = candidates.id
          AND folder."organizationId" IS NULL AND folder."teamId" IS NULL
        """,
        renew_lock=renew_lock,
    )
    files = await _assign_team_tenancy_batched(
        "UserWorkspaceFile",
        """
        WITH candidates AS (
            SELECT file.id, org.id AS org_id, team.id AS team_id
            FROM "UserWorkspaceFile" file
            JOIN "UserWorkspace" workspace ON workspace.id = file."workspaceId"
            JOIN "OrgMember" member ON member."userId" = workspace."userId"
              AND member."isOwner" AND member.status = 'ACTIVE'
            JOIN "Organization" org ON org.id = member."orgId"
              AND org."isPersonal" AND org."deletedAt" IS NULL
            JOIN "Team" team ON team."orgId" = org.id
              AND team."isDefault" AND team."archivedAt" IS NULL
            JOIN "TeamMember" team_member ON team_member."teamId" = team.id
              AND team_member."userId" = workspace."userId" AND team_member.status = 'ACTIVE'
            LEFT JOIN "ChatSession" session ON session.id = file."sessionId"
              AND session."userId" = workspace."userId"
            LEFT JOIN "AgentGraphExecution" execution ON execution.id = file."executionId"
              AND execution."userId" = workspace."userId"
            LEFT JOIN "UserWorkspaceFolder" folder ON folder.id = file."folderId"
            WHERE file."organizationId" IS NULL AND file."teamId" IS NULL
              AND file."scopeResolved" AND NOT file."isDeleted" AND NOT file."isUserGlobalConfig"
              AND file."scopeKey" = '__personal__:__org__'
              AND ((session.id IS NOT NULL AND file."executionId" IS NULL)
                OR (execution.id IS NOT NULL AND file."sessionId" IS NULL))
              AND COALESCE(session."organizationId", execution."organizationId") = org.id
              AND COALESCE(session."teamId", execution."teamId") = team.id
              AND (file."folderId" IS NULL OR (
                  folder."workspaceId" = file."workspaceId" AND folder."scopeResolved"
                  AND NOT folder."isDeleted" AND folder."organizationId" = org.id
                  AND folder."teamId" = team.id
              ))
              AND NOT EXISTS (
                  SELECT 1 FROM "UserWorkspaceFile" existing
                  WHERE existing."workspaceId" = file."workspaceId"
                    AND existing."scopeKey" = org.id || ':' || team.id
                    AND existing.path = file.path
              )
            LIMIT 1000
        )
        UPDATE "UserWorkspaceFile" file
        SET "organizationId" = candidates.org_id, "teamId" = candidates.team_id
        FROM candidates
        WHERE file.id = candidates.id
          AND file."organizationId" IS NULL AND file."teamId" IS NULL
        """,
        renew_lock=renew_lock,
    )
    return {"UserWorkspaceFolder": folders, "UserWorkspaceFile": files}
