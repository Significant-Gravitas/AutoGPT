from uuid import uuid4

import pytest
from prisma.errors import DataError

from backend.data import db, workspace, workspace_folder


@pytest.fixture
async def legacy_artifacts():
    user_id, organization_id, team_id = str(uuid4()), str(uuid4()), str(uuid4())
    try:
        await db.prisma.user.create(
            data={"id": user_id, "email": f"{user_id}@workspace-migration.example"}
        )
        await db.prisma.organization.create(
            data={
                "id": organization_id,
                "name": "Personal",
                "slug": organization_id,
                "isPersonal": True,
                "bootstrapUserId": user_id,
            }
        )
        await db.prisma.orgmember.create(
            data={"orgId": organization_id, "userId": user_id, "isOwner": True}
        )
        await db.prisma.team.create(
            data={
                "id": team_id,
                "orgId": organization_id,
                "name": "Default",
                "isDefault": True,
            }
        )
        await db.prisma.teammember.create(data={"teamId": team_id, "userId": user_id})
        storage = await db.prisma.userworkspace.create(data={"userId": user_id})
        session = await db.prisma.chatsession.create(data={"userId": user_id})
        root = await db.prisma.userworkspacefolder.create(
            data={"workspaceId": storage.id, "name": "Root"}
        )
        child = await db.prisma.userworkspacefolder.create(
            data={"workspaceId": storage.id, "name": "Child", "parentId": root.id}
        )
        artifact = await db.prisma.userworkspacefile.create(
            data={
                "workspaceId": storage.id,
                "name": "report.txt",
                "path": f"/sessions/{session.id}/report.txt",
                "storagePath": "test/report.txt",
                "mimeType": "text/plain",
                "sizeBytes": 12,
                "sessionId": session.id,
                "folderId": child.id,
            }
        )
        # Reproduce pre-migration quarantine rows, which today's writers reject.
        async with db.prisma.tx() as tx:
            await tx.execute_raw("SET LOCAL session_replication_role = replica")
            quarantine = await tx.userworkspacefile.create(
                data={
                    "workspaceId": storage.id,
                    "name": "unknown.txt",
                    "path": "/unknown.txt",
                    "storagePath": "test/unknown.txt",
                    "mimeType": "text/plain",
                    "sizeBytes": 1,
                    "scopeResolved": False,
                    "scopeKey": "__legacy_quarantine__",
                }
            )
        await db.prisma.chatsession.update(
            where={"id": session.id},
            data={"organizationId": organization_id, "teamId": team_id},
        )
        yield user_id, organization_id, team_id, storage, session, root, child, artifact, quarantine
    finally:
        await db.prisma.user.delete_many(where={"id": user_id})
        await db.prisma.organization.delete_many(where={"id": organization_id})


@pytest.mark.asyncio
async def test_legacy_session_artifacts_follow_personal_bootstrap_scope(
    legacy_artifacts,
):
    _, org_id, team_id, storage, _, root, child, artifact, quarantine = legacy_artifacts
    assert (
        await workspace.get_workspace_file_by_path(
            storage.id, artifact.path, org_id, team_id
        )
        is None
    )
    from backend.data.workspace_migration import normalize_legacy_workspace_scopes

    counts = await normalize_legacy_workspace_scopes()
    assert counts["UserWorkspaceFolder"] >= 2
    assert counts["UserWorkspaceFile"] >= 1
    visible = await workspace.get_workspace_file_by_path(
        storage.id, artifact.path, org_id, team_id
    )
    assert visible is not None and visible.id == artifact.id
    folders = await workspace_folder.list_folders(storage.id, org_id, team_id)
    assert {folder.id for folder in folders} == {root.id, child.id}
    untouched = await db.prisma.userworkspacefile.find_unique(
        where={"id": quarantine.id}
    )
    assert untouched is not None and untouched.scopeResolved is False
    assert (untouched.organizationId, untouched.teamId, untouched.scopeKey) == (
        None,
        None,
        "__legacy_quarantine__",
    )
    assert all(
        count == 0 for count in (await normalize_legacy_workspace_scopes()).values()
    )


@pytest.mark.asyncio
async def test_bootstrap_scope_exception_rejects_other_org_source_swap_and_quarantine(
    legacy_artifacts,
):
    user_id, org_id, team_id, storage, session, root, child, artifact, quarantine = (
        legacy_artifacts
    )
    other_session = await db.prisma.chatsession.create(
        data={"userId": user_id, "organizationId": org_id, "teamId": team_id}
    )
    for update in [
        {"organizationId": str(uuid4()), "teamId": team_id},
        {"organizationId": org_id, "teamId": team_id, "sessionId": other_session.id},
        {"organizationId": org_id, "teamId": team_id, "isUserGlobalConfig": True},
    ]:
        with pytest.raises(DataError):
            await db.prisma.userworkspacefile.update(
                where={"id": artifact.id}, data=update
            )
    with pytest.raises(DataError):
        await db.prisma.userworkspacefile.update(
            where={"id": quarantine.id},
            data={"organizationId": org_id, "teamId": team_id, "scopeResolved": True},
        )
    with pytest.raises(DataError):
        await db.prisma.userworkspacefolder.update(
            where={"id": root.id},
            data={"organizationId": str(uuid4()), "teamId": team_id},
        )
    from backend.data.workspace_migration import normalize_legacy_workspace_scopes

    await normalize_legacy_workspace_scopes()
    with pytest.raises(DataError):
        await db.prisma.userworkspacefile.update(
            where={"id": artifact.id}, data={"organizationId": None, "teamId": None}
        )
    with pytest.raises(DataError):
        await db.prisma.userworkspacefolder.update(
            where={"id": child.id}, data={"organizationId": None, "teamId": None}
        )


@pytest.mark.asyncio
async def test_bootstrap_cannot_target_an_active_shared_org_even_when_source_matches(
    legacy_artifacts,
):
    user_id, org_id, team_id, _, session, _, _, artifact, _ = legacy_artifacts
    shared_org, shared_team = str(uuid4()), str(uuid4())
    try:
        await db.prisma.organization.create(
            data={"id": shared_org, "name": "Shared", "slug": shared_org}
        )
        await db.prisma.orgmember.create(
            data={"orgId": shared_org, "userId": user_id, "isOwner": True}
        )
        await db.prisma.team.create(
            data={
                "id": shared_team,
                "orgId": shared_org,
                "name": "Default",
                "isDefault": True,
            }
        )
        await db.prisma.teammember.create(
            data={"teamId": shared_team, "userId": user_id}
        )
        await db.prisma.chatsession.update(
            where={"id": session.id},
            data={"organizationId": shared_org, "teamId": shared_team},
        )
        with pytest.raises(DataError, match="workspace artifact scope is immutable"):
            await db.prisma.userworkspacefile.update(
                where={"id": artifact.id},
                data={"organizationId": shared_org, "teamId": shared_team},
            )
        from backend.data.workspace_migration import normalize_legacy_workspace_scopes

        await normalize_legacy_workspace_scopes()
        untouched = await db.prisma.userworkspacefile.find_unique(
            where={"id": artifact.id}
        )
        assert untouched is not None
        assert (untouched.organizationId, untouched.teamId) == (None, None)
    finally:
        await db.prisma.chatsession.update(
            where={"id": session.id},
            data={"organizationId": org_id, "teamId": team_id},
        )
        await db.prisma.organization.delete_many(where={"id": shared_org})


@pytest.mark.asyncio
async def test_proven_child_and_file_survive_quarantined_legacy_ancestor(
    legacy_artifacts,
):
    _, org_id, team_id, storage, _, root, child, artifact, _ = legacy_artifacts
    async with db.prisma.tx() as tx:
        await tx.execute_raw("SET LOCAL session_replication_role = replica")
        await tx.userworkspacefolder.update(
            where={"id": root.id},
            data={"scopeResolved": False, "scopeKey": "__legacy_quarantine__"},
        )
    from backend.data.workspace_migration import normalize_legacy_workspace_scopes

    await normalize_legacy_workspace_scopes()
    visible = await workspace.get_workspace_file_by_path(
        storage.id, artifact.path, org_id, team_id
    )
    assert visible is not None and visible.id == artifact.id
    normalized_child = await db.prisma.userworkspacefolder.find_unique(
        where={"id": child.id}
    )
    assert normalized_child is not None and normalized_child.parentId is None
    assert (normalized_child.organizationId, normalized_child.teamId) == (
        org_id,
        team_id,
    )
    untouched_root = await db.prisma.userworkspacefolder.find_unique(
        where={"id": root.id}
    )
    assert untouched_root is not None and untouched_root.scopeResolved is False
    assert (untouched_root.organizationId, untouched_root.teamId) == (None, None)
    assert all(
        count == 0 for count in (await normalize_legacy_workspace_scopes()).values()
    )
