import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.api.features.experts import experts_db
from backend.api.features.orgs.db import create_org
from backend.copilot.learning.chat_source import (
    _matches_source,
    record_chat_turn,
    source_revision_from_record,
)
from backend.copilot.learning.nightly import _select_sources
from backend.copilot.learning.owner_actions import restore_version
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.tools import skills
from backend.data import skill_versions
from backend.data.db import prisma
from backend.data.skill_learning import owner_key_for, upsert_source_revision
from backend.data.tenancy import (
    connect_live_transaction_lease_database,
    disconnect_live_transaction_lease_database,
)
from backend.data.workspace import get_or_create_workspace
from backend.util.exceptions import (
    ExpertNotFoundError,
    ExpertPrivateTenancyNotFoundError,
)
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import LocalWorkspaceStorage

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("RUN_ORG_ISOLATION_DB_TESTS") != "1",
        reason="Requires an isolated migrated PostgreSQL database and Redis cluster",
    ),
]


@pytest_asyncio.fixture
async def organizations(mocker, tmp_path):
    # Both clients are shared with the rest of the session: leave them as found.
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    lease_connected_here = await connect_live_transaction_lease_database()
    user = await prisma.user.create(
        data={"id": str(uuid4()), "email": f"{uuid4()}@example.invalid"}
    )
    orgs = [
        await create_org(name, f"org-test-{uuid4()}", user.id)
        for name in ("AutoGPT", "Denton", "Reviewer")
    ]
    await prisma.organization.update(
        where={"id": orgs[0].id}, data={"isPersonal": True}
    )
    mocker.patch(
        "backend.util.workspace.get_workspace_storage",
        return_value=LocalWorkspaceStorage(str(tmp_path)),
    )
    mocker.patch("backend.util.workspace.scan_content_safe", AsyncMock())
    mocker.patch(
        "backend.util.workspace.get_workspace_storage_limit_bytes",
        AsyncMock(return_value=0),
    )
    mocker.patch.object(experts_db, "emit_funnel_event")
    mocker.patch.object(
        experts_db, "spawn_background_task", side_effect=lambda coro, **_: coro.close()
    )
    mocker.patch.object(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    try:
        yield user.id, [org.id for org in orgs]
    finally:
        await prisma.expert.delete_many(where={"ownerUserId": user.id})
        await prisma.user.delete(where={"id": user.id})
        await prisma.organization.delete_many(
            where={"id": {"in": [org.id for org in orgs]}}
        )
        if lease_connected_here:
            await disconnect_live_transaction_lease_database()
        if connected_here:
            await prisma.disconnect()


async def test_same_template_hires_and_rehires_are_independent(organizations, mocker):
    user, (personal, denton, _) = organizations
    template = await prisma.expert.create(
        data={
            "name": "Finance",
            "role": "Finance",
            "identity": "Help with finance",
            "isTemplate": True,
        }
    )
    try:
        a = (await experts_db.hire_expert(user, template.id, None)).expert
        b = (
            await experts_db.hire_expert(
                user, template.id, None, organization_id=denton
            )
        ).expert
        assert a.id != b.id
        assert (a.organization_id, b.organization_id) == (personal, denton)
        assert [
            e.id for e in await experts_db.list_experts(user, with_metrics=False)
        ] == [a.id]
        assert [
            e.id
            for e in await experts_db.list_experts(
                user, organization_id=denton, with_metrics=False
            )
        ] == [b.id]
        assert (
            await experts_db.hire_expert(
                user, template.id, None, organization_id=denton
            )
        ).expert.id == b.id
        assert (await experts_db.resolve_private_expert_tenancy(user, b.id))[
            0
        ] == denton
        await prisma.expert.update(where={"id": b.id}, data={"isArchived": True})
        mocker.patch(
            "backend.api.features.experts.scheduling.resume_expert_schedules",
            AsyncMock(),
        )
        mocker.patch(
            "backend.api.features.experts.scheduling.reattach_expert_triggers",
            AsyncMock(),
        )
        revived = (
            await experts_db.hire_expert(
                user, template.id, None, organization_id=denton
            )
        ).expert
        assert (revived.id, revived.organization_id) == (b.id, denton)
        assert await experts_db.get_expert(user, a.id, organization_id=denton) is None
        assert await experts_db.get_expert(user, b.id, organization_id=None) is None
        assert (await experts_db.get_expert(user, b.id)).organization_id == denton
        await prisma.orgmember.update(
            where={"orgId_userId": {"orgId": denton, "userId": user}},
            data={"status": "SUSPENDED"},
        )
        with pytest.raises(ExpertPrivateTenancyNotFoundError):
            await experts_db.resolve_private_expert_tenancy(user, b.id)
        with pytest.raises(ExpertNotFoundError):
            await experts_db.update_budget(user, b.id, 17)
        assert (
            await prisma.expert.find_unique(where={"id": b.id})
        ).weeklyBudget is None
        assert (
            await experts_db.list_experts(
                user, organization_id=denton, with_metrics=False
            )
            == []
        )
    finally:
        await prisma.expert.delete_many(where={"sourceTemplateId": template.id})
        await prisma.expert.delete(where={"id": template.id})


async def test_skills_package_cache_history_and_delete_are_org_scoped(organizations):
    user, (personal, denton, reviewer) = organizations
    for org, label in ((personal, "AutoGPT"), (denton, "Denton")):
        await skills.store_user_skill(
            user,
            name="invoice",
            description=f"{label} invoices",
            body=f"Follow {label} steps",
            organization_id=org,
            files=[skills.SkillFile(relative_path="notes.txt", content=label.encode())],
        )
    assert await skills.list_user_skills(user, organization_id=reviewer) == []
    # Read twice: the second read must retain the same tenant through the warm cache.
    for _ in range(2):
        assert (await skills.list_user_skills(user))[
            0
        ].description == "AutoGPT invoices"
        assert (await skills.list_user_skills(user, organization_id=denton))[
            0
        ].description == "Denton invoices"
    assert "Denton invoices" not in await skills.build_skills_context(
        user, organization_id=personal
    )
    package = await skills.read_user_skill_package(
        user, "invoice", organization_id=denton
    )
    assert package is not None and package.files[0].content == b"Denton"
    a = await skill_versions.get_head(user, owner_key_for(None, personal), "invoice")
    b = await skill_versions.get_head(user, owner_key_for(None, denton), "invoice")
    assert a is not None and b is not None and a.id != b.id
    assert (a.organization_id, b.organization_id) == (personal, denton)
    await skills.delete_user_skill(user, "invoice", organization_id=denton)
    assert await skills.list_user_skills(user, organization_id=denton) == []
    assert (
        await skills.read_user_skill_with_body(user, "invoice")
    ).body.strip() == "Follow AutoGPT steps"


async def test_session_capture_and_expert_skill_target_keep_saved_org(organizations):
    user, (_, denton, reviewer) = organizations
    expert = (
        await experts_db.create_raised_expert(
            user, "Denton Expert", None, None, organization_id=denton
        )
    ).expert
    await skills.store_user_skill(
        user,
        name="shop",
        description="Shop procedure",
        body="Denton only",
        expert_id=expert.id,
    )
    with pytest.raises(ExpertPrivateTenancyNotFoundError):
        await skills.list_user_skills(user, expert.id, organization_id=reviewer)
    session = ChatSession(
        usage=[],
        messages=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        session_id=str(uuid4()),
        user_id=user,
        organization_id=denton,
        team_id=expert.team_id,
        expert_id=expert.id,
    )
    record = await record_chat_turn(
        user,
        session,
        [ChatMessage(role="user", content="Remember this", sequence=1)],
        "Remember this",
    )
    assert record.organization_id == denton
    assert record.owner_key == owner_key_for(expert.id, denton)
    assert (
        len(
            await skills.list_user_skills(
                user, expert.id, organization_id=session.organization_id
            )
        )
        == 1
    )


async def test_legacy_packages_and_history_stay_in_original_personal_org(organizations):
    user, (personal, denton, _) = organizations
    workspace = await get_or_create_workspace(user)
    legacy = WorkspaceManager(user, workspace.id, user_global_config=True)
    await legacy.write_file(
        filename="SKILL.md",
        path="/skills/legacy/SKILL.md",
        content=b"---\nname: legacy\ndescription: Original private process\n---\nPersonal content\n",
    )
    await prisma.skillhead.create(
        data={"userId": user, "ownerKey": "personal", "skillName": "legacy"}
    )
    migration = (
        Path(__file__).parents[4]
        / "migrations/20261016173000_private_experts_skills_org/migration.sql"
    )
    async with prisma.tx() as tx:
        for statement in migration.read_text().split(";"):
            sql = "\n".join(
                line
                for line in statement.splitlines()
                if not line.lstrip().startswith("--")
            ).strip()
            if sql.startswith("UPDATE") or (
                sql.startswith('ALTER TABLE "UserWorkspaceFile"') and "TRIGGER" in sql
            ):
                await tx.execute_raw(sql)
    assert [s.name for s in await skills.list_user_skills(user)] == ["legacy"]
    assert [
        s.name for s in await skills.list_user_skills(user, organization_id=personal)
    ] == ["legacy"]
    assert await skills.list_user_skills(user, organization_id=denton) == []
    head = await skill_versions.get_head(user, owner_key_for(None, personal), "legacy")
    assert head is not None and head.organization_id == personal
    trigger = await prisma.query_raw(
        "SELECT tgenabled FROM pg_trigger WHERE tgname = 'enforce_workspace_artifact_scope'"
    )
    assert trigger and all(row["tgenabled"] == "O" for row in trigger)


async def test_learning_selection_and_owner_actions_stay_in_saved_org(organizations):
    user, (personal, denton, _) = organizations
    for org in (personal, denton):
        await upsert_source_revision(
            user,
            expert_id=None,
            source_kind="chat_session",
            source_id=str(uuid4()),
            revision="1",
            evidence_refs=[],
            outcome_signals=[],
            organization_id=org,
        )
        await skills.store_user_skill(
            user,
            name="process",
            description="Private process",
            body=org,
            organization_id=org,
        )
    pending = await _select_sources(user, None, organization_id=denton)
    assert len(pending) == 1 and pending[0].organization_id == denton
    head = await skill_versions.get_head(user, owner_key_for(None, personal), "process")
    assert head is not None and head.current_version_id
    denied = await restore_version(
        user_id=user,
        expert_id=None,
        skill_name="process",
        version_id=head.current_version_id,
        actor_user_id=user,
        organization_id=denton,
    )
    assert denied.status == "conflict"
    assert (
        await skills.read_user_skill_with_body(user, "process", organization_id=denton)
    ).body.strip() == denton
    await prisma.orgmember.update(
        where={"orgId_userId": {"orgId": denton, "userId": user}},
        data={"status": "SUSPENDED"},
    )
    with pytest.raises(ExpertPrivateTenancyNotFoundError):
        await skills.list_user_skills(user, organization_id=denton)


async def test_legacy_session_learning_resolves_original_personal_org(organizations):
    user, (personal, denton, _) = organizations
    session = ChatSession.new(user, dry_run=False)
    record = await record_chat_turn(
        user,
        session,
        [ChatMessage(role="user", content="Personal", sequence=1)],
        "Personal",
    )
    source = source_revision_from_record(record)
    assert source.scope.organization_id == personal
    assert await _matches_source(session, source)
    session.organization_id = denton
    assert not await _matches_source(session, source)


async def test_expert_mode_is_private_and_rechecks_saved_org_membership(organizations):
    user, (personal, denton, _) = organizations
    experts = []
    for org in (personal, denton):
        team = await prisma.team.find_first(where={"orgId": org, "isDefault": True})
        experts.append(
            await prisma.expert.create(
                data={
                    "ownerUserId": user,
                    "organizationId": org,
                    "teamId": team.id,
                    "name": "Private helper",
                    "role": "Review",
                    "identity": "Review private documents",
                    "visibility": "PRIVATE",
                }
            )
        )
    await experts_db.update_autopilot_mode(user, experts[1].id, "ask_first")
    assert await experts_db.get_autopilot_mode(user, experts[0].id) is None
    assert await experts_db.get_autopilot_mode(user, experts[1].id) == "ask_first"
    assert [
        row.autopilot_mode
        for row in await experts_db.list_expert_identities(user, organization_id=denton)
    ] == ["ask_first"]
    await prisma.orgmember.update(
        where={"orgId_userId": {"orgId": denton, "userId": user}},
        data={"status": "SUSPENDED"},
    )
    assert await experts_db.get_autopilot_mode(user, experts[1].id) is None
    with pytest.raises(ExpertNotFoundError):
        await experts_db.update_autopilot_mode(user, experts[1].id, "unsupervised")
    assert (
        await prisma.expert.find_unique(where={"id": experts[1].id})
    ).autopilotMode == "ask_first"


async def test_deleted_browser_cleanup_preserves_org_and_session_scope(
    organizations, mocker
):
    from backend.copilot import db as chat_db
    from backend.copilot.tools import agent_browser

    user, (personal, denton, _) = organizations
    workspace = await get_or_create_workspace(user)
    sessions = []
    for org in (personal, denton):
        team = await prisma.team.find_first(where={"orgId": org, "isDefault": True})
        session = await chat_db.create_chat_session(
            session_id=str(uuid4()), user_id=user, organization_id=org, team_id=team.id
        )
        sessions.append((session.session_id, org, team.id))
    files = []
    for session_id, org, team in sessions:
        manager = WorkspaceManager(
            user, workspace.id, session_id, organization_id=org, team_id=team
        )
        files.append(
            await manager.write_file(
                content=b'{"cookies":[]}',
                filename=agent_browser._STATE_FILENAME,
                mime_type="application/json",
            )
        )
    deleted, org, team = sessions[1]
    await chat_db.delete_chat_session(deleted, user, organization_id=org, team_id=team)
    mocker.patch.object(agent_browser, "_run", AsyncMock(return_value=(0, "", "")))
    lookup = mocker.patch.object(
        agent_browser,
        "get_workspace_manager",
        AsyncMock(side_effect=AssertionError("deleted session cannot resolve")),
    )
    await agent_browser.close_browser_session(
        deleted, user, deleted_session_scope=(org, team)
    )
    lookup.assert_not_awaited()
    assert (
        await prisma.userworkspacefile.find_unique(where={"id": files[1].id})
    ).isDeleted
    assert not (
        await prisma.userworkspacefile.find_unique(where={"id": files[0].id})
    ).isDeleted
