import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import prisma
import pytest
import pytest_asyncio

from backend.api.features.experts import routine_jobs, routines
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.model import ChatSession
from backend.data import db
from backend.data.tenancy import has_live_resource_access
from backend.executor import scheduler

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def owner_orgs(monkeypatch):
    # The client is shared with the rest of the session: leave it as found.
    connected_here = not db.is_connected()
    await db.connect()
    owner = str(uuid4())
    orgs = [str(uuid4()) for _ in range(2)]
    teams = [str(uuid4()) for _ in range(2)]
    await db.prisma.user.create(data={"id": owner, "email": f"{owner}@example.invalid"})
    for org, team in zip(orgs, teams):
        await db.prisma.organization.create(data={"id": org, "slug": org, "name": org})
        await db.prisma.orgmember.create(
            data={"orgId": org, "userId": owner, "isOwner": True}
        )
        await db.prisma.team.create(
            data={"id": team, "orgId": org, "name": "Default", "isDefault": True}
        )
        await db.prisma.teammember.create(
            data={"teamId": team, "userId": owner, "isAdmin": True}
        )
    await db.query_raw_with_schema(
        'INSERT INTO {schema_prefix}"PersonalMemoryOrganization" ("userId", "organizationId") VALUES ($1, $2)',
        owner,
        orgs[0],
    )
    jobs = SimpleNamespace(
        add_copilot_turn_schedule=AsyncMock(
            return_value=SimpleNamespace(id=str(uuid4()))
        ),
        delete_schedule=AsyncMock(),
    )
    monkeypatch.setattr(routine_jobs, "get_scheduler_client", lambda: jobs)
    try:
        yield owner, orgs, teams, jobs
    finally:
        await db.prisma.user.delete(where={"id": owner})
        await db.prisma.organization.delete_many(where={"id": {"in": orgs}})
        if connected_here:
            await db.disconnect()


async def create(owner, org=None):
    return await routines.create_routine(
        owner,
        None,
        organization_id=org,
        title="Daily",
        prompt="Private briefing",
        crons=["0 9 * * *"],
    )


async def test_same_owner_two_orgs_crud_and_private_default_team(owner_orgs):
    owner, orgs, teams, jobs = owner_orgs
    personal = await create(owner)
    business = await create(owner, orgs[1])
    assert personal.organization_id == orgs[0]
    assert business.organization_id == orgs[1]
    assert [r.id for r in await routines.list_routines(owner)] == [personal.id]
    assert [
        r.id for r in await routines.list_routines(owner, organization_id=orgs[1])
    ] == [business.id]
    assert (
        await routines.get_routine(personal.id, user_id=owner, organization_id=orgs[1])
        is None
    )
    for action in (
        routines.enable_routine,
        routines.disable_routine,
        routines.delete_routine,
    ):
        with pytest.raises(routines.RoutineNotFoundError):
            await action(owner, None, personal.id, organization_id=orgs[1])
    await routines.enable_routine(
        owner, None, business.id, organization_id=orgs[1], prompt="Business only"
    )
    payload = jobs.add_copilot_turn_schedule.await_args.kwargs
    assert (payload["organization_id"], payload["team_id"]) == (orgs[1], teams[1])
    assert (await routines.get_routine(personal.id)).prompt == "Private briefing"
    assert (await routines.get_routine(business.id)).prompt == "Business only"
    await routines.disable_routine(owner, None, business.id, organization_id=orgs[1])
    await routines.delete_routine(owner, None, business.id, organization_id=orgs[1])
    assert await routines.get_routine(business.id) is None
    await db.prisma.teammember.update_many(
        where={"userId": owner, "teamId": teams[1]}, data={"status": "REMOVED"}
    )
    with pytest.raises(routines.RoutineNotFoundError):
        await create(owner, orgs[1])


async def test_legacy_writer_is_anchored_and_cannot_move_owner_org(owner_orgs):
    owner, orgs, _, _ = owner_orgs
    legacy = await db.prisma.expertroutine.create(
        data={
            "userId": owner,
            "title": "Legacy",
            "prompt": "Original personal business",
            "crons": ["0 9 * * *"],
        }
    )
    assert legacy.organizationId == orgs[0]
    assert [r.id for r in await routines.list_routines(owner)] == [legacy.id]
    assert await routines.list_routines(owner, organization_id=orgs[1]) == []
    with pytest.raises(prisma.errors.PrismaError):
        await db.prisma.expertroutine.update(
            where={"id": legacy.id}, data={"organizationId": orgs[1]}
        )


async def test_another_org_member_cannot_read_or_change_private_routine(owner_orgs):
    owner, orgs, teams, _ = owner_orgs
    mine = await create(owner, orgs[1])
    other = str(uuid4())
    await db.prisma.user.create(data={"id": other, "email": f"{other}@example.invalid"})
    try:
        await db.prisma.orgmember.create(
            data={"orgId": orgs[1], "userId": other, "isAdmin": True}
        )
        await db.prisma.teammember.create(
            data={"teamId": teams[1], "userId": other, "isAdmin": True}
        )
        await db.query_raw_with_schema(
            'INSERT INTO {schema_prefix}"PersonalMemoryOrganization" ("userId", "organizationId") VALUES ($1, $2)',
            other,
            orgs[1],
        )
        assert await routines.list_routines(other, organization_id=orgs[1]) == []
        assert (
            await routines.get_routine(mine.id, user_id=other, organization_id=orgs[1])
            is None
        )
        with pytest.raises(routines.RoutineNotFoundError):
            await routines.disable_routine(
                other, None, mine.id, organization_id=orgs[1]
            )
    finally:
        await db.prisma.user.delete(where={"id": other})


async def test_pinned_session_cannot_cross_org_at_data_boundary(owner_orgs):
    owner, orgs, teams, _ = owner_orgs
    target = await db.prisma.chatsession.create(
        data={
            "userId": owner,
            "organizationId": orgs[1],
            "teamId": teams[1],
        }
    )
    routine = await create(owner, orgs[0])
    with pytest.raises(ValueError, match="chat was not found"):
        await routines.enable_routine(
            owner,
            None,
            routine.id,
            organization_id=orgs[0],
            session_mode="PINNED",
            pinned_session_id=target.id,
        )
    with pytest.raises(ValueError, match="chat was not found"):
        await routines.create_routine(
            owner,
            None,
            organization_id=orgs[0],
            title="Bad pin",
            prompt="Bad pin",
            crons=["0 9 * * *"],
            session_mode="PINNED",
            session_id=target.id,
        )


async def test_saved_worker_after_reload_and_scope_mismatch(owner_orgs, monkeypatch):
    owner, orgs, teams, jobs = owner_orgs
    routine = await create(owner, orgs[1])
    await routines.enable_routine(owner, None, routine.id, organization_id=orgs[1])
    payload = json.loads(json.dumps(jobs.add_copilot_turn_schedule.await_args.kwargs))
    payload.pop("name")
    payload["schedule_id"] = "saved-job"
    await db.disconnect()
    await db.connect()
    delivered = AsyncMock(return_value=True)
    route = AsyncMock(return_value=("platform", None))
    created = []

    async def new_chat(user_id, **kwargs):
        session = ChatSession.new(user_id, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(
        scheduler,
        "experts_db",
        lambda: SimpleNamespace(
            get_routine=routines.get_routine,
            record_routine_thread=routine_jobs.record_routine_thread,
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "get_database_manager_async_client",
        lambda: SimpleNamespace(
            has_live_resource_access=has_live_resource_access,
        ),
    )
    monkeypatch.setattr(scheduler, "resolve_default_chat_route", route)
    monkeypatch.setattr(
        scheduler, "_refuse_if_turn_may_not_start", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(scheduler, "create_chat_session", new_chat)
    monkeypatch.setattr(scheduler, "schedule_turn", delivered)
    monkeypatch.setattr(
        scheduler.product_analytics, "track_schedule_fired", lambda **_: None
    )
    await scheduler._execute_copilot_turn(**payload)
    assert delivered.await_count == 1
    assert delivered.await_args.kwargs["organization_id"] == orgs[1]
    assert delivered.await_args.kwargs["team_id"] == teams[1]
    route.assert_awaited_once_with(owner, organization_id=orgs[1])
    assert (await routines.get_routine(routine.id)).session_id == created[0].session_id
    monkeypatch.setattr(
        scheduler, "get_chat_session", AsyncMock(return_value=created[0])
    )
    await scheduler._execute_copilot_turn(**payload)
    assert delivered.await_count == 2
    await scheduler._execute_copilot_turn(**(payload | {"organization_id": orgs[0]}))
    await scheduler._execute_copilot_turn(**(payload | {"user_id": "another-owner"}))
    assert delivered.await_count == 2
    legacy_payload = {
        k: v for k, v in payload.items() if k not in ("organization_id", "team_id")
    }
    await scheduler._execute_copilot_turn(**legacy_payload)
    assert delivered.await_count == 3
    assert delivered.await_args.kwargs["organization_id"] == orgs[1]
    await db.prisma.teammember.update_many(
        where={"userId": owner, "teamId": teams[1]}, data={"status": "REMOVED"}
    )
    await scheduler._execute_copilot_turn(**payload)
    assert delivered.await_count == 3


async def test_legacy_job_retry_keeps_saved_business_org(owner_orgs, monkeypatch):
    owner, orgs, teams, jobs = owner_orgs
    routine = await routines.create_routine(
        owner,
        None,
        organization_id=orgs[1],
        title="Retry",
        prompt="Later",
        run_at=datetime.now(UTC) + timedelta(hours=1),
        session_mode="FRESH",
    )
    await routines.enable_routine(owner, None, routine.id, organization_id=orgs[1])
    payload = jobs.add_copilot_turn_schedule.await_args.kwargs.copy()
    for key in ("name", "organization_id", "team_id"):
        payload.pop(key)
    payload["schedule_id"] = "saved-retry"
    jobs.add_copilot_turn_schedule.reset_mock()
    monkeypatch.setattr(
        scheduler,
        "experts_db",
        lambda: SimpleNamespace(get_routine=routines.get_routine),
    )
    monkeypatch.setattr(
        scheduler,
        "get_database_manager_async_client",
        lambda: SimpleNamespace(has_live_resource_access=has_live_resource_access),
    )
    monkeypatch.setattr(scheduler, "get_scheduler_client", lambda: jobs)
    monkeypatch.setattr(
        scheduler,
        "resolve_default_chat_route",
        AsyncMock(return_value=("platform", None)),
    )
    monkeypatch.setattr(
        scheduler, "_refuse_if_turn_may_not_start", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        scheduler,
        "create_chat_session",
        AsyncMock(side_effect=ConcurrentTurnLimitError("busy")),
    )
    await scheduler._execute_copilot_turn(**payload)
    retry = jobs.add_copilot_turn_schedule.await_args.kwargs
    assert (retry["organization_id"], retry["team_id"], retry["routine_id"]) == (
        orgs[1],
        teams[1],
        routine.id,
    )
