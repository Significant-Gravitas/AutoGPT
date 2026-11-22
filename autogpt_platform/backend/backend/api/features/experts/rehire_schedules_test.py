"""Re-hiring an expert resumes only the schedules letting it go paused (#14577).

End to end against the scheduler service and its job store, through the
archive and hire flows the API runs. A schedule the user paused stays paused
across the round trip, whether they paused it before letting the expert go or
while it was gone. The scheduler's side of the mark is checked case by case in
``executor/scheduler_archive_mark_test.py``.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator

import prisma.models
import pytest
import pytest_asyncio
from pydantic import BaseModel

from backend.api.features.experts import experts_db, scheduling
from backend.api.model import CreateGraph
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.manage_schedules import (
    PauseScheduleTool,
    ScheduleToggledResponse,
)
from backend.data.user import get_or_create_user
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.usecases.sample import create_test_graph
from backend.util.background import _background_tasks
from backend.util.clients import get_scheduler_client
from backend.util.test import SpinTestServer

# New Year's midnight: no schedule fires while a test runs.
_CRON = "0 0 1 1 *"


class _Hire(BaseModel):
    owner_id: str
    template_id: str
    expert_id: str
    graph_id: str
    graph_version: int


@pytest_asyncio.fixture
async def hire(server: SpinTestServer) -> AsyncIterator[_Hire]:
    """An owner who has hired an expert with no schedules yet, and a workflow
    of their own to schedule for it. Everything is deleted afterwards."""
    # The scheduler service may still be starting when the session's first
    # test gets here, and nothing under test retries a call to it.
    await get_scheduler_client(should_retry=True).health_check_async()
    suffix = uuid.uuid4().hex[:8]
    owner = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"rehire-{suffix}@example.com",
            "name": "Rehire Owner",
        }
    )
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"Frankie {suffix}",
            "role": "Operations",
            "identity": "You are Frankie, an expert.",
            "isTemplate": True,
        }
    )
    try:
        hired = await experts_db.hire_expert(owner.id, template.id, None)
        await _settle(hired.expert.id)
        graph = await server.agent_server.test_create_graph(
            create_graph=CreateGraph(graph=create_test_graph()), user_id=owner.id
        )
        yield _Hire(
            owner_id=owner.id,
            template_id=template.id,
            expert_id=hired.expert.id,
            graph_id=graph.id,
            graph_version=graph.version,
        )
    finally:
        await _delete_everything_of(owner.id, template.id)


async def _settle(expert_id: str) -> None:
    """Wait out the work a hire starts behind it, so none outlives the test."""
    await asyncio.gather(
        *(task for task in _background_tasks if expert_id in task.get_name())
    )


async def _delete_everything_of(owner_id: str, template_id: str) -> None:
    scheduler = get_scheduler_client()
    for job in await scheduler.get_execution_schedules(
        user_id=owner_id, include_paused=True
    ):
        await scheduler.delete_schedule(job.id, user_id=owner_id)
    await prisma.models.Expert.prisma().delete_many(
        where={"OR": [{"ownerUserId": owner_id}, {"id": template_id}]}
    )
    await prisma.models.AgentPreset.prisma().delete_many(where={"userId": owner_id})
    await prisma.models.LibraryAgent.prisma().delete_many(where={"userId": owner_id})
    await prisma.models.User.prisma().delete_many(where={"id": owner_id})


async def _schedule(hire: _Hire, name: str) -> str:
    """A schedule of the user's own on the expert's workflow, as the Create
    schedule dialog on the expert's page makes one."""
    created = await get_scheduler_client().add_execution_schedule(
        user_id=hire.owner_id,
        graph_id=hire.graph_id,
        graph_version=hire.graph_version,
        name=name,
        cron=_CRON,
        input_data={"input": "data"},
        input_credentials={},
        expert_id=hire.expert_id,
    )
    return created.id


async def _schedules(hire: _Hire) -> dict[str, GraphExecutionJobInfo]:
    jobs = await get_scheduler_client().get_execution_schedules(
        user_id=hire.owner_id, kind="graph", include_paused=True
    )
    return {job.id: job for job in jobs if isinstance(job, GraphExecutionJobInfo)}


async def _let_go_and_hire_again(hire: _Hire) -> None:
    await experts_db.archive_expert(hire.owner_id, hire.expert_id)
    await _hire_again(hire)


async def _hire_again(hire: _Hire) -> None:
    rehired = await experts_db.hire_expert(hire.owner_id, hire.template_id, None)
    await _settle(hire.expert_id)
    assert rehired.expert.id == hire.expert_id and not rehired.expert.is_archived


@pytest.mark.asyncio(loop_scope="session")
async def test_rehire_resumes_what_letting_go_paused_and_not_what_the_user_did(
    hire: _Hire,
):
    """The issue's steps: one schedule kept running, one paused from the
    expert's chat, then let the expert go and hire it again."""
    kept = await _schedule(hire, "Kept running")
    paused = await _schedule(hire, "Paused before letting go")
    toggled = await PauseScheduleTool()._execute(
        user_id=hire.owner_id,
        session=make_session(hire.owner_id, expert_id=hire.expert_id),
        schedule_id=paused,
    )
    assert isinstance(toggled, ScheduleToggledResponse) and toggled.changed

    await experts_db.archive_expert(hire.owner_id, hire.expert_id)

    away = await _schedules(hire)
    assert not away[kept].next_run_time and not away[paused].next_run_time

    await _hire_again(hire)

    back = await _schedules(hire)
    assert back[kept].next_run_time, "the schedule letting go paused came back"
    assert not back[paused].next_run_time, "the user's paused schedule resumed"
    # Re-hiring spent the mark: the next round trip starts clean.
    assert not back[kept].paused_by_expert_archive
    assert not back[paused].paused_by_expert_archive

    await _let_go_and_hire_again(hire)

    again = await _schedules(hire)
    assert again[kept].next_run_time and not again[paused].next_run_time


@pytest.mark.parametrize("resumed_first", [False, True])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_schedule_paused_while_the_expert_is_away_stays_paused(
    hire: _Hire, resumed_first: bool
):
    """Letting the expert go paused both. The user then pauses one through
    the scheduler, directly or after resuming it: the chat tools and the
    schedule listings hide an archived expert's paused schedules, so this is
    the rule for any caller that does reach one. Their pause is their word on
    it, and re-hiring leaves it paused."""
    theirs = await _schedule(hire, "Paused while away")
    other = await _schedule(hire, "Left alone")
    await experts_db.archive_expert(hire.owner_id, hire.expert_id)

    scheduler = get_scheduler_client()
    if resumed_first:
        assert await scheduler.resume_schedule(theirs, user_id=hire.owner_id)
        assert await scheduler.pause_schedule(theirs, user_id=hire.owner_id)
    else:
        assert not await scheduler.pause_schedule(theirs, user_id=hire.owner_id)

    await _hire_again(hire)

    back = await _schedules(hire)
    assert not back[theirs].next_run_time, "the user's paused schedule resumed"
    assert back[other].next_run_time, "the schedule letting go paused stayed off"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_schedule_letting_go_paused_without_the_mark_stays_paused(
    hire: _Hire,
):
    """Only marked schedules resume on re-hire. One the expert was let go
    with, but paused by the same unmarked call a user's pause makes, can't
    be told apart from the user's, so it stays paused."""
    first = await _schedule(hire, "Morning report")
    second = await _schedule(hire, "Weekly digest")
    # Let go with its schedules paused unmarked.
    await scheduling.pause_expert_schedules(
        hire.owner_id, hire.expert_id, reason="Expert archived"
    )
    await prisma.models.Expert.prisma().update(
        where={"id": hire.expert_id}, data={"isArchived": True}
    )
    for schedule_id in (first, second):
        assert await get_scheduler_client().pause_schedule(
            schedule_id, user_id=hire.owner_id
        )

    await _hire_again(hire)

    back = await _schedules(hire)
    assert not back[first].next_run_time and not back[second].next_run_time
