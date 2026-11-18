"""A schedule create sent again under its key makes one schedule (#16901).

A create the scheduler applied, whose answer was lost (a timeout, or a 503
while the scheduler restarted), is retried by the user. When every attempt
minted its own id, each retry added another recurring schedule, which then
ran, and charged, again. The first tests run the real route against the real
scheduler service, its jobstore and the database. The per-account limit is too
high to fill there, so the last ones run a scheduler built in this process, on
its real jobstore, with the limit lowered.
"""

import asyncio
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.schedules.model import ScheduleCreationRequest
from backend.api.features.schedules.routes import create_graph_execution_schedule
from backend.api.model import CreateGraph
from backend.data.graph import GraphModel
from backend.data.model import CredentialsMetaInput, User
from backend.executor import schedule_events
from backend.executor import scheduler as scheduler_module
from backend.executor import utils as execution_utils
from backend.executor.scheduler import GraphExecutionJobInfo, Jobstores, Scheduler
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import ConflictError
from backend.util.test import SpinTestServer

INPUTS = {"input_1": "Hello", "input_2": "World"}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_retried_create_returns_the_schedule_the_first_attempt_made(
    server: SpinTestServer,
):
    owner = await create_test_user()
    graph = await _new_graph(server, owner)
    key = uuid.uuid4()
    try:
        first = await _create(owner, graph, key)
        retried = await _create(owner, graph, key)

        assert await _schedule_ids(owner, graph) == [str(key)]
        assert first.id == retried.id == str(key)
    finally:
        await _delete_schedules(owner, graph)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_same_create_sent_twice_at_once_makes_one_schedule(
    server: SpinTestServer,
):
    owner = await create_test_user()
    graph = await _new_graph(server, owner)
    key = uuid.uuid4()
    try:
        sent = await asyncio.gather(
            _create(owner, graph, key), _create(owner, graph, key)
        )

        assert await _schedule_ids(owner, graph) == [str(key)]
        assert [schedule.id for schedule in sent] == [str(key), str(key)]
    finally:
        await _delete_schedules(owner, graph)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"cron": "30 18 * * *"}, id="cron"),
        pytest.param({"timezone": "Europe/Paris"}, id="timezone"),
        pytest.param({"inputs": {**INPUTS, "input_2": "Moon"}}, id="inputs"),
        pytest.param(
            {
                "credentials": {
                    "github": CredentialsMetaInput.model_validate(
                        {
                            "id": str(uuid.uuid4()),
                            "provider": "github",
                            "type": "api_key",
                        }
                    )
                }
            },
            id="credentials",
        ),
        pytest.param({"name": "Evening run"}, id="name"),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_different_create_under_a_used_key_is_a_conflict(
    server: SpinTestServer, change: dict[str, Any]
):
    owner = await create_test_user()
    graph = await _new_graph(server, owner)
    key = uuid.uuid4()
    try:
        await _create(owner, graph, key)

        with pytest.raises(ConflictError):
            await _create(owner, graph, key, **change)

        [kept] = await _schedules(owner, graph)
        assert (kept.id, kept.name, kept.cron, kept.timezone, kept.input_data) == (
            str(key),
            "Morning run",
            "0 9 * * *",
            "UTC",
            INPUTS,
        )
    finally:
        await _delete_schedules(owner, graph)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_used_key_cannot_schedule_another_workflow(server: SpinTestServer):
    owner = await create_test_user()
    graph = await _new_graph(server, owner)
    other_graph = await _new_graph(server, owner)
    key = uuid.uuid4()
    try:
        await _create(owner, graph, key)

        with pytest.raises(ConflictError):
            await _create(owner, other_graph, key)

        assert await _schedule_ids(owner, graph) == [str(key)]
        assert await _schedule_ids(owner, other_graph) == []
    finally:
        await _delete_schedules(owner, graph)
        await _delete_schedules(owner, other_graph)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_key_never_answers_with_or_replaces_another_accounts_schedule(
    server: SpinTestServer,
):
    owner = await create_test_user()
    other = await create_test_user(alt_user=True)
    owner_graph = await _new_graph(server, owner)
    other_graph = await _new_graph(server, other)
    key = uuid.uuid4()
    try:
        await _create(owner, owner_graph, key)

        with pytest.raises(ConflictError):
            await _create(other, other_graph, key)

        [kept] = await _schedules(owner, owner_graph)
        assert (kept.id, kept.user_id, kept.graph_id) == (
            str(key),
            owner.id,
            owner_graph.id,
        )
        assert await _schedule_ids(other, other_graph) == []
    finally:
        await _delete_schedules(owner, owner_graph)
        await _delete_schedules(other, other_graph)


def test_a_retried_create_is_answered_at_the_accounts_limit(
    scheduler: Scheduler, validated: AsyncMock, recorded: MagicMock
):
    first_key, second_key = str(uuid.uuid4()), str(uuid.uuid4())
    first = _add(scheduler, first_key)
    _add(scheduler, second_key)  # The account now holds its limit of two.

    retried = _add(scheduler, first_key)

    assert retried.id == first.id == first_key
    assert _job_ids(scheduler) == sorted([first_key, second_key])
    # Answered from the stored schedule: not validated or recorded again.
    assert validated.await_count == 2
    assert recorded.call_count == 2


def test_a_new_create_at_the_accounts_limit_is_refused(scheduler: Scheduler):
    held = sorted([_add(scheduler, str(uuid.uuid4())).id for _ in range(2)])

    with pytest.raises(ValueError, match="the most one account can have"):
        _add(scheduler, str(uuid.uuid4()))

    assert _job_ids(scheduler) == held


def test_another_account_repeating_a_create_under_its_key_gets_a_conflict(
    scheduler: Scheduler,
):
    """Everything but the account matches, so only the owner check stands
    between this create and the other account's schedule."""
    key = str(uuid.uuid4())
    _add(scheduler, key)

    with pytest.raises(ConflictError):
        _add(scheduler, key, user_id="user-2")

    [job] = scheduler.scheduler.get_jobs(Jobstores.EXECUTION.value)
    assert (job.id, job.kwargs["user_id"]) == (key, "user-1")


def test_two_copies_of_a_create_that_both_miss_the_replay_make_one_schedule(
    scheduler: Scheduler, recorded: MagicMock
):
    """Each copy finds no schedule under the key, then both insert it: the
    jobstore keeps one, and the copy that lost answers with it."""
    key = str(uuid.uuid4())
    both_checked = threading.Barrier(2, timeout=30)
    check_capacity = scheduler._check_schedule_capacity

    def check_then_wait(*args: Any, **kwargs: Any) -> None:
        check_capacity(*args, **kwargs)
        both_checked.wait()

    with (
        patch.object(scheduler, "_check_schedule_capacity", check_then_wait),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        sent = list(pool.map(lambda _: _add(scheduler, key), range(2)))

    assert [schedule.id for schedule in sent] == [key, key]
    assert _job_ids(scheduler) == [key]
    recorded.assert_called_once()


@pytest.fixture
def validated(mocker) -> AsyncMock:
    """Graph validation, which reads the database through a service these
    in-process tests don't start. The route tests above validate for real."""
    return mocker.patch.object(
        execution_utils, "validate_and_construct_node_execution_input", AsyncMock()
    )


@pytest.fixture
def recorded(mocker) -> MagicMock:
    """The feed and analytics record of each new schedule."""
    return mocker.patch.object(schedule_events, "record_schedule_created")


@pytest.fixture
def scheduler(
    monkeypatch: pytest.MonkeyPatch, validated: AsyncMock, recorded: MagicMock
) -> Iterator[Scheduler]:
    """The scheduler ``run_service`` builds, on its own throwaway jobstore,
    with an account allowed two schedules."""
    monkeypatch.setattr(scheduler_module.config, "max_schedules_per_user", 2)
    for name in ("_event_loop", "_event_loop_thread", "_scheduler_instance"):
        monkeypatch.setattr(scheduler_module, name, getattr(scheduler_module, name))
    service = Scheduler(register_system_tasks=False, throwaway_jobstores=True)
    with (
        patch.object(scheduler_module, "load_dotenv"),
        patch("backend.util.service.AppService.run_service"),
        patch("backend.util.service.AppService.cleanup"),
    ):
        service.run_service()
        try:
            yield service
        finally:
            service.cleanup()
            if scheduler_module._event_loop is not None:
                scheduler_module._event_loop.close()


def _add(scheduler: Scheduler, key: str, **change: Any) -> GraphExecutionJobInfo:
    create: dict[str, Any] = {
        "user_id": "user-1",
        "graph_id": "graph-1",
        "graph_version": 1,
        "cron": "0 9 * * *",
        "input_data": INPUTS,
        "input_credentials": {},
        "name": "Morning run",
        "user_timezone": "UTC",
    }
    return scheduler.add_graph_execution_schedule(**(create | change), schedule_id=key)


def _job_ids(scheduler: Scheduler) -> list[str]:
    return sorted(
        job.id for job in scheduler.scheduler.get_jobs(Jobstores.EXECUTION.value)
    )


async def _new_graph(server: SpinTestServer, user: User) -> GraphModel:
    return await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=create_test_graph()), user_id=user.id
    )


async def _create(
    user: User, graph: GraphModel, key: uuid.UUID, **change: Any
) -> GraphExecutionJobInfo:
    """POST /graphs/{graph_id}/schedules, as the schedule dialogs send it."""
    body: dict[str, Any] = {
        "name": "Morning run",
        "cron": "0 9 * * *",
        "inputs": INPUTS,
        "graph_version": graph.version,
        "timezone": "UTC",
        "schedule_id": key,
    }
    return await create_graph_execution_schedule(
        user_id=user.id,
        ctx=RequestContext(
            user_id=user.id,
            org_id="",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=True,
            is_team_admin=False,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        ),
        graph_id=graph.id,
        schedule_params=ScheduleCreationRequest.model_validate(body | change),
    )


async def _schedules(user: User, graph: GraphModel) -> list[GraphExecutionJobInfo]:
    return [
        schedule
        for schedule in await get_scheduler_client().get_execution_schedules(
            graph_id=graph.id, user_id=user.id, include_paused=True
        )
        if isinstance(schedule, GraphExecutionJobInfo)
    ]


async def _schedule_ids(user: User, graph: GraphModel) -> list[str]:
    return sorted(schedule.id for schedule in await _schedules(user, graph))


async def _delete_schedules(user: User, graph: GraphModel) -> None:
    for schedule in await _schedules(user, graph):
        await get_scheduler_client().delete_schedule(schedule.id, user_id=user.id)
