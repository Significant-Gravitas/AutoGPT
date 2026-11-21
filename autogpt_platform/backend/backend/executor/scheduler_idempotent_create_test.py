"""A schedule create sent again under its key makes one schedule (#16901).

A create the scheduler applied, whose answer was lost (a timeout, or a 503
while the scheduler restarted), is retried by the user. When every attempt
minted its own id, each retry added another recurring schedule, which then
ran, and charged, again. The first tests run the real route against the real
scheduler service, its jobstore and the database. The per-account limit is too
high to fill there, so the last ones run a scheduler built in this process, on
its real jobstore, with the limit lowered.
"""

import json
import pickle
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, nullcontext
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs.auth.models import RequestContext
from pydantic import SecretStr

from backend.api.features.schedules.model import ScheduleCreationRequest
from backend.api.features.schedules.routes import create_graph_execution_schedule
from backend.api.model import CreateGraph
from backend.blocks.allquiet.incidents import AllQuietCreateIncidentBlock
from backend.blocks.io import AgentInputBlock
from backend.data.graph import Graph, GraphModel, Link, Node
from backend.data.model import APIKeyCredentials, CredentialsMetaInput, User
from backend.executor import schedule_events
from backend.executor import scheduler as scheduler_module
from backend.executor import utils as execution_utils
from backend.executor.scheduler import (
    GraphExecutionJobInfo,
    Jobstores,
    Scheduler,
    _build_trigger,
    _fires_alike,
)
from backend.integrations.credentials_store import IntegrationCredentialsStore
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
async def test_a_retried_create_with_credentials_returns_the_first_schedule(
    server: SpinTestServer,
):
    """Most schedules carry credentials, which the scheduler stores and reads
    back before comparing them with the retry's."""
    owner = await create_test_user()
    async with _an_api_key(owner) as api_key:
        graph = await _new_graph(server, owner, _graph_needing_a_key())
        [field] = graph.credentials_input_schema["properties"]
        sent: dict[str, Any] = {
            "inputs": {"title": "Checkout is down"},
            "credentials": {
                field: {
                    "id": api_key.id,
                    "provider": api_key.provider,
                    "type": api_key.type,
                    "title": api_key.title,
                }
            },
        }
        key = uuid.uuid4()
        try:
            first = await _create(owner, graph, key, **sent)
            retried = await _create(owner, graph, key, **sent)

            assert await _schedule_ids(owner, graph) == [str(key)]
            assert first.id == retried.id == str(key)
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
    jobstore keeps one, and the copy that lost answers with it. The capacity
    lock keeps two creates from checking at once, so it is taken out here to
    reach the insert race this answer still guards."""
    key = str(uuid.uuid4())
    both_checked = threading.Barrier(2, timeout=30)
    check_capacity = scheduler._check_schedule_capacity

    def check_then_wait(*args: Any, **kwargs: Any) -> None:
        check_capacity(*args, **kwargs)
        both_checked.wait()

    with (
        patch.object(scheduler, "_schedule_capacity_lock", nullcontext()),
        patch.object(scheduler, "_check_schedule_capacity", check_then_wait),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        sent = list(pool.map(lambda _: _add(scheduler, key), range(2)))

    assert [schedule.id for schedule in sent] == [key, key]
    assert _job_ids(scheduler) == [key]
    recorded.assert_called_once()


def test_a_copy_that_takes_the_last_slot_mid_check_answers_the_other(
    scheduler: Scheduler, recorded: MagicMock
):
    """This copy finds its key free, then the other copy gets in before this
    one counts the account's schedules, and takes the last of its two. This
    copy is answered with that schedule, not refused at the limit."""
    held = _add(scheduler, str(uuid.uuid4()))
    key = str(uuid.uuid4())
    count = scheduler._schedule_count
    other_copy_sent = False

    def count_once_the_other_copy_is_in(*args: Any, **kwargs: Any) -> int:
        nonlocal other_copy_sent
        if not other_copy_sent:
            other_copy_sent = True
            _add(scheduler, key)
        return count(*args, **kwargs)

    with patch.object(scheduler, "_schedule_count", count_once_the_other_copy_is_in):
        answered = _add(scheduler, key)

    assert answered.id == key
    assert _job_ids(scheduler) == sorted([held.id, key])
    assert recorded.call_count == 2


def test_a_different_create_that_loses_the_insert_race_gets_a_conflict(
    scheduler: Scheduler, recorded: MagicMock
):
    """Two different creates under one key both find it free; the one the
    jobstore turns away is refused, not answered with the other's schedule.
    The capacity lock is taken out, as above."""
    key = str(uuid.uuid4())
    both_checked = threading.Barrier(2, timeout=30)
    check_capacity = scheduler._check_schedule_capacity

    def check_then_wait(*args: Any, **kwargs: Any) -> None:
        check_capacity(*args, **kwargs)
        both_checked.wait()

    def send(cron: str) -> GraphExecutionJobInfo | ConflictError:
        try:
            return _add(scheduler, key, cron=cron)
        except ConflictError as refused:
            return refused

    with (
        patch.object(scheduler, "_schedule_capacity_lock", nullcontext()),
        patch.object(scheduler, "_check_schedule_capacity", check_then_wait),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        outcomes = list(pool.map(send, ["0 9 * * *", "30 18 * * *"]))

    [made] = [o for o in outcomes if isinstance(o, GraphExecutionJobInfo)]
    assert sum(isinstance(o, ConflictError) for o in outcomes) == 1
    [job] = scheduler.scheduler.get_jobs(Jobstores.EXECUTION.value)
    assert (job.id, job.kwargs["cron"]) == (key, made.cron)
    recorded.assert_called_once()


def test_two_creates_racing_for_the_accounts_last_slot_leave_it_at_its_limit(
    scheduler: Scheduler,
):
    """Two creates under different keys each count one free slot. The first
    is held right after its count until the second reaches the capacity lock,
    or, without the lock, counts too: only one of them may take the slot
    (#16910)."""
    held = _add(scheduler, str(uuid.uuid4()))
    first_counted, release_first = threading.Event(), threading.Event()
    second_reached_the_count = threading.Event()
    count = scheduler._schedule_count
    capacity_lock = scheduler._schedule_capacity_lock

    class NotingTheSecondCreate:
        """The capacity lock, noting the second create as it reaches it."""

        def __enter__(self) -> None:
            if first_counted.is_set():
                second_reached_the_count.set()
            capacity_lock.acquire()

        def __exit__(self, *exc: object) -> None:
            capacity_lock.release()

    def count_and_hold_the_first(*args: Any, **kwargs: Any) -> int:
        counted = count(*args, **kwargs)
        if not first_counted.is_set():
            first_counted.set()
            release_first.wait(30)
        else:
            second_reached_the_count.set()
        return counted

    def send(key: str) -> GraphExecutionJobInfo | ValueError:
        try:
            return _add(scheduler, key)
        except ValueError as refused:
            return refused

    with (
        patch.object(scheduler, "_schedule_capacity_lock", NotingTheSecondCreate()),
        patch.object(scheduler, "_schedule_count", count_and_hold_the_first),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        first = pool.submit(send, str(uuid.uuid4()))
        try:
            assert first_counted.wait(30)
            second = pool.submit(send, str(uuid.uuid4()))
            assert second_reached_the_count.wait(30)
        finally:
            release_first.set()
        outcomes = [first.result(30), second.result(30)]

    made = [o for o in outcomes if isinstance(o, GraphExecutionJobInfo)]
    refused = [o for o in outcomes if isinstance(o, ValueError)]
    assert len(made) == len(refused) == 1
    assert "the most one account can have" in str(refused[0])
    assert _job_ids(scheduler) == sorted([held.id, made[0].id])


def test_a_limit_refusal_under_a_key_another_create_took_is_a_conflict(
    scheduler: Scheduler,
):
    """A different create under this key takes the account's last slot
    between this one's lookup and its count: that's a conflict, not the
    limit, and its schedule is left alone."""
    held = _add(scheduler, str(uuid.uuid4()))
    key = str(uuid.uuid4())
    count = scheduler._schedule_count
    other_create_sent = False

    def count_once_another_create_is_in(*args: Any, **kwargs: Any) -> int:
        nonlocal other_create_sent
        if not other_create_sent:
            other_create_sent = True
            _add(scheduler, key, cron="30 18 * * *")
        return count(*args, **kwargs)

    with (
        patch.object(scheduler, "_schedule_count", count_once_another_create_is_in),
        pytest.raises(ConflictError),
    ):
        _add(scheduler, key)

    assert _job_ids(scheduler) == sorted([held.id, key])
    assert scheduler.scheduler.get_job(key).kwargs["cron"] == "30 18 * * *"


@pytest.mark.parametrize(
    "stored, sent",
    [
        pytest.param({"flag": True}, {"flag": 1}, id="true-vs-1"),
        pytest.param(
            {"rows": [{"on": False}]}, {"rows": [{"on": 0}]}, id="nested-false-vs-0"
        ),
    ],
)
def test_inputs_that_differ_only_in_json_type_are_another_create(
    scheduler: Scheduler, stored: dict[str, Any], sent: dict[str, Any]
):
    """Python has ``True == 1``; the inputs ``true`` and ``1`` still differ."""
    key = str(uuid.uuid4())
    _add(scheduler, key, input_data=stored)

    with pytest.raises(ConflictError):
        _add(scheduler, key, input_data=sent)

    [job] = scheduler.scheduler.get_jobs(Jobstores.EXECUTION.value)
    assert json.dumps(job.kwargs["input_data"]) == json.dumps(stored)


def test_a_retry_with_its_inputs_in_another_order_is_the_same_create(
    scheduler: Scheduler,
):
    key = str(uuid.uuid4())
    first = _add(scheduler, key, input_data={"a": 1, "b": {"c": 2, "d": 3}})

    retried = _add(scheduler, key, input_data={"b": {"d": 3, "c": 2}, "a": 1})

    assert retried.id == first.id == key


def test_a_key_that_is_not_a_uuid_is_refused(scheduler: Scheduler):
    """System jobs share the jobstore's ids, and none of theirs is a UUID."""
    with pytest.raises(ValueError):
        _add(scheduler, "refresh_near_expiry_credentials")

    assert _job_ids(scheduler) == []


def test_a_stored_trigger_fires_alike_its_rebuild_and_unlike_a_changed_one():
    """A retry is matched against the trigger the jobstore unpickles."""

    def build(cron: str = "0-30/5 9 * * 1-5", timezone: str = "Europe/Paris"):
        return _build_trigger(cron=cron, run_at=None, user_timezone=timezone)

    stored = pickle.loads(pickle.dumps(build()))

    assert _fires_alike(stored, build())
    assert not _fires_alike(stored, build(timezone="UTC"))
    assert not _fires_alike(stored, build(cron="0-30/5 10 * * 1-5"))
    # APScheduler's own field equality ignores a range's step.
    assert not _fires_alike(stored, build(cron="0-30/10 9 * * 1-5"))


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
    with an account allowed two schedules. Paused: these tests store jobs and
    never run them, even when a slow run passes their 09:00 UTC."""
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
        service.scheduler.pause()
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


async def _new_graph(
    server: SpinTestServer, user: User, graph: Graph | None = None
) -> GraphModel:
    return await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=graph or create_test_graph()), user_id=user.id
    )


def _graph_needing_a_key() -> Graph:
    """A title input feeding a block that runs on an All Quiet key."""
    nodes = [
        Node(block_id=AgentInputBlock().id, input_default={"name": "title"}),
        Node(block_id=AllQuietCreateIncidentBlock().id),
    ]
    links = [
        Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="result",
            sink_name="title",
        )
    ]
    return Graph(name="IncidentGraph", description="", nodes=nodes, links=links)


@asynccontextmanager
async def _an_api_key(user: User) -> AsyncIterator[APIKeyCredentials]:
    """An All Quiet key the user holds for the test only; nothing runs on it."""
    api_key = APIKeyCredentials(
        id=str(uuid.uuid4()),
        provider="allquiet",
        api_key=SecretStr("not-a-real-key"),
        title="All Quiet",
        expires_at=None,
    )
    store = IntegrationCredentialsStore()
    await store.add_creds(user.id, api_key)
    try:
        yield api_key
    finally:
        await store.delete_creds_by_id(user.id, api_key.id)


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
            # The org `test_create_graph` puts the workflow in.
            org_id=f"test-org-{user.id}",
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
