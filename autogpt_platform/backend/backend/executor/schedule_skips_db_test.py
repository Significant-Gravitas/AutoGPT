"""A schedule whose key was removed is kept, and its owner hears once (#17245).

Before, the first run after the key's removal deleted the schedule and told no
one. These fire the stored schedule the way APScheduler does, through the job
function, against the real scheduler service, database and alert engine. Only
the email queue is stubbed, so what would be sent is what is checked.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import AlertConditionStatus
from prisma.models import AlertCondition
from pydantic import SecretStr

from backend.api.features.library.db import get_library_agent_id_by_graph_id
from backend.api.model import CreateGraph
from backend.blocks.allquiet.incidents import AllQuietCreateIncidentBlock
from backend.blocks.branching import ComparisonOperator, ConditionBlock
from backend.blocks.io import AgentInputBlock
from backend.data.execution import get_graph_executions
from backend.data.graph import Graph, GraphModel, Link, Node
from backend.data.model import APIKeyCredentials, CredentialsMetaInput
from backend.data.user import get_or_create_user
from backend.executor.scheduler import (
    GraphExecutionJobArgs,
    GraphExecutionJobInfo,
    _execute_graph,
)
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.notifications import briefing_runner
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

QUEUE = "backend.notifications.briefing_runner.queue_notification_async"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_schedule_whose_key_was_removed_is_kept_and_its_owner_told_once(
    server: SpinTestServer,
):
    async with _a_schedule_on_a_key(server) as (owner, graph, schedule, field, store):
        removed = schedule.input_credentials[field]
        await store.delete_creds_by_id(owner, removed.id)

        # The first run after the removal: no run, the schedule kept.
        await _fire(owner, graph)
        assert [s.id for s in await _schedules(owner, graph)] == [schedule.id]
        assert await get_graph_executions(graph_id=graph.id, user_id=owner) == []

        # One alert reaches its owner, naming the key and opening the schedule.
        [email] = await _flush_alerts(owner)
        library_agent_id = await get_library_agent_id_by_graph_id(owner, graph.id)
        assert email.data.primary.subject == (
            f"{graph.name} is skipping its scheduled runs: the key it used was "
            "removed"
        )
        assert email.data.primary.preheader == (
            "The All Quiet key “Work” its schedule “Morning run” ran on was removed."
        )
        assert email.data.primary.cta_url.endswith(
            f"/library/agents/{library_agent_id}"
            f"?activeTab=scheduled&activeItem={schedule.id}"
        )

        # The next run fails the same way and tells them nothing more.
        await _fire(owner, graph)
        assert await _flush_alerts(owner) == []
        assert [row.status for row in await _alerts(owner)] == [
            AlertConditionStatus.SENT
        ]

        # Given another key, the next run starts.
        async with _an_api_key(owner, store, "Spare") as spare:
            await get_scheduler_client().update_execution_schedule(
                schedule_id=schedule.id,
                user_id=owner,
                input_credentials={field: _meta(spare)},
            )
            await _fire(owner, graph)

            [run] = await get_graph_executions(graph_id=graph.id, user_id=owner)
            assert run.trigger_ref == schedule.id
            [edited] = await _schedules(owner, graph)
            assert edited.runs_skipped is None
            assert [row.status for row in await _alerts(owner)] == [
                AlertConditionStatus.RESOLVED
            ]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_that_starts_again_withdraws_the_alert_before_it_is_sent(
    server: SpinTestServer,
):
    """The key comes back (a store that couldn't be read, say), and the next
    run starts: the alert still waiting out its debounce is never sent."""
    async with _a_schedule_on_a_key(server) as (owner, graph, schedule, field, store):
        key = schedule.input_credentials[field]
        stored_key = await store.get_creds_by_id(owner, key.id)
        assert stored_key is not None
        await store.delete_creds_by_id(owner, key.id)

        await _fire(owner, graph)
        assert [s.id for s in await _schedules(owner, graph)] == [schedule.id]
        [skipped] = await _schedules(owner, graph)
        assert skipped.runs_skipped is not None

        await store.add_creds(owner, stored_key)
        await _fire(owner, graph)

        [run] = await get_graph_executions(graph_id=graph.id, user_id=owner)
        assert run.trigger_ref == schedule.id
        [resumed] = await _schedules(owner, graph)
        assert resumed.runs_skipped is None
        assert await _flush_alerts(owner) == []


@asynccontextmanager
async def _a_schedule_on_a_key(
    server: SpinTestServer,
) -> AsyncIterator[
    tuple[str, GraphModel, GraphExecutionJobInfo, str, IntegrationCredentialsStore]
]:
    """A new owner's daily schedule of a workflow whose step runs on their
    All Quiet key "Work"."""
    # Untimed: the scheduler may still be starting.
    await get_scheduler_client(should_retry=True).health_check_async()
    owner = str(uuid.uuid4())
    await get_or_create_user({"sub": owner, "email": f"{owner}@example.com"})
    store = IntegrationCredentialsStore()
    async with _an_api_key(owner, store, "Work") as key:
        graph = await server.agent_server.test_create_graph(
            create_graph=CreateGraph(graph=_a_gated_key_step()), user_id=owner
        )
        [field] = graph.credentials_input_schema["properties"]
        schedule = await get_scheduler_client().add_execution_schedule(
            user_id=owner,
            graph_id=graph.id,
            graph_version=graph.version,
            cron="0 9 * * *",
            input_data={"title": "Checkout is down"},
            input_credentials={field: _meta(key)},
            name="Morning run",
            user_timezone="UTC",
        )
        try:
            yield owner, graph, schedule, field, store
        finally:
            try:
                await get_scheduler_client().delete_schedule(schedule.id, user_id=owner)
            except NotFoundError:
                pass  # The code before #17245 deleted it on the first run.


def _a_gated_key_step() -> Graph:
    """An All Quiet step behind a condition never met: each run passes the
    checks on its key, then ends without calling All Quiet."""
    title = Node(block_id=AgentInputBlock().id, input_default={"name": "title"})
    gate = Node(
        block_id=ConditionBlock().id,
        input_default={"operator": ComparisonOperator.EQUAL.value, "value2": "never"},
    )
    incident = Node(block_id=AllQuietCreateIncidentBlock().id)
    links = [
        Link(
            source_id=title.id,
            sink_id=gate.id,
            source_name="result",
            sink_name="value1",
        ),
        Link(
            source_id=gate.id,
            sink_id=incident.id,
            source_name="yes_output",
            sink_name="title",
        ),
    ]
    return Graph(
        name="Incident on a key",
        description="",
        nodes=[title, gate, incident],
        links=links,
    )


@asynccontextmanager
async def _an_api_key(
    owner: str, store: IntegrationCredentialsStore, title: str
) -> AsyncIterator[APIKeyCredentials]:
    key = APIKeyCredentials(
        id=str(uuid.uuid4()),
        provider="allquiet",
        api_key=SecretStr("not-a-real-key"),
        title=title,
        expires_at=None,
    )
    await store.add_creds(owner, key)
    try:
        yield key
    finally:
        await store.delete_creds_by_id(owner, key.id)


def _meta(key: APIKeyCredentials) -> CredentialsMetaInput:
    return CredentialsMetaInput.model_validate(
        {"id": key.id, "provider": key.provider, "type": key.type, "title": key.title}
    )


async def _schedules(owner: str, graph: GraphModel) -> list[GraphExecutionJobInfo]:
    return [
        schedule
        for schedule in await get_scheduler_client().get_execution_schedules(
            graph_id=graph.id, user_id=owner, include_paused=True
        )
        if isinstance(schedule, GraphExecutionJobInfo)
    ]


async def _fire(owner: str, graph: GraphModel) -> None:
    """Run the stored schedule's job the way APScheduler does: with the
    kwargs it stores, read fresh."""
    [stored] = await _schedules(owner, graph)
    await _execute_graph(
        **stored.model_dump(
            mode="json", include=set(GraphExecutionJobArgs.model_fields)
        )
    )


async def _flush_alerts(owner: str) -> list:
    """The alert pass for one owner, past its debounce: the emails it queues."""
    queue = AsyncMock(return_value=SimpleNamespace(success=True))
    with patch(QUEUE, queue):
        await briefing_runner._flush_user_alerts(owner)
    return [call.args[0] for call in queue.await_args_list]


async def _alerts(owner: str) -> list[AlertCondition]:
    return await AlertCondition.prisma().find_many(where={"userId": owner})
