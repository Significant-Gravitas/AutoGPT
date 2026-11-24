"""A schedule for a workflow installed on an expert is refused when it carries a
key the expert was never granted (#17376).

Each run of such a schedule is refused at run time, where the scheduler only
logs the refusal, so the schedule looked active and never ran. These tests send
what the schedule dialogs send through the real routes, the scheduler service
and its job store, and the database.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import fastapi
import httpx
import prisma.models
import pytest
import pytest_asyncio
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from pydantic import BaseModel, SecretStr

from backend.api.features.experts import experts_db
from backend.api.features.schedules.routes import router
from backend.api.model import CreateGraph
from backend.api.rest_api import app as rest_app
from backend.blocks.allquiet.incidents import AllQuietCreateIncidentBlock
from backend.blocks.io import AgentInputBlock
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data.graph import Graph, GraphModel, Link, Node
from backend.data.model import APIKeyCredentials
from backend.data.user import get_or_create_user
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.integrations.credentials_store import (
    IntegrationCredentialsStore,
    openai_credentials,
)
from backend.util.background import _background_tasks
from backend.util.clients import get_scheduler_client
from backend.util.settings import Settings
from backend.util.test import SpinTestServer

REFUSAL = "Grant them on the expert's page to let it run this workflow."
# New Year's midnight: no schedule fires while a test runs.
CRON = "0 0 1 1 *"

# The schedule routes, answering errors as the REST app does.
app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
app.include_router(router)


class Owner(BaseModel):
    user_id: str
    template_id: str
    expert_id: str


@pytest_asyncio.fixture
async def owner(server: SpinTestServer) -> AsyncIterator[Owner]:
    """Someone who hired an expert and granted it none of their keys.
    Everything the test makes is deleted afterwards."""
    # The scheduler service may still be starting when the session's first
    # test gets here, and nothing under test retries a call to it.
    await get_scheduler_client(should_retry=True).health_check_async()
    suffix = uuid.uuid4().hex[:8]
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"expert-grants-{suffix}@example.com",
            "name": "Expert Grants Owner",
        }
    )
    template = await prisma.models.Expert.prisma().create(
        data={
            "name": f"Gale {suffix}",
            "role": "Operations",
            "identity": "You are Gale, an expert.",
            "isTemplate": True,
        }
    )
    try:
        hired = await experts_db.hire_expert(user.id, template.id, None)
        await _settle(hired.expert.id)
        # Settled before any workflow is installed, as opening the expert's
        # page does: a later install then grants nothing by itself.
        await experts_db.settle_credential_seed(user.id, hired.expert.id)
        yield Owner(user_id=user.id, template_id=template.id, expert_id=hired.expert.id)
    finally:
        await _delete_everything_of(user.id, template.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_key_the_expert_was_never_granted_is_refused_and_nothing_is_stored(
    server: SpinTestServer, owner: Owner
):
    """The issue's steps. Such a schedule used to be stored, and its every run
    refused at run time, where nobody saw it."""
    graph = await _install(server, owner, _incident_graph())
    key = await _own_key(owner)

    async with _signed_in(owner) as client:
        response = await _create(client, graph, {"title": "Checkout is down"}, key)

    assert response.status_code == 400, response.text
    assert REFUSAL in response.json()["detail"]
    assert await _schedules(owner) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_a_key_granted_to_the_expert_schedules_its_workflow(
    server: SpinTestServer, owner: Owner
):
    graph = await _install(server, owner, _incident_graph())
    key = await _own_key(owner)
    await experts_db.grant_expert_credentials(owner.user_id, owner.expert_id, [key.id])

    async with _signed_in(owner) as client:
        response = await _create(client, graph, {"title": "Checkout is down"}, key)

    assert response.status_code == 200, response.text
    [stored] = await _schedules(owner)
    assert (stored.id, stored.expert_id) == (response.json()["id"], owner.expert_id)


@pytest.mark.skipif(
    not Settings().secrets.openai_api_key,
    reason="The platform has no OpenAI key of its own here",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_the_platforms_own_key_needs_no_grant(
    server: SpinTestServer, owner: Owner
):
    graph = await _install(server, owner, _summary_graph())

    async with _signed_in(owner) as client:
        response = await _create(
            client, graph, {"prompt": "Sum up the day"}, openai_credentials
        )

    assert response.status_code == 200, response.text
    [stored] = await _schedules(owner)
    assert stored.expert_id == owner.expert_id


@pytest.mark.asyncio(loop_scope="session")
async def test_an_edit_to_a_key_the_expert_was_never_granted_is_refused(
    server: SpinTestServer, owner: Owner
):
    """An edit can change a schedule's keys, so it meets the same grants, and
    the refused edit leaves the schedule on the key it had."""
    graph = await _install(server, owner, _incident_graph())
    granted = await _own_key(owner)
    ungranted = await _own_key(owner)
    await experts_db.grant_expert_credentials(
        owner.user_id, owner.expert_id, [granted.id]
    )
    [field] = graph.credentials_input_schema["properties"]

    async with _signed_in(owner) as client:
        created = await _create(client, graph, {"title": "Checkout is down"}, granted)
        assert created.status_code == 200, created.text
        response = await client.patch(
            f"/schedules/{created.json()['id']}",
            json={"credentials": {field: _meta(ungranted)}},
        )

    assert response.status_code == 400, response.text
    assert REFUSAL in response.json()["detail"]
    [stored] = await _schedules(owner)
    assert stored.input_credentials[field].id == granted.id


@pytest.mark.asyncio(loop_scope="session")
async def test_an_edit_of_inputs_alone_meets_the_grants_of_the_key_it_keeps(
    server: SpinTestServer, owner: Owner
):
    """An edit is checked on the keys the schedule will hold after it, so one
    that sends no key still meets a grant revoked since, and changes nothing."""
    graph = await _install(server, owner, _incident_graph())
    key = await _own_key(owner)
    await experts_db.grant_expert_credentials(owner.user_id, owner.expert_id, [key.id])

    async with _signed_in(owner) as client:
        created = await _create(client, graph, {"title": "Checkout is down"}, key)
        assert created.status_code == 200, created.text
        await experts_db.revoke_expert_credential(
            owner.user_id, owner.expert_id, key.id
        )
        response = await client.patch(
            f"/schedules/{created.json()['id']}",
            json={"inputs": {"title": "Checkout is back"}},
        )

    assert response.status_code == 400, response.text
    assert REFUSAL in response.json()["detail"]
    [stored] = await _schedules(owner)
    assert stored.input_data == {"title": "Checkout is down"}


@pytest.mark.asyncio(loop_scope="session")
async def test_an_edit_of_the_name_or_the_timing_meets_no_grants(
    server: SpinTestServer, owner: Owner
):
    """The edit dialog sends a name and a cron, never inputs or keys. A
    schedule whose key lost its grant can still be renamed and retimed."""
    graph = await _install(server, owner, _incident_graph())
    key = await _own_key(owner)
    await experts_db.grant_expert_credentials(owner.user_id, owner.expert_id, [key.id])

    async with _signed_in(owner) as client:
        created = await _create(client, graph, {"title": "Checkout is down"}, key)
        assert created.status_code == 200, created.text
        await experts_db.revoke_expert_credential(
            owner.user_id, owner.expert_id, key.id
        )
        response = await client.patch(
            f"/schedules/{created.json()['id']}",
            json={"name": "Evening run", "cron": "30 0 1 1 *"},
        )

    assert response.status_code == 200, response.text
    [stored] = await _schedules(owner)
    assert (stored.name, stored.cron) == ("Evening run", "30 0 1 1 *")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_workflow_no_expert_has_installed_is_scheduled_on_any_key(
    server: SpinTestServer, owner: Owner
):
    """Grants bind an expert's workflows only. The owner's other workflows are
    scheduled, and edited, on keys no expert was granted."""
    graph = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=_incident_graph()), user_id=owner.user_id
    )
    first = await _own_key(owner)
    second = await _own_key(owner)
    [field] = graph.credentials_input_schema["properties"]

    async with _signed_in(owner) as client:
        created = await _create(client, graph, {"title": "Checkout is down"}, first)
        assert created.status_code == 200, created.text
        edited = await client.patch(
            f"/schedules/{created.json()['id']}",
            json={"credentials": {field: _meta(second)}},
        )

    assert edited.status_code == 200, edited.text
    [stored] = await _schedules(owner)
    assert stored.expert_id is None
    assert stored.input_credentials[field].id == second.id


async def _settle(expert_id: str) -> None:
    """Wait out the work a hire starts behind it, so none outlives the test."""
    await asyncio.gather(
        *(task for task in _background_tasks if expert_id in task.get_name())
    )


async def _delete_everything_of(user_id: str, template_id: str) -> None:
    scheduler = get_scheduler_client()
    for job in await scheduler.get_execution_schedules(
        user_id=user_id, include_paused=True
    ):
        await scheduler.delete_schedule(job.id, user_id=user_id)
    await prisma.models.Expert.prisma().delete_many(
        where={"OR": [{"ownerUserId": user_id}, {"id": template_id}]}
    )
    await prisma.models.LibraryAgent.prisma().delete_many(where={"userId": user_id})
    await prisma.models.User.prisma().delete_many(where={"id": user_id})


async def _install(server: SpinTestServer, owner: Owner, graph: Graph) -> GraphModel:
    """The workflow, in the owner's library and installed on their expert."""
    created = await server.agent_server.test_create_graph(
        create_graph=CreateGraph(graph=graph), user_id=owner.user_id
    )
    library_agent = await prisma.models.LibraryAgent.prisma().find_first_or_raise(
        where={"userId": owner.user_id, "agentGraphId": created.id}
    )
    await experts_db.install_workflow(
        owner.user_id, owner.expert_id, library_agent_id=library_agent.id
    )
    return created


def _incident_graph() -> Graph:
    """A title input feeding a block that runs on an All Quiet key."""
    return _graph(
        "IncidentGraph",
        Node(block_id=AgentInputBlock().id, input_default={"name": "title"}),
        Node(block_id=AllQuietCreateIncidentBlock().id),
        sink_name="title",
    )


def _summary_graph() -> Graph:
    """A prompt input feeding an LLM block on an OpenAI model."""
    return _graph(
        "SummaryGraph",
        Node(block_id=AgentInputBlock().id, input_default={"name": "prompt"}),
        Node(
            block_id=AITextGeneratorBlock().id,
            input_default={"model": LLMModel.GPT5_MINI.value},
        ),
        sink_name="prompt",
    )


def _graph(name: str, source: Node, sink: Node, *, sink_name: str) -> Graph:
    link = Link(
        source_id=source.id,
        sink_id=sink.id,
        source_name="result",
        sink_name=sink_name,
    )
    return Graph(name=name, description="", nodes=[source, sink], links=[link])


async def _own_key(owner: Owner) -> APIKeyCredentials:
    """An All Quiet key of the owner's own; nothing runs on it."""
    key = APIKeyCredentials(
        id=str(uuid.uuid4()),
        provider="allquiet",
        api_key=SecretStr("not-a-real-key"),
        title="All Quiet",
        expires_at=None,
    )
    await IntegrationCredentialsStore().add_creds(owner.user_id, key)
    return key


@asynccontextmanager
async def _signed_in(owner: Owner) -> AsyncIterator[httpx.AsyncClient]:
    """A client of the schedule routes, signed in as the owner. It runs on the
    test's own event loop, which the database client is bound to."""
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": owner.user_id,
        "role": "user",
        "email": "owner@example.com",
    }
    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        user_id=owner.user_id,
        # The org `test_create_graph` puts the workflow in.
        org_id=f"test-org-{owner.user_id}",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


async def _create(
    client: httpx.AsyncClient,
    graph: GraphModel,
    inputs: dict[str, Any],
    key: APIKeyCredentials,
) -> httpx.Response:
    """POST /graphs/{graph_id}/schedules as the schedule dialogs send it, the
    expert's page's included: no expert named, so the one the workflow is
    installed on is found."""
    [field] = graph.credentials_input_schema["properties"]
    return await client.post(
        f"/graphs/{graph.id}/schedules",
        json={
            "name": "Morning run",
            "cron": CRON,
            "inputs": inputs,
            "credentials": {field: _meta(key)},
            "graph_version": graph.version,
            "timezone": "UTC",
        },
    )


def _meta(key: APIKeyCredentials) -> dict[str, Any]:
    return {
        "id": key.id,
        "provider": key.provider,
        "type": key.type,
        "title": key.title,
    }


async def _schedules(owner: Owner) -> list[GraphExecutionJobInfo]:
    return [
        schedule
        for schedule in await get_scheduler_client().get_execution_schedules(
            user_id=owner.user_id, include_paused=True
        )
        if isinstance(schedule, GraphExecutionJobInfo)
    ]
