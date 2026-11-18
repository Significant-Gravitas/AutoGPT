"""A row written without an org lands in its owner's personal org and default
team, the tenancy the org bootstrap gives every older row: each writer of the
tables it tags, against Postgres.

Lives outside ``orgs/``, whose conftest stubs the ``server`` fixture; these
tests need only Postgres, so they connect Prisma themselves.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
import pytest_asyncio
from prisma.enums import APIKeyPermission
from pydantic import BaseModel

from backend.api.features.builder.db import update_search
from backend.api.features.builder.model import SearchEntry
from backend.api.features.library import db as library_db
from backend.api.features.store import db as store_db
from backend.copilot import db as chat_db
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data import human_review
from backend.data.auth.api_key import create_api_key
from backend.data.db import prisma
from backend.data.integrations import Webhook, create_webhook
from backend.data.tenancy import tenancy_for_new_row
from backend.integrations.providers import ProviderName


class Tenancy(BaseModel):
    organization_id: str | None
    team_id: str | None


@pytest_asyncio.fixture(loop_scope="session")
async def user_id() -> AsyncIterator[str]:
    """A user with no personal org yet, so each writer has to make one."""
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    uid = f"new-row-{uuid.uuid4().hex[:12]}"
    await prisma.user.create(data={"id": uid, "email": f"{uid}@example.com"})
    try:
        yield uid
    finally:
        try:
            await prisma.integrationwebhook.delete_many(where={"userId": uid})
            await prisma.storelisting.delete_many(where={"owningUserId": uid})
            await prisma.user.delete_many(where={"id": uid})
            await prisma.organization.delete_many(where={"bootstrapUserId": uid})
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


@pytest.mark.asyncio(loop_scope="session")
async def test_without_an_org_a_new_row_gets_the_owners_personal_org(user_id):
    first = await tenancy_for_new_row(user_id, None)
    again = await tenancy_for_new_row(user_id, None)

    assert first == again == await _personal_tenancy(user_id)
    assert await tenancy_for_new_row(user_id, "org-given", None) == ("org-given", None)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "writer",
    [
        "graph",
        "graph_execution",
        "chat_session",
        "plain_message_chat",
        "library_agent",
        "library_folder",
        "webhook",
        "human_review",
        "gate_approval",
    ],
)
async def test_a_writer_given_no_org_tags_the_row_with_the_personal_org(
    user_id, writer
):
    written = await _WRITERS[writer](user_id)

    assert written == Tenancy.model_validate(
        dict(zip(("organization_id", "team_id"), await _personal_tenancy(user_id)))
    )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("writer", ["api_key", "builder_search"])
async def test_an_org_only_writer_given_no_org_tags_the_personal_org(user_id, writer):
    written = await _WRITERS[writer](user_id)

    assert written.organization_id == (await _personal_tenancy(user_id))[0]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_auto_approval_record_takes_its_runs_org(user_id):
    org = await prisma.organization.create(
        data={"name": "Shared", "slug": f"shared-{uuid.uuid4().hex[:8]}"}
    )
    try:
        graph = await _graph(user_id)
        run = await execution_db.create_graph_execution(
            graph_id=graph.id,
            graph_version=graph.version,
            starting_nodes_input=[],
            inputs={},
            user_id=user_id,
            organization_id=org.id,
        )
        node_id = str(uuid.uuid4())
        await human_review.create_auto_approval_record(
            user_id=user_id,
            node_id=node_id,
            payload={},
            graph_exec_id=run.id,
            graph_id=graph.id,
            graph_version=graph.version,
        )
        record = await prisma.pendinghumanreview.find_unique_or_raise(
            where={"nodeExecId": human_review.get_auto_approve_key(run.id, node_id)}
        )
        assert (record.organizationId, record.teamId) == (org.id, None)
    finally:
        await prisma.organization.delete_many(where={"id": org.id})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_store_submission_tags_its_listing_and_version(user_id):
    await prisma.profile.create(
        data={
            "userId": user_id,
            "username": user_id,
            "name": "Creator",
            "description": "Makes workflows",
            "links": [],
        }
    )
    graph = await _graph(user_id)
    personal_org_id, _ = await _personal_tenancy(user_id)

    submission = await store_db.create_store_submission(
        user_id=user_id,
        graph_id=graph.id,
        graph_version=graph.version,
        slug=f"briefing-{uuid.uuid4().hex[:8]}",
        name="Briefing",
        sub_heading="Summarises the week",
        description="Summarises the week",
    )

    version = await prisma.storelistingversion.find_unique_or_raise(
        where={"id": submission.listing_version_id}, include={"StoreListing": True}
    )
    assert version.StoreListing is not None
    assert (version.organizationId, version.StoreListing.owningOrgId) == (
        personal_org_id,
        personal_org_id,
    )


async def _personal_tenancy(user_id: str) -> tuple[str, str]:
    member = await prisma.orgmember.find_first_or_raise(
        where={"userId": user_id, "isOwner": True, "Org": {"is": {"isPersonal": True}}}
    )
    team = await prisma.team.find_first_or_raise(
        where={"orgId": member.orgId, "isDefault": True}
    )
    return member.orgId, team.id


async def _graph(user_id: str) -> graph_db.GraphModel:
    return await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(name="Briefing", description="Summarises the week"),
            user_id,
        ),
        user_id,
    )


async def _write_graph(user_id: str) -> Tenancy:
    graph = await _graph(user_id)
    row = await prisma.agentgraph.find_unique_or_raise(
        where={"graphVersionId": {"id": graph.id, "version": graph.version}}
    )
    return Tenancy(organization_id=row.organizationId, team_id=row.teamId)


async def _write_graph_execution(user_id: str) -> Tenancy:
    graph = await _graph(user_id)
    run = await execution_db.create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={},
        user_id=user_id,
    )
    return Tenancy(organization_id=run.organization_id, team_id=run.team_id)


async def _write_chat_session(user_id: str) -> Tenancy:
    session_id = str(uuid.uuid4())
    await chat_db.create_chat_session(session_id, user_id)
    return await _chat_tenancy(session_id)


async def _write_plain_message_chat(user_id: str) -> Tenancy:
    session_id = await chat_db.append_plain_session_message(
        user_id, "Your weekly briefing is ready.", str(uuid.uuid4())
    )
    assert session_id is not None
    return await _chat_tenancy(session_id)


async def _chat_tenancy(session_id: str) -> Tenancy:
    row = await prisma.chatsession.find_unique_or_raise(where={"id": session_id})
    return Tenancy(organization_id=row.organizationId, team_id=row.teamId)


async def _write_library_agent(user_id: str) -> Tenancy:
    [entry] = await library_db.create_library_agent(await _graph(user_id), user_id)
    row = await prisma.libraryagent.find_unique_or_raise(where={"id": entry.id})
    return Tenancy(organization_id=row.organizationId, team_id=row.teamId)


async def _write_library_folder(user_id: str) -> Tenancy:
    folder = await library_db.create_folder(user_id, "Reports")
    row = await prisma.libraryfolder.find_unique_or_raise(where={"id": folder.id})
    return Tenancy(organization_id=row.organizationId, team_id=row.teamId)


async def _write_webhook(user_id: str) -> Tenancy:
    webhook = await create_webhook(
        Webhook(
            id=str(uuid.uuid4()),
            user_id=user_id,
            provider=ProviderName.GITHUB,
            credentials_id="",
            webhook_type="repo",
            resource="owner/repo",
            events=["push"],
            secret="not-a-secret",
            provider_webhook_id="",
        )
    )
    return Tenancy(organization_id=webhook.organization_id, team_id=webhook.team_id)


async def _write_human_review(user_id: str) -> Tenancy:
    graph = await _graph(user_id)
    run = await execution_db.create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={},
        user_id=user_id,
    )
    node_exec_id = str(uuid.uuid4())
    await human_review.get_or_create_human_review(
        user_id=user_id,
        node_exec_id=node_exec_id,
        input_data={},
        message="Send it?",
        editable=False,
        graph_exec_id=run.id,
        graph_id=graph.id,
        graph_version=graph.version,
        alert=False,
    )
    return await _review_tenancy(node_exec_id)


async def _write_gate_approval(user_id: str) -> Tenancy:
    graph = await _graph(user_id)
    run = await execution_db.create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[],
        inputs={},
        user_id=user_id,
    )
    node_exec_id = str(uuid.uuid4())
    await human_review.record_gate_approval(
        user_id=user_id,
        node_exec_id=node_exec_id,
        graph_exec_id=run.id,
        graph_id=graph.id,
        graph_version=graph.version,
        input_data={},
        block_name="Send email",
        message="Approved in chat",
    )
    return await _review_tenancy(node_exec_id)


async def _review_tenancy(node_exec_id: str) -> Tenancy:
    row = await prisma.pendinghumanreview.find_unique_or_raise(
        where={"nodeExecId": node_exec_id}
    )
    return Tenancy(organization_id=row.organizationId, team_id=row.teamId)


async def _write_api_key(user_id: str) -> Tenancy:
    key, _ = await create_api_key(
        "Reports", user_id, permissions=[APIKeyPermission.READ_GRAPH]
    )
    row = await prisma.apikey.find_unique_or_raise(where={"id": key.id})
    return Tenancy(organization_id=row.organizationId, team_id=row.teamId)


async def _write_builder_search(user_id: str) -> Tenancy:
    search_id = await update_search(user_id, SearchEntry(search_query="weekly"))
    row = await prisma.buildersearchhistory.find_unique_or_raise(
        where={"id": search_id}
    )
    return Tenancy(organization_id=row.organizationId, team_id=None)


_WRITERS: dict[str, Callable[[str], Awaitable[Tenancy]]] = {
    "graph": _write_graph,
    "graph_execution": _write_graph_execution,
    "chat_session": _write_chat_session,
    "plain_message_chat": _write_plain_message_chat,
    "library_agent": _write_library_agent,
    "library_folder": _write_library_folder,
    "webhook": _write_webhook,
    "human_review": _write_human_review,
    "gate_approval": _write_gate_approval,
    "api_key": _write_api_key,
    "builder_search": _write_builder_search,
}
