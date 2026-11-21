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
from backend.blocks.agent import AgentExecutorBlock
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
            await prisma.organization.delete_many(
                where={"OR": [{"bootstrapUserId": uid}, {"slug": f"team-{uid}"}]}
            )
        finally:
            if connected_here and prisma.is_connected():
                await prisma.disconnect()


@pytest_asyncio.fixture(loop_scope="session")
async def reviewer_id(user_id: str) -> AsyncIterator[str]:
    """A second user to review with, so the creator's org can't stand in for it."""
    uid = f"reviewer-{uuid.uuid4().hex[:12]}"
    await prisma.user.create(data={"id": uid, "email": f"{uid}@example.com"})
    try:
        yield uid
    finally:
        await prisma.user.delete_many(where={"id": uid})
        await prisma.organization.delete_many(where={"bootstrapUserId": uid})


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


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("submitted_in", ["personal", "untagged", "team"])
async def test_an_approved_sub_workflow_is_listed_in_its_workflows_org(
    user_id, reviewer_id, submitted_in
):
    """The sub-workflow of an approved submission is listed in the submission's
    org: the team org it was made in, or the creator's personal org for one
    written before its version carried an org. Never the reviewer's."""
    await _creator_profile(user_id)
    team_org_id = await _team_org(user_id) if submitted_in == "team" else None
    sub = await _graph(user_id, team_org_id)
    submission = await _submit(
        user_id, await _workflow_using(user_id, sub, team_org_id), team_org_id
    )
    if submitted_in == "untagged":
        await prisma.storelistingversion.update(
            where={"id": submission.listing_version_id},
            data={"organizationId": None},
        )

    await _approve(submission, reviewer_id)

    expected = team_org_id or (await _personal_tenancy(user_id))[0]
    sub_listing = await prisma.storelisting.find_first_or_raise(
        where={"agentGraphId": sub.id}, include={"Versions": True}
    )
    assert sub_listing.owningOrgId == expected
    assert [v.organizationId for v in sub_listing.Versions or []] == [expected]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_sub_workflows_new_version_joins_its_listings_org(user_id, reviewer_id):
    """A sub-workflow listed from a team org keeps its later versions there, even
    when the workflow that brings a new one was submitted from the personal org."""
    await _creator_profile(user_id)
    team_org_id = await _team_org(user_id)
    sub = await _graph(user_id, team_org_id)
    await _approve(
        await _submit(
            user_id, await _workflow_using(user_id, sub, team_org_id), team_org_id
        ),
        reviewer_id,
    )
    sub_v2 = await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(
                id=sub.id, version=2, name="Briefing", description="Summarises it"
            ),
            user_id,
        ),
        user_id,
        organization_id=team_org_id,
    )

    await _approve(
        await _submit(user_id, await _workflow_using(user_id, sub_v2, None), None),
        reviewer_id,
    )

    sub_listing = await prisma.storelisting.find_first_or_raise(
        where={"agentGraphId": sub.id}, include={"Versions": True}
    )
    assert sorted(
        (v.agentGraphVersion, v.organizationId) for v in sub_listing.Versions or []
    ) == [(1, team_org_id), (2, team_org_id)]


async def _personal_tenancy(user_id: str) -> tuple[str, str]:
    member = await prisma.orgmember.find_first_or_raise(
        where={"userId": user_id, "isOwner": True, "Org": {"is": {"isPersonal": True}}}
    )
    team = await prisma.team.find_first_or_raise(
        where={"orgId": member.orgId, "isDefault": True}
    )
    return member.orgId, team.id


async def _graph(
    user_id: str, organization_id: str | None = None
) -> graph_db.GraphModel:
    return await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(name="Briefing", description="Summarises the week"),
            user_id,
        ),
        user_id,
        organization_id=organization_id,
    )


async def _creator_profile(user_id: str) -> None:
    await prisma.profile.create(
        data={
            "userId": user_id,
            "username": user_id,
            "name": "Creator",
            "description": "Makes workflows",
            "links": [],
        }
    )


async def _team_org(user_id: str) -> str:
    """A team org *user_id* owns besides their personal one."""
    org = await prisma.organization.create(
        data={"name": "Creator's team", "slug": f"team-{user_id}"}
    )
    await prisma.orgmember.create(
        data={
            "orgId": org.id,
            "userId": user_id,
            "status": "ACTIVE",
            "isOwner": True,
            "isAdmin": True,
        }
    )
    return org.id


async def _workflow_using(
    user_id: str, sub: graph_db.GraphModel, organization_id: str | None
) -> graph_db.GraphModel:
    """A workflow whose one node runs *sub*."""
    return await graph_db.create_graph(
        graph_db.make_graph_model(
            graph_db.Graph(
                name="Briefing with a helper",
                description="Summarises the week",
                nodes=[
                    graph_db.Node(
                        block_id=AgentExecutorBlock().id,
                        input_default={
                            "user_id": user_id,
                            "graph_id": sub.id,
                            "graph_version": sub.version,
                            "inputs": {},
                            "input_schema": {},
                            "output_schema": {},
                        },
                    )
                ],
            ),
            user_id,
        ),
        user_id,
        organization_id=organization_id,
    )


async def _submit(
    user_id: str, graph: graph_db.GraphModel, organization_id: str | None
):
    return await store_db.create_store_submission(
        user_id=user_id,
        graph_id=graph.id,
        graph_version=graph.version,
        slug=f"briefing-{uuid.uuid4().hex[:8]}",
        name="Briefing",
        sub_heading="Summarises the week",
        description="Summarises the week",
        organization_id=organization_id,
    )


async def _approve(submission, reviewer_id: str) -> None:
    await store_db.review_store_submission(
        submission.listing_version_id,
        is_approved=True,
        external_comments="",
        internal_comments="",
        reviewer_id=reviewer_id,
        notify_creator=False,
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
