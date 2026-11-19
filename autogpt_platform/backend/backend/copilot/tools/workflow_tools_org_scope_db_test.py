"""Otto runs a workflow, a preset or a marketplace listing of the user's own, and
sets up a trigger on a workflow, only from a chat in the organization it belongs
to, and bills that organization: the real run_agent and trigger tools against
workflows, library entries and presets in Postgres. Only the executor's queue is
stubbed."""

import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import pytest_asyncio
from pydantic import BaseModel

from backend.api.features.library import db as library_db
from backend.api.features.library.model import (
    LibraryAgent,
    LibraryAgentPreset,
    LibraryAgentPresetCreatable,
)
from backend.api.features.orgs.db import get_user_default_team
from backend.blocks.basic import StoreValueBlock
from backend.copilot.gate.subject import NO_OP
from backend.data import graph as graph_db
from backend.data.db import prisma
from backend.data.execution import ExecutionStatus

from ._test_data import (
    _create_user_with_profile,
    _ensure_db_connected,
    _publish_to_store,
    make_session,
)
from .run_agent import RunAgentTool
from .setup_agent_webhook_trigger import SetupAgentWebhookTriggerTool


class World(BaseModel):
    user: str
    orgs: dict[str, tuple[str, str]]
    calls: dict[str, dict[str, str]]
    trigger_calls: dict[str, dict[str, str]]


@pytest_asyncio.fixture(loop_scope="session")
async def world(mocker) -> AsyncIterator[World]:
    mocker.patch("backend.api.features.store.db.ensure_embedding", AsyncMock())
    await _ensure_db_connected()
    user = await _create_user_with_profile("Org scope")
    creator = await _create_user_with_profile("Org scope creator")
    users = {"in": [user.id, creator.id]}
    personal = await get_user_default_team(user.id)
    work = await prisma.organization.create(
        data={"name": "Work", "slug": f"work-{uuid.uuid4().hex[:8]}"}
    )
    try:
        work_team = await prisma.team.create(
            data={"name": "General", "orgId": work.id, "isDefault": True}
        )
        await prisma.orgmember.create(
            data={"orgId": work.id, "userId": user.id, "status": "ACTIVE"}
        )
        graph, library_agent = await _workflow(user.id, *personal)
        preset = await _preset(user.id, graph)
        listed, _ = await _workflow(user.id, *personal)
        others, _ = await _workflow(
            creator.id, *await get_user_default_team(creator.id)
        )
        others_slug = await _slug(creator, others)
        await library_db.create_library_agent(
            others,
            user.id,
            create_library_agents_for_sub_graphs=False,
            organization_id=personal[0],
            team_id=personal[1],
        )
        # Its own org check alone refuses this one: the lookup finds a
        # marketplace workflow in the library from any org.
        marketplace_preset = await _preset(user.id, others)
        # The org-scoped lookup alone refuses this one: untagged, it passes the
        # preset's own org check.
        untagged_preset = await _preset(user.id, graph)
        await prisma.execute_raw(
            'UPDATE "AgentPreset" SET "organizationId" = NULL, "teamId" = NULL '
            'WHERE "id" = $1',
            untagged_preset.id,
        )
        calls = {
            "library agent": {"library_agent_id": library_agent.id},
            "graph id": {"library_agent_id": graph.id},
            "preset": {"preset_id": preset.id},
            "own marketplace listing": {
                "username_agent_slug": await _slug(user, listed)
            },
            "someone else's marketplace listing": {"username_agent_slug": others_slug},
            "preset on a marketplace workflow": {"preset_id": marketplace_preset.id},
            "untagged preset": {"preset_id": untagged_preset.id},
        }
        yield World(
            user=user.id,
            orgs={"personal": personal, "work": (work.id, work_team.id)},
            calls=calls,
            trigger_calls={
                "library agent": {"library_agent_id": library_agent.id},
                "graph id": {"graph_id": graph.id},
            },
        )
    finally:
        await prisma.storelisting.delete_many(where={"owningUserId": users})
        await prisma.agentpreset.delete_many(where={"userId": users})
        await prisma.libraryagent.delete_many(where={"userId": users})
        await prisma.user.delete_many(where={"id": users})
        await prisma.organization.delete_many(where={"id": work.id})


@pytest.fixture
def queue(mocker) -> AsyncMock:
    mocker.patch("backend.copilot.tools.run_agent.track_chat_outcome")
    mocker.patch(
        "backend.copilot.tools.run_agent._safe_link_to_chat_share", AsyncMock()
    )
    return mocker.patch(
        "backend.copilot.tools.run_agent.execution_utils.add_graph_execution",
        AsyncMock(return_value=MagicMock(id="exec-1", status=ExecutionStatus.QUEUED)),
    )


@pytest.mark.parametrize(
    "call, chat_org, runs",
    [
        ("library agent", "personal", True),
        ("library agent", "work", False),
        ("graph id", "personal", True),
        ("graph id", "work", False),
        ("preset", "personal", True),
        ("preset", "work", False),
        ("own marketplace listing", "personal", True),
        ("own marketplace listing", "work", False),
        ("someone else's marketplace listing", "work", True),
        ("preset on a marketplace workflow", "personal", True),
        ("preset on a marketplace workflow", "work", False),
        ("untagged preset", "personal", True),
        ("untagged preset", "work", False),
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_otto_runs_a_workflow_only_from_its_organization(
    world: World, queue: AsyncMock, call: str, chat_org: str, runs: bool
):
    session = make_session(user_id=world.user)
    session.organization_id, session.team_id = world.orgs[chat_org]

    response = await RunAgentTool().execute(
        user_id=world.user,
        session_id=session.session_id,
        tool_call_id=str(uuid.uuid4()),
        session=session,
        **world.calls[call],
    )

    assert isinstance(response.output, str)
    result = orjson.loads(response.output)
    if runs:
        assert result.get("execution_id") == "exec-1", result
        assert queue.await_args.kwargs["organization_id"] == world.orgs[chat_org][0]
    else:
        assert result.get("error") == "workflow_in_another_organization", result
        queue.assert_not_awaited()


@pytest.mark.parametrize(
    "call",
    [
        "library agent",
        "graph id",
        "preset",
        "own marketplace listing",
        "preset on a marketplace workflow",
        "untagged preset",
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_the_gate_asks_nothing_for_a_workflow_the_run_refuses(
    world: World, call: str
):
    """Were the lookup to raise past the gate, the call would be refused as
    unchecked instead of telling the user where the workflow is."""
    session = make_session(user_id=world.user)
    session.organization_id, session.team_id = world.orgs["work"]

    subject = await RunAgentTool().gate_subject(
        world.user, session, dict(world.calls[call])
    )

    assert subject is NO_OP


@pytest.mark.parametrize("call", ["library agent", "graph id"])
@pytest.mark.parametrize(
    "chat_org, error",
    [("personal", "no_webhook_trigger"), ("work", "workflow_in_another_organization")],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_otto_sets_up_a_trigger_only_from_its_workflows_organization(
    world: World, call: str, chat_org: str, error: str
):
    """The workflow has no trigger block: from its own org, the tool gets past
    the lookup to that check."""
    session = make_session(user_id=world.user)
    session.organization_id, session.team_id = world.orgs[chat_org]

    response = await SetupAgentWebhookTriggerTool().execute(
        user_id=world.user,
        session_id=session.session_id,
        tool_call_id=str(uuid.uuid4()),
        session=session,
        name="On push",
        **world.trigger_calls[call],
    )

    assert isinstance(response.output, str)
    assert orjson.loads(response.output).get("error") == error


async def _workflow(
    user_id: str, org_id: str | None, team_id: str | None
) -> tuple[graph_db.GraphModel, LibraryAgent]:
    return await library_db.create_graph_in_library(
        graph_db.Graph(
            name="Briefing",
            description="",
            nodes=[
                graph_db.Node(
                    block_id=StoreValueBlock().id, input_default={"input": "x"}
                )
            ],
        ),
        user_id,
        organization_id=org_id,
        team_id=team_id,
    )


async def _preset(user_id: str, graph: graph_db.GraphModel) -> LibraryAgentPreset:
    return await library_db.create_preset(
        user_id,
        LibraryAgentPresetCreatable(
            graph_id=graph.id,
            graph_version=graph.version,
            inputs={},
            credentials={},
            name="Weekly",
            description="",
        ),
    )


async def _slug(user, graph: graph_db.GraphModel) -> str:
    listing = await _publish_to_store(
        user.id,
        graph,
        slug_prefix="briefing",
        name="Briefing",
        description="",
        sub_heading="",
        categories=["testing"],
    )
    return f"{user.email.split('@')[0]}/{listing.slug}"
