"""A marketplace workflow runs as whoever runs it, so the sub-workflows it
starts must be ones its creator could use: the creator's own, or versions
published in the Marketplace, and never one of the person running it.

A copy of one belongs to whoever makes it, so it keeps none of the sub-workflow
nodes that name its maker's own workflows: those are blanked as it is made.

These go through the real store, library and executor service.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import prisma.models
import pytest
from pydantic import BaseModel

import backend.util.json
from backend.api.features.library import db as library_db
from backend.api.features.library.db import add_store_agent_to_library
from backend.api.features.store import db as store_db
from backend.api.model import CreateGraph
from backend.blocks.agent import NO_WORKFLOW_TO_RUN, AgentExecutorBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.blocks.text import FillTextTemplateBlock
from backend.copilot.tools.agent_generator.core import (
    get_agent_as_json,
    save_agent_to_library,
)
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionWithNodes,
    get_graph_execution,
)
from backend.data.graph import (
    BaseGraph,
    Graph,
    GraphModel,
    Link,
    Node,
    copy_graph,
    create_graph,
    get_graph,
    graph_owner_id,
)
from backend.data.user import get_or_create_user
from backend.util.exceptions import PreconditionFailed
from backend.util.test import SpinTestServer


@pytest.fixture(autouse=True)
def _runners_belong_to_their_org():
    """Check memberships for real, as manager_test.py does: these runs are
    picked up by the executor service, which checks them in its own process."""


@pytest.fixture(autouse=True)
def _default_team_untenanted():
    """Look up the user's own org and team for real, as manager_test.py does."""


@pytest.fixture(autouse=True)
def _no_embeddings(mocker):
    mocker.patch(
        "backend.api.features.store.db.ensure_embedding",
        new_callable=AsyncMock,
        return_value=True,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_listing_runs_none_of_its_installers_own_workflows(
    server: SpinTestServer,
):
    author, installer, other_creator = [await _new_user() for _ in range(3)]
    authors_own = await create_graph(_labelled_echo("author"), author)
    others = await create_graph(_labelled_echo("other"), other_creator)
    await _publish(others, other_creator, reviewer_id=author)
    installers = await create_graph(_labelled_echo("installer"), installer)
    await _publish(installers, installer, reviewer_id=author)
    installers_draft = await create_graph(
        _labelled_echo("installer-draft", graph_id=installers.id, version=2),
        installer,
    )
    listing = await create_graph(
        _workflow_running(
            {
                "from_author": authors_own,
                "from_other_creator": others,
                "from_installer": installers,
                "from_installer_draft": installers_draft,
            }
        ),
        author,
    )
    # Approved as a listing could be before submission and review checked the
    # sub-workflows it runs.
    with patch.object(store_db, "foreign_sub_graph_refs", AsyncMock(return_value=[])):
        listing_version_id = await _publish(listing, author, reviewer_id=other_creator)
    await add_store_agent_to_library(
        store_listing_version_id=listing_version_id, user_id=installer
    )

    graph_exec = await server.agent_server.test_execute_graph(
        user_id=installer,
        graph_id=listing.id,
        graph_version=listing.version,
        node_input={"topic": "hello"},
    )
    run = await _finished_run(installer, graph_exec.id)

    assert run.outputs == {
        "from_author": ["author:hello"],
        "from_other_creator": ["other:hello"],
    }
    refused = {
        node_exec.node_id: node_exec.output_data["error"]
        for node_exec in run.node_executions
        if node_exec.status == ExecutionStatus.FAILED
    }
    refusal = (
        f"Workflow #{installers.id} is one of yours, so a workflow someone "
        "else created can't run it"
    )
    assert refused == {
        _node_running(listing, installers).id: [refusal],
        _node_running(listing, installers_draft).id: [refusal],
    }
    assert (
        await prisma.models.AgentGraphExecution.prisma().count(
            where={"agentGraphId": installers.id}
        )
        == 0
    )
    sub_runs = await prisma.models.AgentGraphExecution.prisma().find_many(
        where={"parentGraphExecutionId": run.id}
    )
    assert sorted(
        (sub_run.agentGraphId, sub_run.userId, sub_run.executionStatus)
        for sub_run in sub_runs
    ) == sorted(
        [
            (authors_own.id, installer, ExecutionStatus.COMPLETED),
            (others.id, installer, ExecutionStatus.COMPLETED),
        ]
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_listing_may_run_only_its_creators_or_published_workflows(
    server: SpinTestServer,
):
    author, other_creator = [await _new_user() for _ in range(2)]
    authors_own = await create_graph(_labelled_echo("author"), author)
    others = await create_graph(_labelled_echo("other"), other_creator)
    others_version_id = await _publish(others, other_creator, reviewer_id=author)
    others_draft = await create_graph(
        _labelled_echo("other-draft", graph_id=others.id, version=2), other_creator
    )

    with pytest.raises(PreconditionFailed, match=f"#{others.id} v2"):
        await _submit(
            await create_graph(
                _workflow_running({"a": authors_own, "b": others_draft}), author
            ),
            author,
        )

    await _publish(
        await create_graph(_workflow_running({"a": authors_own, "b": others}), author),
        author,
        reviewer_id=other_creator,
    )

    pending = await _submit(
        await create_graph(_workflow_running({"a": authors_own, "b": others}), author),
        author,
    )
    await _review(others_version_id, reviewer_id=author, approve=False)
    with pytest.raises(PreconditionFailed, match=f"#{others.id} v1"):
        await _review(pending, reviewer_id=other_creator, approve=True)


@pytest.fixture
def _library_side_effects(mocker):
    """A new library entry reads the user's schedules and starts image and
    embedding work, none of which these tests look at."""
    scheduler = AsyncMock()
    scheduler.get_graph_execution_schedules = AsyncMock(return_value=[])
    for site in ("db", "_schedule_info"):
        mocker.patch(
            f"backend.api.features.library.{site}.get_scheduler_client",
            return_value=scheduler,
        )
    mocker.patch.object(library_db, "add_generated_agent_image", AsyncMock())
    mocker.patch.object(library_db, "schedule_library_agent_embedding", MagicMock())


class _Listing(BaseModel):
    """An approved listing of the author's, in the installer's library. Its
    sub-workflow nodes name the author's own workflow, another creator's
    published one, the installer's published workflow and that workflow's
    unpublished next version, and a sub-workflow of the author's that names
    the installer's workflow in turn."""

    author: str
    installer: str
    others: GraphModel
    installers: GraphModel
    sub_workflows: dict[str, GraphModel]
    graph: GraphModel
    version_id: str
    library_agent_id: str


@pytest.mark.asyncio(loop_scope="session")
async def test_a_copy_of_a_listing_runs_none_of_its_copiers_own_workflows(
    server: SpinTestServer, _library_side_effects
):
    """The installer duplicates the listing and runs the copy, which is
    theirs, so the check above no longer stands between it and their own
    workflows. The nodes that named those were blanked when the copy was made,
    in the copy and in the sub-workflow of the author's it carries along: they
    fail and say why, and the rest runs."""
    listing = await _a_listing_naming_the_installers_workflows()

    copy = await library_db.fork_library_agent(
        listing.library_agent_id, listing.installer
    )
    graph_exec = await server.agent_server.test_execute_graph(
        user_id=listing.installer,
        graph_id=copy.graph_id,
        graph_version=copy.graph_version,
        node_input={"topic": "hello"},
    )
    run = await _finished_run(listing.installer, graph_exec.id)

    assert run.outputs == {
        "from_author": ["author:hello"],
        "from_other_creator": ["other:hello"],
    }
    copied = await get_graph(
        copy.graph_id,
        copy.graph_version,
        user_id=listing.installer,
        include_subgraphs=True,
    )
    assert copied is not None
    relay = _sub_graph_named_by(copied, "via_authors_relay")
    relay_run_row = await prisma.models.AgentGraphExecution.prisma().find_first(
        where={"parentGraphExecutionId": run.id, "agentGraphId": relay.id}
    )
    assert relay_run_row is not None
    relay_run = await _finished_run(listing.installer, relay_run_row.id)
    failed = {
        node_exec.node_id: node_exec.output_data.get("error")
        for graph_run in (run, relay_run)
        for node_exec in graph_run.node_executions
        if node_exec.status == ExecutionStatus.FAILED
    }
    blanked = [
        _sub_workflow_feeding(copied, "from_installer").id,
        _sub_workflow_feeding(copied, "from_installer_draft").id,
        _sub_workflow_feeding(relay, "result").id,
    ]
    assert {node_id: failed.get(node_id) for node_id in blanked} == {
        node_id: [NO_WORKFLOW_TO_RUN] for node_id in blanked
    }
    assert (
        await prisma.models.AgentGraphExecution.prisma().count(
            where={"agentGraphId": listing.installers.id}
        )
        == 0
    )


async def _duplicate(listing: _Listing, server: SpinTestServer) -> str:
    copy = await library_db.fork_library_agent(
        listing.library_agent_id, listing.installer
    )
    return copy.graph_id


async def _copy_for_a_team(listing: _Listing, server: SpinTestServer) -> str:
    copy = await copy_graph(listing.graph.id, listing.graph.version, listing.installer)
    return copy.id


async def _upload_its_file(listing: _Listing, server: SpinTestServer) -> str:
    """Download the listing's file from the Marketplace and upload it, as the
    library's import does."""
    downloaded = backend.util.json.dumps(await store_db.get_agent(listing.version_id))
    upload = CreateGraph.model_validate(
        {"graph": json.loads(downloaded), "source": "upload"}
    )
    return (await server.agent_server.test_create_graph(upload, listing.installer)).id


async def _save_an_edit(listing: _Listing, server: SpinTestServer) -> str:
    """Save an edit of the listing, as Otto does, which makes it a copy."""
    edited = _workflow_running(listing.sub_workflows)
    edited.id = listing.graph.id
    copy, _ = await library_db.update_graph_in_library(edited, listing.installer)
    assert copy.forked_from_id == listing.graph.id
    return copy.id


async def _save_as_a_new_workflow(listing: _Listing, server: SpinTestServer) -> str:
    """Save the listing's JSON, as Otto's tools hand it out, as a new workflow,
    as Otto does when it makes the user their own version of one."""
    agent_json = await get_agent_as_json(listing.library_agent_id, listing.installer)
    assert agent_json is not None and agent_json["id"] == listing.graph.id
    saved, _ = await save_agent_to_library(agent_json, listing.installer)
    return saved.id


@pytest.mark.parametrize(
    "make_copy",
    [
        _duplicate,
        _copy_for_a_team,
        _upload_its_file,
        _save_an_edit,
        _save_as_a_new_workflow,
    ],
    ids=[
        "duplicate",
        "copy-for-a-team",
        "upload-its-file",
        "save-an-edit",
        "save-as-a-new-workflow",
    ],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_every_copy_of_a_listing_blanks_its_copiers_own_workflows(
    server: SpinTestServer,
    _library_side_effects,
    make_copy: Callable[[_Listing, SpinTestServer], Awaitable[str]],
):
    listing = await _a_listing_naming_the_installers_workflows()

    copy_id = await make_copy(listing, server)

    copied = await get_graph(
        copy_id, None, user_id=listing.installer, include_subgraphs=True
    )
    assert copied is not None and copy_id != listing.graph.id
    assert {
        output_name: _sub_workflow_feeding(copied, output_name).input_default[
            "graph_id"
        ]
        for output_name in (
            "from_installer",
            "from_installer_draft",
            "from_other_creator",
        )
    } == {
        "from_installer": "",
        "from_installer_draft": "",
        "from_other_creator": listing.others.id,
    }
    assert listing.installers.id not in {
        node.input_default.get("graph_id")
        for graph in [copied, *copied.sub_graphs]
        for node in graph.nodes
        if node.block_id == AgentExecutorBlock().id
    }


@pytest.mark.parametrize(
    "save",
    [library_db.update_graph_in_library, library_db.create_graph_in_library],
    ids=["as-an-edit", "as-a-new-workflow"],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_an_edited_copy_keeps_the_sub_workflows_its_editor_added(
    server: SpinTestServer,
    _library_side_effects,
    save: Callable[[Graph, str], Awaitable[tuple[GraphModel, object]]],
):
    listing = await _a_listing_naming_the_installers_workflows()
    installers_other = await create_graph(
        _labelled_echo("installer-other"), listing.installer
    )
    edited = _workflow_running(
        {
            "from_installer": listing.installers,
            "added_by_installer": installers_other,
        }
    )
    edited.id = listing.graph.id

    copy, _ = await save(edited, listing.installer)

    assert copy.id != listing.graph.id
    assert _sub_workflow_feeding(copy, "from_installer").input_default["graph_id"] == ""
    assert (
        _sub_workflow_feeding(copy, "added_by_installer").input_default["graph_id"]
        == installers_other.id
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_workflow_of_ones_own_keeps_its_sub_workflows(
    server: SpinTestServer, _library_side_effects
):
    """Saving a workflow that runs another of one's own keeps it, and so does
    saving its JSON as a new workflow with Otto, and uploading its exported
    file, which carries a copy of that one to run."""
    owner = await _new_user()
    own = await create_graph(_labelled_echo("own"), owner)

    saved = await server.agent_server.test_create_graph(
        CreateGraph(graph=_workflow_running({"mine": own}), source="builder"), owner
    )
    assert _sub_workflow_feeding(saved, "mine").input_default["graph_id"] == own.id

    agent_json = await get_agent_as_json(saved.id, owner)
    assert agent_json is not None and agent_json["id"] == saved.id
    resaved, _ = await save_agent_to_library(agent_json, owner)
    assert resaved.id != saved.id
    assert _sub_workflow_feeding(resaved, "mine").input_default["graph_id"] == own.id

    exported = await get_graph(saved.id, saved.version, user_id=owner, for_export=True)
    assert exported is not None
    uploaded = await server.agent_server.test_create_graph(
        CreateGraph.model_validate(
            {"graph": json.loads(backend.util.json.dumps(exported)), "source": "upload"}
        ),
        owner,
    )
    carried = _sub_workflow_feeding(uploaded, "mine").input_default["graph_id"]
    assert carried not in ("", own.id)
    assert await graph_owner_id(carried) == owner


async def _a_listing_naming_the_installers_workflows() -> _Listing:
    author, installer, other_creator = [await _new_user() for _ in range(3)]
    others = await create_graph(_labelled_echo("other"), other_creator)
    await _publish(others, other_creator, reviewer_id=author)
    installers = await create_graph(_labelled_echo("installer"), installer)
    await _publish(installers, installer, reviewer_id=author)
    sub_workflows = {
        "from_author": await create_graph(_labelled_echo("author"), author),
        "from_other_creator": others,
        "from_installer": installers,
        "from_installer_draft": await create_graph(
            _labelled_echo("installer-draft", graph_id=installers.id, version=2),
            installer,
        ),
        "via_authors_relay": await create_graph(
            _workflow_running({"result": installers}), author
        ),
    }
    graph = await create_graph(_workflow_running(sub_workflows), author)
    # Approved as a listing could be before submission and review checked the
    # sub-workflows it runs.
    with patch.object(store_db, "foreign_sub_graph_refs", AsyncMock(return_value=[])):
        version_id = await _publish(graph, author, reviewer_id=other_creator)
    library_agent = await add_store_agent_to_library(
        store_listing_version_id=version_id, user_id=installer
    )
    return _Listing(
        author=author,
        installer=installer,
        others=others,
        installers=installers,
        sub_workflows=sub_workflows,
        graph=graph,
        version_id=version_id,
        library_agent_id=library_agent.id,
    )


async def _new_user() -> str:
    user = await get_or_create_user(
        {"sub": str(uuid4()), "email": f"sub-workflows-{uuid4()}@example.com"}
    )
    return user.id


def _labelled_echo(label: str, graph_id: str | None = None, version: int = 1) -> Graph:
    """A workflow whose output `result` is "<label>:<its input topic>"."""
    topic = Node(
        block_id=AgentInputBlock().id,
        input_default={"name": "topic", "title": "Topic", "value": ""},
    )
    fill = Node(
        block_id=FillTextTemplateBlock().id,
        input_default={"format": label + ":{{topic}}"},
    )
    result = Node(
        block_id=AgentOutputBlock().id,
        input_default={"name": "result", "title": "Result", "value": ""},
    )
    return Graph(
        id=graph_id or str(uuid4()),
        version=version,
        is_active=True,
        name=f"{label} workflow",
        description=f"Labels its topic with {label}",
        nodes=[topic, fill, result],
        links=[
            Link(
                source_id=topic.id,
                sink_id=fill.id,
                source_name="result",
                sink_name="values_#_topic",
            ),
            Link(
                source_id=fill.id,
                sink_id=result.id,
                source_name="output",
                sink_name="value",
            ),
        ],
    )


def _workflow_running(sub_graphs: dict[str, GraphModel]) -> Graph:
    """A workflow that hands its topic to each of *sub_graphs* and outputs what
    each returns under its key."""
    topic = Node(
        block_id=AgentInputBlock().id,
        input_default={"name": "topic", "title": "Topic", "value": ""},
    )
    nodes, links = [topic], []
    for output_name, sub_graph in sub_graphs.items():
        run = Node(
            block_id=AgentExecutorBlock().id,
            input_default={
                "user_id": "",
                "inputs": {},
                "graph_id": sub_graph.id,
                "graph_version": sub_graph.version,
                "input_schema": sub_graph.input_schema,
                "output_schema": sub_graph.output_schema,
            },
        )
        output = Node(
            block_id=AgentOutputBlock().id,
            input_default={"name": output_name, "title": output_name, "value": ""},
        )
        nodes += [run, output]
        links += [
            Link(
                source_id=topic.id,
                sink_id=run.id,
                source_name="result",
                sink_name="topic",
            ),
            Link(
                source_id=run.id,
                sink_id=output.id,
                source_name="result",
                sink_name="value",
            ),
        ]
    return Graph(
        id=str(uuid4()),
        version=1,
        is_active=True,
        name="Workflow with sub-workflows",
        description="Runs its sub-workflows",
        nodes=nodes,
        links=links,
    )


def _node_running(graph: GraphModel, sub_graph: GraphModel) -> Node:
    return next(
        node
        for node in graph.nodes
        if node.input_default.get("graph_id") == sub_graph.id
        and node.input_default.get("graph_version") == sub_graph.version
    )


def _sub_workflow_feeding(graph: BaseGraph, output_name: str) -> Node:
    """The sub-workflow node whose result *graph* outputs as *output_name*."""
    output = next(
        node
        for node in graph.nodes
        if node.block_id == AgentOutputBlock().id
        and node.input_default.get("name") == output_name
    )
    source_id = next(
        link.source_id for link in graph.links if link.sink_id == output.id
    )
    return next(node for node in graph.nodes if node.id == source_id)


def _sub_graph_named_by(graph: GraphModel, output_name: str) -> BaseGraph:
    graph_id = _sub_workflow_feeding(graph, output_name).input_default["graph_id"]
    return next(sub_graph for sub_graph in graph.sub_graphs if sub_graph.id == graph_id)


async def _submit(graph: GraphModel, user_id: str) -> str:
    submission = await store_db.create_store_submission(
        user_id=user_id,
        graph_id=graph.id,
        graph_version=graph.version,
        slug=f"sub-workflows-{uuid4().hex[:12]}",
        name=graph.name,
        sub_heading="Sub-workflow test",
        description=graph.description,
        categories=["testing"],
        image_urls=["https://example.com/image.jpg"],
    )
    assert submission.listing_version_id is not None
    return submission.listing_version_id


async def _review(listing_version_id: str, reviewer_id: str, approve: bool) -> None:
    await store_db.review_store_submission(
        store_listing_version_id=listing_version_id,
        is_approved=approve,
        external_comments="",
        internal_comments="",
        reviewer_id=reviewer_id,
        notify_creator=False,
    )


async def _publish(graph: GraphModel, user_id: str, reviewer_id: str) -> str:
    listing_version_id = await _submit(graph, user_id)
    await _review(listing_version_id, reviewer_id=reviewer_id, approve=True)
    return listing_version_id


async def _finished_run(user_id: str, graph_exec_id: str) -> GraphExecutionWithNodes:
    """The run once it has ended, however it ended."""
    for _ in range(240):
        run = await get_graph_execution(
            user_id=user_id, execution_id=graph_exec_id, include_node_executions=True
        )
        if isinstance(run, GraphExecutionWithNodes) and run.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.TERMINATED,
        ):
            return run
        await asyncio.sleep(0.5)
    raise AssertionError(f"Run #{graph_exec_id} did not end within 2 minutes")
