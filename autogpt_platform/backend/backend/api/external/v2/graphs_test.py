from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import prisma.models
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission, SubmissionStatus

from backend.api.features.library import db as library_db
from backend.api.features.library.exceptions import (
    LibraryAgentInAnotherOrganizationError,
)
from backend.api.features.orgs.db import get_user_default_team
from backend.blocks import _base
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import NotAuthorizedError
from backend.util.test import SpinTestServer

from .graphs import (
    create_graph,
    get_graph,
    list_graph_blocks,
    list_graph_credential_requirements,
    list_graph_versions,
    set_active_version,
    update_graph,
)
from .library.agents import update_library_agent as patch_library_agent
from .models import (
    GraphCreateRequest,
    GraphNode,
    GraphSetActiveVersionRequest,
    LibraryAgentUpdateRequest,
)
from .pagination import PageRequest
from .tenancy import TenantContext

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)
_ALL_BUT_RUN_AGENT = [p for p in APIKeyPermission if p != APIKeyPermission.RUN_AGENT]
_REQUEST = GraphCreateRequest(name="Agent", nodes=[], links=[])


@pytest.fixture
def graph_in_org_1(mocker: pytest_mock.MockFixture) -> AsyncMock:
    """The credential acts in org-1, where every version of the graph is."""
    mocker.patch(
        "backend.api.external.v2.graphs.writable_organization_ids",
        new_callable=AsyncMock,
        return_value=frozenset({"org-1"}),
    )
    return mocker.patch(
        "backend.data.graph.get_graph_version_owners",
        new_callable=AsyncMock,
        return_value={(_AUTH.user_id, "org-1")},
    )


@pytest.fixture
def writes(
    mocker: pytest_mock.MockFixture, graph_in_org_1: AsyncMock
) -> dict[str, AsyncMock]:
    mocker.patch(
        "backend.data.graph.get_graph_all_versions",
        new_callable=AsyncMock,
        return_value=[Mock(version=1, is_active=True, user_id=_AUTH.user_id)],
    )
    return {
        name: mocker.patch(target, new_callable=AsyncMock)
        for name, target in {
            "create_graph": "backend.data.graph.create_graph",
            "create_library_agent": "backend.api.features.library.db.create_library_agent",
            "update_agent_version_in_library": "backend.api.features.library.db.update_agent_version_in_library",
        }.items()
    }


@pytest.mark.parametrize(
    "scopes, moves", [(list(APIKeyPermission), True), (_ALL_BUT_RUN_AGENT, False)]
)
async def test_activating_moves_webhook_presets_only_for_a_credential_that_may_run(
    mocker: pytest_mock.MockFixture,
    graph_in_org_1: AsyncMock,
    scopes: list[APIKeyPermission],
    moves: bool,
) -> None:
    graph = Mock(
        user_id=_AUTH.user_id, organization_id="org-1", id="graph-1", version=2
    )
    mocker.patch(
        "backend.data.graph.get_graph", new_callable=AsyncMock, return_value=graph
    )
    mocker.patch(
        "backend.data.graph.get_graph_all_versions",
        new_callable=AsyncMock,
        return_value=[graph],
    )
    mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        return_value=graph,
    )
    mocker.patch("backend.data.graph.set_graph_active_version", new_callable=AsyncMock)
    move_entry = mocker.patch(
        "backend.api.features.library.db.update_agent_version_in_library",
        new_callable=AsyncMock,
    )
    migrate = mocker.patch(
        "backend.api.features.library.db.migrate_webhook_presets_to_new_version",
        new_callable=AsyncMock,
    )

    await set_active_version(
        "graph-1",
        GraphSetActiveVersionRequest(active_graph_version=2),
        auth=_AUTH.model_copy(update={"scopes": scopes}),
    )

    # Only the credential's organization's entry and presets follow it.
    assert move_entry.await_args.kwargs["organization_ids"] == {"org-1"}
    assert migrate.await_count == (1 if moves else 0)
    if moves:
        assert migrate.await_args.kwargs["organization_ids"] == {"org-1"}


@pytest.mark.parametrize(
    "change",
    [
        lambda: update_graph("graph-1", _REQUEST, auth=_AUTH),
        lambda: set_active_version(
            "graph-1", GraphSetActiveVersionRequest(active_graph_version=1), auth=_AUTH
        ),
    ],
    ids=["update", "set-active-version"],
)
@pytest.mark.parametrize(
    "other_version",
    [
        # An untagged version is the personal org's, and org-1 isn't that.
        (_AUTH.user_id, None),
        # A colleague's version would be deactivated with the caller's
        # credentials, and stay active beside the new one.
        ("user-2", "org-1"),
    ],
    ids=["another-organization", "a-colleagues-version"],
)
async def test_a_graph_with_a_version_beyond_the_caller_is_not_changed(
    mocker: pytest_mock.MockFixture,
    writes: dict[str, AsyncMock],
    graph_in_org_1: AsyncMock,
    change,
    other_version: tuple[str, str | None],
) -> None:
    """Its new active version would move another organization's library entry
    and presets, and be the active version there too."""
    graph_in_org_1.return_value = {(_AUTH.user_id, "org-1"), other_version}
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(user_id=_AUTH.user_id, organization_id=None, version=1),
    )
    activate = mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate", new_callable=AsyncMock
    )

    with pytest.raises(NotAuthorizedError):
        await change()

    activate.assert_not_awaited()
    writes["create_graph"].assert_not_awaited()
    writes["update_agent_version_in_library"].assert_not_awaited()


async def test_a_refused_library_move_leaves_the_active_version_alone(
    mocker: pytest_mock.MockFixture, graph_in_org_1: AsyncMock
) -> None:
    """The move can answer 409; after it, the graph would be active on a
    version its library entry isn't on."""
    graph = Mock(
        user_id=_AUTH.user_id, organization_id="org-1", id="graph-1", version=2
    )
    mocker.patch(
        "backend.data.graph.get_graph", new_callable=AsyncMock, return_value=graph
    )
    mocker.patch(
        "backend.data.graph.get_graph_all_versions",
        new_callable=AsyncMock,
        return_value=[graph],
    )
    mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        return_value=graph,
    )
    mocker.patch(
        "backend.api.features.library.db.update_agent_version_in_library",
        new_callable=AsyncMock,
        side_effect=LibraryAgentInAnotherOrganizationError("elsewhere"),
    )
    flip = mocker.patch(
        "backend.data.graph.set_graph_active_version", new_callable=AsyncMock
    )
    deactivate = mocker.patch(
        "backend.api.external.v2.graphs.on_graph_deactivate", new_callable=AsyncMock
    )

    with pytest.raises(LibraryAgentInAnotherOrganizationError):
        await set_active_version(
            "graph-1", GraphSetActiveVersionRequest(active_graph_version=2), auth=_AUTH
        )

    flip.assert_not_awaited()
    # The activation hooks wrote nothing to undo, and the version may be the
    # one already active: deactivating it would detach its live webhooks.
    deactivate.assert_not_awaited()


@pytest.mark.parametrize(
    "save",
    [
        lambda: create_graph(_REQUEST, auth=_AUTH),
        lambda: update_graph("graph-1", _REQUEST, auth=_AUTH),
    ],
    ids=["create", "update"],
)
async def test_a_graph_that_fails_activation_is_not_saved(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock], save
) -> None:
    mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        side_effect=GraphActivationError("missing credential"),
    )

    with pytest.raises(GraphActivationError):
        await save()

    for write in writes.values():
        write.assert_not_awaited()


async def test_a_created_graph_is_saved_with_its_activation_edits(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock]
) -> None:
    activated = Mock()
    mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        return_value=activated,
    )
    mocker.patch("backend.api.external.v2.graphs.Graph.from_internal")

    await create_graph(_REQUEST, auth=_AUTH)

    assert writes["create_graph"].await_args.args[0] is activated
    assert writes["create_library_agent"].await_args.args[0] is activated


@pytest.mark.parametrize(
    "save",
    [
        lambda request: create_graph(request, auth=_AUTH),
        lambda request: update_graph("graph-1", request, auth=_AUTH),
    ],
    ids=["create", "update"],
)
async def test_an_inactive_version_is_saved_without_another_users_credential_refs(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock], save
) -> None:
    activate = mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        side_effect=GraphActivationError("missing credential"),
    )
    clear = mocker.patch(
        "backend.api.external.v2.graphs.clear_unowned_auto_credentials",
        new_callable=AsyncMock,
    )
    mocker.patch("backend.data.graph.get_graph", new_callable=AsyncMock)
    mocker.patch("backend.api.external.v2.graphs.Graph.from_internal")
    writes["create_graph"].return_value = Mock(is_active=False)
    inactive = _REQUEST.model_copy(update={"is_active": False})

    await save(inactive)

    clear.assert_awaited_once()
    assert clear.await_args.args[0] is writes["create_graph"].await_args.args[0]
    activate.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_listed_versions_carry_the_credentials_their_sub_graphs_need(
    server: SpinTestServer,
) -> None:
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    parent = await _parent_of_a_sub_graph(user_id)
    auth = _AUTH.model_copy(update={"user_id": user_id})

    single = await get_graph(parent.id, version=None, auth=auth)
    listed = await list_graph_versions(parent.id, page=PageRequest(limit=10), auth=auth)

    assert single.credentials_input_schema["properties"]
    assert listed.items[0].credentials_input_schema == single.credentials_input_schema


@pytest.mark.asyncio(loop_scope="session")
async def test_a_version_change_returns_the_library_agent_a_fresh_read_does(
    server: SpinTestServer,
) -> None:
    """A sub-graph's credentials included, which the active-version routes rely on."""
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    parent = await _parent_of_a_sub_graph(user_id)
    await library_db.create_library_agent(parent, user_id)

    changed = await library_db.update_agent_version_in_library(
        user_id, parent.id, parent.version
    )
    fresh = await library_db.get_library_agent_by_graph_id(user_id, parent.id)

    assert fresh and fresh.credentials_input_schema["properties"]
    assert changed.credentials_input_schema == fresh.credentials_input_schema


@pytest.mark.asyncio(loop_scope="session")
async def test_a_page_of_versions_resolves_only_its_own_sub_graphs(
    server: SpinTestServer, mocker: pytest_mock.MockFixture
) -> None:
    """Every version's sub-graphs cost round trips, so a page pays only for its own."""
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    first = await graph_db.create_graph(
        graph_db.Graph(name="Versions", description="", nodes=[]), user_id
    )
    for version in (2, 3):
        later = graph_db.make_graph_model(
            graph_db.Graph(
                id=first.id, version=version, name="Versions", description=""
            ),
            user_id,
        )
        later.reassign_ids(user_id=user_id, reassign_graph_id=False)
        await graph_db.create_graph(later, user_id)
    resolve = mocker.patch.object(
        graph_db, "get_sub_graphs_of_each", wraps=graph_db.get_sub_graphs_of_each
    )
    auth = _AUTH.model_copy(update={"user_id": user_id})

    newest = await list_graph_versions(first.id, page=PageRequest(limit=1), auth=auth)
    older = await list_graph_versions(
        first.id, page=PageRequest(limit=1, cursor=newest.next_cursor), auth=auth
    )

    assert [v.version for v in newest.items + older.items] == [3, 2]
    assert newest.total_count == 3
    assert [[g.version for g in call.args[0]] for call in resolve.await_args_list] == [
        [3],
        [2],
    ]


_ACTIVATING_V2 = pytest.mark.parametrize(
    "activate_v2",
    [
        lambda graph_id, request, auth: update_graph(graph_id, request, auth=auth),
        lambda graph_id, request, auth: _save_inactive_then_activate(
            graph_id, request, auth
        ),
    ],
    ids=["update", "set-active-version"],
)


@pytest.mark.asyncio(loop_scope="session")
@_ACTIVATING_V2
async def test_activating_a_version_moves_its_webhook_presets_onto_it(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, activate_v2
) -> None:
    """Left on v1, a preset's webhook URL keeps running the deactivated version."""
    assert await _preset_version_once_v2_is_active(monkeypatch, activate_v2, _AUTH) == 2


@pytest.mark.asyncio(loop_scope="session")
@_ACTIVATING_V2
async def test_a_credential_that_cannot_start_runs_leaves_webhook_presets_in_place(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, activate_v2
) -> None:
    """Moving a preset decides what its next delivery runs, with the
    credentials stored on it: a start, which WRITE_GRAPH alone doesn't grant."""
    auth = _AUTH.model_copy(update={"scopes": _ALL_BUT_RUN_AGENT})

    assert await _preset_version_once_v2_is_active(monkeypatch, activate_v2, auth) == 1


async def _preset_version_once_v2_is_active(
    monkeypatch: pytest.MonkeyPatch, activate_v2, auth: TenantContext
) -> int:
    """Pin a webhook preset to a triggered graph's v1, activate v2 with `auth`,
    and read which version the preset runs."""
    # Trigger blocks are disabled without a platform URL, and CI sets none.
    monkeypatch.setattr(_base.app_config, "platform_base_url", "https://example.com")
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    auth = auth.model_copy(update={"user_id": user_id})
    triggered = GraphCreateRequest(
        name="Triggered",
        nodes=[GraphNode(id="trigger", block_id=GenericWebhookTriggerBlock().id)],
        links=[],
    )
    v1 = await create_graph(triggered, auth=auth)
    webhook = await prisma.models.IntegrationWebhook.prisma().create(
        data={
            "userId": user_id,
            "provider": "generic_webhook",
            "credentialsId": "",
            "webhookType": "plain",
            "resource": "",
            "events": [],
            "config": "{}",
            "secret": "",
            "providerWebhookId": "",
        }
    )
    preset = await prisma.models.AgentPreset.prisma().create(
        data={
            "userId": user_id,
            "name": "On webhook",
            "description": "",
            "agentGraphId": v1.id,
            "agentGraphVersion": v1.version,
            "webhookId": webhook.id,
            # Presets take their graph's organization.
            "organizationId": auth.organization_id,
        }
    )

    await activate_v2(v1.id, triggered, auth)

    preset = await prisma.models.AgentPreset.prisma().find_unique_or_raise(
        where={"id": preset.id}
    )
    return preset.agentGraphVersion


@pytest.mark.asyncio(loop_scope="session")
async def test_a_graph_read_without_a_version_is_of_the_active_one(
    server: SpinTestServer,
) -> None:
    auth = await _new_user_auth()
    graph_id = await _active_v1_and_a_newer_draft(auth)

    graph = await get_graph(graph_id, version=None, auth=auth)

    assert graph.version == 1


@pytest.mark.parametrize(
    "read", [list_graph_blocks, list_graph_credential_requirements]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_page_read_of_a_graph_is_of_its_active_version(
    server: SpinTestServer, read
) -> None:
    auth = await _new_user_auth()
    graph_id = await _active_v1_and_a_newer_draft(auth)

    page = await read(graph_id, page=PageRequest(limit=10), auth=auth)

    # The active v1 has no nodes; the draft's code block needs credentials.
    assert page.items == []


@pytest.mark.asyncio(loop_scope="session")
async def test_a_library_agent_saved_past_its_listed_version_lists_as_published(
    server: SpinTestServer,
) -> None:
    """As its single read reports it: a later save is still the listed agent."""
    auth = await _new_user_auth()
    v1 = await create_graph(_REQUEST, auth=auth)
    listing = await prisma.models.StoreListing.prisma().create(
        data={
            "slug": f"listed-{v1.id}",
            "agentGraphId": v1.id,
            "owningUserId": auth.user_id,
            "hasApprovedVersion": True,
        }
    )
    await prisma.models.StoreListingVersion.prisma().create(
        data={
            "agentGraphId": v1.id,
            "agentGraphVersion": 1,
            "name": "Agent",
            "subHeading": "",
            "description": "",
            "storeListingId": listing.id,
            "submissionStatus": SubmissionStatus.APPROVED,
            "isAvailable": True,
            "imageUrls": [],
            "categories": [],
        }
    )
    await update_graph(v1.id, _REQUEST, auth=auth)

    listed = await library_db.list_library_agents(auth.user_id, published=True)
    unlisted = await library_db.list_library_agents(auth.user_id, published=False)

    assert [agent.graph_version for agent in listed.agents] == [2]
    assert unlisted.agents == []


@pytest.mark.asyncio(loop_scope="session")
async def test_activating_a_draft_deactivates_the_version_that_was_active(
    server: SpinTestServer, mocker: pytest_mock.MockFixture
) -> None:
    """Left active, the old version's webhook triggers keep firing."""
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    auth = _AUTH.model_copy(update={"user_id": user_id})
    v1 = await create_graph(_REQUEST, auth=auth)
    deactivate = mocker.patch(
        "backend.api.external.v2.graphs.on_graph_deactivate", new_callable=AsyncMock
    )

    await _save_inactive_then_activate(v1.id, _REQUEST, auth)

    assert [call.args[0].version for call in deactivate.await_args_list] == [1]


async def _new_user_auth() -> TenantContext:
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    return _AUTH.model_copy(update={"user_id": user_id})


async def _active_v1_and_a_newer_draft(auth: TenantContext) -> str:
    v1 = await create_graph(_REQUEST, auth=auth)
    draft = GraphCreateRequest(
        name="Agent",
        nodes=[GraphNode(id="code", block_id=ExecuteCodeBlock().id)],
        links=[],
        is_active=False,
    )
    await update_graph(v1.id, draft, auth=auth)
    return v1.id


async def _parent_of_a_sub_graph(user_id: str) -> graph_db.GraphModel:
    sub = await graph_db.create_graph(
        graph_db.Graph(
            name="Sub",
            description="",
            nodes=[graph_db.Node(block_id=ExecuteCodeBlock().id)],
        ),
        user_id,
    )
    return await graph_db.create_graph(
        graph_db.Graph(
            name="Parent",
            description="",
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
    )


async def _save_inactive_then_activate(
    graph_id: str, request: GraphCreateRequest, auth: TenantContext
) -> None:
    await update_graph(
        graph_id, request.model_copy(update={"is_active": False}), auth=auth
    )
    await set_active_version(
        graph_id, GraphSetActiveVersionRequest(active_graph_version=2), auth=auth
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_another_organizations_key_cannot_version_a_personal_graph(
    server: SpinTestServer,
) -> None:
    """A graph saved without an organization is the personal org's. A key in
    another organization can read it, but a version it saved or activated would
    move the personal org's library entry and the graph's active version."""
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    personal_org, _ = await get_user_default_team(user_id)
    assert personal_org
    v1 = await _untagged_version(user_id, str(uuid4()), version=1)
    await library_db.create_library_agent(v1, user_id=user_id)
    await _untagged_version(user_id, v1.id, version=2, is_active=False)
    activate_v2 = GraphSetActiveVersionRequest(active_graph_version=2)
    elsewhere = _AUTH.model_copy(
        update={"user_id": user_id, "organization_id": str(uuid4())}
    )

    with pytest.raises(NotAuthorizedError):
        await update_graph(v1.id, _REQUEST, auth=elsewhere)
    with pytest.raises(NotAuthorizedError):
        await set_active_version(v1.id, activate_v2, auth=elsewhere)

    versions = await graph_db.get_graph_all_versions(v1.id, user_id=user_id)
    assert sorted(v.version for v in versions) == [1, 2]
    assert await _library_version(user_id, v1.id) == 1

    personal = _AUTH.model_copy(
        update={"user_id": user_id, "organization_id": personal_org}
    )
    await set_active_version(v1.id, activate_v2, auth=personal)

    assert await _library_version(user_id, v1.id) == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_a_library_agent_is_not_pointed_at_another_organizations_version(
    server: SpinTestServer,
) -> None:
    """The web app saves a version in whichever organization is selected, so a
    graph can span two. A library agent pointed at the other one's version
    would run it, billed to the first."""
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    personal_org, _ = await get_user_default_team(user_id)
    assert personal_org
    v1 = await _untagged_version(user_id, str(uuid4()), version=1)
    [entry] = await library_db.create_library_agent(v1, user_id=user_id)
    await _untagged_version(
        user_id, v1.id, version=2, is_active=False, organization_id=str(uuid4())
    )
    await _untagged_version(user_id, v1.id, version=3, is_active=False)
    personal = _AUTH.model_copy(
        update={"user_id": user_id, "organization_id": personal_org}
    )

    with pytest.raises(NotAuthorizedError):
        await patch_library_agent(
            LibraryAgentUpdateRequest(graph_version=2), entry.id, auth=personal
        )
    assert await _library_version(user_id, v1.id) == 1

    await patch_library_agent(
        LibraryAgentUpdateRequest(graph_version=3), entry.id, auth=personal
    )
    assert await _library_version(user_id, v1.id) == 3


async def _untagged_version(
    user_id: str,
    graph_id: str,
    *,
    version: int,
    is_active: bool = True,
    organization_id: str | None = None,
) -> graph_db.GraphModel:
    """A version saved with no organization, as the app's chat saves them, or
    in *organization_id*."""
    internal = _REQUEST.model_copy(update={"is_active": is_active}).to_internal(
        id=graph_id, version=version
    )
    graph = graph_db.make_graph_model(internal, user_id)
    graph.reassign_ids(user_id=user_id, reassign_graph_id=False)
    return await graph_db.create_graph(
        graph, user_id=user_id, organization_id=organization_id
    )


async def _library_version(user_id: str, graph_id: str) -> int:
    entry = await library_db.get_library_agent_by_graph_id(user_id, graph_id)
    assert entry
    return entry.graph_version


_OWNERS_SECRET = "owner-secret-do-not-leak"


@pytest.mark.parametrize("reader", ["get_graph", "list_graph_versions"])
@pytest.mark.parametrize("owner, shared", [("user-1", False), ("user-2", True)])
async def test_a_graph_someone_else_owns_is_read_as_its_shared_copy(
    mocker: pytest_mock.MockFixture, reader: str, owner: str, shared: bool
) -> None:
    """As on the internal routes: no owner's ids, credentials or secret inputs."""
    graph = graph_db.make_graph_model(
        GraphCreateRequest(
            name="Agent",
            nodes=[
                GraphNode(
                    id="trigger",
                    block_id=GenericWebhookTriggerBlock().id,
                    input_default={"secret_token": _OWNERS_SECRET},
                )
            ],
            links=[],
        ).to_internal(id="graph-1", version=1),
        owner,
    ).model_copy(update={"organization_id": "org-1"})
    mocker.patch(
        "backend.api.external.v2.graphs._get_graph_version",
        new_callable=AsyncMock,
        return_value=graph,
    )
    mocker.patch.object(
        graph_db, "count_graph_versions", new_callable=AsyncMock, return_value=1
    )
    mocker.patch.object(
        graph_db, "get_graph_all_versions", new_callable=AsyncMock, return_value=[graph]
    )

    if reader == "get_graph":
        body = (
            await get_graph(graph_id="graph-1", version=None, auth=_AUTH)
        ).model_dump_json()
    else:
        page = await list_graph_versions(
            graph_id="graph-1", page=PageRequest(limit=20), auth=_AUTH
        )
        body = page.model_dump_json()

    assert (_OWNERS_SECRET not in body) is shared
    assert ('"organization_id":"org-1"' not in body) is shared
