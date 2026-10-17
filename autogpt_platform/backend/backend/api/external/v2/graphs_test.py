from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import prisma.models
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.blocks import _base
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.test import SpinTestServer

from .graphs import (
    create_graph,
    get_graph,
    list_graph_versions,
    set_active_version,
    update_graph,
)
from .models import GraphCreateRequest, GraphNode, GraphSetActiveVersionRequest
from .pagination import PageRequest
from .tenancy import TenantContext

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)
_REQUEST = GraphCreateRequest(name="Agent", nodes=[], links=[])


@pytest.fixture
def writes(mocker: pytest_mock.MockFixture) -> dict[str, AsyncMock]:
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
    sub = await graph_db.create_graph(
        graph_db.Graph(
            name="Sub",
            description="",
            nodes=[graph_db.Node(block_id=ExecuteCodeBlock().id)],
        ),
        user_id,
    )
    parent = await graph_db.create_graph(
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
    auth = _AUTH.model_copy(update={"user_id": user_id})

    single = await get_graph(parent.id, version=None, auth=auth)
    listed = await list_graph_versions(parent.id, page=PageRequest(limit=10), auth=auth)

    assert single.credentials_input_schema["properties"]
    assert listed.items[0].credentials_input_schema == single.credentials_input_schema


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
        graph_db, "get_sub_graphs", wraps=graph_db.get_sub_graphs
    )
    auth = _AUTH.model_copy(update={"user_id": user_id})

    newest = await list_graph_versions(first.id, page=PageRequest(limit=1), auth=auth)
    older = await list_graph_versions(
        first.id, page=PageRequest(limit=1, cursor=newest.next_cursor), auth=auth
    )

    assert [v.version for v in newest.items + older.items] == [3, 2]
    assert newest.total_count == 3
    assert resolve.await_count == 2


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "activate_v2",
    [
        lambda graph_id, request, auth: update_graph(graph_id, request, auth=auth),
        lambda graph_id, request, auth: _save_inactive_then_activate(
            graph_id, request, auth
        ),
    ],
    ids=["update", "set-active-version"],
)
async def test_activating_a_version_moves_its_webhook_presets_onto_it(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, activate_v2
) -> None:
    """Left on v1, a preset's webhook URL keeps running the deactivated version."""
    # Trigger blocks are disabled without a platform URL, and CI sets none.
    monkeypatch.setattr(_base.app_config, "platform_base_url", "https://example.com")
    user_id = str(uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@example.com"})
    auth = _AUTH.model_copy(update={"user_id": user_id})
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
        }
    )

    await activate_v2(v1.id, triggered, auth)

    moved = await prisma.models.AgentPreset.prisma().find_unique_or_raise(
        where={"id": preset.id}
    )
    assert moved.agentGraphVersion == 2


async def _save_inactive_then_activate(
    graph_id: str, request: GraphCreateRequest, auth: TenantContext
) -> None:
    await update_graph(
        graph_id, request.model_copy(update={"is_active": False}), auth=auth
    )
    await set_active_version(
        graph_id, GraphSetActiveVersionRequest(active_graph_version=2), auth=auth
    )
