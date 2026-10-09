from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_mock
from fastapi import HTTPException
from prisma.enums import APIKeyPermission

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

from .graphs import (
    create_graph,
    get_graph,
    list_graph_versions,
    set_active_version,
    update_graph,
)
from .models import GraphCreateRequest, GraphSetActiveVersionRequest
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
        return_value=[Mock(version=1, is_active=True, organization_id="org-1")],
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


async def test_an_inactive_version_is_saved_without_another_users_credential_refs(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock]
) -> None:
    activate = mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate", new_callable=AsyncMock
    )
    clear = mocker.patch(
        "backend.api.external.v2.graphs.clear_unowned_auto_credentials",
        new_callable=AsyncMock,
    )
    mocker.patch("backend.data.graph.get_graph", new_callable=AsyncMock)
    mocker.patch("backend.api.external.v2.graphs.Graph.from_internal")
    writes["create_graph"].return_value = Mock(is_active=False)
    inactive = _REQUEST.model_copy(update={"is_active": False})

    await update_graph("graph-1", inactive, auth=_AUTH)

    clear.assert_awaited_once()
    assert clear.await_args.args[0] is writes["create_graph"].await_args.args[0]
    activate.assert_not_awaited()


async def test_a_new_version_is_only_added_to_the_callers_own_graph(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock]
) -> None:
    """A teammate's graph is readable in the org, not ours to add versions to."""
    versions = mocker.patch(
        "backend.data.graph.get_graph_all_versions",
        new_callable=AsyncMock,
        return_value=[],
    )

    with pytest.raises(HTTPException) as raised:
        await update_graph("graph-1", _REQUEST, auth=_AUTH)

    assert raised.value.status_code == 404
    versions.assert_awaited_once_with("graph-1", user_id="user-1")
    writes["create_graph"].assert_not_awaited()


async def test_a_graph_tagged_with_another_organization_is_not_found(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock]
) -> None:
    mocker.patch(
        "backend.data.graph.get_graph_all_versions",
        new_callable=AsyncMock,
        return_value=[Mock(version=1, is_active=True, organization_id="org-2")],
    )

    with pytest.raises(NotFoundError):
        await update_graph("graph-1", _REQUEST, auth=_AUTH)

    writes["create_graph"].assert_not_awaited()


async def test_a_new_active_version_takes_over_its_webhook_presets(
    mocker: pytest_mock.MockFixture, writes: dict[str, AsyncMock]
) -> None:
    mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        side_effect=lambda graph, user_id: graph,
    )
    mocker.patch("backend.data.graph.set_graph_active_version", new_callable=AsyncMock)
    mocker.patch(
        "backend.api.external.v2.graphs.on_graph_deactivate", new_callable=AsyncMock
    )
    mocker.patch("backend.data.graph.get_graph", new_callable=AsyncMock)
    mocker.patch("backend.api.external.v2.graphs.Graph.from_internal")
    migrate = mocker.patch(
        "backend.api.features.library.db.migrate_webhook_presets_to_new_version",
        new_callable=AsyncMock,
    )
    new_version = Mock(is_active=True, webhook_input_node=Mock())
    writes["create_graph"].return_value = new_version

    await update_graph("graph-1", _REQUEST, auth=_AUTH)

    migrate.assert_awaited_once_with(user_id="user-1", new_graph=new_version)


async def test_only_the_owner_can_activate_a_version(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Activating registers webhooks with the caller's credentials."""
    mocker.patch(
        "backend.data.graph.get_graph",
        new_callable=AsyncMock,
        return_value=Mock(user_id="teammate", organization_id="org-1"),
    )
    activate = mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate", new_callable=AsyncMock
    )

    with pytest.raises(HTTPException) as raised:
        await set_active_version(
            "graph-1",
            GraphSetActiveVersionRequest(active_graph_version=2),
            auth=_AUTH,
        )

    assert raised.value.status_code == 404
    activate.assert_not_awaited()


async def test_an_activated_version_takes_over_its_webhook_presets(
    mocker: pytest_mock.MockFixture,
) -> None:
    graph = Mock(
        user_id="user-1", organization_id="org-1", version=2, webhook_input_node=Mock()
    )
    mocker.patch(
        "backend.data.graph.get_graph", new_callable=AsyncMock, return_value=graph
    )
    mocker.patch(
        "backend.api.external.v2.graphs.before_graph_activate",
        new_callable=AsyncMock,
        return_value=graph,
    )
    mocker.patch("backend.data.graph.set_graph_active_version", new_callable=AsyncMock)
    mocker.patch(
        "backend.api.features.library.db.update_agent_version_in_library",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "backend.api.external.v2.graphs.on_graph_deactivate", new_callable=AsyncMock
    )
    migrate = mocker.patch(
        "backend.api.features.library.db.migrate_webhook_presets_to_new_version",
        new_callable=AsyncMock,
    )

    await set_active_version(
        "graph-1", GraphSetActiveVersionRequest(active_graph_version=2), auth=_AUTH
    )

    migrate.assert_awaited_once_with(user_id="user-1", new_graph=graph)


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
