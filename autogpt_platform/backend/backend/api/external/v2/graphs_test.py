from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.code_executor import ExecuteCodeBlock
from backend.data import graph as graph_db
from backend.data.user import get_or_create_user
from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError
from backend.util.test import SpinTestServer

from .graphs import create_graph, get_graph, list_graph_versions, update_graph
from .models import GraphCreateRequest
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
        return_value=[Mock(version=1, is_active=True)],
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
