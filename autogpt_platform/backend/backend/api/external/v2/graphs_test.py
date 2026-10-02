from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.integrations.webhooks.graph_lifecycle_hooks import GraphActivationError

from .graphs import create_graph, update_graph
from .models import GraphCreateRequest
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
