"""Bad marketplace searches are the caller's error, not a 500."""

from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from .errors import add_v2_exception_handlers
from .marketplace import marketplace_router
from .tenancy import TenantContext, require_auth

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)


@pytest.fixture
def creators(mocker: pytest_mock.MockFixture) -> AsyncMock:
    return mocker.patch(
        "backend.api.external.v2.marketplace.store_cache.get_cached_store_creators",
        new_callable=AsyncMock,
        return_value=Mock(creators=[], pagination=Mock(total_items=0)),
    )


@pytest.fixture
def client() -> fastapi.testclient.TestClient:
    app = fastapi.FastAPI()
    app.include_router(marketplace_router, prefix="/marketplace")
    app.dependency_overrides[require_auth] = lambda: _AUTH
    add_v2_exception_handlers(app)
    return fastapi.testclient.TestClient(app, raise_server_exceptions=False)


def test_a_blank_creator_search_lists_every_creator(
    client: fastapi.testclient.TestClient, creators: AsyncMock
) -> None:
    response = client.get("/marketplace/creators", params={"search_query": "   "})

    assert response.status_code == 200, response.text
    assert creators.await_args.kwargs["search_query"] is None


def test_an_overlong_creator_search_is_refused(
    client: fastapi.testclient.TestClient, creators: AsyncMock
) -> None:
    response = client.get("/marketplace/creators", params={"search_query": "x" * 101})

    assert response.status_code == 422
    creators.assert_not_awaited()
