"""Pagination bounds on the builder routes (#15277)."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.dependencies import (
    get_request_context,
    get_user_id,
    requires_user,
)
from autogpt_libs.auth.models import RequestContext

from backend.api.features.builder import model as builder_model
from backend.api.features.builder import routes as builder_routes
from backend.util.models import Pagination

app = fastapi.FastAPI()
app.include_router(builder_routes.router)
app.dependency_overrides[requires_user] = lambda: None
app.dependency_overrides[get_user_id] = lambda: "test-user"
app.dependency_overrides[get_request_context] = lambda: RequestContext(
    user_id="test-user",
    org_id="test-org",
    team_id=None,
    is_org_owner=True,
    is_org_admin=True,
    is_org_billing_manager=False,
    is_team_admin=True,
    is_team_billing_manager=False,
    seat_status="ACTIVE",
)
client = fastapi.testclient.TestClient(app)

ROUTES = ["/blocks", "/providers", "/search"]
BAD_PARAMS = [
    {"page_size": 0},
    {"page_size": -5},
    {"page_size": builder_routes.MAX_PAGE_SIZE + 1},
    {"page": 0},
    {"page": -1},
]


def _pagination(page: int = 1, page_size: int = 50) -> Pagination:
    return Pagination(
        total_items=0, total_pages=0, current_page=page, page_size=page_size
    )


@pytest.fixture
def mock_db(mocker):
    mocker.patch.object(
        builder_routes.builder_db,
        "get_blocks",
        MagicMock(
            return_value=builder_model.BlockResponse(
                blocks=[], pagination=_pagination()
            )
        ),
    )
    mocker.patch.object(
        builder_routes.builder_db,
        "get_providers",
        MagicMock(
            return_value=builder_model.ProviderResponse(
                providers=[], pagination=_pagination()
            )
        ),
    )
    mocker.patch.object(
        builder_routes.builder_db,
        "get_sorted_search_results",
        AsyncMock(return_value=MagicMock(items=[], total_items={})),
    )
    mocker.patch.object(
        builder_routes.builder_db,
        "update_search",
        AsyncMock(return_value="search-id"),
    )


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("params", BAD_PARAMS)
def test_out_of_range_pagination_is_422(mock_db, route, params):
    response = client.get(route, params=params)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("route", ROUTES)
def test_default_pagination_still_ok(mock_db, route):
    response = client.get(route)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("route", ROUTES)
def test_max_page_size_is_accepted(mock_db, route):
    response = client.get(
        route, params={"page": 1, "page_size": builder_routes.MAX_PAGE_SIZE}
    )
    assert response.status_code == 200, response.text
