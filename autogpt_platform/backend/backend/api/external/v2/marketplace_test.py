"""Bad marketplace searches are the caller's error, not a 500."""

import io
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi import HTTPException, UploadFile
from prisma.enums import APIKeyPermission

from backend.api.features.library.exceptions import (
    LibraryAgentInAnotherOrganizationError,
)

from .errors import add_v2_exception_handlers
from .marketplace import (
    _read_media_within_limit,
    add_agent_to_library,
    marketplace_router,
)
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


async def test_adding_a_listing_lands_in_the_credentials_organization(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.external.v2.marketplace.store_cache.get_cached_agent_details",
        new_callable=AsyncMock,
        return_value=Mock(store_listing_version_id="slv-1"),
    )
    add = mocker.patch(
        "backend.api.external.v2.marketplace.library_db.add_store_agent_to_library",
        new_callable=AsyncMock,
    )
    mocker.patch("backend.api.external.v2.marketplace.LibraryAgent.from_internal")

    await add_agent_to_library(username="someone", agent_name="agent", auth=_AUTH)

    assert add.await_args.kwargs["organization_id"] == "org-1"


def test_a_listing_already_in_another_organizations_library_is_a_conflict(
    client: fastapi.testclient.TestClient, mocker: pytest_mock.MockFixture
) -> None:
    """The entry is unique per user, so it can be neither duplicated nor taken."""
    mocker.patch(
        "backend.api.external.v2.marketplace.store_cache.get_cached_agent_details",
        new_callable=AsyncMock,
        return_value=Mock(store_listing_version_id="slv-1"),
    )
    mocker.patch(
        "backend.api.external.v2.marketplace.library_db.add_store_agent_to_library",
        new_callable=AsyncMock,
        side_effect=LibraryAgentInAnotherOrganizationError("elsewhere"),
    )

    response = client.post("/marketplace/agents/someone/agent/add-to-library")

    assert response.status_code == 409


async def test_an_oversized_media_upload_is_refused_without_reading_it_all(
    mocker: pytest_mock.MockFixture,
) -> None:
    """The whole body was read into memory before the 10 MB check."""
    upload = UploadFile(file=io.BytesIO(b"x" * (10 * 1024 * 1024 + 1)), size=None)
    read = mocker.patch.object(upload, "read", wraps=upload.read)

    with pytest.raises(HTTPException) as raised:
        await _read_media_within_limit(upload)

    assert raised.value.status_code == 413
    assert all(call.args == (64 * 1024,) for call in read.await_args_list)


async def test_a_declared_oversized_media_upload_is_refused_before_reading(
    mocker: pytest_mock.MockFixture,
) -> None:
    upload = UploadFile(file=io.BytesIO(b""), size=11 * 1024 * 1024)
    read = mocker.patch.object(upload, "read", new_callable=AsyncMock)

    with pytest.raises(HTTPException) as raised:
        await _read_media_within_limit(upload)

    assert raised.value.status_code == 413
    read.assert_not_awaited()
