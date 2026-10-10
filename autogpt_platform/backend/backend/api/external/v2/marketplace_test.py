"""The v2 marketplace routes: searches, library adds and media uploads."""

import io
from typing import Optional
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from fastapi import HTTPException, Response, UploadFile
from prisma.enums import APIKeyPermission

from backend.api.features.library.exceptions import (
    LibraryAgentInAnotherOrganizationError,
)

from .errors import add_v2_exception_handlers
from .marketplace import (
    add_agent_to_library,
    marketplace_router,
    upload_submission_media,
)
from .tenancy import TenantContext, require_auth

_AUTH = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
)
_LIMIT = 10 * 1024 * 1024
_ELEVEN_MB = 11 * 1024 * 1024


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


@pytest.mark.parametrize(
    "declared_size", [_ELEVEN_MB, None], ids=["size-known", "size-unknown"]
)
async def test_oversized_media_is_refused_without_being_read_whole(
    mocker: pytest_mock.MockFixture, declared_size: Optional[int]
) -> None:
    """The whole body in memory is what the cap exists to prevent."""
    mocker.patch(
        "backend.api.external.v2.marketplace.media_upload_limiter.check",
        new_callable=AsyncMock,
        return_value=None,
    )
    scan = mocker.patch(
        "backend.api.external.v2.marketplace.scan_content_safe",
        new_callable=AsyncMock,
    )
    store = mocker.patch(
        "backend.api.features.store.media.upload_media", new_callable=AsyncMock
    )
    body = _CountingBody(b"\0" * _ELEVEN_MB)

    with pytest.raises(HTTPException) as refused:
        await upload_submission_media(
            response=Response(),
            file=UploadFile(file=body, size=declared_size, filename="big.png"),
            auth=TenantContext(
                user_id="user-1",
                scopes=[APIKeyPermission.WRITE_STORE],
                type="api_key",
                organization_id="org-1",
            ),
        )

    assert refused.value.status_code == 413
    assert body.bytes_read <= (0 if declared_size else _LIMIT + 64 * 1024)
    scan.assert_not_awaited()
    store.assert_not_awaited()


class _CountingBody(io.BytesIO):
    bytes_read = 0

    def read(self, size: Optional[int] = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk
