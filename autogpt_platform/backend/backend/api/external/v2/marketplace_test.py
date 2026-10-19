import io
from typing import Optional
from unittest import mock

import fastapi
import fastapi.testclient
import prisma.models
import pytest
import pytest_mock
from fastapi import HTTPException, Response, UploadFile
from prisma.enums import APIKeyPermission, SubmissionStatus

from .errors import add_v2_exception_handlers
from .marketplace import upload_submission_media
from .routes import v2_router
from .tenancy import TenantContext, require_auth

_LIMIT = 10 * 1024 * 1024
_ELEVEN_MB = 11 * 1024 * 1024


@pytest.mark.parametrize(
    "declared_size", [_ELEVEN_MB, None], ids=["size-known", "size-unknown"]
)
async def test_oversized_media_is_refused_without_being_read_whole(
    mocker: pytest_mock.MockFixture, declared_size: Optional[int]
) -> None:
    """The whole body in memory is what the cap exists to prevent."""
    mocker.patch(
        "backend.api.external.v2.marketplace.media_upload_limiter.check",
        new_callable=mock.AsyncMock,
        return_value=None,
    )
    scan = mocker.patch(
        "backend.api.external.v2.marketplace.scan_content_safe",
        new_callable=mock.AsyncMock,
    )
    store = mocker.patch(
        "backend.api.features.store.media.upload_media", new_callable=mock.AsyncMock
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


@pytest.mark.parametrize(
    "owning_org_id, status, expected",
    [
        ("org-1", SubmissionStatus.APPROVED, 400),
        ("org-2", SubmissionStatus.PENDING, 404),
        ("org-1", SubmissionStatus.PENDING, 204),
    ],
    ids=["approved", "another-org", "deleted"],
)
def test_a_submission_that_cannot_be_deleted_says_why(
    mocker: pytest_mock.MockFixture,
    owning_org_id: str,
    status: SubmissionStatus,
    expected: int,
):
    version = mock.Mock(
        id="slv-1",
        storeListingId="sl-1",
        submissionStatus=status,
        StoreListing=mock.Mock(owningOrgId=owning_org_id, owningUserId="user-1"),
    )
    client = mock.AsyncMock()
    client.find_unique.return_value = version
    client.count.return_value = 1
    mocker.patch.object(
        prisma.models.StoreListingVersion, "prisma", return_value=client
    )
    app = fastapi.FastAPI()
    app.include_router(v2_router)
    add_v2_exception_handlers(app)
    app.dependency_overrides[require_auth] = lambda: TenantContext(
        user_id="user-1",
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id="org-1",
    )

    response = fastapi.testclient.TestClient(app).delete(
        "/marketplace/submissions/slv-1"
    )

    assert response.status_code == expected, response.text
