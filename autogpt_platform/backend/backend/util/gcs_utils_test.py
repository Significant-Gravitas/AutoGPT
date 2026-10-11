from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from google.auth.credentials import Signing
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from backend.util.gcs_utils import (
    download_range,
    download_with_fresh_session,
    generate_iam_signed_url,
    parse_gcs_path,
)


def _mock_client(mocker, *, download: AsyncMock) -> MagicMock:
    """Patch the async GCS Storage client and session used by gcs_utils."""
    client = MagicMock()
    client.download = download
    client.close = AsyncMock(return_value=None)
    mocker.patch(
        "backend.util.gcs_utils.async_gcs_storage.Storage",
        return_value=client,
    )
    session = MagicMock()
    session.close = AsyncMock(return_value=None)
    mocker.patch(
        "backend.util.gcs_utils.aiohttp.ClientSession",
        return_value=session,
    )
    mocker.patch(
        "backend.util.gcs_utils.aiohttp.TCPConnector", return_value=MagicMock()
    )
    return client


def test_parse_gcs_path_splits_bucket_and_blob():
    assert parse_gcs_path("gcs://my-bucket/path/to/file") == (
        "my-bucket",
        "path/to/file",
    )


def test_parse_gcs_path_rejects_invalid_prefix():
    with pytest.raises(ValueError):
        parse_gcs_path("s3://my-bucket/file")


@pytest.mark.asyncio
async def test_generate_iam_signed_url_uses_adc_service_account(mocker):
    credentials = MagicMock()
    credentials.valid = False
    credentials.service_account_email = "default"

    def refresh(_request):
        credentials.valid = True
        credentials.service_account_email = "backend@example.iam.gserviceaccount.com"
        credentials.token = "access-token"

    credentials.refresh.side_effect = refresh
    mocker.patch(
        "backend.util.gcs_utils.google_auth_default",
        return_value=(credentials, "project"),
    )
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generate_signed_url.return_value = "https://signed.example/object"

    result = await generate_iam_signed_url(client, "bucket", "object", 60)

    assert result == "https://signed.example/object"
    credentials.refresh.assert_called_once()
    assert blob.generate_signed_url.call_args.kwargs["service_account_email"] == (
        "backend@example.iam.gserviceaccount.com"
    )
    assert blob.generate_signed_url.call_args.kwargs["access_token"] == "access-token"
    assert blob.generate_signed_url.call_args.kwargs["version"] == "v4"
    assert blob.generate_signed_url.call_args.kwargs["method"] == "GET"


@pytest.mark.asyncio
async def test_generate_iam_signed_url_uses_locally_signing_credentials(mocker):
    credentials = MagicMock(spec=Signing)
    mocker.patch(
        "backend.util.gcs_utils.google_auth_default",
        return_value=(credentials, "project"),
    )
    client = MagicMock()
    blob = client.bucket.return_value.blob.return_value
    blob.generate_signed_url.return_value = "https://signed.example/object"

    result = await generate_iam_signed_url(client, "bucket", "object", 60)

    assert result == "https://signed.example/object"
    blob.generate_signed_url.assert_called_once()
    assert blob.generate_signed_url.call_args.kwargs["credentials"] is credentials
    assert "service_account_email" not in blob.generate_signed_url.call_args.kwargs
    assert "access_token" not in blob.generate_signed_url.call_args.kwargs


@pytest.mark.asyncio
async def test_download_range_sends_range_header_and_slices(mocker):
    download = AsyncMock(return_value=b"0123456789ABCDEF")
    client = _mock_client(mocker, download=download)

    result = await download_range("bucket", "blob", 8)

    assert result == b"01234567"
    download.assert_awaited_once_with("bucket", "blob", headers={"Range": "bytes=0-7"})
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_download_range_falls_back_when_headers_unsupported(mocker):
    download = AsyncMock(
        side_effect=[TypeError("unexpected kwarg 'headers'"), b"full-content-here"]
    )
    _mock_client(mocker, download=download)

    result = await download_range("bucket", "blob", 4)

    assert result == b"full"
    assert download.await_count == 2


def _gcs_http_error(status: int, url: str) -> aiohttp.ClientResponseError:
    """The error gcloud-aio raises for a non-2xx GCS response."""
    request_info = aiohttp.RequestInfo(
        url=URL(url), method="GET", headers=CIMultiDictProxy(CIMultiDict())
    )
    return aiohttp.ClientResponseError(
        request_info, (), status=status, message="GCS error"
    )


# File ids are UUIDs, so "404" turns up in object URLs by chance.
_URL_CONTAINING_404 = (
    "https://storage.googleapis.com/storage/v1/b/bucket/o/"
    "workspaces%2Fws%2F1aa50e64-f289-404d-889a-935d443c0ca0%2Fa.png?alt=media"
)


@pytest.mark.asyncio
async def test_download_range_maps_404_to_file_not_found(mocker):
    download = AsyncMock(side_effect=_gcs_http_error(404, _URL_CONTAINING_404))
    _mock_client(mocker, download=download)

    with pytest.raises(FileNotFoundError):
        await download_range("bucket", "missing", 16)


@pytest.mark.asyncio
async def test_download_maps_404_to_file_not_found(mocker):
    download = AsyncMock(side_effect=_gcs_http_error(404, _URL_CONTAINING_404))
    _mock_client(mocker, download=download)

    with pytest.raises(FileNotFoundError):
        await download_with_fresh_session("bucket", "missing")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 500, 503])
async def test_download_keeps_other_errors_when_url_contains_404(mocker, status):
    download = AsyncMock(side_effect=_gcs_http_error(status, _URL_CONTAINING_404))
    _mock_client(mocker, download=download)

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await download_with_fresh_session("bucket", "blob")
    assert exc_info.value.status == status

    with pytest.raises(aiohttp.ClientResponseError):
        await download_range("bucket", "blob", 16)
