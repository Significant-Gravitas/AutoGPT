from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.util.gcs_utils import download_range, parse_gcs_path


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


@pytest.mark.asyncio
async def test_download_range_maps_404_to_file_not_found(mocker):
    download = AsyncMock(side_effect=Exception("404 Not Found"))
    _mock_client(mocker, download=download)

    with pytest.raises(FileNotFoundError):
        await download_range("bucket", "missing", 16)
