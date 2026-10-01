from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from backend.util.gcs_utils_test import _URL_CONTAINING_404, _gcs_http_error
from backend.util.workspace_storage import GCSWorkspaceStorage, LocalWorkspaceStorage


@pytest.mark.asyncio
async def test_retrieve_partial_returns_only_leading_bytes(tmp_path):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    storage_path = await storage.store("ws", "file", "data.txt", b"x" * 10_000)

    partial = await storage.retrieve_partial(storage_path, 512)

    assert partial == b"x" * 512


@pytest.mark.asyncio
async def test_retrieve_partial_returns_whole_file_when_smaller_than_cap(tmp_path):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    storage_path = await storage.store("ws", "file", "data.txt", b"hello")

    partial = await storage.retrieve_partial(storage_path, 4096)

    assert partial == b"hello"


@pytest.mark.asyncio
async def test_retrieve_partial_raises_when_missing(tmp_path):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        await storage.retrieve_partial("local://ws/file/missing.txt", 256)


@pytest.mark.asyncio
async def test_gcs_retrieve_partial_delegates_to_download_range(mocker):
    download_range = mocker.patch(
        "backend.util.workspace_storage.download_range",
        AsyncMock(return_value=b"head"),
    )
    storage = GCSWorkspaceStorage(bucket_name="my-bucket")

    result = await storage.retrieve_partial("gcs://my-bucket/path/file.txt", 4)

    assert result == b"head"
    download_range.assert_awaited_once_with("my-bucket", "path/file.txt", 4)


def _gcs_storage_whose_delete_raises(mocker, error: Exception) -> GCSWorkspaceStorage:
    storage = GCSWorkspaceStorage(bucket_name="my-bucket")
    client = MagicMock()
    client.delete = AsyncMock(side_effect=error)
    mocker.patch.object(storage, "_get_async_client", AsyncMock(return_value=client))
    return storage


@pytest.mark.asyncio
async def test_gcs_delete_ignores_already_deleted_file(mocker):
    storage = _gcs_storage_whose_delete_raises(
        mocker, _gcs_http_error(404, _URL_CONTAINING_404)
    )

    await storage.delete("gcs://my-bucket/path/file.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 503])
async def test_gcs_delete_raises_other_errors_when_url_contains_404(mocker, status):
    storage = _gcs_storage_whose_delete_raises(
        mocker, _gcs_http_error(status, _URL_CONTAINING_404)
    )

    with pytest.raises(aiohttp.ClientResponseError):
        await storage.delete("gcs://my-bucket/path/file.txt")
