from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from backend.util import workspace_storage, workspace_storage_local
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


@pytest.mark.asyncio
async def test_gcs_store_wraps_custom_metadata(mocker):
    storage = GCSWorkspaceStorage(bucket_name="my-bucket")
    client = MagicMock()
    client.upload = AsyncMock()
    mocker.patch.object(storage, "_get_async_client", AsyncMock(return_value=client))

    await storage.store("workspace", "file", "data.txt", b"content")

    custom_metadata = client.upload.await_args.kwargs["metadata"]["metadata"]
    assert custom_metadata["uploaded_at"]
    assert custom_metadata["workspace_id"] == "workspace"
    assert custom_metadata["file_id"] == "file"


@pytest.mark.asyncio
async def test_workspace_storage_uses_private_bucket(mocker):
    config = MagicMock()
    config.resolved_private_user_data_bucket = "private-data"
    mocker.patch("backend.util.workspace_storage.Config", return_value=config)
    workspace_storage._gcs_storages.clear()

    storage = await workspace_storage.get_workspace_storage()

    assert isinstance(storage, GCSWorkspaceStorage)
    assert storage.bucket_name == "private-data"
    workspace_storage._gcs_storages.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "partial", "delete", "url"])
async def test_gcs_workspace_operations_reject_a_different_bucket(operation):
    storage = GCSWorkspaceStorage(bucket_name="private-bucket")
    path = "gcs://other-bucket/workspaces/ws/file/data.txt"

    with pytest.raises(PermissionError, match="configured private bucket"):
        if operation == "retrieve":
            await storage.retrieve(path)
        elif operation == "partial":
            await storage.retrieve_partial(path, 10)
        elif operation == "delete":
            await storage.delete(path)
        else:
            await storage.get_download_url(path)


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


@pytest.mark.asyncio
async def test_local_store_accepts_name_over_255_utf8_bytes(tmp_path):
    """A name under the char cap but over NAME_MAX bytes stores fine (#15651)."""
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    filename = "夜" * 86 + ".txt"  # 90 chars, 262 bytes
    assert len(filename.encode("utf-8")) > 255

    storage_path = await storage.store("ws", "file", filename, b"hello")

    assert storage_path.endswith(".txt")
    assert await storage.retrieve(storage_path) == b"hello"


@pytest.mark.asyncio
async def test_local_store_failure_leaves_no_empty_directory(tmp_path, mocker):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    mocker.patch.object(
        workspace_storage_local.aiofiles,
        "open",
        side_effect=OSError(28, "No space left on device"),
    )

    with pytest.raises(OSError):
        await storage.store("ws", "file", "data.txt", b"hello")

    assert not (tmp_path / "ws" / "file").exists()
    assert not (tmp_path / "ws").exists()
    assert tmp_path.exists()


@pytest.mark.asyncio
async def test_local_store_failure_keeps_sibling_files(tmp_path, mocker):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    kept = await storage.store("ws", "other", "keep.txt", b"keep")
    mocker.patch.object(
        workspace_storage_local.aiofiles, "open", side_effect=OSError("boom")
    )

    with pytest.raises(OSError):
        await storage.store("ws", "file", "data.txt", b"hello")

    assert not (tmp_path / "ws" / "file").exists()
    mocker.stopall()
    assert await storage.retrieve(kept) == b"keep"
