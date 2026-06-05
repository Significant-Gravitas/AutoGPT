from unittest.mock import AsyncMock

import pytest

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
