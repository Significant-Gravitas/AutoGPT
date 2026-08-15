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


@pytest.mark.asyncio
async def test_local_copy_duplicates_bytes_and_leaves_source_intact(tmp_path):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))
    source = await storage.store("ws", "file-1", "report.pdf", b"payload")

    copied = await storage.copy(source, "ws", "file-2", "summary.pdf")

    assert copied == "local://ws/file-2/summary.pdf"
    assert await storage.retrieve(copied) == b"payload"
    assert await storage.retrieve(source) == b"payload"


@pytest.mark.asyncio
async def test_local_copy_raises_when_source_missing(tmp_path):
    storage = LocalWorkspaceStorage(base_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        await storage.copy("local://ws/file-1/gone.txt", "ws", "file-2", "gone.txt")


@pytest.mark.asyncio
async def test_gcs_copy_uses_server_side_copy_without_downloading(mocker):
    """The blob is copied inside GCS; no bytes pass through this process."""
    client = AsyncMock()
    mocker.patch.object(
        GCSWorkspaceStorage,
        "_get_async_client",
        AsyncMock(return_value=client),
    )
    download = mocker.patch(
        "backend.util.workspace_storage.download_with_fresh_session",
        AsyncMock(),
    )
    storage = GCSWorkspaceStorage(bucket_name="my-bucket")

    result = await storage.copy(
        "gcs://my-bucket/workspaces/ws/file-1/report.pdf",
        "ws",
        "file-2",
        "summary.pdf",
    )

    assert result == "gcs://my-bucket/workspaces/ws/file-2/summary.pdf"
    client.copy.assert_awaited_once()
    args, kwargs = client.copy.await_args
    assert args[0] == "my-bucket"
    assert args[1] == "workspaces/ws/file-1/report.pdf"
    assert args[2] == "my-bucket"
    assert kwargs["new_name"] == "workspaces/ws/file-2/summary.pdf"
    download.assert_not_called()
