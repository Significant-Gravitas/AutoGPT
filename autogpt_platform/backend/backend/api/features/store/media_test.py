import io
import unittest.mock
from unittest.mock import AsyncMock

import fastapi
import pytest
import starlette.datastructures

from backend.util.settings import Settings

from . import exceptions as store_exceptions
from . import media as store_media


@pytest.fixture
def mock_settings(monkeypatch):
    settings = Settings()
    settings.config.media_gcs_bucket_name = "test-bucket"
    settings.config.google_application_credentials = "test-credentials"
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)
    return settings


@pytest.fixture
def mock_storage_client(mocker):
    # Mock the async gcloud.aio.storage.Storage client
    mock_client = AsyncMock()
    mock_client.upload = AsyncMock()

    # Mock context manager methods
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Mock the constructor to return our mock client
    mocker.patch(
        "backend.api.features.store.media.async_storage.Storage",
        return_value=mock_client,
    )

    # Mock virus scanner to avoid actual scanning
    mocker.patch(
        "backend.api.features.store.media.scan_content_safe", new_callable=AsyncMock
    )

    return mock_client


async def test_upload_media_success(mock_settings, mock_storage_client):
    # Create test JPEG data with valid signature
    test_data = b"\xff\xd8\xff" + b"test data"

    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(test_data),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    result = await store_media.upload_media("test-user", test_file)

    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/users/test-user/images/"
    )
    assert result.endswith(".jpeg")
    mock_storage_client.upload.assert_called_once()


async def test_upload_media_invalid_type(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.txt",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(store_exceptions.InvalidFileTypeError):
        await store_media.upload_media("test-user", test_file)

    mock_storage_client.upload.assert_not_called()


@pytest.fixture
def local_storage_settings(monkeypatch, tmp_path):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.google_application_credentials = ""
    settings.config.workspace_storage_dir = str(tmp_path / "workspaces")
    settings.config.platform_base_url = ""
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)
    monkeypatch.setattr(
        "backend.api.features.store.media.scan_content_safe",
        unittest.mock.AsyncMock(),
    )
    return settings


async def test_upload_media_missing_credentials(local_storage_settings, tmp_path):
    test_data = b"\xff\xd8\xff" + b"test data"
    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(test_data),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    result = await store_media.upload_media("test-user", test_file)

    assert result.startswith("/api/store/media/test-user/images/")
    assert result.endswith(".jpeg")

    stored = tmp_path / "store-media" / "users" / "test-user" / "images"
    files = list(stored.iterdir())
    assert len(files) == 1
    assert files[0].read_bytes() == test_data


async def test_check_media_exists_without_gcs(local_storage_settings, tmp_path):
    filename = "agent_graph-1.jpeg"
    missing = await store_media.check_media_exists("test-user", filename)
    assert missing is None

    test_file = fastapi.UploadFile(
        filename=filename,
        file=io.BytesIO(b"\xff\xd8\xff" + b"existing"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )
    uploaded = await store_media.upload_media(
        "test-user", test_file, use_file_name=True
    )
    assert uploaded == f"/api/store/media/test-user/images/{filename}"
    assert (
        tmp_path / "store-media" / "users" / "test-user" / "images" / filename
    ).is_file()

    found = await store_media.check_media_exists("test-user", filename)
    assert found == uploaded

    assert await store_media.check_media_exists("test-user", "../secret.jpeg") is None


def test_get_local_media_path_rejects_traversal(local_storage_settings):
    with pytest.raises(ValueError):
        store_media.get_local_media_path("test-user", "images", "../secret.jpeg")
    with pytest.raises(ValueError):
        store_media.get_local_media_path("test-user", "other", "file.jpeg")


async def test_upload_media_video_type(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.mp4",
        file=io.BytesIO(b"\x00\x00\x00\x18ftypmp42"),  # Valid MP4 signature
        headers=starlette.datastructures.Headers({"content-type": "video/mp4"}),
    )

    result = await store_media.upload_media("test-user", test_file)

    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/users/test-user/videos/"
    )
    assert result.endswith(".mp4")
    mock_storage_client.upload.assert_called_once()


async def test_upload_media_file_too_large(mock_settings, mock_storage_client):
    large_data = b"\xff\xd8\xff" + b"x" * (
        50 * 1024 * 1024 + 1
    )  # 50MB + 1 byte with valid JPEG signature
    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(large_data),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(store_exceptions.FileSizeTooLargeError):
        await store_media.upload_media("test-user", test_file)


async def test_upload_media_file_read_error(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(b""),  # Empty file that will raise error on read
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )
    test_file.read = unittest.mock.AsyncMock(side_effect=Exception("Read error"))

    with pytest.raises(store_exceptions.FileReadError):
        await store_media.upload_media("test-user", test_file)


async def test_upload_media_png_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),  # Valid PNG signature
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    result = await store_media.upload_media("test-user", test_file)
    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/users/test-user/images/"
    )
    assert result.endswith(".png")


async def test_upload_media_gif_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.gif",
        file=io.BytesIO(b"GIF89a"),  # Valid GIF signature
        headers=starlette.datastructures.Headers({"content-type": "image/gif"}),
    )

    result = await store_media.upload_media("test-user", test_file)
    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/users/test-user/images/"
    )
    assert result.endswith(".gif")


async def test_upload_media_webp_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.webp",
        file=io.BytesIO(b"RIFF\x00\x00\x00\x00WEBP"),  # Valid WebP signature
        headers=starlette.datastructures.Headers({"content-type": "image/webp"}),
    )

    result = await store_media.upload_media("test-user", test_file)
    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/users/test-user/images/"
    )
    assert result.endswith(".webp")


async def test_upload_media_webm_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.webm",
        file=io.BytesIO(b"\x1a\x45\xdf\xa3"),  # Valid WebM signature
        headers=starlette.datastructures.Headers({"content-type": "video/webm"}),
    )

    result = await store_media.upload_media("test-user", test_file)
    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/users/test-user/videos/"
    )
    assert result.endswith(".webm")


async def test_upload_media_mismatched_signature(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),  # PNG signature with JPEG content type
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(store_exceptions.InvalidFileTypeError):
        await store_media.upload_media("test-user", test_file)


async def test_upload_media_invalid_signature(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"invalid signature"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(store_exceptions.InvalidFileTypeError):
        await store_media.upload_media("test-user", test_file)
