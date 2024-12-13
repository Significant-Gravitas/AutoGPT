import io
import unittest.mock

import fastapi
import pytest
import starlette.datastructures

import backend.server.v2.store.exceptions
import backend.server.v2.store.media
from backend.util.settings import Settings


@pytest.fixture
def mock_settings(monkeypatch):
    settings = Settings()
    settings.config.media_gcs_bucket_name = "test-bucket"
    settings.config.google_application_credentials = "test-credentials"
    monkeypatch.setattr("backend.server.v2.store.media.Settings", lambda: settings)
    return settings


@pytest.fixture
def mock_storage_client(mocker):
    mock_client = unittest.mock.MagicMock()
    mock_bucket = unittest.mock.MagicMock()
    mock_blob = unittest.mock.MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.public_url = "http://test-url/media/laptop.jpeg"

    mocker.patch("google.cloud.storage.Client", return_value=mock_client)

    return mock_client


async def test_upload_media_success(mock_settings, mock_storage_client):
    # Create test JPEG data with valid signature
    test_data = b"\xFF\xD8\xFF" + b"test data"

    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(test_data),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)

    assert result == "http://test-url/media/laptop.jpeg"
    mock_bucket = mock_storage_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.upload_from_string.assert_called_once()


async def test_upload_media_invalid_type(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.txt",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.InvalidFileTypeError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)

    mock_bucket = mock_storage_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.upload_from_string.assert_not_called()


async def test_upload_media_missing_credentials(monkeypatch):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.google_application_credentials = ""
    monkeypatch.setattr("backend.server.v2.store.media.Settings", lambda: settings)

    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(b"\xFF\xD8\xFF" + b"test data"),  # Valid JPEG signature
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.StorageConfigError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)


async def test_upload_media_video_type(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.mp4",
        file=io.BytesIO(b"\x00\x00\x00\x18ftypmp42"),  # Valid MP4 signature
        headers=starlette.datastructures.Headers({"content-type": "video/mp4"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)

    assert result == "http://test-url/media/laptop.jpeg"
    mock_bucket = mock_storage_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.upload_from_string.assert_called_once()


async def test_upload_media_file_too_large(mock_settings, mock_storage_client):
    large_data = b"\xFF\xD8\xFF" + b"x" * (
        50 * 1024 * 1024 + 1
    )  # 50MB + 1 byte with valid JPEG signature
    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(large_data),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.FileSizeTooLargeError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)


async def test_upload_media_file_read_error(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(b""),  # Empty file that will raise error on read
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )
    test_file.read = unittest.mock.AsyncMock(side_effect=Exception("Read error"))

    with pytest.raises(backend.server.v2.store.exceptions.FileReadError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)


async def test_upload_media_png_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),  # Valid PNG signature
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)
    assert result == "http://test-url/media/laptop.jpeg"


async def test_upload_media_gif_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.gif",
        file=io.BytesIO(b"GIF89a"),  # Valid GIF signature
        headers=starlette.datastructures.Headers({"content-type": "image/gif"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)
    assert result == "http://test-url/media/laptop.jpeg"


async def test_upload_media_webp_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.webp",
        file=io.BytesIO(b"RIFF\x00\x00\x00\x00WEBP"),  # Valid WebP signature
        headers=starlette.datastructures.Headers({"content-type": "image/webp"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)
    assert result == "http://test-url/media/laptop.jpeg"


async def test_upload_media_webm_success(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.webm",
        file=io.BytesIO(b"\x1a\x45\xdf\xa3"),  # Valid WebM signature
        headers=starlette.datastructures.Headers({"content-type": "video/webm"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)
    assert result == "http://test-url/media/laptop.jpeg"


async def test_upload_media_mismatched_signature(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),  # PNG signature with JPEG content type
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.InvalidFileTypeError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)


async def test_upload_media_invalid_signature(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"invalid signature"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.InvalidFileTypeError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)
