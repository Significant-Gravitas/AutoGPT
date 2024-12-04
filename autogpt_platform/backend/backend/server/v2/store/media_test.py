import io
import unittest.mock

import fastapi
import pytest
import starlette.datastructures

import backend.server.v2.store.exceptions
import backend.server.v2.store.media


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "test-credentials")


@pytest.fixture
def mock_storage_client(mocker):
    mock_client = unittest.mock.MagicMock()
    mock_bucket = unittest.mock.MagicMock()
    mock_blob = unittest.mock.MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.public_url = "http://test-url/media/test.jpg"

    mocker.patch("google.cloud.storage.Client", return_value=mock_client)

    return mock_client


async def test_upload_media_success(mock_env_vars, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)

    assert result == "http://test-url/media/test.jpg"
    mock_bucket = mock_storage_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.upload_from_string.assert_called_once()


async def test_upload_media_invalid_type(mock_env_vars, mock_storage_client):
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
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)

    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.StorageConfigError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)


async def test_upload_media_video_type(mock_env_vars, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.mp4",
        file=io.BytesIO(b"test video data"),
        headers=starlette.datastructures.Headers({"content-type": "video/mp4"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)

    assert result == "http://test-url/media/test.jpg"
    mock_bucket = mock_storage_client.bucket.return_value
    mock_blob = mock_bucket.blob.return_value
    mock_blob.upload_from_string.assert_called_once()


async def test_upload_media_file_too_large(mock_env_vars, mock_storage_client):
    large_data = b"x" * (50 * 1024 * 1024 + 1)  # 50MB + 1 byte
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(large_data),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.FileSizeTooLargeError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)
