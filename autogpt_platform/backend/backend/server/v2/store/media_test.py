import io
import unittest.mock

import fastapi
import pytest
import starlette.datastructures

import backend.server.v2.store.exceptions
import backend.server.v2.store.media


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://test-url")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")


@pytest.fixture
def mock_supabase(mocker):
    mock_client = unittest.mock.MagicMock()
    mock_storage = unittest.mock.MagicMock()
    mock_bucket = unittest.mock.MagicMock()

    mock_storage.from_.return_value = mock_bucket
    mock_bucket.upload.return_value = {"Key": "test-key"}
    mock_bucket.get_public_url.return_value = "http://test-url/media/test.jpg"

    mock_client.storage = mock_storage
    mock_client.storage.from_ = unittest.mock.MagicMock(return_value=mock_bucket)

    mocker.patch("supabase.create_client", return_value=mock_client)

    return mock_client


async def test_upload_media_success(mock_env_vars, mock_supabase):
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)

    assert result == "http://test-url/media/test.jpg"
    mock_bucket = mock_supabase.storage.from_.return_value
    mock_bucket.upload.assert_called_once()
    mock_bucket.get_public_url.assert_called_once()


async def test_upload_media_invalid_type(mock_env_vars, mock_supabase):
    test_file = fastapi.UploadFile(
        filename="test.txt",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.InvalidFileTypeError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)

    mock_bucket = mock_supabase.storage.from_.return_value
    mock_bucket.upload.assert_not_called()


async def test_upload_media_missing_credentials():
    test_file = fastapi.UploadFile(
        filename="test.jpeg",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(backend.server.v2.store.exceptions.StorageConfigError):
        await backend.server.v2.store.media.upload_media("test-user", test_file)


async def test_upload_media_video_type(mock_env_vars, mock_supabase):
    test_file = fastapi.UploadFile(
        filename="test.mp4",
        file=io.BytesIO(b"test video data"),
        headers=starlette.datastructures.Headers({"content-type": "video/mp4"}),
    )

    result = await backend.server.v2.store.media.upload_media("test-user", test_file)

    assert result == "http://test-url/media/test.jpg"
    mock_bucket = mock_supabase.storage.from_.return_value
    mock_bucket.upload.assert_called_once()
    mock_bucket.get_public_url.assert_called_once()
