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


async def test_upload_media_org_scoped_storage_path(mock_settings, mock_storage_client):
    """With organization_id set, files land under orgs/{org_id}/ instead of
    users/{user_id}/ — the org avatar upload path."""
    test_file = fastapi.UploadFile(
        filename="logo.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    result = await store_media.upload_media(
        "test-user", test_file, organization_id="org-123"
    )

    assert result.startswith(
        "https://storage.googleapis.com/test-bucket/orgs/org-123/images/"
    )
    assert result.endswith(".png")
    assert "users/test-user" not in result
    mock_storage_client.upload.assert_called_once()


async def test_upload_media_org_scoped_local_storage(tmp_path, monkeypatch, mocker):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.media_storage_dir = str(tmp_path)
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)
    mocker.patch(
        "backend.api.features.store.media.scan_content_safe", new_callable=AsyncMock
    )
    test_file = fastapi.UploadFile(
        filename="logo.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    result = await store_media.upload_media(
        "test-user", test_file, organization_id="org-123"
    )

    filename = result.rsplit("/", 1)[-1]
    assert result == f"/api/orgs/org-123/avatar/{filename}"
    assert (tmp_path / "orgs" / "org-123" / "images" / filename).read_bytes() == (
        b"\x89PNG\r\n\x1a\n"
    )


@pytest.mark.parametrize(
    ("platform_base_url", "team_id", "expected_url", "expected_parts"),
    [
        (
            "",
            None,
            "/api/store/media/orgs/org-123/images/listing.png",
            ("store", "orgs", "org-123", "home", "images", "listing.png"),
        ),
        (
            "https://autogpt.example/_agpt",
            "team-456",
            "/_agpt/api/store/media/orgs/org-123/teams/team-456/images/listing.png",
            (
                "store",
                "orgs",
                "org-123",
                "teams",
                "team-456",
                "images",
                "listing.png",
            ),
        ),
        (
            "https://autogpt.example",
            None,
            "/api/store/media/orgs/org-123/images/listing.png",
            ("store", "orgs", "org-123", "home", "images", "listing.png"),
        ),
        (
            "https://autogpt.example//attacker.example\\_agpt/../safe",
            None,
            "/attacker.example/safe/api/store/media/orgs/org-123/images/listing.png",
            ("store", "orgs", "org-123", "home", "images", "listing.png"),
        ),
    ],
)
async def test_upload_store_media_uses_tenant_scoped_local_storage(
    tmp_path,
    monkeypatch,
    mocker,
    platform_base_url,
    team_id,
    expected_url,
    expected_parts,
):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.media_storage_dir = str(tmp_path)
    settings.config.platform_base_url = platform_base_url
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)
    mocker.patch(
        "backend.api.features.store.media.scan_content_safe", new_callable=AsyncMock
    )
    test_file = fastapi.UploadFile(
        filename="listing.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    result = await store_media.upload_media(
        "test-user",
        test_file,
        use_file_name=True,
        organization_id="org-123",
        team_id=team_id,
        local_store_media=True,
    )

    assert result == expected_url
    assert tmp_path.joinpath(*expected_parts).read_bytes() == b"\x89PNG\r\n\x1a\n"
    assert (
        await store_media.check_media_exists(
            "test-user",
            "listing.png",
            organization_id="org-123",
            team_id=team_id,
            local_store_media=True,
        )
        == expected_url
    )


def test_local_store_media_url_encodes_dynamic_path_segments(monkeypatch):
    settings = Settings()
    settings.config.platform_base_url = (
        "https://autogpt.example/_agpt?next=//bad#fragment"
    )
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)

    result = store_media.get_local_store_media_url(
        "org?admin=true", "team#other", "images", "listing name%.png"
    )

    assert result == (
        "/_agpt/api/store/media/orgs/org%3Fadmin%3Dtrue/teams/team%23other/"
        "images/listing%20name%25.png"
    )


async def test_local_store_media_does_not_cross_team_scopes(
    tmp_path, monkeypatch, mocker
):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.media_storage_dir = str(tmp_path)
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)
    mocker.patch(
        "backend.api.features.store.media.scan_content_safe", new_callable=AsyncMock
    )
    for team_id, content in (("team-a", b"A"), ("team-b", b"B")):
        test_file = fastapi.UploadFile(
            filename="listing.png",
            file=io.BytesIO(b"\x89PNG\r\n\x1a\n" + content),
            headers=starlette.datastructures.Headers({"content-type": "image/png"}),
        )
        await store_media.upload_media(
            "test-user",
            test_file,
            use_file_name=True,
            organization_id="org-123",
            team_id=team_id,
            local_store_media=True,
        )

    team_a = store_media.get_local_store_media_path(
        "org-123", "team-a", "images", "listing.png"
    )
    team_b = store_media.get_local_store_media_path(
        "org-123", "team-b", "images", "listing.png"
    )
    assert team_a != team_b
    assert team_a.read_bytes().endswith(b"A")
    assert team_b.read_bytes().endswith(b"B")


async def test_gcs_store_media_preserves_hosted_org_path(
    mock_settings, mock_storage_client
):
    test_file = fastapi.UploadFile(
        filename="listing.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    result = await store_media.upload_media(
        "test-user",
        test_file,
        use_file_name=True,
        organization_id="org-123",
        team_id="team-456",
        local_store_media=True,
    )

    assert result == (
        "https://storage.googleapis.com/test-bucket/orgs/org-123/images/listing.png"
    )
    upload = mock_storage_client.upload.await_args
    assert upload.args[:2] == ("test-bucket", "orgs/org-123/images/listing.png")


async def test_upload_media_rejects_filename_scope_escape(
    mock_settings, mock_storage_client
):
    test_file = fastapi.UploadFile(
        filename="../../other-org/images/avatar.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),
        headers=starlette.datastructures.Headers({"content-type": "image/png"}),
    )

    with pytest.raises(
        store_exceptions.InvalidFileTypeError, match="Invalid file name"
    ):
        await store_media.upload_media(
            "test-user",
            test_file,
            use_file_name=True,
            organization_id="org-123",
        )

    mock_storage_client.upload.assert_not_called()


def test_local_media_path_rejects_escape(tmp_path, monkeypatch):
    settings = Settings()
    settings.config.media_storage_dir = str(tmp_path)
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)

    with pytest.raises(ValueError, match="Invalid media path"):
        store_media.get_local_media_path("../outside.png")

    sibling = tmp_path.parent / f"{tmp_path.name}-outside" / "avatar.png"
    with pytest.raises(ValueError, match="Invalid media path"):
        store_media.get_local_media_path(str(sibling))


@pytest.mark.parametrize(
    ("organization_id", "team_id", "media_type", "filename"),
    [
        ("../other-org", None, "images", "listing.png"),
        ("org-123", "../other-team", "images", "listing.png"),
        ("org-123", None, "../videos", "listing.png"),
        ("org-123", None, "images", "../listing.png"),
    ],
)
def test_local_store_media_rejects_scope_escape(
    organization_id, team_id, media_type, filename
):
    with pytest.raises(ValueError, match="Invalid media path"):
        store_media.get_local_store_media_path(
            organization_id, team_id, media_type, filename
        )


async def test_upload_media_invalid_type(mock_settings, mock_storage_client):
    test_file = fastapi.UploadFile(
        filename="test.txt",
        file=io.BytesIO(b"test data"),
        headers=starlette.datastructures.Headers({"content-type": "text/plain"}),
    )

    with pytest.raises(store_exceptions.InvalidFileTypeError):
        await store_media.upload_media("test-user", test_file)

    mock_storage_client.upload.assert_not_called()


async def test_upload_media_missing_credentials(monkeypatch):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.google_application_credentials = ""
    monkeypatch.setattr("backend.api.features.store.media.Settings", lambda: settings)

    test_file = fastapi.UploadFile(
        filename="laptop.jpeg",
        file=io.BytesIO(b"\xff\xd8\xff" + b"test data"),  # Valid JPEG signature
        headers=starlette.datastructures.Headers({"content-type": "image/jpeg"}),
    )

    with pytest.raises(store_exceptions.StorageConfigError):
        await store_media.upload_media("test-user", test_file)


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
