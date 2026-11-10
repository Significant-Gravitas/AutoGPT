import io
from pathlib import Path
from unittest.mock import AsyncMock

import fastapi
import pytest
from PIL import Image
from starlette.datastructures import Headers

from backend.util.settings import Settings

from . import exceptions as store_exceptions
from . import media

FIXTURES = Path(__file__).with_name("media_test_data")


@pytest.fixture
def storage(monkeypatch, tmp_path):
    settings = Settings()
    settings.config.media_gcs_bucket_name = "test-bucket"
    settings.config.public_site_media_bucket = ""
    settings.config.private_user_data_bucket = ""
    settings.config.workspace_storage_dir = str(tmp_path / "workspaces")
    settings.config.platform_base_url = ""
    monkeypatch.setattr(media, "Settings", lambda: settings)
    monkeypatch.setattr(media.local_media, "Settings", lambda: settings)
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(media.async_storage, "Storage", lambda: client)
    scanner = AsyncMock()
    monkeypatch.setattr(media, "scan_content_safe", scanner)
    return settings.config, client, scanner


@pytest.mark.parametrize("storage_mode", ["legacy", "private", "local"])
@pytest.mark.parametrize("use_file_name", [False, True])
@pytest.mark.parametrize(
    "image_format, declared, filename, expected_type, extension",
    [
        ("JPEG", "image/png", "image.png", "image/jpeg", ".jpeg"),
        ("PNG", "image/jpeg", "image.jpg", "image/png", ".png"),
        ("GIF", "image/png", "image.png", "image/gif", ".gif"),
        ("WEBP", "image/jpeg", "image.jpg", "image/webp", ".webp"),
        ("WEBP", "image/png", "image.png", "image/webp", ".webp"),
        ("PNG", None, "image", "image/png", ".png"),
        ("PNG", "application/octet-stream", "image.bin", "image/png", ".png"),
        ("PNG", "video/mp4", "image.mp4", "image/png", ".png"),
    ],
)
async def test_misnamed_image_upload(
    storage,
    tmp_path,
    storage_mode,
    use_file_name,
    image_format,
    declared,
    filename,
    expected_type,
    extension,
):
    config, client, scanner = storage
    if storage_mode == "private":
        config.public_site_media_bucket = "public-media"
        config.private_user_data_bucket = "private-media"
    elif storage_mode == "local":
        config.media_gcs_bucket_name = ""
    content = _image(image_format)
    upload = _upload(content, filename, declared)

    result = await media.upload_media("test-user", upload, use_file_name=use_file_name)

    stored_name = result.rsplit("/", 1)[-1]
    assert stored_name.endswith(extension)
    if use_file_name:
        assert stored_name == "image" + extension
    scanner.assert_awaited_once_with(content, filename=stored_name)
    if storage_mode == "local":
        client.upload.assert_not_awaited()
        assert result.startswith("/api/store/media/test-user/images/")
        stored = (
            tmp_path / "store-media" / "users" / "test-user" / "images" / stored_name
        )
        assert stored.read_bytes() == content
    else:
        bucket = "private-media" if storage_mode == "private" else "test-bucket"
        client.upload.assert_awaited_once_with(
            bucket,
            f"users/test-user/images/{stored_name}",
            content,
            content_type=expected_type,
        )
        if storage_mode == "private":
            assert result.startswith("/api/store/submissions/media/test-user/images/")


@pytest.mark.parametrize(
    "filename, content_type",
    [
        ("still.avif", "image/avif"),
        ("animated.avif", "image/avif"),
        ("header.heic", "image/heic"),
        ("video.mov", "video/quicktime"),
        ("video.m4v", "video/x-m4v"),
        ("video.3gp", "video/3gpp"),
        ("audio.m4a", "audio/mp4"),
        ("video.mkv", "video/x-matroska"),
    ],
)
@pytest.mark.parametrize("declared_mode", ["truthful", "untyped", "generic"])
async def test_unsupported_containers_are_not_inferred_as_video(
    storage, filename, content_type, declared_mode
):
    _, client, scanner = storage
    declared = {
        "truthful": content_type,
        "untyped": None,
        "generic": "application/octet-stream",
    }[declared_mode]
    upload = _upload((FIXTURES / filename).read_bytes(), filename, declared)

    with pytest.raises(store_exceptions.InvalidFileTypeError):
        await media.upload_media("test-user", upload)

    client.upload.assert_not_awaited()
    scanner.assert_not_awaited()


@pytest.mark.parametrize(
    "content, declared",
    [
        (b"\x00\x00\x00\x18ftyp", "application/octet-stream"),
        (b"RIFF\x00\x00\x00\x00WAVE", "image/webp"),
        (b"RIFF", "image/webp"),
        (b"\x00\x00\x00\x18ftypmp42", "video/webm"),
        (b"\x1a\x45\xdf\xa3", "video/mp4"),
        (b"invalid video", "video/mp4"),
    ],
)
async def test_invalid_or_mismatched_signatures_still_reject(
    storage, content, declared
):
    _, client, scanner = storage
    with pytest.raises(store_exceptions.InvalidFileTypeError):
        await media.upload_media("test-user", _upload(content, "image", declared))
    client.upload.assert_not_awaited()
    scanner.assert_not_awaited()


async def test_corrected_image_uses_private_image_limit(storage):
    config, client, scanner = storage
    config.public_site_media_bucket = "public-media"
    config.private_user_data_bucket = "private-media"
    content = _image("PNG") + b"x" * media.MAX_PRIVATE_IMAGE_FILE_SIZE

    with pytest.raises(
        store_exceptions.FileSizeTooLargeError, match="Maximum size is 4MB"
    ):
        await media.upload_media(
            "test-user", _upload(content, "image.mp4", "video/mp4")
        )

    client.upload.assert_not_awaited()
    scanner.assert_not_awaited()


async def test_gif_cannot_bypass_avatar_restriction(storage):
    _, client, scanner = storage
    with pytest.raises(fastapi.HTTPException) as error:
        await media.upload_media(
            "test-user",
            _upload(_image("GIF"), "avatar.png", "image/png"),
            is_avatar=True,
        )
    assert error.value.status_code == 400
    assert (
        error.value.detail == "Choose a PNG, JPEG, or WebP image for your appearance."
    )
    client.upload.assert_not_awaited()
    scanner.assert_not_awaited()


def _image(image_format: str) -> bytes:
    content = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(content, format=image_format)
    return content.getvalue()


def _upload(content: bytes, filename: str, declared: str | None) -> fastapi.UploadFile:
    return fastapi.UploadFile(
        file=io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": declared} if declared else {}),
    )
