import io
from pathlib import Path
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from starlette.datastructures import Headers

from backend.api.features.store import local_media, media, routes
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings


@pytest.fixture
def local_settings(monkeypatch, tmp_path):
    settings = Settings()
    settings.config.media_gcs_bucket_name = ""
    settings.config.workspace_storage_dir = str(tmp_path / "workspaces")
    settings.config.platform_base_url = ""
    monkeypatch.setattr(local_media, "Settings", lambda: settings)
    monkeypatch.setattr(media, "Settings", lambda: settings)
    return settings


@pytest.fixture
def client():
    app = fastapi.FastAPI()
    app.add_exception_handler(NotFoundError, _handle_not_found)
    app.include_router(routes.router, prefix="/api/store")
    return fastapi.testclient.TestClient(app)


def _handle_not_found(request: fastapi.Request, exc: Exception):
    return fastapi.responses.JSONResponse({"detail": str(exc)}, status_code=404)


@pytest.mark.parametrize(
    "content_type,content,extension,media_type",
    [
        ("image/jpeg", b"\xff\xd8\xffimage", ".jpeg", "images"),
        ("image/png", b"\x89PNG\r\n\x1a\nimage", ".png", "images"),
        ("image/gif", b"GIF89aimage", ".gif", "images"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBPimage", ".webp", "images"),
        ("video/mp4", b"\x00\x00\x00\x18ftypmp42video", ".mp4", "videos"),
        ("video/webm", b"\x1a\x45\xdf\xa3video", ".webm", "videos"),
    ],
)
@pytest.mark.parametrize("use_file_name", [False, True])
async def test_upload_and_public_download(
    local_settings,
    client,
    monkeypatch,
    content_type,
    content,
    extension,
    media_type,
    use_file_name,
):
    scanner = AsyncMock()
    monkeypatch.setattr(media, "scan_content_safe", scanner)
    upload = fastapi.UploadFile(
        filename="picture.html",
        file=io.BytesIO(content),
        headers=Headers({"content-type": content_type}),
    )
    url = await media.upload_media("user-id", upload, use_file_name=use_file_name)
    assert url.startswith(f"/api/store/media/user-id/{media_type}/")
    assert url.endswith(extension)
    assert ".html" not in url
    scanner.assert_awaited_once_with(content, filename=url.rsplit("/", 1)[1])
    response = client.get(url)
    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"] == content_type
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("workspace", ["", "/data/workspaces"])
def test_media_root_uses_persistent_location(
    local_settings, monkeypatch, tmp_path, workspace
):
    local_settings.config.workspace_storage_dir = workspace
    monkeypatch.setattr(local_media, "get_data_path", lambda: tmp_path)
    expected = Path("/data") if workspace else tmp_path
    assert local_media.media_root() == expected / "store-media"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8006",
        "https://backend.example.test/",
        "https://autogpt.example.test/_agpt",
    ],
)
def test_media_url_uses_backend_origin(local_settings, client, base_url):
    local_settings.config.platform_base_url = base_url
    destination = local_media.get_media_path("u", "images", "image.png")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"image")
    url = local_media.media_url("u", "images", "image.png")
    assert url == f"{base_url.rstrip('/')}/api/store/media/u/images/image.png"
    if not base_url.endswith("/_agpt"):
        assert client.get(url).content == b"image"


@pytest.mark.parametrize(
    "component", [".", "..", "../secret", "a/b", "a\\b", "", "a%2fb"]
)
def test_media_path_rejects_invalid_components(local_settings, component):
    with pytest.raises(ValueError):
        local_media.get_media_path(component, "images", "file.png")
    with pytest.raises(ValueError):
        local_media.get_media_path("user", "images", component)


def test_media_path_rejects_symlink_escape(local_settings, tmp_path):
    images = local_media.media_root() / "users/user/images"
    images.mkdir(parents=True)
    secret = tmp_path / "private.png"
    secret.write_bytes(b"private")
    (images / "image.png").symlink_to(secret)
    with pytest.raises(ValueError):
        local_media.get_media_path("user", "images", "image.png")


@pytest.mark.parametrize(
    "path", ["images/missing.png", "documents/file.png", "images/file.html"]
)
def test_public_download_rejects_missing_or_unsupported_media(
    local_settings, client, path
):
    response = client.get(f"/api/store/media/user/{path}")
    assert response.status_code == 404
    assert response.json() == {"detail": "Media not found"}
    assert not local_media.media_root().exists()


def test_public_download_rejects_symlink_escape(local_settings, client, tmp_path):
    images = local_media.media_root() / "users/user/images"
    images.mkdir(parents=True)
    secret = tmp_path / "private.png"
    secret.write_bytes(b"private")
    (images / "image.png").symlink_to(secret)
    response = client.get("/api/store/media/user/images/image.png")
    assert response.status_code == 404
    assert b"private" not in response.content


async def test_rejected_virus_scan_never_writes_media(local_settings, monkeypatch):
    monkeypatch.setattr(
        media, "scan_content_safe", AsyncMock(side_effect=RuntimeError("virus"))
    )
    upload = fastapi.UploadFile(
        filename="image.png",
        file=io.BytesIO(b"\x89PNG\r\n\x1a\n"),
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(media.store_exceptions.MediaUploadError):
        await media.upload_media("user", upload)
    assert not local_media.media_root().exists()


async def test_failed_replacement_preserves_existing_media(local_settings, monkeypatch):
    await local_media.store_media("user", "images", "image.png", b"old")
    destination = local_media.get_media_path("user", "images", "image.png")

    def failed_replace(source, target):
        assert Path(source).read_bytes() == b"new"
        assert Path(target).read_bytes() == b"old"
        raise OSError("disk error")

    monkeypatch.setattr(local_media.os, "replace", failed_replace)
    with pytest.raises(OSError, match="disk error"):
        await local_media.store_media("user", "images", "image.png", b"new")
    assert destination.read_bytes() == b"old"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize(
    "filename", ["image.jpg", "image.JPG", "image.JPEG", "image.jpeg"]
)
async def test_named_jpeg_aliases_reuse_upload(local_settings, filename):
    await local_media.store_media("user", "images", "image.jpeg", b"image")
    assert await local_media.check_media_exists("user", filename) == (
        "/api/store/media/user/images/image.jpeg"
    )


@pytest.mark.parametrize("filename", ["../image.png", "image.html", "missing.png"])
async def test_invalid_or_missing_lookup_returns_none(local_settings, filename):
    assert await local_media.check_media_exists("user", filename) is None
