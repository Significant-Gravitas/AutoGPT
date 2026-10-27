import asyncio
import base64
import io
import math
import threading
import time
import zipfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from fpdf import FPDF
from PIL import Image

from backend.api.features.workspace import preview as preview_module
from backend.api.features.workspace.routes import router
from backend.data.workspace import Workspace, WorkspaceFile

app = fastapi.FastAPI()
app.include_router(router)
client = fastapi.testclient.TestClient(app)

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _make_workspace() -> Workspace:
    return Workspace(
        id="ws-001",
        user_id="test-user-id",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_file(**overrides) -> WorkspaceFile:
    defaults = {
        "id": "file-001",
        "workspace_id": "ws-001",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "name": "test.bin",
        "path": "/test.bin",
        "storage_path": "local://test.bin",
        "mime_type": "application/octet-stream",
        "size_bytes": 100,
        "checksum": "abc123",
        "is_deleted": False,
        "deleted_at": None,
        "metadata": {},
    }
    defaults.update(overrides)
    return WorkspaceFile(**defaults)


def _png_bytes(width: int = 200, height: int = 200) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(out, format="PNG")
    return out.getvalue()


def _pdf_bytes() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=24)
    pdf.cell(40, 10, "Hello preview")
    return bytes(pdf.output())


def _zip_bytes(*, with_thumbnail: bool, thumbnail: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        if with_thumbnail:
            archive.writestr("docProps/thumbnail.png", thumbnail or _png_bytes(120, 90))
        archive.writestr("ppt/presentation.xml", "<p/>")
    return buf.getvalue()


def _oversized_png() -> bytes:
    # A source well above the render budget (~50 MP), kept cheap on disk and in
    # memory by using a single-channel solid image.
    side = int((preview_module.MAX_RENDER_PIXELS * 1.1) ** 0.5)
    out = io.BytesIO()
    Image.new("L", (side, side)).save(out, format="PNG")
    return out.getvalue()


def _decompression_bomb_png() -> bytes:
    # Above 2x the budget, where Pillow raises its own error at open, before the
    # explicit size check runs.
    side = int((2 * preview_module.MAX_RENDER_PIXELS) ** 0.5) + 100
    out = io.BytesIO()
    Image.new("L", (side, side)).save(out, format="PNG")
    return out.getvalue()


def _large_page_pdf() -> bytes:
    # A page far larger than a thumbnail; a fixed render scale would rasterise it
    # to a bitmap many times the budget.
    pdf = FPDF(unit="mm", format=(3000, 3000))
    pdf.add_page()
    pdf.set_font("helvetica", size=24)
    pdf.cell(40, 10, "Hello preview")
    return bytes(pdf.output())


def _oversized_page_pdf() -> bytes:
    # A compact PDF declaring a page so large that even the minimum render scale
    # would exceed the budget; the preview is declined before rendering.
    pdf = FPDF(unit="pt", format=(1_000_000, 1_000_000))
    pdf.add_page()
    return bytes(pdf.output())


def _mock_lookups(mocker, file: WorkspaceFile) -> None:
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_file",
        AsyncMock(return_value=file),
    )


def _mock_storage(mocker, content: bytes):
    storage = mocker.MagicMock()
    storage.retrieve = AsyncMock(return_value=content)
    storage.retrieve_partial = AsyncMock(side_effect=lambda _path, n: content[:n])
    mocker.patch(
        "backend.api.features.workspace.preview.get_workspace_storage",
        AsyncMock(return_value=storage),
    )
    return storage


def _mock_redis(mocker, *, cached: bytes | None = None):
    redis = mocker.MagicMock()
    redis.get = AsyncMock(
        return_value=base64.b64encode(cached).decode() if cached else None
    )
    redis.set = AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.workspace.preview.get_redis_async",
        AsyncMock(return_value=redis),
    )
    return redis


def test_image_preview_returns_smaller_webp(mocker):
    png = _png_bytes(800, 600)
    _mock_lookups(mocker, _make_file(mime_type="image/png"))
    _mock_storage(mocker, png)
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=100")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.headers["cache-control"] == "private, max-age=86400"
    assert 0 < len(response.content) < len(png)


def test_pdf_preview_returns_webp(mocker):
    _mock_lookups(mocker, _make_file(mime_type="application/pdf"))
    _mock_storage(mocker, _pdf_bytes())
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=200")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content[:4] == b"RIFF"


def test_office_preview_extracts_embedded_thumbnail(mocker):
    _mock_lookups(mocker, _make_file(mime_type=PPTX_MIME))
    _mock_storage(mocker, _zip_bytes(with_thumbnail=True))
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"


@pytest.mark.parametrize("mime_type", ["text/plain", "application/octet-stream"])
@pytest.mark.parametrize("name", ["deck.pptx", "report.docx", "sheet.xlsx"])
def test_office_preview_by_extension_despite_wrong_mime(mocker, name, mime_type):
    # Office files often store with a wrong MIME; the extension must still
    # route them to thumbnail extraction instead of raw text bytes or a 415.
    _mock_lookups(mocker, _make_file(name=name, path=f"/{name}", mime_type=mime_type))
    storage = _mock_storage(mocker, _zip_bytes(with_thumbnail=True))
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    assert response.content[:4] == b"RIFF"
    storage.retrieve_partial.assert_not_called()


def test_office_preview_without_thumbnail_returns_415(mocker):
    _mock_lookups(mocker, _make_file(mime_type=PPTX_MIME))
    _mock_storage(mocker, _zip_bytes(with_thumbnail=False))
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 415


def test_office_preview_corrupt_zip_returns_415(mocker):
    _mock_lookups(mocker, _make_file(mime_type=PPTX_MIME))
    _mock_storage(mocker, b"not-a-zip-file")
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 415
    assert response.json()["detail"] == "Cannot render preview"


def test_preview_404_when_bytes_missing_from_storage(mocker):
    _mock_lookups(mocker, _make_file(mime_type="text/plain"))
    storage = mocker.MagicMock()
    storage.retrieve_partial = AsyncMock(side_effect=FileNotFoundError("gone"))
    storage.retrieve = AsyncMock(side_effect=FileNotFoundError("gone"))
    mocker.patch(
        "backend.api.features.workspace.preview.get_workspace_storage",
        AsyncMock(return_value=storage),
    )
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 404


def test_text_preview_caps_bytes(mocker):
    _mock_lookups(mocker, _make_file(mime_type="text/plain"))
    storage = _mock_storage(mocker, b"x" * 10_000)
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?bytes=512")

    assert response.status_code == 200
    assert len(response.content) == 512
    storage.retrieve_partial.assert_awaited_once_with("local://test.bin", 512)


def test_markdown_preview_served_despite_non_text_mime(mocker):
    # `.md` often stores as application/octet-stream (mimetypes can't guess it),
    # but the extension fallback must still serve it as text.
    _mock_lookups(
        mocker,
        _make_file(
            name="README.md",
            path="/README.md",
            mime_type="application/octet-stream",
        ),
    )
    storage = _mock_storage(mocker, b"# Title\n\nbody")
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?bytes=4096")

    assert response.status_code == 200
    assert response.content == b"# Title\n\nbody"
    storage.retrieve_partial.assert_awaited_once()


def test_preview_404_when_file_missing(mocker):
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace",
        AsyncMock(return_value=_make_workspace()),
    )
    mocker.patch(
        "backend.api.features.workspace.routes.get_workspace_file",
        AsyncMock(return_value=None),
    )

    response = client.get("/files/missing/preview")

    assert response.status_code == 404


def test_preview_415_for_unsupported_type(mocker):
    _mock_lookups(mocker, _make_file(mime_type="application/octet-stream"))
    _mock_storage(mocker, b"\x00\x01\x02")
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 415


def test_preview_413_when_file_too_large(mocker):
    huge = _make_file(mime_type="image/png", size_bytes=20_000_000)
    _mock_lookups(mocker, huge)
    storage = _mock_storage(mocker, _png_bytes())
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=100")

    assert response.status_code == 413
    storage.retrieve.assert_not_called()


def test_image_preview_cache_hit_skips_retrieve(mocker):
    cached = b"RIFF-cached-webp-bytes"
    _mock_lookups(mocker, _make_file(mime_type="image/png"))
    storage = _mock_storage(mocker, _png_bytes())
    _mock_redis(mocker, cached=cached)

    response = client.get("/files/file-001/preview?w=100")

    assert response.status_code == 200
    assert response.content == cached
    storage.retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Render budget: a thumbnail never rasterises more than MAX_RENDER_PIXELS, so
# each renderer stays within the budget whatever the source's dimensions.
# ---------------------------------------------------------------------------


def test_render_scale_keeps_a_large_page_within_budget():
    # A ~3000 mm page is ~8503 pt on a side.
    page_pt = 3000 / 25.4 * 72
    scale = preview_module._render_scale(page_pt, page_pt, 400)

    rendered_pixels = (page_pt * scale) ** 2
    assert rendered_pixels <= preview_module.MAX_RENDER_PIXELS

    # Without the budget, a fixed render scale rasterises the whole page and
    # blows past it — this is what the budget exists to prevent.
    fixed_scale = max(1, min(3, round(400 / page_pt)))
    assert (page_pt * fixed_scale) ** 2 > preview_module.MAX_RENDER_PIXELS


def test_render_scale_caps_a_sliver_page():
    # A page with a sub-pixel side: the renderer rounds each rendered side up, so
    # the continuous-area budget alone underestimates the bitmap. The scale
    # ceiling keeps the rounded bitmap within the budget.
    page_width, page_height = 0.000025, 10_000.0
    scale = preview_module._render_scale(page_width, page_height, 400)

    assert scale <= preview_module.MAX_RENDER_SCALE
    rounded_pixels = math.ceil(page_width * scale) * math.ceil(page_height * scale)
    assert rounded_pixels <= preview_module.MAX_RENDER_PIXELS


def test_pdf_preview_renders_a_large_page_as_a_thumbnail(mocker):
    _mock_lookups(mocker, _make_file(mime_type="application/pdf"))
    _mock_storage(mocker, _large_page_pdf())
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=200")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    with Image.open(io.BytesIO(response.content)) as rendered:
        assert max(rendered.size) <= 200 * 4


def test_pdf_preview_declines_a_page_too_large_for_the_budget(mocker):
    _mock_lookups(mocker, _make_file(mime_type="application/pdf"))
    _mock_storage(mocker, _oversized_page_pdf())
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=400")

    assert response.status_code == 413


def test_image_preview_declines_an_oversized_source(mocker):
    _mock_lookups(mocker, _make_file(mime_type="image/png"))
    _mock_storage(mocker, _oversized_png())
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=100")

    assert response.status_code == 413


def test_image_preview_declines_a_decompression_bomb_source(mocker):
    # Above 2x the budget Pillow raises before the size check; it must still be a
    # 413 (too large), not a generic 415 render failure.
    _mock_lookups(mocker, _make_file(mime_type="image/png"))
    _mock_storage(mocker, _decompression_bomb_png())
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview?w=100")

    assert response.status_code == 413


def test_office_preview_declines_an_oversized_thumbnail(mocker):
    _mock_lookups(mocker, _make_file(mime_type=PPTX_MIME))
    _mock_storage(mocker, _zip_bytes(with_thumbnail=True, thumbnail=_oversized_png()))
    _mock_redis(mocker)

    response = client.get("/files/file-001/preview")

    assert response.status_code == 413


async def test_preview_render_concurrency_is_bounded(mocker):
    # Instrument the rasteriser (it runs in a worker thread) to record the peak
    # number of simultaneous renders, then fire more previews than the cap.
    peak = 0
    in_flight = 0
    lock = threading.Lock()

    def instrumented(_content: bytes, _width: int) -> bytes:
        nonlocal peak, in_flight
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.1)
        with lock:
            in_flight -= 1
        return b"RIFFfake-webp"

    mocker.patch.object(preview_module, "_resize_webp", instrumented)

    storage = mocker.MagicMock()
    storage.retrieve = AsyncMock(return_value=b"image-bytes")
    mocker.patch.object(
        preview_module, "get_workspace_storage", AsyncMock(return_value=storage)
    )
    redis = mocker.MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=None)
    mocker.patch.object(
        preview_module, "get_redis_async", AsyncMock(return_value=redis)
    )

    files = [
        _make_file(id=f"file-{i}", mime_type="image/png", checksum=f"sum-{i}")
        for i in range(10)
    ]

    await asyncio.gather(
        *(
            preview_module.build_preview_response(f, width=100, max_bytes=4096)
            for f in files
        )
    )

    assert peak > 1  # the renders really did overlap
    assert peak <= preview_module.MAX_CONCURRENT_RENDERS
