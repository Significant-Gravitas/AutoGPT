"""
Preview generation for workspace files.

Produces small, cheap previews for the Artifacts page instead of shipping the
whole file to the browser:

* raster images        -> resized WebP thumbnail (Pillow)
* PDFs                 -> first page rasterised to WebP (pypdfium2)
* Office openxml docs  -> embedded ``docProps/thumbnail`` re-encoded to WebP
* text-like files      -> first N bytes (partial read, no full download)

Image/PDF/Office thumbnails are cached in Redis keyed by file checksum so
repeat views and re-renders are free. All CPU-bound work runs in a thread so
the event loop is never blocked.
"""

import asyncio
import base64
import io
import logging
import zipfile
from typing import Awaitable, Callable, Optional

import fastapi
import pypdfium2
from fastapi.responses import Response
from PIL import Image

from backend.data.redis_client import get_redis_async
from backend.data.workspace import WorkspaceFile
from backend.util.workspace_storage import get_workspace_storage

logger = logging.getLogger(__name__)

PREVIEW_CACHE_TTL = 86_400  # 1 day
WEBP_QUALITY = 80
MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard (~50 MP)
CACHE_HEADERS = {"Cache-Control": "private, max-age=86400"}

# Per-kind size ceilings. The frontend mirrors these exactly so it never
# requests a preview for a file this big (it shows an illustration instead);
# these server-side checks are defence-in-depth against direct calls.
PREVIEW_MAX_IMAGE_BYTES = 10_000_000
PREVIEW_MAX_DOC_BYTES = 50_000_000
PREVIEW_MAX_TEXT_BYTES = 50_000_000

OFFICE_MIMES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
)
_TEXT_HINTS = (
    "json",
    "xml",
    "csv",
    "yaml",
    "javascript",
    "typescript",
    "markdown",
    "svg",
)
# Extension fallback for Markdown only. ``mimetypes.guess_type`` returns
# ``None``/``application/octet-stream`` for ``.md`` on many systems, which would
# 415 a file the Artifacts card renders as a content preview. Other text/code
# types fall back to a static illustration on the frontend and never request a
# preview, so they don't need an extension fallback here.
_TEXT_EXTENSIONS = frozenset({"md", "markdown", "mdx"})
_EMBEDDED_THUMBNAILS = (
    "docprops/thumbnail.jpeg",
    "docprops/thumbnail.jpg",
    "docprops/thumbnail.png",
)


async def build_preview_response(
    file: WorkspaceFile, *, width: int, max_bytes: int
) -> Response:
    """Dispatch on MIME type and return the smallest useful preview."""
    try:
        return await _dispatch_preview(file, width=width, max_bytes=max_bytes)
    except FileNotFoundError as e:
        # File row exists but its bytes are gone from storage.
        raise fastapi.HTTPException(status_code=404, detail="File not found") from e


async def _dispatch_preview(
    file: WorkspaceFile, *, width: int, max_bytes: int
) -> Response:
    mime = (file.mime_type or "").lower()

    if mime.startswith("image/") and "svg" not in mime:
        _ensure_size(file, PREVIEW_MAX_IMAGE_BYTES)
        return _webp_response(await _image_thumbnail(file, width))
    if mime == "application/pdf":
        _ensure_size(file, PREVIEW_MAX_DOC_BYTES)
        return _webp_response(await _pdf_thumbnail(file, width))
    if mime in OFFICE_MIMES:
        _ensure_size(file, PREVIEW_MAX_DOC_BYTES)
        return _webp_response(await _office_thumbnail(file, width))
    if _is_text_like(mime, file.name):
        _ensure_size(file, PREVIEW_MAX_TEXT_BYTES)
        content = await _text_preview(file, max_bytes)
        return Response(
            content=content,
            media_type=file.mime_type or "text/plain",
            headers=CACHE_HEADERS,
        )

    raise fastapi.HTTPException(
        status_code=415, detail="Preview not available for this file type"
    )


def _is_text_like(mime: str, name: str = "") -> bool:
    if mime.startswith("text/") or any(hint in mime for hint in _TEXT_HINTS):
        return True
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return ext in _TEXT_EXTENSIONS


def _ensure_size(file: WorkspaceFile, limit: int) -> None:
    if file.size_bytes > limit:
        raise fastapi.HTTPException(status_code=413, detail="File too large to preview")


def _webp_response(content: bytes) -> Response:
    return Response(content=content, media_type="image/webp", headers=CACHE_HEADERS)


# ---------------------------------------------------------------------------
# Renderers (cached)
# ---------------------------------------------------------------------------


async def _image_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    async def render() -> bytes:
        content = await _retrieve(file)
        return await _to_thread_webp(_resize_webp, content, width)

    return await _cached(file, "img", width, render)


async def _pdf_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    async def render() -> bytes:
        content = await _retrieve(file)
        return await _to_thread_webp(_pdfium_first_page, content, width)

    return await _cached(file, "pdf", width, render)


async def _office_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    async def render() -> bytes:
        content = await _retrieve(file)
        try:
            raw = await asyncio.to_thread(_extract_zip_thumbnail, content)
        except Exception as e:
            raise fastapi.HTTPException(
                status_code=415, detail="Cannot render preview"
            ) from e
        if raw is None:
            raise fastapi.HTTPException(status_code=415, detail="No embedded thumbnail")
        return await _to_thread_webp(_resize_webp, raw, width)

    return await _cached(file, "office", width, render)


async def _text_preview(file: WorkspaceFile, max_bytes: int) -> bytes:
    storage = await get_workspace_storage()
    return await storage.retrieve_partial(file.storage_path, max_bytes)


async def _retrieve(file: WorkspaceFile) -> bytes:
    storage = await get_workspace_storage()
    return await storage.retrieve(file.storage_path)


async def _to_thread_webp(fn: Callable[..., bytes], *args) -> bytes:
    """Run a sync renderer in a thread, mapping any failure to a 415."""
    try:
        return await asyncio.to_thread(fn, *args)
    except fastapi.HTTPException:
        raise
    except Exception as e:
        raise fastapi.HTTPException(
            status_code=415, detail="Cannot render preview"
        ) from e


async def _cached(
    file: WorkspaceFile,
    kind: str,
    width: int,
    render: Callable[[], Awaitable[bytes]],
) -> bytes:
    key = f"wsfile:preview:{file.id}:{kind}:{width}:{(file.checksum or '')[:12]}"
    redis = None
    try:
        redis = await get_redis_async()
        cached = await redis.get(key)
        if cached:
            return base64.b64decode(cached)
    except Exception as e:
        logger.warning(f"Preview cache read failed for {file.id}: {e}")

    content = await render()

    if redis is not None:
        try:
            await redis.set(
                key, base64.b64encode(content).decode(), ex=PREVIEW_CACHE_TTL
            )
        except Exception as e:
            logger.warning(f"Preview cache write failed for {file.id}: {e}")
    return content


# ---------------------------------------------------------------------------
# Sync helpers (run in a thread)
# ---------------------------------------------------------------------------


def _resize_webp(content: bytes, width: int) -> bytes:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    with Image.open(io.BytesIO(content)) as img:
        # Image.open positions at the first frame, so animated GIFs/WebP already
        # render their frame 0 without an explicit seek.
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        img.thumbnail((width, width * 4), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="WEBP", quality=WEBP_QUALITY, method=4)
        return out.getvalue()


def _pdfium_first_page(content: bytes, width: int) -> bytes:
    pdf = pypdfium2.PdfDocument(content)
    try:
        page = pdf[0]
        page_width = page.get_width() or width
        # Integer render scale (pypdfium2's scale defaults to int); PIL downscales
        # to the exact target width afterwards.
        scale = max(1, min(3, round(width / page_width)))
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        out = io.BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.thumbnail((width, width * 4), Image.Resampling.LANCZOS)
        image.save(out, format="WEBP", quality=WEBP_QUALITY, method=4)
        return out.getvalue()
    finally:
        pdf.close()


def _extract_zip_thumbnail(content: bytes) -> Optional[bytes]:
    """Return the embedded ``docProps/thumbnail.*`` image from an openxml file."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = {name.lower(): name for name in archive.namelist()}
        for candidate in _EMBEDDED_THUMBNAILS:
            if candidate in names:
                return archive.read(names[candidate])
    return None
