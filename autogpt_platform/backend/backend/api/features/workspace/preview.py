"""
Preview generation for workspace files.

Produces small, cheap previews for the Artifacts page instead of shipping the
whole file to the browser:

* raster images        -> resized WebP thumbnail (Pillow)
* PDFs                 -> first page rasterised to WebP (pypdfium2)
* Office openxml docs  -> embedded ``docProps/thumbnail`` re-encoded to WebP
                         (matched by extension too, whatever the stored MIME)
* text-like files      -> first N bytes (partial read, no full download)

Image/PDF/Office thumbnails are cached in Redis keyed by file checksum so
repeat views and re-renders are free. All CPU-bound work runs in a thread so
the event loop is never blocked.
"""

import asyncio
import base64
import io
import logging
import math
import zipfile
from typing import Awaitable, Callable, Optional, cast

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
# A preview is a small thumbnail, so rasterising a source larger than this spends
# memory and CPU for no visible gain. Every renderer stays within this budget:
# images decline a source above it, PDFs pick a render scale from the page size.
MAX_RENDER_PIXELS = 50_000_000
# Thumbnails rasterise in worker threads; cap how many run at once so a burst of
# large previews can't hold many full-resolution bitmaps in memory together and
# slow every other request on the process.
MAX_CONCURRENT_RENDERS = 4
# Upper bound on the render scale. A PDF page with a sub-pixel side has that
# side rounded up to a whole pixel, so without a scale ceiling the other side
# could still rasterise well past the pixel budget.
MAX_RENDER_SCALE = 3
_MIN_RENDER_SCALE = 0.01
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
# Office files are often stored with a wrong MIME (text/plain,
# application/octet-stream), so the extension wins over the stored MIME.
_OFFICE_EXTENSIONS = frozenset({"pptx", "docx", "xlsx"})
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

    if mime in OFFICE_MIMES or _extension(file.name) in _OFFICE_EXTENSIONS:
        _ensure_size(file, PREVIEW_MAX_DOC_BYTES)
        return _webp_response(await _office_thumbnail(file, width))
    if mime.startswith("image/") and "svg" not in mime:
        _ensure_size(file, PREVIEW_MAX_IMAGE_BYTES)
        return _webp_response(await _image_thumbnail(file, width))
    if mime == "application/pdf":
        _ensure_size(file, PREVIEW_MAX_DOC_BYTES)
        return _webp_response(await _pdf_thumbnail(file, width))
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
    return _extension(name) in _TEXT_EXTENSIONS


def _extension(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _ensure_size(file: WorkspaceFile, limit: int) -> None:
    if file.size_bytes > limit:
        raise fastapi.HTTPException(status_code=413, detail="File too large to preview")


def _webp_response(content: bytes) -> Response:
    return Response(content=content, media_type="image/webp", headers=CACHE_HEADERS)


# ---------------------------------------------------------------------------
# Renderers (cached)
# ---------------------------------------------------------------------------

_render_semaphores: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _render_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _render_semaphores.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)
        _render_semaphores[loop] = semaphore
    return semaphore


async def _image_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    async def render() -> bytes:
        async with _render_semaphore():
            content = await _retrieve(file)
            return await _to_thread_webp(_resize_webp, content, width)

    return await _cached(file, "img", width, render)


async def _pdf_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    async def render() -> bytes:
        async with _render_semaphore():
            content = await _retrieve(file)
            return await _to_thread_webp(_pdfium_first_page, content, width)

    return await _cached(file, "pdf", width, render)


async def _office_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    async def render() -> bytes:
        async with _render_semaphore():
            content = await _retrieve(file)
            try:
                raw = await asyncio.to_thread(_extract_zip_thumbnail, content)
            except Exception as e:
                raise fastapi.HTTPException(
                    status_code=415, detail="Cannot render preview"
                ) from e
            if raw is None:
                raise fastapi.HTTPException(
                    status_code=415, detail="No embedded thumbnail"
                )
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
    Image.MAX_IMAGE_PIXELS = MAX_RENDER_PIXELS
    try:
        with Image.open(io.BytesIO(content)) as img:
            _ensure_within_render_budget(img.size)
            # Image.open positions at the first frame, so animated GIFs/WebP
            # already render their frame 0 without an explicit seek.
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")
            img.thumbnail((width, width * 4), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            img.save(out, format="WEBP", quality=WEBP_QUALITY, method=4)
            return out.getvalue()
    except Image.DecompressionBombError as e:
        # Pillow raises this above 2x the pixel limit, before the size check
        # above runs; treat it as the same over-budget case, not a 415.
        raise fastapi.HTTPException(
            status_code=413, detail="File too large to preview"
        ) from e


def _ensure_within_render_budget(size: tuple[int, int]) -> None:
    source_width, source_height = size
    if source_width * source_height > MAX_RENDER_PIXELS:
        raise fastapi.HTTPException(status_code=413, detail="File too large to preview")


def _pdfium_first_page(content: bytes, width: int) -> bytes:
    pdf = pypdfium2.PdfDocument(content)
    try:
        page = pdf[0]
        page_width, page_height = page.get_width(), page.get_height()
        scale = _render_scale(page_width, page_height, width)
        # A page can declare dimensions so large that even the minimum scale
        # exceeds the budget; pypdfium2 rounds each side up, so check the bitmap
        # it would allocate rather than the scaled area, and decline if over.
        _ensure_within_render_budget(
            (math.ceil(page_width * scale), math.ceil(page_height * scale))
        )
        # pypdfium2's ``scale`` is documented as a float but defaults to ``1``,
        # so the inferred type is ``int``; a sub-unit scale is what shrinks a
        # large page to thumbnail size.
        render_page = cast("Callable[..., pypdfium2.PdfBitmap]", page.render)
        bitmap = render_page(scale=scale)
        image = bitmap.to_pil()
        out = io.BytesIO()
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.thumbnail((width, width * 4), Image.Resampling.LANCZOS)
        image.save(out, format="WEBP", quality=WEBP_QUALITY, method=4)
        return out.getvalue()
    finally:
        pdf.close()


def _render_scale(page_width: float, page_height: float, target_width: int) -> float:
    """Render scale that reaches ~target_width while staying within the budget.

    Rendering a large page at a fixed scale produces a proportionally large
    bitmap; scaling from the page's own size keeps the output near thumbnail
    size whatever the page dimensions. A fixed upper bound also caps the scale,
    so a page with a sub-pixel side (which the renderer rounds up to one pixel)
    cannot rasterise past the budget on its other side.
    """
    if page_width <= 0 or page_height <= 0:
        return 1.0
    to_target_width = target_width / page_width
    within_budget = (MAX_RENDER_PIXELS / (page_width * page_height)) ** 0.5
    return max(min(to_target_width, within_budget, MAX_RENDER_SCALE), _MIN_RENDER_SCALE)


def _extract_zip_thumbnail(content: bytes) -> Optional[bytes]:
    """Return the embedded ``docProps/thumbnail.*`` image from an openxml file."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = {name.lower(): name for name in archive.namelist()}
        for candidate in _EMBEDDED_THUMBNAILS:
            if candidate in names:
                return archive.read(names[candidate])
    return None
