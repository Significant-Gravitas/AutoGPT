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
repeat views and re-renders are free. Every render of a file's bytes runs in a
short-lived child process with a time and memory limit (see ``preview_render``),
so neither the event loop nor its thread pool waits on one.
"""

import asyncio
import base64
import logging
from typing import Awaitable, Callable

import fastapi
from fastapi.responses import Response

from backend.api.features.workspace import preview_render
from backend.data.redis_client import get_redis_async
from backend.data.workspace import WorkspaceFile
from backend.util.workspace_storage import get_workspace_storage

logger = logging.getLogger(__name__)

PREVIEW_CACHE_TTL = 86_400  # 1 day
# Each thumbnail renders in its own child process; cap how many run at once so a
# burst of large previews can't hold many full-resolution bitmaps in memory
# together.
MAX_CONCURRENT_RENDERS = 4
CACHE_HEADERS = {"Cache-Control": "private, max-age=86400"}
# A text preview is always served as plain text.
TEXT_PREVIEW_HEADERS = {
    **CACHE_HEADERS,
    "Content-Security-Policy": "sandbox",
    "X-Content-Type-Options": "nosniff",
}

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
            content=content, media_type="text/plain", headers=TEXT_PREVIEW_HEADERS
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
    return await _cached(
        file, "img", width, _render(file, preview_render.image_thumbnail, width)
    )


async def _pdf_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    return await _cached(
        file, "pdf", width, _render(file, preview_render.pdf_thumbnail, width)
    )


async def _office_thumbnail(file: WorkspaceFile, width: int) -> bytes:
    return await _cached(
        file, "office", width, _render(file, preview_render.office_thumbnail, width)
    )


def _render(
    file: WorkspaceFile, renderer: preview_render.Renderer, width: int
) -> Callable[[], Awaitable[bytes]]:
    async def render() -> bytes:
        async with _render_semaphore():
            content = await _retrieve(file)
            return await _render_in_child(renderer, content, width)

    return render


async def _text_preview(file: WorkspaceFile, max_bytes: int) -> bytes:
    storage = await get_workspace_storage()
    return await storage.retrieve_partial(file.storage_path, max_bytes)


async def _retrieve(file: WorkspaceFile) -> bytes:
    storage = await get_workspace_storage()
    return await storage.retrieve(file.storage_path)


async def _render_in_child(
    renderer: preview_render.Renderer, content: bytes, width: int
) -> bytes:
    """Run a renderer in a child process, mapping each outcome to a response."""
    try:
        return await preview_render.render_in_child(renderer, content, width)
    except preview_render.PreviewTooLarge as e:
        raise fastapi.HTTPException(
            status_code=413, detail="File too large to preview"
        ) from e
    except preview_render.NoEmbeddedThumbnail as e:
        raise fastapi.HTTPException(
            status_code=415, detail="No embedded thumbnail"
        ) from e
    except preview_render.RenderFailed as e:
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
