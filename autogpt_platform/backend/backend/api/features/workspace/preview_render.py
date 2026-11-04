"""
Thumbnail renderers for workspace previews, run in a short-lived child process.

Each render of a file's bytes runs in a fresh interpreter started with
``python -m`` (never a fork of the threaded API process), is killed once it
passes ``RENDER_TIMEOUT_SECONDS``, and is capped at ``RENDER_MAX_MEMORY_BYTES``
of address space; a child that cannot set the cap does not render. The parent
awaits it on the event loop, so a render never holds a thread of the default
executor. The child receives only the environment variables in
``CHILD_ENV_ALLOWLIST``.

The module imports only what rendering needs, so a child starts in a fraction
of a second. The child answers with an exit code and the raw thumbnail bytes,
never a pickle, so nothing the child sends is executed by the parent.
"""

import asyncio
import functools
import importlib
import io
import logging
import math
import os
import sys
import zipfile
from collections.abc import Callable
from typing import Optional, cast

import pypdfium2
from PIL import Image

from backend.util.zip_member import ZipMemberTooLarge, read_member

if sys.platform != "win32":
    import resource

logger = logging.getLogger(__name__)

RENDER_TIMEOUT_SECONDS = 10
RENDER_MAX_MEMORY_BYTES = 768 * 1024 * 1024
# A thumbnail is at most 1024 px wide; anything bigger did not come from a
# renderer here.
MAX_THUMBNAIL_BYTES = 16 * 1024 * 1024
# The embedded thumbnail of an Office file: the same cap as an image preview.
MAX_EMBEDDED_THUMBNAIL_BYTES = 10_000_000
WEBP_QUALITY = 80
# A preview is a small thumbnail, so rasterising a source larger than this spends
# memory and CPU for no visible gain. Every renderer stays within this budget:
# images decline a source above it, PDFs pick a render scale from the page size.
MAX_RENDER_PIXELS = 50_000_000
# Upper bound on the render scale. A PDF page with a sub-pixel side has that
# side rounded up to a whole pixel, so without a scale ceiling the other side
# could still rasterise well past the pixel budget.
MAX_RENDER_SCALE = 3
_MIN_RENDER_SCALE = 0.01
_EMBEDDED_THUMBNAILS = (
    "docprops/thumbnail.jpeg",
    "docprops/thumbnail.jpg",
    "docprops/thumbnail.png",
)

# The only environment variables a render child receives: what the interpreter
# needs to start, find the backend package and write temporary files. The
# rest of the API process environment stays in the parent.
CHILD_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SYSTEMROOT",
)

_CHILD_MODULE = "backend.api.features.workspace.preview_render"
_EXIT_OK = 0
_EXIT_TOO_LARGE = 3
_EXIT_NO_THUMBNAIL = 4
_EXIT_NO_MEMORY_CAP = 5

Renderer = Callable[[bytes, int], bytes]


class PreviewTooLarge(Exception):
    """The source is over the render budget, or the render ran out of memory."""


class NoEmbeddedThumbnail(Exception):
    """An Office file carries no ``docProps/thumbnail`` image."""


class RenderFailed(Exception):
    """The render failed, timed out or crashed."""


async def render_in_child(renderer: Renderer, content: bytes, width: int) -> bytes:
    """Run ``renderer(content, width)`` in a child process and return its output."""
    try:
        process = await _start_child(renderer, width)
    except OSError as e:
        logger.warning(f"Preview render process could not start: {e}")
        raise RenderFailed() from e
    try:
        stdout = await asyncio.wait_for(
            _exchange(process, content), timeout=RENDER_TIMEOUT_SECONDS
        )
    except TimeoutError as e:
        logger.warning(f"Preview render timed out after {RENDER_TIMEOUT_SECONDS}s")
        raise RenderFailed() from e
    finally:
        if process.returncode is None:
            process.kill()
            # The exit is only reported once the output pipe closes, and reading
            # stops when the output is refused or the render is cancelled, so
            # discard what the child left in the pipe first.
            await _discard(cast(asyncio.StreamReader, process.stdout))
            await process.wait()
    return _result(process.returncode, stdout)


async def _start_child(renderer: Renderer, width: int) -> asyncio.subprocess.Process:
    _check_memory_cap()
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        _CHILD_MODULE,
        f"{renderer.__module__}:{renderer.__qualname__}",
        str(width),
        str(RENDER_MAX_MEMORY_BYTES),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=_child_env(),
    )


def _child_env() -> dict[str, str]:
    return {
        name: os.environ[name] for name in CHILD_ENV_ALLOWLIST if name in os.environ
    }


async def _exchange(process: asyncio.subprocess.Process, content: bytes) -> bytes:
    stdin = cast(asyncio.StreamWriter, process.stdin)
    stdout = cast(asyncio.StreamReader, process.stdout)
    feed = asyncio.create_task(_feed(stdin, content))
    try:
        output = await _read_bounded(stdout, MAX_THUMBNAIL_BYTES)
        await process.wait()
    finally:
        feed.cancel()
    return output


async def _feed(stdin: asyncio.StreamWriter, content: bytes) -> None:
    # A child that exits early (over budget, out of memory) closes its end
    # before reading everything; its exit code carries the outcome.
    try:
        stdin.write(content)
        await stdin.drain()
        stdin.close()
    except (BrokenPipeError, ConnectionResetError):
        pass


async def _read_bounded(stream: asyncio.StreamReader, limit: int) -> bytes:
    out = bytearray()
    while chunk := await stream.read(64 * 1024):
        out += chunk
        if len(out) > limit:
            raise RenderFailed()
    return bytes(out)


async def _discard(stream: asyncio.StreamReader) -> None:
    while await stream.read(64 * 1024):
        pass


def _result(returncode: int | None, stdout: bytes) -> bytes:
    if returncode == _EXIT_OK and stdout:
        return stdout
    if returncode == _EXIT_TOO_LARGE:
        raise PreviewTooLarge()
    if returncode == _EXIT_NO_THUMBNAIL:
        raise NoEmbeddedThumbnail()
    if returncode == _EXIT_NO_MEMORY_CAP:
        if not _check_memory_cap():
            _warn_child_could_not_cap_memory()
        raise RenderFailed()
    logger.warning(f"Preview render exited with code {returncode}")
    raise RenderFailed()


@functools.cache
def _check_memory_cap() -> bool:
    """Warn once when a child cannot apply the memory cap; return whether it did.

    On Windows the child renders without one. Elsewhere a child that cannot
    set it exits without rendering, so every preview is "not available".
    """
    if sys.platform == "win32":
        logger.warning(
            "RLIMIT_AS is not available on this platform; preview renders run "
            "without a memory cap"
        )
        return True
    try:
        _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    except (ValueError, OSError) as e:
        logger.warning(f"RLIMIT_AS is not available ({e}); previews will not render")
        return True
    if hard_limit != resource.RLIM_INFINITY and hard_limit < RENDER_MAX_MEMORY_BYTES:
        logger.warning(
            f"RLIMIT_AS hard limit {hard_limit} is below the preview memory cap "
            f"{RENDER_MAX_MEMORY_BYTES}; previews will not render"
        )
        return True
    return False


@functools.cache
def _warn_child_could_not_cap_memory() -> None:
    # The limits checked above allowed the cap, yet the child could not set it.
    logger.warning(
        "Preview render child could not set its memory cap; previews will not render"
    )


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


def _child_main(argv: list[str]) -> int:
    target, width, max_memory = argv
    if not _limit_memory(int(max_memory)):
        return _EXIT_NO_MEMORY_CAP
    module_name, _, attr = target.partition(":")
    renderer: Renderer = getattr(importlib.import_module(module_name), attr)
    content = sys.stdin.buffer.read()
    try:
        output = renderer(content, int(width))
    except (PreviewTooLarge, MemoryError):
        return _EXIT_TOO_LARGE
    except NoEmbeddedThumbnail:
        return _EXIT_NO_THUMBNAIL
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    return _EXIT_OK


def _limit_memory(max_bytes: int) -> bool:
    if sys.platform == "win32":
        return True
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except (ValueError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# Renderers (run in the child)
# ---------------------------------------------------------------------------


def image_thumbnail(content: bytes, width: int) -> bytes:
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
        # above runs; treat it as the same over-budget case.
        raise PreviewTooLarge() from e


def pdf_thumbnail(content: bytes, width: int) -> bytes:
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


def office_thumbnail(content: bytes, width: int) -> bytes:
    raw = _extract_zip_thumbnail(content)
    if raw is None:
        raise NoEmbeddedThumbnail()
    return image_thumbnail(raw, width)


def _ensure_within_render_budget(size: tuple[int, int]) -> None:
    source_width, source_height = size
    if source_width * source_height > MAX_RENDER_PIXELS:
        raise PreviewTooLarge()


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
                info = archive.getinfo(names[candidate])
                try:
                    return read_member(
                        archive, info, max_bytes=MAX_EMBEDDED_THUMBNAIL_BYTES
                    )
                except ZipMemberTooLarge as e:
                    raise PreviewTooLarge() from e
    return None


if __name__ == "__main__":
    # Run the copy imported under the module's own name, so the exceptions a
    # renderer raises are the same classes the child catches.
    sys.exit(importlib.import_module(_CHILD_MODULE)._child_main(sys.argv[1:]))
