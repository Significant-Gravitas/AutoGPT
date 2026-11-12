"""Images for nace: loaded one at a time, typed by content, bounded by size."""

import base64
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from backend.data.execution import ExecutionContext
from backend.util.file import MediaFileType, get_exec_file_path, store_media_file

from ._client import MAX_REQUEST_BYTES

# Pillow's JPEG opener also opens multi-picture JPEGs (such as iPhone HDR
# photos) and names them MPO; there is no separate MPO opener to allow.
OPENERS = ("PNG", "JPEG", "WEBP")
MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "MPO": "image/jpeg",
    "WEBP": "image/webp",
}


async def resolve_images(
    images: list[MediaFileType], execution_context: ExecutionContext
) -> list[str]:
    """PNG, JPEG or WebP data URLs for nace, in input order.

    Images load one at a time. Each is read only up to the room left in the
    request and refused before it is typed or encoded when it does not fit, so
    later images are never fetched once the request is full.
    """
    if not images:
        return []
    graph_exec_id = execution_context.graph_exec_id or ""
    resolved: list[str] = []
    used = 0
    for index, image in enumerate(images, start=1):
        name = await store_media_file(
            file=image,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        content = _read_image(get_exec_file_path(graph_exec_id, name), index, used)
        url = _data_url(content, index)
        used += len(url)
        resolved.append(url)
    return resolved


def image_mime_type(content: bytes) -> str | None:
    """The MIME type of a PNG, JPEG or WebP image, or None for anything else."""
    try:
        with Image.open(BytesIO(content), formats=OPENERS) as image:
            image.verify()
            return MIME_TYPES.get(image.format or "")
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ):
        return None


def _read_image(path: str, index: int, used: int) -> bytes:
    """The file's bytes, if their base64 form fits beside ``used`` characters."""
    raw_cap = max(0, MAX_REQUEST_BYTES - used) // 4 * 3
    with open(path, "rb") as handle:
        content = handle.read(raw_cap + 1)
    if len(content) <= raw_cap:
        return content
    if used == 0:
        raise ValueError(
            f"Image {index} is too large: nace accepts at most "
            f"{MAX_REQUEST_BYTES:,} bytes of base64-encoded images per request. "
            "Send fewer or smaller images."
        )
    raise ValueError(
        f"Image {index} brings the images past {MAX_REQUEST_BYTES:,} bytes once "
        "base64-encoded, the most nace accepts in one request. "
        "Send fewer or smaller images."
    )


def _data_url(content: bytes, index: int) -> str:
    mime_type = image_mime_type(content)
    if mime_type is None:
        raise ValueError(f"Image {index} must be a PNG, JPEG or WebP image.")
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
