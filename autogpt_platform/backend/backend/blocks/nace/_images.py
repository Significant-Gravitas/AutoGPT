"""Images for nace: loaded one at a time, typed by content, bounded by size."""

import base64
import os
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
# Every image reserves room for the longest prefix, "data:image/jpeg;base64,",
# so the data URLs together never pass MAX_REQUEST_BYTES.
PREFIX_ROOM = max(len(f"data:{mime};base64,") for mime in MIME_TYPES.values())
# The largest raw image whose data URL fits in a request on its own.
FULL_RAW_CAP = (MAX_REQUEST_BYTES - PREFIX_ROOM) // 4 * 3


async def resolve_images(
    images: list[MediaFileType], execution_context: ExecutionContext
) -> list[str]:
    """PNG, JPEG or WebP data URLs for nace, in input order.

    Images load one at a time against a budget of ``MAX_REQUEST_BYTES``
    data-URL characters. Each is read only up to the room left and refused
    before it is typed or encoded when its data URL would not fit; once no room
    is left, the next image is refused without being fetched. State, questions
    and JSON syntax are not counted here: ``_encode`` checks the whole body.
    """
    if not images:
        return []
    graph_exec_id = execution_context.graph_exec_id or ""
    resolved: list[str] = []
    used = 0
    for index, image in enumerate(images, start=1):
        raw_cap = max(0, MAX_REQUEST_BYTES - used - PREFIX_ROOM) // 4 * 3
        if raw_cap == 0:
            raise _past_the_budget(index)
        name = await store_media_file(
            file=image,
            execution_context=execution_context,
            return_format="for_local_processing",
        )
        path = get_exec_file_path(graph_exec_id, name)
        content = _read_image(path, index, raw_cap)
        url = _data_url(content, index)
        used += len(url)
        resolved.append(url)
    return resolved


def image_mime_type(content: bytes) -> str | None:
    """The MIME type of a PNG, JPEG or WebP image, or None for anything else.

    Pillow recognises the format from the file header, and ``verify()`` checks
    PNG chunk checksums. It reads little of a JPEG or WebP, so a truncated one
    still passes.
    """
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


def _read_image(path: str, index: int, raw_cap: int) -> bytes:
    """The file's bytes, read no further than one byte past ``raw_cap``."""
    with open(path, "rb") as handle:
        content = handle.read(raw_cap + 1)
        size = os.fstat(handle.fileno()).st_size
    if len(content) <= raw_cap:
        return content
    if size > FULL_RAW_CAP:
        raise ValueError(
            f"Image {index} is too large: its data URL alone would pass "
            f"{MAX_REQUEST_BYTES:,} bytes, the most nace accepts in one request. "
            "Send fewer or smaller images."
        )
    raise _past_the_budget(index)


def _past_the_budget(index: int) -> ValueError:
    return ValueError(
        f"Image {index} brings the images past {MAX_REQUEST_BYTES:,} bytes as "
        "data URLs, the most nace accepts in one request. "
        "Send fewer or smaller images."
    )


def _data_url(content: bytes, index: int) -> str:
    mime_type = image_mime_type(content)
    if mime_type is None:
        raise ValueError(f"Image {index} must be a PNG, JPEG or WebP image.")
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('ascii')}"
