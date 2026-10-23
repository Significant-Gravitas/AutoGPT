"""Fetches the images an owner email's body names, so they travel as
attachments of the email itself.

Runs where the block runs: a ``workspace://`` image is read with the run's own
workspace scope, and an ``https://`` one through the request guard that refuses
private and internal addresses. An image that can't be used is left out, and
the body shows its alt text instead.
"""

import base64
import logging
from typing import TYPE_CHECKING

from backend.notifications.owner_email_body import MAX_IMAGES, ImageType, InlineImage
from backend.util.file import parse_data_uri, store_media_file
from backend.util.type import MediaFileType

if TYPE_CHECKING:
    from backend.data.execution import ExecutionContext

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 2 * 1024 * 1024
# Base64 grows it by a third, which keeps a full email well under Postmark's 10 MB.
MAX_TOTAL_BYTES = 4 * 1024 * 1024


async def fetch_images(
    sources: list[str], execution_context: "ExecutionContext"
) -> list[InlineImage]:
    images: list[InlineImage] = []
    total = 0
    for source in sources[:MAX_IMAGES]:
        content = await _read(source, execution_context)
        content_type = sniff_image_type(content) if content else None
        if content is None or content_type is None:
            continue
        if len(content) > MAX_IMAGE_BYTES or total + len(content) > MAX_TOTAL_BYTES:
            logger.info(f"Leaving out an email image of {len(content)} bytes")
            continue
        total += len(content)
        images.append(
            InlineImage(
                source=source,
                content_type=content_type,
                data=base64.b64encode(content).decode("ascii"),
            )
        )
    return images


def sniff_image_type(content: bytes) -> ImageType | None:
    """The type the bytes are, whatever the file's name or header claims."""
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _read(source: str, execution_context: "ExecutionContext") -> bytes | None:
    try:
        data_uri = await store_media_file(
            MediaFileType(source), execution_context, return_format="for_external_api"
        )
    # Missing, refused, blocked, unreachable or infected: the image is left out
    # either way, and the email still goes.
    except Exception as e:
        logger.info(f"Leaving out an email image that couldn't be read: {e}")
        return None
    parsed = parse_data_uri(data_uri)
    return base64.b64decode(parsed[1]) if parsed else None
