"""Where an expert's picture comes from when the expert is packaged.

Everything else in a package is read out of our own database. An avatar is a
URL, and a URL is the one field of an expert that could make the backend fetch
somewhere of the caller's choosing. So bytes are read only from media we host —
a local upload off disk, an object in our own bucket — a path that ships with
the frontend travels as a URL, and anything else is dropped rather than
followed. A missing picture is a far smaller loss than an export that can be
turned into a request.
"""

import asyncio
import logging
import re

from gcloud.aio import storage as async_storage

from backend.api.features.experts.package_model import (
    AVATAR_EXTENSIONS,
    MAX_AVATAR_BYTES,
    PackagedAvatar,
)
from backend.api.features.store import local_media
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

_MEDIA_PATH = re.compile(r"^/api/store/media/([^/]+)/([^/]+)/([^/]+)$")
_GCS_PREFIX = "https://storage.googleapis.com/"


async def packaged_avatar(
    url: str | None, owner_user_id: str | None
) -> tuple[PackagedAvatar | None, bytes | None]:
    """How *url* travels in a package, and its bytes when we may read them.

    *owner_user_id* owns the expert being packaged. Both media backends key an
    upload by its uploader, so a stored ``avatarUrl`` naming somebody else's
    file is a URL the owner can write but must not be able to read through: the
    export would hand them bytes their own account never held. Ownership is
    therefore checked here, against the URL, rather than assumed from the fact
    that the route already checked who owns the expert.
    """
    if not url or not (stripped := url.strip()):
        return None, None
    base = settings.config.platform_base_url.rstrip("/")
    path = stripped[len(base) :] if base and stripped.startswith(base) else stripped
    if media := _MEDIA_PATH.match(path):
        return _packaged(await _local_media_bytes(owner_user_id, *media.groups()), path)
    if stripped.startswith(_GCS_PREFIX):
        return _packaged(await _bucket_bytes(stripped, owner_user_id), stripped)
    if path.startswith("/"):
        # A roster template's /experts/*.svg: it ships with the frontend, so a
        # URL restores it anywhere this platform runs.
        return PackagedAvatar(kind="url", url=path), None
    logger.info("Expert avatar is not our own media; packaging without it")
    return None, None


def _packaged(
    content: bytes | None, source: str
) -> tuple[PackagedAvatar | None, bytes | None]:
    extension = source.rsplit(".", 1)[-1].lower()
    if content is None or extension not in AVATAR_EXTENSIONS:
        logger.info("Expert avatar '%s' could not be packaged", source[:120])
        return None, None
    if len(content) > MAX_AVATAR_BYTES:
        logger.info("Expert avatar is %s bytes; packaging without it", len(content))
        return None, None
    return PackagedAvatar(kind="file", path=f"avatar.{extension}"), content


async def _local_media_bytes(
    owner_user_id: str | None, user_id: str, media_type: str, filename: str
) -> bytes | None:
    if not owner_user_id or user_id != owner_user_id:
        logger.info("Expert avatar is not the owner's media; packaging without it")
        return None
    try:
        path = local_media.get_media_path(user_id, media_type, filename)
        return await asyncio.to_thread(path.read_bytes)
    except (ValueError, OSError) as exc:
        logger.warning("Expert avatar could not be read from media storage: %s", exc)
        return None


async def _bucket_bytes(url: str, owner_user_id: str | None) -> bytes | None:
    """Only our own bucket, and only the owner's own objects within it, so a
    doctored avatarUrl cannot turn a package into a fetch of somewhere else —
    nor into a read of a neighbour's upload."""
    bucket = settings.config.media_gcs_bucket_name
    prefix = f"{_GCS_PREFIX}{bucket}/"
    if not bucket or not url.startswith(prefix):
        logger.info("Expert avatar is not in our media bucket; packaging without it")
        return None
    if not owner_user_id or not url[len(prefix) :].startswith(
        f"users/{owner_user_id}/"
    ):
        logger.info("Expert avatar is not the owner's object; packaging without it")
        return None
    try:
        async with async_storage.Storage() as client:
            return await client.download(bucket, url[len(prefix) :])
    except Exception as exc:
        logger.warning("Expert avatar could not be downloaded: %s", exc)
        return None
