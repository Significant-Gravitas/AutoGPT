"""Deleting brain-dump voice recordings after the retention window.

The privacy policy promises the recording is gone within 7 days; the
transcript stays while the account exists. This sweep is what keeps that
promise, and it works from the bucket rather than from the table:

- A row only points at the user's *current* take. Re-recording resets
  ``audioPath``, and deleting the account cascades the row away, but
  neither touches the blob — so the bucket holds recordings no row knows.
- Recordings stored before this shipped carry no expiry the generic
  cleanup can read (``store_file`` used to drop it), so only their age in
  the bucket says when they are due.

Rows that point at an expired recording are cleared *before* the blob is
deleted. If clearing fails nothing is deleted and the next run retries; if
a delete fails the row is already clear and the blob is found again next
run, so the database never points at a recording that is gone.
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable

from gcloud.aio import storage as async_gcs_storage
from pydantic import BaseModel

from backend.api.features.onboarding_dump.models import (
    AUDIO_FILE_EXTENSIONS,
    RECORDING_FILENAME_PREFIX,
)
from backend.api.features.onboarding_dump.storage import AUDIO_RETENTION
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.gcs_utils import fresh_gcs_client, is_not_found_error

logger = logging.getLogger(__name__)

# ``store_file`` puts user files at ``uploads/users/{user_id}/{uuid}/{name}``.
_RECORDINGS_GLOB = f"uploads/users/*/*/{RECORDING_FILENAME_PREFIX}*"
_RECORDING_NAME = re.compile(
    r"^uploads/users/[^/]+/[^/]+/"
    + re.escape(RECORDING_FILENAME_PREFIX)
    + r"[A-Za-z0-9_-]{1,64}\.(?:"
    + "|".join(sorted(set(AUDIO_FILE_EXTENSIONS.values())))
    + r")$"
)
_DELETE_CONCURRENCY = 10

ClearAudioPaths = Callable[[list[str]], Awaitable[int]]


class RecordingPurgeResult(BaseModel):
    expired: int = 0
    rows_cleared: int = 0
    deleted: int = 0
    failed: int = 0


async def purge_expired_recordings(
    clear_audio_paths: ClearAudioPaths,
    now: datetime | None = None,
) -> RecordingPurgeResult:
    """Delete every recording older than ``AUDIO_RETENTION``.

    ``clear_audio_paths`` nulls ``audioPath`` on rows that point at the
    given paths: the database manager's in the scheduler, which has no
    Prisma connection of its own. Logs counts only, never paths: they
    carry user IDs.
    """
    handler = await get_cloud_storage_handler()
    bucket = handler.config.gcs_bucket_name
    if not bucket:
        # Without a bucket ``store_audio`` fails, so nothing was ever stored.
        return RecordingPurgeResult()

    cutoff = (now or datetime.now(timezone.utc)) - AUDIO_RETENTION
    async with fresh_gcs_client() as client:
        names = await _list_expired_recordings(client, bucket, cutoff)
        if not names:
            return RecordingPurgeResult()
        rows_cleared = await clear_audio_paths([f"gcs://{bucket}/{n}" for n in names])
        outcomes = await _delete_blobs(client, bucket, names)

    deleted = sum(outcomes)
    result = RecordingPurgeResult(
        expired=len(names),
        rows_cleared=rows_cleared,
        deleted=deleted,
        failed=len(names) - deleted,
    )
    logger.info(f"[BrainDumpRetention] Purged expired recordings: {result}")
    return result


async def _list_expired_recordings(
    client: async_gcs_storage.Storage, bucket: str, cutoff: datetime
) -> list[str]:
    names: list[str] = []
    page_token: str | None = None
    while True:
        params = {"prefix": "uploads/users/", "matchGlob": _RECORDINGS_GLOB}
        if page_token:
            params["pageToken"] = page_token
        page = await client.list_objects(bucket, params=params)
        names.extend(
            item["name"]
            for item in page.get("items", [])
            if _RECORDING_NAME.match(item.get("name", ""))
            and _created_at(item) <= cutoff
        )
        page_token = page.get("nextPageToken")
        if not page_token:
            return names


def _created_at(item: dict) -> datetime:
    # RFC 3339 with a trailing "Z", which ``fromisoformat`` only reads
    # from Python 3.11 on.
    created = datetime.fromisoformat(item["timeCreated"].replace("Z", "+00:00"))
    return created if created.tzinfo else created.replace(tzinfo=timezone.utc)


async def _delete_blobs(
    client: async_gcs_storage.Storage, bucket: str, names: list[str]
) -> list[bool]:
    semaphore = asyncio.Semaphore(_DELETE_CONCURRENCY)

    async def delete(name: str) -> bool:
        async with semaphore:
            try:
                await client.delete(bucket, name)
            except Exception as e:
                # Already gone is what we wanted.
                if is_not_found_error(e):
                    return True
                logger.warning(
                    f"[BrainDumpRetention] Recording delete failed: "
                    f"{type(e).__name__}"
                )
                return False
            return True

    return await asyncio.gather(*(delete(name) for name in names))
