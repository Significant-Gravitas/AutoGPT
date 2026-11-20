"""Deleting brain-dump voice recordings after the retention window.

The privacy policy promises the recording is deleted as soon as it is
transcribed, within 48 hours at most; the transcript stays while the
account exists. The pipeline deletes a recording once its transcript is
stored (``service.finalize_voice_dump``). This sweep catches every other
one: takes that failed or never finished, which keep their recording so
the user can download it or retry, and transcribed ones whose delete
failed. It works from the bucket rather than from the table:

- A row only points at the user's *current* take. Re-recording resets
  ``audioPath``, and deleting the account cascades the row away, but
  neither touches the blob — so the bucket holds recordings no row knows.
- Recordings stored before this shipped carry no expiry the generic
  cleanup can read (``store_file`` used to drop it), so only their age in
  the bucket says when they are due.

What counts as a recording: ``POST /files/upload`` stores a user's file
under its own name in the same ``uploads/users/{user_id}/{uuid}/`` layout,
so a name like ``brain-dump-<id>.webm`` (the name a downloaded recording
is saved under) proves nothing. ``store_audio`` marks recordings in their
custom metadata, which an upload can't set. Recordings from before the
marker existed are recognised by name, but only if they were created
before ``_UNMARKED_RECORDINGS_BEFORE``, a date on or after the release
that added the marker. Each run counts the name-matched, unmarked objects
it skipped as too new (``unmarked_skipped``), so a recording missed that
way shows up in the log.

Rows that point at an expired recording are cleared *before* the blob is
deleted. If clearing fails nothing is deleted and the next run retries; if
a delete fails the row is already clear and the blob is found again next
run, so the database never points at a recording that is gone. (A blob
that goes some other way can leave a row behind; the download route
answers 404 for that and clears the row.)

Nothing else deletes recordings: the generic expiry cleanup reads only the
first page of the bucket listing. A run that fails raises out of the job,
which APScheduler logs at ERROR, so it reaches Sentry.
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
from backend.api.features.onboarding_dump.storage import (
    RECORDING_MARKER_KEY,
    RECORDING_MARKER_VALUE,
    audio_max_age,
)
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.gcs_utils import (
    fresh_gcs_client,
    is_not_found_error,
    object_created_at,
)

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
# Must not be earlier than the production release that started marking
# recordings: an unmarked recording stored in between would never be
# deleted (``unmarked_skipped`` counts what this date leaves out). Later
# only risks an upload named like a recording, stored after that release
# and before this date.
_UNMARKED_RECORDINGS_BEFORE = datetime(2027, 1, 1, tzinfo=timezone.utc)
_DELETE_CONCURRENCY = 10
# Keeps one run well inside the scheduler's 5-minute timeout. The first runs
# after this shipped have every recording ever stored to delete; at one run
# an hour that is 48,000 a day. Each run takes the most recently due first,
# so a take that just reached its max age never waits behind that backlog.
_MAX_PURGED_PER_RUN = 2000

ClearAudioPaths = Callable[[list[str]], Awaitable[int]]


class RecordingPurgeResult(BaseModel):
    expired: int = 0
    rows_cleared: int = 0
    deleted: int = 0
    failed: int = 0
    # Named like a recording but unmarked, and created on or after
    # ``_UNMARKED_RECORDINGS_BEFORE``: a user's upload, or a recording the
    # cutoff missed.
    unmarked_skipped: int = 0


async def purge_expired_recordings(
    clear_audio_paths: ClearAudioPaths,
    now: datetime | None = None,
) -> RecordingPurgeResult:
    """Delete every recording that reached ``audio_max_age()``.

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

    cutoff = (now or datetime.now(timezone.utc)) - audio_max_age()
    async with fresh_gcs_client() as client:
        listing = await _list_expired_recordings(client, bucket, cutoff)
        result = RecordingPurgeResult(unmarked_skipped=listing.unmarked_skipped)
        names = listing.names
        if names:
            result.rows_cleared = await clear_audio_paths(
                [f"gcs://{bucket}/{n}" for n in names]
            )
            outcomes = await _delete_blobs(client, bucket, names)
            result.expired = len(names)
            result.deleted = sum(outcomes)
            result.failed = len(names) - result.deleted

    logger.info(f"[BrainDumpRetention] Purged expired recordings: {result}")
    return result


class _Listing(BaseModel):
    names: list[str]
    unmarked_skipped: int


async def _list_expired_recordings(
    client: async_gcs_storage.Storage, bucket: str, cutoff: datetime
) -> _Listing:
    """Up to ``_MAX_PURGED_PER_RUN`` due recordings, newest first.

    The bucket lists in name (user ID) order, so the whole listing is read
    before any are picked.
    """
    due: list[tuple[datetime, str]] = []
    unmarked_skipped = 0
    page_token: str | None = None
    while True:
        params = {"prefix": "uploads/users/", "matchGlob": _RECORDINGS_GLOB}
        if page_token:
            params["pageToken"] = page_token
        page = await client.list_objects(bucket, params=params)
        for item in page.get("items", []):
            if not _RECORDING_NAME.match(item.get("name", "")):
                continue
            created = object_created_at(item)
            if not _is_marked(item) and created >= _UNMARKED_RECORDINGS_BEFORE:
                unmarked_skipped += 1
                continue
            if created <= cutoff:
                due.append((created, item["name"]))
        page_token = page.get("nextPageToken")
        if not page_token:
            break
    due.sort(reverse=True)
    return _Listing(
        names=[name for _, name in due[:_MAX_PURGED_PER_RUN]],
        unmarked_skipped=unmarked_skipped,
    )


def _is_marked(item: dict) -> bool:
    custom = item.get("metadata") or {}
    return custom.get(RECORDING_MARKER_KEY) == RECORDING_MARKER_VALUE


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
                    f"[BrainDumpRetention] Recording delete failed: {type(e).__name__}"
                )
                return False
            return True

    return await asyncio.gather(*(delete(name) for name in names))
