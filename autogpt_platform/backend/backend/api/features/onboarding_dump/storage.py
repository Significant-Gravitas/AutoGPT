"""Part buffering and durable storage for brain-dump audio.

MediaRecorder emits timeslice blobs that, concatenated in order, form a
valid stream — so parts are buffered in Redis (keyed by user + recording,
indexed by part) while the user is still talking, then concatenated once
on finalize and written to cloud storage as a single object.

Redis rather than one blob per part: a 30-minute dump is ~600 parts, and
600 GCS objects per user would need their own lifecycle management. The
buffer is disposable — the browser holds every part in IndexedDB until
finalize reports ``completed``, so losing the buffer costs a re-upload,
never a recording.
"""

import base64
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, cast

from backend.data import redis_client
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.gcs_utils import (
    blob_created_at,
    fresh_gcs_client,
    is_not_found_error,
    parse_gcs_path,
)

logger = logging.getLogger(__name__)

# Long enough to cover a paused-and-resumed session or a client that
# reconnects after a network drop, short enough that abandoned takes don't
# accumulate.
PART_BUFFER_TTL_SECONDS = 6 * 60 * 60

# Privacy policy: "We delete the recording as soon as it is transcribed
# (within 48 hours at most) and keep the transcript while your account
# exists." The pipeline deletes a recording once its transcript is stored;
# ``retention.py`` deletes every other one (failed, unfinished or orphaned
# takes, and deletes that failed) once it reaches this age.
AUDIO_RETENTION = timedelta(hours=48)

# The blob's own expiry, in the custom metadata the generic cleanup reads.
# Don't count on it: that cleanup lists only the first page of the bucket
# (1,000 objects), so in a bucket of any size it never reaches recordings,
# and the sweep is the only thing that deletes them. It sits a day past the
# sweep's reach, so where the cleanup does reach it, it never races the sweep
# for a recording whose row still points at it.
AUDIO_EXPIRATION_HOURS = int(
    (AUDIO_RETENTION + timedelta(days=1)).total_seconds() // 3600
)

# Set only by ``store_audio``, so a user's own upload that happens to be named
# like a recording (say, one they downloaded) is never taken for one.
RECORDING_MARKER_KEY = "brain_dump_recording"
RECORDING_MARKER_VALUE = "true"


# How often the scheduler runs the sweep. It is its own job rather than part
# of the generic file cleanup, which every scheduler start re-arms at start +
# interval, so frequent deploys could postpone it past 48 hours. This job
# keeps its next run across restarts (``Scheduler._register_recording_purge``).
RECORDING_PURGE_INTERVAL = timedelta(hours=1)

# A recording just short of the max age at one run goes at the next, once
# that run reaches it. The scheduler gives a run 5 minutes; the rest covers
# a run that starts late.
_PURGE_RUN_ALLOWANCE = timedelta(minutes=30)


def audio_max_age() -> timedelta:
    """The age at which a recording is deleted and no longer served.

    Taking the sweep's interval and run time off the retention keeps the
    deletion inside ``AUDIO_RETENTION``.
    """
    return AUDIO_RETENTION - RECORDING_PURGE_INTERVAL - _PURGE_RUN_ALLOWANCE


def _buffer_key(user_id: str, recording_id: str, suffix: str) -> str:
    return f"onboarding:braindump:{{{user_id}:{recording_id}}}:{suffix}"


def _parts_key(user_id: str, recording_id: str) -> str:
    return _buffer_key(user_id, recording_id, "parts")


def _sizes_key(user_id: str, recording_id: str) -> str:
    return _buffer_key(user_id, recording_id, "sizes")


def _discarded_key(user_id: str, recording_id: str) -> str:
    return _buffer_key(user_id, recording_id, "discarded")


async def _read_parts(user_id: str, recording_id: str) -> dict[str, str]:
    redis = await redis_client.get_redis_async()
    # The shared client runs with ``decode_responses=True``, so values come
    # back as ``str`` — which is why parts are base64 on the way in.
    return await cast(
        Awaitable[dict[str, str]], redis.hgetall(_parts_key(user_id, recording_id))
    )


async def append_part(
    user_id: str,
    recording_id: str,
    part_index: int,
    content: bytes,
) -> int:
    """Buffer one part and return the recording's cumulative byte count.

    Re-uploading the same ``part_index`` overwrites it, so the client's
    retry queue can replay a part it isn't sure landed.

    Parts are base64-encoded because the shared Redis client decodes
    responses as UTF-8 text — handing it raw opus bytes would round-trip
    them through a lossy decode and corrupt the recording.

    The running total comes from a companion hash of per-part *sizes*
    rather than from the parts themselves: summing ``hvals`` on the parts
    would pull every buffered byte back over the wire on every single
    upload — ~136 MB of live buffers at the 50 MB cap, and quadratic
    traffic across the ~600 parts of a 30-minute dump. Sizes are keyed by
    part index, so a replayed part overwrites its own entry instead of
    being counted twice, and the sum is read inside the same transaction
    that wrote the part, which is what makes it authoritative.
    """
    redis = await redis_client.get_redis_async()
    parts_key = _parts_key(user_id, recording_id)
    sizes_key = _sizes_key(user_id, recording_id)
    encoded = base64.b64encode(content).decode("ascii")
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(parts_key, str(part_index), encoded)
        pipe.hset(sizes_key, str(part_index), str(len(content)))
        pipe.expire(parts_key, PART_BUFFER_TTL_SECONDS)
        pipe.expire(sizes_key, PART_BUFFER_TTL_SECONDS)
        pipe.hvals(sizes_key)
        results = await cast(Awaitable[list[Any]], pipe.execute())
    return _sum_sizes(cast(list[str], results[-1]))


async def buffered_size(user_id: str, recording_id: str) -> int:
    redis = await redis_client.get_redis_async()
    sizes = await cast(
        Awaitable[list[str]], redis.hvals(_sizes_key(user_id, recording_id))
    )
    return _sum_sizes(sizes)


def _sum_sizes(sizes: list[str]) -> int:
    return sum(int(size) for size in sizes)


async def assemble_parts(user_id: str, recording_id: str) -> bytes:
    """Concatenate the buffered parts in ``part_index`` order."""
    entries = await _read_parts(user_id, recording_id)
    if not entries:
        return b""
    ordered = sorted(entries.items(), key=lambda kv: int(kv[0]))
    return b"".join(base64.b64decode(value) for _, value in ordered)


async def discard_parts(user_id: str, recording_id: str) -> None:
    redis = await redis_client.get_redis_async()
    await redis.delete(
        _parts_key(user_id, recording_id), _sizes_key(user_id, recording_id)
    )


async def mark_discarded(user_id: str, recording_id: str) -> None:
    """Remember a take the client threw away, while its parts could still arrive."""
    redis = await redis_client.get_redis_async()
    await redis.set(
        _discarded_key(user_id, recording_id), "1", ex=PART_BUFFER_TTL_SECONDS
    )


async def was_discarded(user_id: str, recording_id: str) -> bool:
    redis = await redis_client.get_redis_async()
    return bool(await redis.exists(_discarded_key(user_id, recording_id)))


async def store_audio(user_id: str, content: bytes, filename: str) -> str:
    handler = await get_cloud_storage_handler()
    return await handler.store_file(
        content,
        filename,
        expiration_hours=AUDIO_EXPIRATION_HOURS,
        user_id=user_id,
        delete_when_expired=True,
        custom_metadata={RECORDING_MARKER_KEY: RECORDING_MARKER_VALUE},
    )


async def audio_available(audio_path: str, now: datetime | None = None) -> bool:
    """Whether the stored recording can still be handed back.

    It can't once it is gone, nor once it is old enough for the sweep to
    delete: until the next run it is still in the bucket, but the policy
    already counts it as deleted. A path outside the configured private
    bucket can't be served either.
    """
    handler = await get_cloud_storage_handler()
    bucket, blob = parse_gcs_path(audio_path)
    if bucket != handler.config.gcs_bucket_name:
        return False
    created = await blob_created_at(bucket, blob)
    if created is None:
        return False
    return (now or datetime.now(timezone.utc)) - created < audio_max_age()


async def delete_audio(audio_path: str) -> None:
    """Delete one stored recording; one that is already gone counts as deleted.

    Any other failure raises, for the caller to leave the file to the
    retention sweep. A path outside the configured private bucket is
    refused rather than deleted.
    """
    handler = await get_cloud_storage_handler()
    bucket, blob = parse_gcs_path(audio_path)
    if bucket != handler.config.gcs_bucket_name:
        raise ValueError("Recording is not in the configured private bucket")
    async with fresh_gcs_client() as client:
        try:
            await client.delete(bucket, blob)
        except Exception as e:
            if not is_not_found_error(e):
                raise


async def audio_download_url(user_id: str, audio_path: str) -> str:
    handler = await get_cloud_storage_handler()
    return await handler.generate_signed_url(
        audio_path, expiration_hours=1, user_id=user_id
    )
