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
from typing import Any, Awaitable, cast

from backend.data import redis_client
from backend.util.cloud_storage import get_cloud_storage_handler

logger = logging.getLogger(__name__)

# Long enough to cover a paused-and-resumed session or a client that
# reconnects after a network drop, short enough that abandoned takes don't
# accumulate.
PART_BUFFER_TTL_SECONDS = 6 * 60 * 60

# The audio has to outlive the onboarding session by enough for the user
# to come back and download it. 48h is the ceiling ``store_file`` accepts.
AUDIO_EXPIRATION_HOURS = 48


def _parts_key(user_id: str, recording_id: str) -> str:
    return f"onboarding:braindump:parts:{user_id}:{recording_id}"


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
    """
    redis = await redis_client.get_redis_async()
    key = _parts_key(user_id, recording_id)
    encoded = base64.b64encode(content).decode("ascii")
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, str(part_index), encoded)
        pipe.expire(key, PART_BUFFER_TTL_SECONDS)
        pipe.hvals(key)
        results = await cast(Awaitable[list[Any]], pipe.execute())
    values = cast(list[str], results[-1])
    return sum(_decoded_size(v) for v in values)


async def buffered_size(user_id: str, recording_id: str) -> int:
    return sum(
        _decoded_size(v) for v in (await _read_parts(user_id, recording_id)).values()
    )


async def assemble_parts(user_id: str, recording_id: str) -> bytes:
    """Concatenate the buffered parts in ``part_index`` order."""
    entries = await _read_parts(user_id, recording_id)
    if not entries:
        return b""
    ordered = sorted(entries.items(), key=lambda kv: int(kv[0]))
    return b"".join(base64.b64decode(value) for _, value in ordered)


def _decoded_size(encoded: str) -> int:
    """Byte length the base64 payload will have once decoded."""
    padding = encoded.count("=")
    return (len(encoded) * 3) // 4 - padding


async def discard_parts(user_id: str, recording_id: str) -> None:
    redis = await redis_client.get_redis_async()
    await redis.delete(_parts_key(user_id, recording_id))


async def store_audio(user_id: str, content: bytes, filename: str) -> str:
    handler = await get_cloud_storage_handler()
    return await handler.store_file(
        content,
        filename,
        expiration_hours=AUDIO_EXPIRATION_HOURS,
        user_id=user_id,
    )


async def audio_download_url(user_id: str, audio_path: str) -> str:
    handler = await get_cloud_storage_handler()
    return await handler.generate_signed_url(
        audio_path, expiration_hours=1, user_id=user_id
    )


async def retrieve_audio(user_id: str, audio_path: str) -> bytes:
    handler = await get_cloud_storage_handler()
    return await handler.retrieve_file(audio_path, user_id=user_id)
