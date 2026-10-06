from collections.abc import AsyncIterator, Mapping

import aiohttp
from gcloud.aio import storage as async_storage

from backend.util.gcs_utils import is_not_found_error
from backend.util.settings import Settings

from . import local_media

PRIVATE_MEDIA_PREFIX = "/api/store/submissions/media/"

# The library's default is a 10-second total timeout, which covers the whole
# body and so cuts off any video a client can't download in 10 seconds.
_STREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)


def url(user_id: str, media_type: str, filename: str) -> str:
    object_path(user_id, media_type, filename)
    return (
        f"{PRIVATE_MEDIA_PREFIX}{local_media.validate_path_component(user_id)}"
        f"/{media_type}/{local_media.validate_path_component(filename)}"
    )


async def metadata(
    user_id: str, media_type: str, filename: str
) -> Mapping[str, object]:
    bucket_name = Settings().config.resolved_private_user_data_bucket
    if not bucket_name:
        return await local_media.media_metadata(user_id, media_type, filename)

    storage_path = object_path(user_id, media_type, filename)
    try:
        async with async_storage.Storage() as async_client:
            return await async_client.download_metadata(bucket_name, storage_path)
    except Exception as error:
        if is_not_found_error(error):
            raise FileNotFoundError("Private media not found") from error
        raise


async def stream(
    user_id: str,
    media_type: str,
    filename: str,
    byte_range: tuple[int, int] | None = None,
) -> AsyncIterator[bytes]:
    bucket_name = Settings().config.resolved_private_user_data_bucket
    if not bucket_name:
        async for chunk in local_media.stream_media(
            user_id, media_type, filename, byte_range
        ):
            yield chunk
        return

    storage_path = object_path(user_id, media_type, filename)
    headers = None
    if byte_range is not None:
        start, end = byte_range
        headers = {"Range": f"bytes={start}-{end}"}
    async with async_storage.Storage() as async_client:
        stream = await async_client.download_stream(
            bucket_name,
            storage_path,
            headers=headers,
            timeout=_STREAM_TIMEOUT,  # type: ignore[arg-type]
        )
        while chunk := await stream.read(64 * 1024):
            yield chunk


def object_path(user_id: str, media_type: str, filename: str) -> str:
    if media_type not in local_media.MEDIA_TYPES:
        raise ValueError("Invalid media type")
    safe_user_id = local_media.validate_path_component(user_id)
    safe_filename = local_media.validate_path_component(filename)
    if local_media.content_type_for_filename(safe_filename) is None:
        raise ValueError("Invalid media filename")
    return f"users/{safe_user_id}/{media_type}/{safe_filename}"
