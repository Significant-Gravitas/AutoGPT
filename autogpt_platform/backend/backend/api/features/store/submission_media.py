from collections.abc import AsyncIterator, Mapping

import aiohttp
import prisma.enums
import prisma.models
from autogpt_libs.auth import OrgAction, RequestContext, User, check_org_permission
from gcloud.aio import storage as async_storage
from google.cloud import storage as gcs_storage

from backend.util.gcs_utils import generate_iam_signed_url, is_not_found_error
from backend.util.settings import Settings

from . import local_media

PRIVATE_MEDIA_PREFIX = "/api/store/submissions/media/"
PRIVATE_MEDIA_PROXY_BUFFER_BYTES = 4 * 1024 * 1024
PRIVATE_MEDIA_SIGNED_URL_TTL_SECONDS = 60

# The library's default is a 10-second total timeout, which covers the whole
# body and so cuts off any video a client can't download in 10 seconds.
_STREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)


async def can_read(user: User, owner_user_id: str) -> bool:
    """
    Whether `user` may read media stored under `owner_user_id`: their own, an
    admin, or someone sharing an active organization with the owner whose role
    there covers its workflows, as listing its submissions requires. Leaving the
    organization ends access.
    """
    if user.user_id == owner_user_id or user.role == "admin":
        return True
    shared_memberships = await prisma.models.OrgMember.prisma().find_many(
        where={
            "userId": user.user_id,
            "status": prisma.enums.OrgMemberStatus.ACTIVE,
            "Org": {
                "is": {
                    "deletedAt": None,
                    "Members": {
                        "some": {
                            "userId": owner_user_id,
                            "status": prisma.enums.OrgMemberStatus.ACTIVE,
                        }
                    },
                }
            },
        }
    )
    return any(
        check_org_permission(_org_home_context(member), OrgAction.VIEW_RESOURCES)
        for member in shared_memberships
    )


def _org_home_context(member: prisma.models.OrgMember) -> RequestContext:
    return RequestContext(
        user_id=member.userId,
        org_id=member.orgId,
        team_id=None,
        is_org_owner=member.isOwner,
        is_org_admin=member.isAdmin,
        is_org_billing_manager=member.isBillingManager,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


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


async def signed_url(user_id: str, media_type: str, filename: str) -> str | None:
    bucket_name = Settings().config.resolved_private_user_data_bucket
    if not bucket_name:
        return None
    storage_path = object_path(user_id, media_type, filename)
    return await generate_iam_signed_url(
        gcs_storage.Client.create_anonymous_client(),
        bucket_name,
        storage_path,
        PRIVATE_MEDIA_SIGNED_URL_TTL_SECONDS,
    )


def object_path(user_id: str, media_type: str, filename: str) -> str:
    if media_type not in local_media.MEDIA_TYPES:
        raise ValueError("Invalid media type")
    safe_user_id = local_media.validate_path_component(user_id)
    safe_filename = local_media.validate_path_component(filename)
    if local_media.content_type_for_filename(safe_filename) is None:
        raise ValueError("Invalid media filename")
    return f"users/{safe_user_id}/{media_type}/{safe_filename}"
