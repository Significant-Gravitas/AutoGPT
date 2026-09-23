import base64
import hashlib
import re

from fastapi import HTTPException

from backend.data.redis_client import get_redis_async
from backend.executor.automod.models import AutoModResponse
from backend.util.request import Requests
from backend.util.settings import Settings

settings = Settings()
_MANAGED_AVATAR = re.compile(
    r"^/avatars/notion/\d+(?:-\d+){9}\.(?:rose|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|fuchsia)\.svg$"
    r"|^/experts/(?:maria|max|frankie)\.svg$"
    r"|^/autogpt-characters/v1\.1/expert-(?:maria|mina)/neutral/(?:24|32|40|48|64|80|96|128|192|256|512|1024)\.(?:png|webp)$"
)


async def moderate_avatar_image(
    user_id: str, content: bytes, content_type: str
) -> None:
    if not settings.config.automod_api_url or not settings.secrets.automod_api_key:
        raise HTTPException(
            503, "Appearance review is unavailable. Please try again later."
        )
    image = base64.b64encode(content).decode("ascii")
    try:
        response = await Requests(
            extra_headers={"X-API-Key": settings.secrets.automod_api_key.strip()},
        ).post(
            f"{settings.config.automod_api_url.rstrip('/')}/moderate",
            json={
                "type": "image",
                "content": f"data:{content_type};base64,{image}",
                "metadata": {"user_id": user_id, "purpose": "expert-avatar"},
            },
            timeout=settings.config.automod_timeout,
        )
        result = AutoModResponse.model_validate(response.json())
    except Exception as error:
        raise HTTPException(
            503, "We couldn't review this image. Please try again later."
        ) from error
    if (
        not result.success
        or result.status != "approved"
        or any(item.decision != "approved" for item in result.moderation_results)
    ):
        raise HTTPException(
            422,
            "This image wasn't approved for an Expert appearance. Choose another image.",
        )


async def record_approved_avatar(user_id: str, url: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.setex(_receipt_key(user_id, url), 86400, "approved")
    except Exception as error:
        raise HTTPException(
            503, "Appearance review is unavailable. Please try again later."
        ) from error


async def require_approved_avatar(user_id: str, url: str | None) -> None:
    if not url or _MANAGED_AVATAR.fullmatch(url):
        return
    redis = await get_redis_async()
    if await redis.get(_receipt_key(user_id, url)):
        return
    raise HTTPException(
        400, "Upload this image through Change appearance so it can be reviewed first."
    )


def _receipt_key(user_id: str, url: str) -> str:
    digest = hashlib.sha256(url.encode()).hexdigest()
    return f"expert-avatar-approval:{user_id}:{digest}"
