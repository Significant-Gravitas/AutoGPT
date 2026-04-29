from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Security
from starlette.status import HTTP_204_NO_CONTENT, HTTP_400_BAD_REQUEST

from backend.api.features.push.model import (
    PushSubscribeRequest,
    PushUnsubscribeRequest,
    VapidPublicKeyResponse,
)
from backend.data.push_subscription import (
    delete_push_subscription,
    upsert_push_subscription,
    validate_push_endpoint,
)
from backend.util.settings import Settings

router = APIRouter()
_settings = Settings()


@router.get(
    "/vapid-key",
    summary="Get VAPID public key for push subscription",
)
async def get_vapid_public_key() -> VapidPublicKeyResponse:
    return VapidPublicKeyResponse(public_key=_settings.secrets.vapid_public_key)


@router.post(
    "/subscribe",
    summary="Register a push subscription for the current user",
    status_code=HTTP_204_NO_CONTENT,
    dependencies=[Security(requires_user)],
)
async def subscribe_push(
    user_id: Annotated[str, Security(get_user_id)],
    body: PushSubscribeRequest,
) -> None:
    try:
        await validate_push_endpoint(body.endpoint)
        await upsert_push_subscription(
            user_id=user_id,
            endpoint=body.endpoint,
            p256dh=body.keys.p256dh,
            auth=body.keys.auth,
            user_agent=body.user_agent,
        )
    except ValueError as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@router.post(
    "/unsubscribe",
    summary="Remove a push subscription",
    status_code=HTTP_204_NO_CONTENT,
    dependencies=[Security(requires_user)],
)
async def unsubscribe_push(
    user_id: Annotated[str, Security(get_user_id)],
    body: PushUnsubscribeRequest,
) -> None:
    await delete_push_subscription(user_id, body.endpoint)
