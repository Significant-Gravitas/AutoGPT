import asyncio
import logging
from urllib.parse import quote

from pydantic import BaseModel

from backend.api.model import NotificationPayload
from backend.data.native_push_subscription import NativePushSubscriptionDTO
from backend.data.native_push_transport import send_apns, send_fcm
from backend.util.clients import get_database_manager_async_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class NativePushMessage(BaseModel):
    body: str
    path: str


def build_native_push(payload: NotificationPayload) -> NativePushMessage | None:
    data = payload.model_dump(mode="json")
    session_id = data.get("session_id")
    path = (
        f"/home?sessionId={quote(session_id, safe='')}"
        if isinstance(session_id, str) and 0 < len(session_id) <= 128
        else "/mobile?tab=attention"
    )
    if payload.type == "attention":
        return NativePushMessage(
            body="Your team needs a response. Open AutoGPT to continue.", path=path
        )
    if payload.type == "copilot_completion":
        return NativePushMessage(body="There's an update in your chat.", path=path)
    return None


async def send_native_push_for_user(user_id: str, payload: NotificationPayload) -> None:
    message = build_native_push(payload)
    config = Settings().config
    if message is None or not (
        config.apns_private_key_path or config.fcm_service_account_path
    ):
        return
    subscriptions = (
        await get_database_manager_async_client().get_native_push_subscriptions(user_id)
    )
    await asyncio.gather(*(deliver_native_push(sub, message) for sub in subscriptions))


async def deliver_native_push(
    sub: NativePushSubscriptionDTO, message: NativePushMessage
) -> None:
    try:
        send = send_apns if sub.provider == "apns" else send_fcm
        response = await send(sub, message.body, message.path)
        if response is None or response.is_success:
            return
        invalid = sub.provider == "apns" and response.status_code == 410
        if sub.provider == "apns" and response.status_code == 400:
            invalid = response.json().get("reason") == "BadDeviceToken"
        if sub.provider == "fcm" and response.status_code == 404:
            invalid = any(
                detail.get("errorCode") == "UNREGISTERED"
                for detail in response.json().get("error", {}).get("details", [])
            )
        if invalid:
            await get_database_manager_async_client().delete_native_push_subscription(
                sub.id
            )
        else:
            logger.warning(
                "Native push provider returned HTTP %s", response.status_code
            )
    except Exception:
        # Provider exceptions can contain tokens or credential material.
        logger.warning(
            "Native push delivery failed; registration retained for the next event"
        )
