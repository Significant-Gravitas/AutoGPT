import hashlib
import logging

from backend.api.model import AttentionNotificationPayload
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)
from backend.data.redis_client import get_redis_async
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


async def notify_attention(
    user_id: str, event_id: str, session_id: str | None = None
) -> None:
    config = Settings().config
    if not (config.apns_private_key_path or config.fcm_service_account_path):
        return
    try:
        redis = await get_redis_async()
        key = (
            "mobile-attention:"
            + hashlib.sha256(f"{user_id}:{event_id}".encode()).hexdigest()
        )
        if not await redis.set(key, "1", nx=True, ex=86400):
            return
        try:
            await AsyncRedisNotificationEventBus().publish(
                NotificationEvent(
                    user_id=user_id,
                    payload=AttentionNotificationPayload(
                        type="attention",
                        event="response_required",
                        session_id=session_id,
                    ),
                )
            )
        except Exception:
            await redis.delete(key)
            raise
    except Exception:
        logger.warning("Could not publish a mobile attention notification")
