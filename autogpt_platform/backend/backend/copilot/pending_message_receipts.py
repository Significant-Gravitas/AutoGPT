"""Short-lived admission receipts survive draining an in-flight message."""

from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async


class PendingMessageReceiptUnavailable(Exception):
    pass


def pending_message_receipt_key(session_id: str, message_id: str) -> str:
    return f"copilot:pending-received:{{{session_id}}}:{message_id}"


async def pending_message_was_received(session_id: str, message_id: str | None) -> bool:
    if message_id is None:
        return False
    try:
        redis = await get_redis_async()
        return bool(
            await redis.exists(pending_message_receipt_key(session_id, message_id))
        )
    except (RedisError, RedisClusterException, ConnectionError, OSError) as exc:
        raise PendingMessageReceiptUnavailable() from exc
