"""Authorship record for messages the bot posted proactively.

``edit_message`` may only touch a message *this account* asked the bot to
post. Channel authorization alone can't establish that: two users linked to
the same server both pass a "is this channel in a server you linked?" check,
so without a record of who sent what, either could rewrite the other's post —
or any reply the bot made to a third party, turning the bot's own history
into something forgeable.

So every proactive send records ``(platform, channel_id, ref_id) -> sender``
here, and the edit path requires a matching record. Inbound replies the
handler makes on a user's behalf are deliberately *not* recorded: they are
conversation, not the caller's post, and nothing should rewrite them.

Records live in Redis with a 30-day TTL. Expiry only ever costs an edit the
model could have made (it gets ``not_sender`` and can post afresh); it can
never grant one.
"""

import json
import logging

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

SENT_MESSAGE_TTL = 30 * 86400  # 30 days


def _key(platform: str, channel_id: str, ref_id: str) -> str:
    return f"copilot-bot:sent:{platform}:{channel_id}:{ref_id}"


async def record_sent(
    platform: str,
    channel_id: str,
    ref_id: str,
    user_id: str,
    *,
    chunk_count: int = 1,
    editable: bool = True,
) -> None:
    """Record that ``user_id`` had the bot post ``ref_id`` in ``channel_id``.

    ``chunk_count``/``editable`` come off the adapter's ``PostedRef`` and are
    stored so the edit path can refuse the cases it cannot honour rather than
    silently half-applying them.

    Never raises: a proactive post that already landed must not be reported as
    failed because Redis was briefly unreachable. The cost of a lost record is
    a refused edit, which is the safe direction.
    """
    try:
        redis = await get_redis_async()
        await redis.set(
            _key(platform, channel_id, ref_id),
            json.dumps(
                {"user_id": user_id, "chunks": chunk_count, "editable": editable}
            ),
            ex=SENT_MESSAGE_TTL,
        )
    except Exception:
        logger.exception(
            "Failed to record sent message %s/%s on %s; edits to it will be refused",
            channel_id,
            ref_id,
            platform,
        )


async def sender_of(
    platform: str, channel_id: str, ref_id: str
) -> tuple[str, int, bool] | None:
    """``(user_id, chunk_count, editable)`` for a recorded post, else ``None``.

    ``None`` means "not a message this bot posted proactively, or the record
    aged out" — both of which must refuse an edit.
    """
    try:
        redis = await get_redis_async()
        raw = await redis.get(_key(platform, channel_id, ref_id))
    except Exception:
        logger.exception(
            "Failed to read sent-message record %s/%s on %s",
            channel_id,
            ref_id,
            platform,
        )
        return None
    if not raw:
        return None
    try:
        record = json.loads(raw)
        return (
            str(record["user_id"]),
            int(record.get("chunks", 1)),
            bool(record.get("editable", True)),
        )
    except (ValueError, TypeError, KeyError):
        logger.warning(
            "Malformed sent-message record for %s/%s on %s",
            channel_id,
            ref_id,
            platform,
        )
        return None
