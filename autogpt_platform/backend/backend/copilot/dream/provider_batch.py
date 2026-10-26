"""Anthropic's side of a batch dream pass: the key the callbacks submit and
cancel with, cancelling one of the pass's batches, and making sure one has
stopped.

The callbacks run in the BatchExecutor, a process of their own, so they look
the key up themselves the way ``batch_executor._default_api_key_for`` does:
the copilot config's direct Anthropic key first, then the shared settings key.
"""

import asyncio
import logging

from backend.copilot.config import ChatConfig
from backend.data import redis_client
from backend.util.llm.providers import cancel_batch, poll_batch
from backend.util.settings import Settings

from .locks import LOCK_CHECK_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# How long a best-effort cancel, or a status read, may take, the client's
# retries included.
PROVIDER_CANCEL_TIMEOUT_SECONDS = 10.0
# Anthropic ends every batch within 24 hours of its creation, expiring what
# it has not processed; an hour more for the end to show.
PROVIDER_BATCH_WINDOW_SECONDS = 25 * 60 * 60
# A batch whose cancel was sent without being acknowledged, kept in Redis so
# the next stop of it reads its status before sending another.
CANCEL_SENT_PREFIX = "dream:batch:cancel_sent"


def anthropic_api_key() -> str | None:
    """The copilot config's direct Anthropic key, else the shared settings
    key; ``None`` when neither is set or can be read."""
    try:
        key = ChatConfig().direct_anthropic_api_key
        if key:
            return key
    except Exception:
        logger.debug("ChatConfig unavailable during dream batch callback")
    try:
        return Settings().secrets.anthropic_api_key or None
    except Exception:
        return None


async def provider_batch_stopped(provider_batch_id: str) -> bool:
    """Whether *provider_batch_id* has stopped at Anthropic: the cancel was
    acknowledged (the batch stops within minutes), or the batch has already
    ended. ``False`` when neither can be confirmed: no key, the provider
    unreachable or out of time, or the batch still running after a refused
    cancel. Logged, never raised.

    A cancel and a status read can each take ``PROVIDER_CANCEL_TIMEOUT_SECONDS``,
    more together than a reaper attempt has, so they may be spread over two
    calls: a cancel not acknowledged is noted (``CANCEL_SENT_PREFIX``), and
    while the note stands a call reads the status without sending another.
    A status read that answers clears it, and so does one that fails: the
    note outlives only a call cut short during the read (an attempt's
    deadline), so the next one goes straight to the read it did not finish,
    and a batch still running gets the cancel again."""
    if not await _cancel_sent(provider_batch_id):
        if await cancel_provider_batch(provider_batch_id):
            return True
        await _note_cancel_sent(provider_batch_id, True)
    stopped = await _batch_ended(provider_batch_id)
    if not stopped:
        await _note_cancel_sent(provider_batch_id, False)
    return stopped


async def _batch_ended(provider_batch_id: str) -> bool:
    """Whether Anthropic reads the batch ended; ``False`` when it reads it
    running, or cannot be asked or answer within
    ``PROVIDER_CANCEL_TIMEOUT_SECONDS``."""
    api_key = anthropic_api_key()
    if api_key is None:
        return False
    try:
        async with asyncio.timeout(PROVIDER_CANCEL_TIMEOUT_SECONDS):
            status = await poll_batch(
                provider="anthropic",
                provider_batch_id=provider_batch_id,
                api_key=api_key,
            )
    except Exception:
        logger.warning(
            f"Could not read the status of dream batch {provider_batch_id}",
            exc_info=True,
        )
        return False
    if status in ("ended", "failed"):
        logger.info(f"Dream batch {provider_batch_id} has already ended")
        return True
    logger.warning(f"Dream batch {provider_batch_id} is still {status}")
    return False


async def _cancel_sent(provider_batch_id: str) -> bool:
    """Whether a cancel of the batch was sent and not acknowledged since its
    status last read running; ``False`` when Redis cannot say."""
    try:
        async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
            redis = await redis_client.get_redis_async()
            return bool(await redis.exists(f"{CANCEL_SENT_PREFIX}:{provider_batch_id}"))
    except Exception:
        logger.warning(
            f"Could not read whether dream batch {provider_batch_id}'s cancel was "
            "sent",
            exc_info=True,
        )
        return False


async def _note_cancel_sent(provider_batch_id: str, sent: bool) -> None:
    """Note, or clear, that the batch's cancel was sent and not
    acknowledged. Best-effort: without the note the next call sends the
    cancel again."""
    key = f"{CANCEL_SENT_PREFIX}:{provider_batch_id}"
    try:
        async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
            redis = await redis_client.get_redis_async()
            if sent:
                await redis.set(key, "1", ex=PROVIDER_BATCH_WINDOW_SECONDS)
            else:
                await redis.delete(key)
    except Exception:
        logger.warning(
            f"Could not note dream batch {provider_batch_id}'s cancel",
            exc_info=True,
        )


async def cancel_provider_batch(provider_batch_id: str) -> bool:
    """Ask Anthropic to cancel *provider_batch_id* through the client's
    ``messages.batches.cancel``, and say whether it acknowledged. Best-effort:
    a missing key, a refusal (a batch that has already ended cannot be
    cancelled) or the deadline is logged, never raised, so it never stands in
    the way of the stop that asked for it."""
    api_key = anthropic_api_key()
    if api_key is None:
        logger.warning(f"No Anthropic key to cancel dream batch {provider_batch_id}")
        return False
    try:
        async with asyncio.timeout(PROVIDER_CANCEL_TIMEOUT_SECONDS):
            acknowledged = await cancel_batch(
                provider="anthropic",
                provider_batch_id=provider_batch_id,
                api_key=api_key,
            )
    except Exception:
        logger.warning(
            f"Cancelling dream batch {provider_batch_id} failed", exc_info=True
        )
        return False
    if acknowledged:
        logger.info(f"Anthropic is cancelling dream batch {provider_batch_id}")
    else:
        logger.warning(f"Anthropic did not cancel dream batch {provider_batch_id}")
    return acknowledged
