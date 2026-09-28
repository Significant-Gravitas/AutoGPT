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
from backend.util.llm.providers import cancel_batch, poll_batch
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

# How long a best-effort cancel, or a status read, may take, the client's
# retries included.
PROVIDER_CANCEL_TIMEOUT_SECONDS = 10.0
# Anthropic ends every batch within 24 hours of its creation, expiring what
# it has not processed; an hour more for the end to show.
PROVIDER_BATCH_WINDOW_SECONDS = 25 * 60 * 60


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
    cancel. Logged, never raised."""
    if await cancel_provider_batch(provider_batch_id):
        return True
    api_key = anthropic_api_key()
    if api_key is None:
        return False
    try:
        status = await asyncio.wait_for(
            poll_batch(
                provider="anthropic",
                provider_batch_id=provider_batch_id,
                api_key=api_key,
            ),
            timeout=PROVIDER_CANCEL_TIMEOUT_SECONDS,
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
        acknowledged = await asyncio.wait_for(
            cancel_batch(
                provider="anthropic",
                provider_batch_id=provider_batch_id,
                api_key=api_key,
            ),
            timeout=PROVIDER_CANCEL_TIMEOUT_SECONDS,
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
