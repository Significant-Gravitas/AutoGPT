"""Shared token-usage persistence and rate-limit recording.

Both the baseline (OpenRouter) and SDK (Anthropic) service layers need to:
  1. Append a ``Usage`` record to the session.
  2. Log the turn's token counts.
  3. Record weighted usage in Redis for rate-limiting.

This module extracts that common logic so both paths stay in sync.
"""

import logging

from .model import ChatSession, Usage
from .rate_limit import record_token_usage

logger = logging.getLogger(__name__)


async def persist_and_record_usage(
    *,
    session: ChatSession | None,
    user_id: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    log_prefix: str = "",
    cost_usd: float | str | None = None,
) -> int:
    """Persist token usage to session and record for rate limiting.

    Args:
        session: The chat session to append usage to (may be None on error).
        user_id: User ID for rate-limit counters (skipped if None).
        prompt_tokens: Uncached input tokens.
        completion_tokens: Output tokens.
        cache_read_tokens: Tokens served from prompt cache (Anthropic only).
        cache_creation_tokens: Tokens written to prompt cache (Anthropic only).
        log_prefix: Prefix for log messages (e.g. "[SDK]", "[Baseline]").
        cost_usd: Optional cost for logging (float from SDK, str otherwise).

    Returns:
        The computed total_tokens (prompt + completion; cache excluded).
    """
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0

    # total_tokens = prompt + completion. Cache tokens are tracked
    # separately and excluded from total so both baseline and SDK
    # paths share the same semantics.
    total_tokens = prompt_tokens + completion_tokens

    if session is not None:
        session.usage.append(
            Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )
        )

    if cache_read_tokens or cache_creation_tokens:
        logger.info(
            "%s Turn usage: uncached=%d, cache_read=%d, cache_create=%d, "
            "output=%d, total=%d, cost_usd=%s",
            log_prefix,
            prompt_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
        )
    else:
        logger.info(
            "%s Turn usage: prompt=%d, completion=%d, total=%d",
            log_prefix,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

    if user_id:
        try:
            await record_token_usage(
                user_id=user_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            )
        except Exception as usage_err:
            logger.warning(
                "%s Failed to record token usage: %s",
                log_prefix,
                usage_err,
            )

    return total_tokens
