"""A new chat's first-message context, read beside the turn's setup.

A session's first user message carries three blocks the SDK engine reads
from other services: the session block (the session's pending follow-ups,
``build_session_context``), the skills index (``build_skills_context``) and
the expert or team block (``first_message_expert_context``). They need only
the turn's ids, not anything setup produces, so they start before the setup
gather and the first turn awaits them where it builds its message, instead of
reading them one after another once setup has finished.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class FirstTurnContext:
    """The three reads, started; ``results`` awaits them, ``close`` stops
    whatever a turn that ends early never awaited."""

    def __init__(
        self,
        session_ctx: Coroutine[Any, Any, str] | None,
        skills_ctx: Coroutine[Any, Any, str],
        expert_ctx: Coroutine[Any, Any, str],
    ) -> None:
        self._session_ctx = asyncio.create_task(_or_empty(session_ctx))
        self._skills_ctx = asyncio.create_task(_skills_index(skills_ctx))
        self._expert_ctx = asyncio.create_task(expert_ctx)

    async def results(self) -> tuple[str, str, str]:
        """The session block, the skills index and the expert block. A
        failed session or expert read raises here, where the turn read them
        before; a failed skills read is logged and leaves the index empty."""
        return await self._session_ctx, await self._skills_ctx, await self._expert_ctx

    def close(self) -> None:
        """Cancel the reads without waiting for them, so it adds no await to
        the turn's cleanup, where a second cancellation could cut the rest
        of it short."""
        for task in (self._session_ctx, self._skills_ctx, self._expert_ctx):
            # Also on a read that already failed: cancelling it marks the
            # error handled, so it isn't logged as never retrieved.
            task.cancel()


async def _or_empty(read: Coroutine[Any, Any, str] | None) -> str:
    return "" if read is None else await read


async def _skills_index(read: Coroutine[Any, Any, str]) -> str:
    # Failures here MUST NOT block the turn: log and go on with no index.
    try:
        return await read
    except Exception:
        logger.exception("[skills] failed to build skills_ctx — proceeding without it")
        return ""
