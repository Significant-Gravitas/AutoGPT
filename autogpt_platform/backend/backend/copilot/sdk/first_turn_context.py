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
        self._session_ctx = (
            asyncio.create_task(session_ctx) if session_ctx is not None else None
        )
        self._skills_ctx = asyncio.create_task(skills_ctx)
        self._expert_ctx = asyncio.create_task(expert_ctx)

    async def results(self) -> tuple[str, str, str]:
        """Await the blocks in their original order. Skills failures leave an
        empty index; the session and expert builders handle their own fallbacks."""
        session_ctx = await self._session_ctx if self._session_ctx is not None else ""
        try:
            skills_ctx = await self._skills_ctx
        except Exception:
            logger.exception(
                "[skills] failed to build skills_ctx — proceeding without it"
            )
            skills_ctx = ""
        return session_ctx, skills_ctx, await self._expert_ctx

    def close(self) -> None:
        """Cancel the reads without waiting for them, so it adds no await to
        the turn's cleanup, where a second cancellation could cut the rest
        of it short."""
        for task in (self._session_ctx, self._skills_ctx, self._expert_ctx):
            if task is not None:
                # cancel() also suppresses asyncio's unretrieved-exception
                # report for a task that finished before cleanup.
                task.cancel()
