"""Which model writes the greeting: Sonnet, backed by a fast Haiku answer.

The greeting writes the plan's tasks, and the plan step waits about 30 s for
them before it settles on the role's starter tasks instead. Sonnet writes the
better copy, but a long take can keep it past that wait, and a deploy where
its call fails would give everyone the role's tasks rather than ones written
from what they said. So Sonnet goes first and the backup, which the team job
already calls the same way, joins it: at once when Sonnet fails, or alongside
it once Sonnet has run ``hedge_after`` seconds. The first full answer wins and
the other call is cancelled.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel

from backend.api.features.onboarding_dump.models import SuggestedPrompt

logger = logging.getLogger(__name__)


class IntroReply(BaseModel):
    """One model's answer: a greeting and its tasks, a greeting whose tasks
    were too few to use, or nothing."""

    greeting: str | None = None
    prompts: list[SuggestedPrompt] = []

    @property
    def full(self) -> bool:
        return bool(self.greeting) and bool(self.prompts)


class Attempt(BaseModel):
    model: str
    timeout: float


Generate = Callable[[str, float], Coroutine[Any, Any, IntroReply]]


async def race(
    primary: Attempt, backup: Attempt, generate: Generate, *, hedge_after: float
) -> IntroReply:
    """The first full reply from ``primary`` or ``backup``; failing that, the
    first reply with a greeting; failing that, an empty one.

    ``generate`` never raises: a failed call is an empty reply.
    """
    started = time.monotonic()
    running = {asyncio.create_task(generate(primary.model, primary.timeout)): primary}
    backup_started = False
    fallback = IntroReply()
    try:
        done, _ = await asyncio.wait(running, timeout=hedge_after)
        while True:
            for task in done:
                attempt = running.pop(task)
                reply = task.result()
                if reply.full:
                    logger.info(
                        "Brain dump greeting from %s in %.1f s",
                        attempt.model,
                        time.monotonic() - started,
                    )
                    return reply
                if reply.greeting and not fallback.greeting:
                    fallback = reply
            if not backup_started:
                backup_started = True
                task = asyncio.create_task(generate(backup.model, backup.timeout))
                running[task] = backup
            if not running:
                return fallback
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in running:
            task.cancel()
        # Let the losing call finish cancelling, so nothing outlives the race.
        await asyncio.gather(*running, return_exceptions=True)
