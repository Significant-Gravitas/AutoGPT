"""Recall history protects a memory from the dream pass; it never condemns one.

Each destructive write the pass makes (``demotions.py``) leaves alone, in its
own statement, a live fact the user recalled within
``Config.dream_demotion_protect_days`` (``RecallProtection``,
``graphiti/recall_stamp.py``), unless the write's reason says the fact is
wrong rather than stale: the user retracted it (``user_signal``), or another
fact this pass read contradicts it (``contradicted_by:<uuid>``). Usage
disproves staleness, not wrongness. Both are syntactic checks on what the
model wrote, not verified intent or evidence, and the model reads content an
attacker may control: a contradiction overrides only when it cites a fact the
pass fetched other than the one it demotes, and any other reason,
``web_contradicted:`` included, is no override.

This window is the deterministic guard, and a setting of 0 turns off this
guard only. The sanitize prompt's recall rule (``prompts.py``) is separate and
stricter: it asks the model not to demote any fact with recalls for
staleness, however old, and the setting does not change it.

Why usage can never raise the number of facts a pass demotes: the guard never
changes which operations the pass attempts or their order (``clamp.py``
selects them from the proposals alone), and each write can only leave out a
fact it would otherwise change. A fact changed with usage data was live when
its write ran, so it was live when the first write reaching it ran, and
without usage data that first write would have changed it. Nothing is read
beforehand to decide, so no failed or stale read can let a protected fact
through: protection holds as long as the write itself runs.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict

from backend.copilot.graphiti.recall import USER_FORGET_REASON
from backend.copilot.graphiti.recall_stamp import RecallProtection, stamp_time
from backend.util.settings import Settings

# The reasons that say a fact is wrong, not stale: the user's own retraction
# (the reason a forget records) and a contradiction citing a fact the pass read.
USER_RETRACTION_REASON = USER_FORGET_REASON
CONTRADICTION_PREFIX = "contradicted_by:"


def demotion_protect_window() -> timedelta:
    """How recent a recall protects a fact
    (``Config.dream_demotion_protect_days``); zero protects nothing."""
    return timedelta(days=Settings().config.dream_demotion_protect_days)


class DemotionGuard(BaseModel):
    """The recall guard as one pass's writes apply it: where the protection
    window starts (``None``: no protection), and the facts a contradiction
    may cite, the ones the pass read."""

    model_config = ConfigDict(frozen=True)

    recalled_since: str | None
    citable: frozenset[str]

    @classmethod
    def at(cls, now: datetime, citable: Collection[str]) -> DemotionGuard:
        """The guard for writes made at *now*."""
        window = demotion_protect_window()
        since = stamp_time(now - window) if window > timedelta(0) else None
        return cls(recalled_since=since, citable=frozenset(citable))

    def protection(self, reason: str) -> RecallProtection:
        """What a write made for *reason* leaves alone."""
        if reason == USER_RETRACTION_REASON:
            return RecallProtection(recalled_since=self.recalled_since, override=True)
        if reason.startswith(CONTRADICTION_PREFIX):
            cited = reason.removeprefix(CONTRADICTION_PREFIX).strip()
            return RecallProtection(
                recalled_since=self.recalled_since,
                override=cited in self.citable,
                cited=cited,
            )
        return RecallProtection(recalled_since=self.recalled_since)
