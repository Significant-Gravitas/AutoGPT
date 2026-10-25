"""The user's delegation settings, as the gate applies them.

``ask_before_external`` makes an outward call ask in every mode, including
Unsupervised, in Otto's own chat and in any thread Otto delegated. With it off
the mode table decides, as for every other effect. An expert's own chat is not
Otto's and keeps its mode.

The delegation mode and ``new_experts_ask_first`` decide whether Otto's
hand-off itself waits for approval, whatever mode Otto's chat runs in.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.copilot.context import turn_delegation_settings_cache
from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.model import AutopilotMode, ChatSession
from backend.data.db_accessors import delegation_db

from .handoff import resolve_expert
from .policy import Effect, Verdict, effect_for, verdict_for_effect

logger = logging.getLogger(__name__)

# Both leave the platform. The toggle stops either from running unchecked;
# a cheap one Auto hands the supervisor stays with the supervisor.
_OUTWARD = frozenset({Effect.EXTERNAL, Effect.EXTERNAL_JUDGED})
# How long a new hire's hand-offs wait for approval, when the user asks for it.
NEW_HIRE_WINDOW = timedelta(days=7)
_HAND_OFF_ASKS_FIRST = (
    "Your delegation settings say Ask first, so this hand-off needs your approval."
)
_SETTINGS_UNREADABLE = (
    "Your delegation settings couldn't be read, so this hand-off needs your "
    "approval."
)
# What a lost read falls back to: the strictest the settings can be. It asks
# before anything leaves the platform and holds every hand-off, so a read
# that fails never makes the gate more permissive than the user set it.
UNREADABLE_SETTINGS = DelegationSettings(
    mode="ask_first", ask_before_external=True, new_experts_ask_first=True
)
_NEW_HIRE_ASKS_FIRST = (
    "{name} joined your team this week and new experts start in Ask first, "
    "so this hand-off needs your approval."
)


async def verdict_for(
    mode: AutopilotMode, effect: Effect, user_id: str, session: ChatSession
) -> Verdict:
    """The mode's verdict, with the user's ask-before-external override."""
    verdict = verdict_for_effect(mode, effect)
    if effect not in _OUTWARD or verdict is not Verdict.RUN:
        return verdict
    if not _ottos(session):
        return verdict
    settings = await turn_delegation_settings(user_id)
    return Verdict.ASK if settings.ask_before_external else verdict


async def hand_off_hold(
    tool_name: str, args: dict[str, Any], user_id: str, session: ChatSession
) -> str | None:
    """Why the user's delegation settings hold this hand-off for approval, or
    None when they leave it to the chat's mode.

    Otto's own chat only, as for the mode a thread Otto opens starts in: an
    expert's chat keeps its own mode. A teammate that cannot be found is left
    to the tool, which refuses the hand-off itself.
    """
    if effect_for(tool_name) is not Effect.TEAM or session.expert_id is not None:
        return None
    settings = await turn_delegation_settings(user_id)
    if settings is UNREADABLE_SETTINGS:
        return _SETTINGS_UNREADABLE
    if settings.mode == "ask_first":
        return _HAND_OFF_ASKS_FIRST
    if not settings.new_experts_ask_first:
        return None
    reference = str(args.get("expert_id") or "").strip()
    target = await resolve_expert(user_id, reference) if reference else None
    if target is None or not await hired_lately(user_id, target.id):
        return None
    return _NEW_HIRE_ASKS_FIRST.format(name=target.name)


async def hired_lately(user_id: str, expert_id: str) -> bool:
    """Hired less than a week ago. An unknown hire date counts as new."""
    hired_at = await delegation_db().get_expert_hired_at(user_id, expert_id)
    return hired_at is None or datetime.now(UTC) - hired_at < NEW_HIRE_WINDOW


async def turn_delegation_settings(user_id: str) -> DelegationSettings:
    """The user's settings, read once per turn.

    Unreadable settings fall back to ``UNREADABLE_SETTINGS``, the strictest
    the settings can be: a lost read must not make the gate more permissive.
    """
    cache = turn_delegation_settings_cache()
    if user_id in cache:
        return cache[user_id]
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
    except Exception:
        logger.warning(f"Delegation settings unreadable for {user_id}", exc_info=True)
        return UNREADABLE_SETTINGS
    cache[user_id] = settings
    return settings


def _ottos(session: ChatSession) -> bool:
    """Otto's own chat, or a thread Otto (no expert) delegated."""
    if session.expert_id is None:
        return True
    meta = session.metadata
    return (
        meta.delegated_by_session_id is not None
        and meta.delegated_by_expert_id is None
        and meta.handed_off_from_expert_id is None
    )
