"""The user's delegation settings, as the gate applies them.

``ask_before_external`` makes an outward call ask in every mode, including
Unsupervised, in Otto's own chat and in any thread Otto delegated. With it off
the mode table decides, as for every other effect. An expert's own chat is not
Otto's and keeps its mode.
"""

import logging

from backend.copilot.context import turn_delegation_settings_cache
from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.model import AutopilotMode, ChatSession
from backend.data.db_accessors import delegation_db

from .policy import Effect, Verdict, verdict_for_effect

logger = logging.getLogger(__name__)


async def verdict_for(
    mode: AutopilotMode, effect: Effect, user_id: str, session: ChatSession
) -> Verdict:
    """The mode's verdict, with the user's ask-before-external override."""
    verdict = verdict_for_effect(mode, effect)
    if effect is not Effect.EXTERNAL or verdict is Verdict.ASK:
        return verdict
    if not _ottos(session):
        return verdict
    settings = await turn_delegation_settings(user_id)
    return Verdict.ASK if settings.ask_before_external else verdict


async def turn_delegation_settings(user_id: str) -> DelegationSettings:
    """The user's settings, read once per turn.

    Unreadable settings fall back to the defaults, which ask before anything
    leaves the platform: a lost read must not make the gate more permissive.
    """
    cache = turn_delegation_settings_cache()
    if user_id in cache:
        return cache[user_id]
    try:
        settings = await delegation_db().get_delegation_settings(user_id)
    except Exception:
        logger.warning(f"Delegation settings unreadable for {user_id}", exc_info=True)
        return DelegationSettings()
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
