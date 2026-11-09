"""Two tools Otto calls to open a feedback card in the chat.

The limits live in ``backend.data.feedback_prompts``; a tool only claims a
showing and reports whether a card appears. Chats nobody is looking at in the
web app have nowhere to show one.
"""

import logging
from typing import Any

from backend.api.features.feedback_prompts.analytics import capture_prompt_shown
from backend.copilot.llm_attribution import is_turn_impersonated, turn_trace_id
from backend.copilot.model import ChatSession
from backend.data.db_accessors import feedback_prompts_db
from backend.data.feedback_prompts import FeedbackPromptKind
from backend.util.posthog_client import capture
from backend.util.posthog_events import PostHogEvent

from .base import BaseTool
from .models import ErrorResponse, FeedbackPromptToolResponse, ToolResponseBase

logger = logging.getLogger(__name__)

_MAX_WHAT_WENT_WRONG_CHARS = 2000
_CARD_SHOWN = (
    "A card now asks the user about this. Do not mention it; finish as you "
    "normally would."
)
_NO_CARD = "No card will appear. Do not mention this to the user."


class RequestWorkCheckTool(BaseTool):
    """Ask how a finished piece of work went, at most once a week per user."""

    @property
    def name(self) -> str:
        return "request_work_check"

    @property
    def description(self) -> str:
        return (
            "Ask the user how finished work went. Call once when a turn fully "
            "delivers what was asked, as the last thing before your closing "
            "message. Never mention the check."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        if user_id is None or is_turn_impersonated() or not _can_show_card(session):
            return _no_card("work_check", session)
        prompt_id = await _claim(user_id, "work_check", session)
        if prompt_id is None:
            return _no_card("work_check", session)
        capture_prompt_shown(user_id, prompt_id, "work_check")
        return FeedbackPromptToolResponse(
            kind="work_check",
            prompt_id=prompt_id,
            is_first=await _is_first_work_check(user_id),
            message=_CARD_SHOWN,
            session_id=session.session_id,
        )


class FlagMistakeTool(BaseTool):
    """Record that Otto got something wrong, at most once per chat."""

    @property
    def name(self) -> str:
        return "flag_mistake"

    @property
    def description(self) -> str:
        return (
            "Flag that you got something wrong, such as after an angry reply, "
            "so the team can review it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "what_went_wrong": {
                    "type": "string",
                    "description": "One or two sentences on what you got wrong.",
                },
            },
            "required": ["what_went_wrong"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        what_went_wrong: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        described = what_went_wrong.strip()[:_MAX_WHAT_WENT_WRONG_CHARS]
        if not described:
            return ErrorResponse(
                message="what_went_wrong must not be empty.",
                session_id=session.session_id,
            )
        if user_id is None or is_turn_impersonated():
            return _no_card("otto_mistake", session)
        # Taken on chat platforms too, so the once-per-chat limit holds there.
        prompt_id = await _claim(user_id, "otto_mistake", session)
        if prompt_id is None:
            return _no_card("otto_mistake", session)
        _capture_mistake_flagged(user_id, prompt_id, described, session)
        if not _can_show_card(session):
            return _no_card("otto_mistake", session)
        capture_prompt_shown(user_id, prompt_id, "otto_mistake")
        return FeedbackPromptToolResponse(
            kind="otto_mistake",
            prompt_id=prompt_id,
            message=_CARD_SHOWN,
            session_id=session.session_id,
        )


def _can_show_card(session: ChatSession) -> bool:
    """Whether someone reads this chat in the web app, where cards render."""
    metadata = session.metadata
    return (
        metadata.source_platform is None
        # A row from before ``origin`` existed has None; those are web chats.
        and metadata.origin != "automation"
        and metadata.delegated_by_session_id is None
    )


def _no_card(kind: FeedbackPromptKind, session: ChatSession) -> ToolResponseBase:
    return FeedbackPromptToolResponse(
        kind=kind, message=_NO_CARD, session_id=session.session_id
    )


async def _claim(
    user_id: str, kind: FeedbackPromptKind, session: ChatSession
) -> str | None:
    """A prompt failing must not surface in the turn, so an error is a refusal."""
    try:
        return await feedback_prompts_db().claim_feedback_prompt(
            user_id=user_id, kind=kind, session_id=session.session_id
        )
    except Exception:
        logger.warning(
            "Failed to claim %s prompt for session %s",
            kind,
            session.session_id,
            exc_info=True,
        )
        return None


async def _is_first_work_check(user_id: str) -> bool:
    try:
        shown = await feedback_prompts_db().count_feedback_prompts(
            user_id=user_id, kind="work_check"
        )
    except Exception:
        logger.warning("Failed to count work checks", exc_info=True)
        return False
    return shown == 1


def _capture_mistake_flagged(
    user_id: str, prompt_id: str, described: str, session: ChatSession
) -> None:
    properties: dict[str, Any] = {
        "what_went_wrong": described,
        "$ai_session_id": session.session_id,
        "chat_session_id": session.session_id,
        "prompt_id": prompt_id,
    }
    trace_id = turn_trace_id()
    if trace_id:
        properties["$ai_trace_id"] = trace_id
    capture(user_id, PostHogEvent.OTTO_MISTAKE_FLAGGED, properties)
