"""PostHog analytics tracking for the chat system.

Event names are the product analytics plan's ``chat_*`` family. The events
captured here carry ``source: chat_copilot``; ``chat_message_sent`` goes
through ``product_analytics`` and carries ``source: platform``. Events are only
sent for a known user: a synthetic distinct id would create a PostHog person
nobody can merge.
"""

from typing import Any, Literal

from backend.util import posthog_client, product_analytics
from backend.util.posthog_events import PostHogEvent

SOURCE = "chat_copilot"

ChatOutcomeType = Literal["agent_run_success", "schedule_created"]


def track_user_message(
    user_id: str | None,
    session_id: str,
    message_length: int,
    *,
    expert_id: str | None = None,
    origin: str | None = None,
    surface: str | None = None,
) -> None:
    """Track when a user sends a message in chat.

    One ``chat_message_sent`` per turn (``expert_id`` set in an expert chat),
    carrying the message length.

    Args:
        user_id: The user's ID; no event without one
        session_id: The chat session ID
        message_length: Length of the user's message
        expert_id: Expert the session is scoped to, if any
        origin: Session origin ("interactive" | "automation"), when known
        surface: Where the message came from (web chat, slack, telegram, ...)
    """
    product_analytics.track_chat_turn(
        user_id=user_id,
        session_id=session_id,
        expert_id=expert_id,
        origin=origin,
        surface=surface,
        message_length=message_length,
    )


def track_tool_called(
    user_id: str | None,
    session_id: str,
    tool_name: str,
    tool_call_id: str,
) -> None:
    """Track when a tool is called in chat.

    Args:
        user_id: The user's ID; no event without one
        session_id: The chat session ID
        tool_name: Name of the tool being called
        tool_call_id: Unique ID of the tool call
    """
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_TOOL_CALLED,
        {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        },
        source=SOURCE,
    )


def track_chat_outcome(
    user_id: str,
    session_id: str,
    outcome_type: ChatOutcomeType,
    **properties: Any,
) -> None:
    """Track a moment of value in a chat: the copilot ran or scheduled
    something for the user. ``outcome_type`` uses the analytics plan's values.

    Args:
        user_id: The user's ID
        session_id: The chat session the outcome happened in
        outcome_type: Which kind of result the chat produced
        properties: Ids describing the result (``graph_id``, ...)
    """
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_OUTCOME,
        {**properties, "session_id": session_id, "outcome_type": outcome_type},
        source=SOURCE,
    )


def track_library_check_outcome(
    user_id: str,
    session_id: str | None,
    outcome: str,
    matches_count: int = 0,
    top_score: float | None = None,
) -> None:
    """Track the create-time library-similarity gate's outcome so we can
    measure how often the LLM bypasses it vs. reuses an existing agent.

    Args:
        outcome: One of ``"matches_shown"``, ``"no_matches"``,
            ``"bypassed_ack"`` (LLM set ``library_check_ack=true``),
            or ``"soft_failed"`` (search threw, gate silently disabled).
        matches_count: Number of candidates returned above threshold.
        top_score: Highest ``combined_score`` seen (above OR below
            threshold) — included so sub-threshold near-misses can
            inform retuning ``LIBRARY_SIMILARITY_THRESHOLD``.
    """
    posthog_client.capture(
        user_id,
        PostHogEvent.CHAT_LIBRARY_CHECK_OUTCOME,
        {
            "session_id": session_id,
            "outcome": outcome,
            "matches_count": matches_count,
            "top_score": top_score,
        },
        source=SOURCE,
    )
