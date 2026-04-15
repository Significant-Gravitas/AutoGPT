"""PostHog analytics tracking for the chat system."""

import atexit
import logging
from typing import Any

from posthog import Posthog

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# PostHog client instance (lazily initialized)
_posthog_client: Posthog | None = None


def _shutdown_posthog() -> None:
    """Flush and shutdown PostHog client on process exit."""
    if _posthog_client is not None:
        _posthog_client.flush()
        _posthog_client.shutdown()


atexit.register(_shutdown_posthog)


def _get_posthog_client() -> Posthog | None:
    """Get or create the PostHog client instance."""
    global _posthog_client
    if _posthog_client is not None:
        return _posthog_client

    if not settings.secrets.posthog_api_key:
        logger.debug("PostHog API key not configured, analytics disabled")
        return None

    _posthog_client = Posthog(
        settings.secrets.posthog_api_key,
        host=settings.secrets.posthog_host,
    )
    logger.info(
        f"PostHog client initialized with host: {settings.secrets.posthog_host}"
    )
    return _posthog_client


def _get_base_properties() -> dict[str, Any]:
    """Get base properties included in all events."""
    return {
        "environment": settings.config.app_env.value,
        "source": "chat_copilot",
    }


def track_user_message(
    user_id: str | None,
    session_id: str,
    message_length: int,
) -> None:
    """Track when a user sends a message in chat.

    Args:
        user_id: The user's ID (or None for anonymous)
        session_id: The chat session ID
        message_length: Length of the user's message
    """
    client = _get_posthog_client()
    if not client:
        return

    try:
        properties = {
            **_get_base_properties(),
            "session_id": session_id,
            "message_length": message_length,
        }
        client.capture(
            distinct_id=user_id or f"anonymous_{session_id}",
            event="copilot_message_sent",
            properties=properties,
        )
    except Exception as e:
        logger.warning(f"Failed to track user message: {e}")


def track_tool_called(
    user_id: str | None,
    session_id: str,
    tool_name: str,
    tool_call_id: str,
) -> None:
    """Track when a tool is called in chat.

    Args:
        user_id: The user's ID (or None for anonymous)
        session_id: The chat session ID
        tool_name: Name of the tool being called
        tool_call_id: Unique ID of the tool call
    """
    client = _get_posthog_client()
    if not client:
        logger.info("PostHog client not available for tool tracking")
        return

    try:
        properties = {
            **_get_base_properties(),
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }
        distinct_id = user_id or f"anonymous_{session_id}"
        logger.info(
            f"Sending copilot_tool_called event to PostHog: distinct_id={distinct_id}, "
            f"tool_name={tool_name}"
        )
        client.capture(
            distinct_id=distinct_id,
            event="copilot_tool_called",
            properties=properties,
        )
    except Exception as e:
        logger.warning(f"Failed to track tool call: {e}")


def track_agent_run_success(
    user_id: str,
    session_id: str,
    graph_id: str,
    graph_name: str,
    execution_id: str,
    library_agent_id: str,
) -> None:
    """Track when an agent is successfully run.

    Args:
        user_id: The user's ID
        session_id: The chat session ID
        graph_id: ID of the agent graph
        graph_name: Name of the agent
        execution_id: ID of the execution
        library_agent_id: ID of the library agent
    """
    client = _get_posthog_client()
    if not client:
        return

    try:
        properties = {
            **_get_base_properties(),
            "session_id": session_id,
            "graph_id": graph_id,
            "graph_name": graph_name,
            "execution_id": execution_id,
            "library_agent_id": library_agent_id,
        }
        client.capture(
            distinct_id=user_id,
            event="copilot_agent_run_success",
            properties=properties,
        )
    except Exception as e:
        logger.warning(f"Failed to track agent run: {e}")


def track_agent_scheduled(
    user_id: str,
    session_id: str,
    graph_id: str,
    graph_name: str,
    schedule_id: str,
    schedule_name: str,
    cron: str,
    library_agent_id: str,
) -> None:
    """Track when an agent is successfully scheduled.

    Args:
        user_id: The user's ID
        session_id: The chat session ID
        graph_id: ID of the agent graph
        graph_name: Name of the agent
        schedule_id: ID of the schedule
        schedule_name: Name of the schedule
        cron: Cron expression for the schedule
        library_agent_id: ID of the library agent
    """
    client = _get_posthog_client()
    if not client:
        return

    try:
        properties = {
            **_get_base_properties(),
            "session_id": session_id,
            "graph_id": graph_id,
            "graph_name": graph_name,
            "schedule_id": schedule_id,
            "schedule_name": schedule_name,
            "cron": cron,
            "library_agent_id": library_agent_id,
        }
        client.capture(
            distinct_id=user_id,
            event="copilot_agent_scheduled",
            properties=properties,
        )
    except Exception as e:
        logger.warning(f"Failed to track agent schedule: {e}")


def track_trigger_setup(
    user_id: str,
    session_id: str,
    graph_id: str,
    graph_name: str,
    trigger_type: str,
    library_agent_id: str,
) -> None:
    """Track when a trigger is set up for an agent.

    Args:
        user_id: The user's ID
        session_id: The chat session ID
        graph_id: ID of the agent graph
        graph_name: Name of the agent
        trigger_type: Type of trigger (e.g., 'webhook')
        library_agent_id: ID of the library agent
    """
    client = _get_posthog_client()
    if not client:
        return

    try:
        properties = {
            **_get_base_properties(),
            "session_id": session_id,
            "graph_id": graph_id,
            "graph_name": graph_name,
            "trigger_type": trigger_type,
            "library_agent_id": library_agent_id,
        }
        client.capture(
            distinct_id=user_id,
            event="copilot_trigger_setup",
            properties=properties,
        )
    except Exception as e:
        logger.warning(f"Failed to track trigger setup: {e}")
