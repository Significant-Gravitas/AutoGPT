"""CoPilot service — shared helpers used by both SDK and baseline paths.

This module contains:
- System prompt building (Langfuse + default fallback)
- Session title generation
- Session assignment
- Shared config and client instances
"""

import asyncio
import logging
from typing import Any

from langfuse import get_client
from langfuse.openai import (
    AsyncOpenAI as LangfuseAsyncOpenAI,  # pyright: ignore[reportPrivateImportUsage]
)

from backend.data.db_accessors import understanding_db
from backend.data.understanding import format_understanding_for_prompt
from backend.util.exceptions import NotAuthorizedError, NotFoundError
from backend.util.settings import AppEnvironment, Settings

from .config import ChatConfig
from .model import (
    ChatSessionInfo,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)

logger = logging.getLogger(__name__)

config = ChatConfig()
settings = Settings()

_client: LangfuseAsyncOpenAI | None = None
_langfuse = None


def _get_openai_client() -> LangfuseAsyncOpenAI:
    global _client
    if _client is None:
        _client = LangfuseAsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    return _client


def _get_langfuse():
    global _langfuse
    if _langfuse is None:
        _langfuse = get_client()
    return _langfuse


# Default system prompt used when Langfuse is not configured
# Provides minimal baseline tone and personality - all workflow, tools, and
# technical details are provided via the supplement.
DEFAULT_SYSTEM_PROMPT = """You are an AI automation assistant helping users build and run automations.

Here is everything you know about the current user from previous interactions:

<users_information>
{users_information}
</users_information>

Your goal is to help users automate tasks by:
- Understanding their needs and business context
- Building and running working automations
- Delivering tangible value through action, not just explanation

Be concise, proactive, and action-oriented. Bias toward showing working solutions over lengthy explanations."""


# ---------------------------------------------------------------------------
# Shared helpers (used by SDK service and baseline)
# ---------------------------------------------------------------------------


def _is_langfuse_configured() -> bool:
    """Check if Langfuse credentials are configured."""
    return bool(
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    )


async def _get_system_prompt_template(context: str) -> str:
    """Get the system prompt, trying Langfuse first with fallback to default.

    Args:
        context: The user context/information to compile into the prompt.

    Returns:
        The compiled system prompt string.
    """
    if _is_langfuse_configured():
        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            # In non-production environments, fetch the latest prompt version
            # instead of the production-labeled version for easier testing
            label = (
                None
                if settings.config.app_env == AppEnvironment.PRODUCTION
                else "latest"
            )
            prompt = await asyncio.to_thread(
                _get_langfuse().get_prompt,
                config.langfuse_prompt_name,
                label=label,
                cache_ttl_seconds=config.langfuse_prompt_cache_ttl,
            )
            return prompt.compile(users_information=context)
        except Exception as e:
            logger.warning(f"Failed to fetch prompt from Langfuse, using default: {e}")

    # Fallback to default prompt
    return DEFAULT_SYSTEM_PROMPT.format(users_information=context)


async def _build_system_prompt(
    user_id: str | None, has_conversation_history: bool = False
) -> tuple[str, Any]:
    """Build the full system prompt including business understanding if available.

    Args:
        user_id: The user ID for fetching business understanding.
        has_conversation_history: Whether there's existing conversation history.
            If True, we don't tell the model to greet/introduce (since they're
            already in a conversation).

    Returns:
        Tuple of (compiled prompt string, business understanding object)
    """
    # If user is authenticated, try to fetch their business understanding
    understanding = None
    if user_id:
        try:
            understanding = await understanding_db().get_business_understanding(user_id)
        except Exception as e:
            logger.warning(f"Failed to fetch business understanding: {e}")
            understanding = None

    if understanding:
        context = format_understanding_for_prompt(understanding)
    elif has_conversation_history:
        context = "No prior understanding saved yet. Continue the existing conversation naturally."
    else:
        context = "This is the first time you are meeting the user. Greet them and introduce them to the platform"

    compiled = await _get_system_prompt_template(context)
    return compiled, understanding


async def _generate_session_title(
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """Generate a concise title for a chat session based on the first message.

    Args:
        message: The first user message in the session
        user_id: User ID for OpenRouter tracing (optional)
        session_id: Session ID for OpenRouter tracing (optional)

    Returns:
        A short title (3-6 words) or None if generation fails
    """
    try:
        # Build extra_body for OpenRouter tracing and PostHog analytics
        extra_body: dict[str, Any] = {}
        if user_id:
            extra_body["user"] = user_id[:128]  # OpenRouter limit
            extra_body["posthogDistinctId"] = user_id
        if session_id:
            extra_body["session_id"] = session_id[:128]  # OpenRouter limit
        extra_body["posthogProperties"] = {
            "environment": settings.config.app_env.value,
        }

        response = await _get_openai_client().chat.completions.create(
            model=config.title_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (3-6 words) for a chat conversation "
                        "based on the user's first message. The title should capture the "
                        "main topic or intent. Return ONLY the title, no quotes or punctuation."
                    ),
                },
                {"role": "user", "content": message[:500]},  # Limit input length
            ],
            max_tokens=20,
            extra_body=extra_body,
        )
        title = response.choices[0].message.content
        if title:
            # Clean up the title
            title = title.strip().strip("\"'")
            # Limit length
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        return None
    except Exception as e:
        logger.warning(f"Failed to generate session title: {e}")
        return None


async def _update_title_async(
    session_id: str, message: str, user_id: str | None = None
) -> None:
    """Generate and persist a session title in the background.

    Shared by both the SDK and baseline execution paths.
    """
    try:
        title = await _generate_session_title(message, user_id, session_id)
        if title and user_id:
            await update_session_title(session_id, user_id, title, only_if_empty=True)
            logger.debug("Generated title for session %s", session_id)
    except Exception as e:
        logger.warning("Failed to update session title for %s: %s", session_id, e)


async def assign_user_to_session(
    session_id: str,
    user_id: str,
) -> ChatSessionInfo:
    """
    Assign a user to a chat session.
    """
    session = await get_chat_session(session_id, None)
    if not session:
        raise NotFoundError(f"Session {session_id} not found")
    if session.user_id is not None and session.user_id != user_id:
        logger.warning(
            f"[SECURITY] Attempt to claim session {session_id} by user {user_id}, "
            f"but it already belongs to user {session.user_id}"
        )
        raise NotAuthorizedError(f"Not authorized to claim session {session_id}")
    session.user_id = user_id
    session = await upsert_chat_session(session)
    return session
