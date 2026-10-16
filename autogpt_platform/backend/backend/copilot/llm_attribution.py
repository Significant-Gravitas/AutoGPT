"""Who an AutoPilot LLM call is for, in the fields OpenRouter Broadcast reads.

OpenRouter Broadcast sends every generation to PostHog LLM analytics as an
``$ai_generation`` event (and to Langfuse). It maps these request body fields
(https://openrouter.ai/docs/guides/features/broadcast/posthog):

- ``user`` becomes the event's ``distinct_id``. It is the platform user id,
  the distinct id every other backend and browser event uses, so the
  generation lands on the same person as their product events and replays.
- ``session_id`` becomes ``$ai_session_id``: the chat session id.
- ``trace.generation_name`` becomes ``$ai_span_name``; every other ``trace``
  key arrives as ``metadata_<key>`` (``metadata_environment``,
  ``metadata_impersonated``).

Only calls that go through OpenRouter are covered. Direct Anthropic, Claude
Code subscription and user-connected (Codex) routes never pass OpenRouter, so
Broadcast sees none of their generations.
"""

from contextvars import ContextVar
from typing import Any, Literal

from autogpt_libs.auth.dependencies import IMPERSONATION_HEADER_NAME
from fastapi import Request

from backend.util.settings import Settings

settings = Settings()

# OpenRouter's limit on ``user``. ``session_id`` allows 256; chat session ids
# are UUIDs, so the shorter cap never cuts one.
_MAX_ID_CHARS = 128

GenerationName = Literal["copilot-title", "copilot-sdk", "copilot-baseline"]

# Whether an admin impersonating the user started the running turn. The
# executor sets it for every turn, so a value never carries over.
_turn_impersonated: ContextVar[bool] = ContextVar("_turn_impersonated", default=False)


def openrouter_attribution(
    *,
    user_id: str | None,
    session_id: str | None,
    generation_name: GenerationName,
) -> dict[str, Any]:
    """The body fields that attribute one OpenRouter request in Broadcast."""
    body: dict[str, Any] = {
        "trace": {
            "generation_name": generation_name,
            "environment": _environment(),
            "impersonated": is_turn_impersonated(),
        }
    }
    if user_id:
        body["user"] = user_id[:_MAX_ID_CHARS]
    if session_id:
        body["session_id"] = session_id[:_MAX_ID_CHARS]
    return body


def set_turn_impersonated(impersonated: bool) -> None:
    """Mark the running turn as started by an impersonating admin, or not."""
    _turn_impersonated.set(impersonated)


def is_turn_impersonated() -> bool:
    return _turn_impersonated.get()


def is_impersonated_request(request: Request) -> bool:
    """Whether an admin made *request* as another user.

    The auth dependencies have already refused the header from anyone but an
    admin by the time a route runs, so its presence is enough.
    """
    return bool(request.headers.get(IMPERSONATION_HEADER_NAME, "").strip())


def _environment() -> str:
    return settings.config.app_env.value
