"""Who an AutoPilot LLM call is for, in the fields OpenRouter Broadcast reads.

OpenRouter Broadcast sends every generation to PostHog LLM analytics as an
``$ai_generation`` event (and to Langfuse). It maps these request body fields
(https://openrouter.ai/docs/guides/features/broadcast/posthog):

- ``user`` becomes the event's ``distinct_id``. It is the platform user id,
  the distinct id every other backend and browser event uses, so the
  generation lands on the same person as their product events and replays.
- ``session_id`` becomes ``$ai_session_id``: the chat session id.
- ``trace.trace_id`` becomes ``$ai_trace_id``: the reply's trace, so every
  model call behind one AutoPilot reply (the main loop, sub-agents,
  compaction, the chat title, and the expert subtasks it spawns) groups into
  one trace. It is the id of the turn that made the call, or for a subtask
  the trace of the turn that spawned it (``ai_spans.trace_id_for``);
  ``ai_spans`` sends the trace's tool spans and root from the backend.
- ``trace.generation_name`` becomes ``$ai_span_name``; every other ``trace``
  key arrives as ``metadata_<key>`` (``metadata_environment``,
  ``metadata_impersonated``, and ``metadata_revision``: the deployed build's
  git sha from ``config.sentry_release``, left out when that is blank).
  Broadcast sends these events itself, not through our PostHog client, so
  they never get a property the client adds to every event it captures.

Only calls that go through OpenRouter are covered. Direct Anthropic, Claude
Code subscription and user-connected (Codex) routes never pass OpenRouter, so
Broadcast sees none of their generations.
"""

from contextvars import ContextVar
from typing import Any, Literal

from autogpt_libs.auth.dependencies import IMPERSONATION_HEADER_NAME
from fastapi import Request

from backend.util import posthog_client
from backend.util.settings import Settings

settings = Settings()

# OpenRouter's limit on ``user``. ``session_id`` allows 256; chat session ids
# are UUIDs, so the shorter cap never cuts one.
_MAX_ID_CHARS = 128

GenerationName = Literal["copilot-title", "copilot-sdk", "copilot-baseline"]

# The running turn's trace, sent as the trace id of every call it makes. The
# executor sets it for every turn, so a value never carries over; a call made
# outside a turn sends none.
_turn_trace_id: ContextVar[str | None] = ContextVar("_turn_trace_id", default=None)


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
    revision = settings.config.sentry_release.strip()
    if revision:
        body["trace"]["revision"] = revision
    trace_id = turn_trace_id()
    if trace_id:
        body["trace"]["trace_id"] = trace_id[:_MAX_ID_CHARS]
    if user_id:
        body["user"] = user_id[:_MAX_ID_CHARS]
    if session_id:
        body["session_id"] = session_id[:_MAX_ID_CHARS]
    return body


def set_turn_impersonated(impersonated: bool) -> None:
    """Mark the running turn as started by an impersonating admin, or not.

    The mark is the PostHog client's, so every event the turn sends carries
    it too. The executor sets it for every turn, so a value never carries
    over."""
    posthog_client.set_impersonated(impersonated)


def is_turn_impersonated() -> bool:
    return posthog_client.is_impersonated()


def set_turn_trace_id(trace_id: str | None) -> None:
    """Make *trace_id* the trace of every model call the running turn makes."""
    _turn_trace_id.set(trace_id or None)


def turn_trace_id() -> str | None:
    return _turn_trace_id.get()


def is_impersonated_request(request: Request) -> bool:
    """Whether an admin made *request* as another user.

    The auth dependencies have already refused the header from anyone but an
    admin by the time a route runs, so its presence is enough.
    """
    return bool(request.headers.get(IMPERSONATION_HEADER_NAME, "").strip())


def _environment() -> str:
    return settings.config.app_env.value
