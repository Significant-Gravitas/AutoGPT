"""Who an AutoPilot LLM call is for, in the fields OpenRouter Broadcast reads.

OpenRouter Broadcast sends every generation to PostHog LLM analytics as an
``$ai_generation`` event. It maps these request body fields
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
  key arrives as ``metadata_<key>``:

  - ``metadata_environment``: the app environment;
  - ``metadata_impersonated``: whether an admin started the turn as the
    user;
  - ``metadata_turn_origin``: what started the turn that made the call
    (``chat``, ``scheduled``, ``spawned``, ``automation`` or ``system``,
    ``turn_tracing.turn_origin``), left out for a call made outside a turn.
    A reply's own calls match its ``$ai_trace``'s ``turn_origin``; a
    subtask's calls join the reply's trace but carry the subtask's origin,
    as on its ``$ai_span``;
  - ``metadata_revision``: the deployed build's git sha from
    ``config.sentry_release``, left out when that is blank;
  - ``metadata_prompt_name``, ``metadata_prompt_version`` and
    ``metadata_prompt_source``: the system prompt the turn runs on
    (``posthog_prompts.PromptRef``), on the engines' own calls only, not the
    title's. On the SDK engine every call the CLI makes carries it, its
    sub-agents' and compaction's included. Broadcast maps no request field to
    ``$ai_prompt_name``, so those properties go on the turn's ``$ai_trace``
    (``ai_spans.capture_turn``) instead.

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
from pydantic import BaseModel

from backend.util import posthog_client
from backend.util.posthog_prompts import PromptRef
from backend.util.settings import Settings

settings = Settings()

# OpenRouter's limit on ``user``. ``session_id`` allows 256; chat session ids
# are UUIDs, so the shorter cap never cuts one.
_MAX_ID_CHARS = 128

GenerationName = Literal["copilot-title", "copilot-sdk", "copilot-baseline"]
# The calls made with the turn's system prompt.
_SYSTEM_PROMPT_GENERATIONS: frozenset[GenerationName] = frozenset(
    {"copilot-sdk", "copilot-baseline"}
)


class _TurnPrompt(BaseModel):
    """The running turn's record of the system prompt it runs on."""

    ref: PromptRef | None = None


# The running turn's trace, sent as the trace id of every call it makes. The
# executor sets it for every turn, so a value never carries over; a call made
# outside a turn sends none.
_turn_trace_id: ContextVar[str | None] = ContextVar("_turn_trace_id", default=None)
# What started the running turn (``turn_tracing.turn_origin``), so analytics
# can tell a person's message from a schedule's. Set for every turn, like the
# trace id; a call made outside a turn sends none.
_turn_origin: ContextVar[str | None] = ContextVar("_turn_origin", default=None)
# The running turn's prompt record. The record itself is shared, not copied,
# with the tasks the turn starts, so the prompt build fills it for the turn
# from inside the ``asyncio.gather`` the engines run it under.
_turn_prompt: ContextVar[_TurnPrompt | None] = ContextVar("_turn_prompt", default=None)


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
    origin = current_turn_origin()
    if origin:
        body["trace"]["turn_origin"] = origin
    prompt = turn_prompt()
    if prompt is not None and generation_name in _SYSTEM_PROMPT_GENERATIONS:
        body["trace"].update(prompt.broadcast_trace())
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


def set_turn_origin(origin: str | None) -> None:
    """Make *origin* the turn origin every model call the running turn makes
    reports."""
    _turn_origin.set(origin or None)


def current_turn_origin() -> str | None:
    return _turn_origin.get()


def start_turn_prompt() -> None:
    """Give the running turn an empty record of its system prompt."""
    _turn_prompt.set(_TurnPrompt())


def record_turn_prompt(ref: PromptRef) -> None:
    """Note the system prompt the running turn runs on; outside a turn,
    nothing."""
    record = _turn_prompt.get()
    if record is not None:
        record.ref = ref


def turn_prompt() -> PromptRef | None:
    record = _turn_prompt.get()
    return record.ref if record is not None else None


def is_impersonated_request(request: Request) -> bool:
    """Whether an admin made *request* as another user.

    The auth dependencies have already refused the header from anyone but an
    admin by the time a route runs, so its presence is enough.
    """
    return bool(request.headers.get(IMPERSONATION_HEADER_NAME, "").strip())


def _environment() -> str:
    return settings.config.app_env.value
