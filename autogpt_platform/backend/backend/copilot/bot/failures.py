"""What a chat user is told when a bot turn fails, and how the team finds it.

A failed turn used to post one fixed sentence ("AutoGPT ran into an error"),
whatever the cause, so neither the user nor the team had anything to go on.
The reply now names the kind of failure in plain words and ends with a short
reference; the same reference is a Sentry tag and sits on the log line, so a
screenshot of the chat leads straight to the event.

The raw exception never reaches chat: provider bodies, file paths and ids stay
in the logs. Only the bounded category and the reference are shown.
"""

import logging
from uuid import uuid4

import sentry_sdk
from pydantic import BaseModel, ConfigDict

from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.util.exceptions import NotFoundError

from .bot_backend import BotStreamError

logger = logging.getLogger(__name__)

AGAIN_SOON = "Try again in a moment."


class FailureCategory(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    reason: str
    advice: str


INTERNAL = FailureCategory(key="internal", reason="AutoGPT hit an internal error", advice=AGAIN_SOON)
PROVIDER_BUSY = FailureCategory(key="provider_busy", reason="the model provider is busy", advice=AGAIN_SOON)
PROVIDER_LIMIT = FailureCategory(key="provider_limit", reason="the model provider's usage limit was reached", advice="Try again later.")
SERVICE_REJECTED = FailureCategory(key="service_rejected", reason="a connected service rejected the request", advice="Check your connected accounts in AutoGPT, then try again.")
REQUEST_DECLINED = FailureCategory(key="request_declined", reason="the model provider declined this request", advice="Try rephrasing it.")
MODEL_UNAVAILABLE = FailureCategory(key="model_unavailable", reason="the model isn't available right now", advice="Try again later.")
NOT_IN_PLAN = FailureCategory(key="not_in_plan", reason="your plan doesn't include this", advice="Check your plan in AutoGPT.")
TOO_LONG = FailureCategory(key="too_long", reason="the request took too long", advice="Try again, or break it into smaller steps.")
STEP_LIMIT = FailureCategory(key="step_limit", reason="the task hit its step limit", advice="Break it into smaller steps and try again.")
EMPTY_REPLY = FailureCategory(key="empty_reply", reason="the model returned an empty reply", advice=AGAIN_SOON)
STOPPED = FailureCategory(key="stopped", reason="the run was stopped before it finished", advice="Send it again to retry.")
START_FAILED = FailureCategory(key="start_failed", reason="AutoGPT couldn't start this conversation", advice=AGAIN_SOON)
LINK_CHECK_FAILED = FailureCategory(key="link_check_failed", reason="AutoGPT couldn't check this account's link", advice=AGAIN_SOON)

# Stream error codes, from the copilot's StreamError.code: the SDK/baseline
# codes plus ProviderFailureKind values. An unlisted code is INTERNAL.
_BY_CODE: dict[str, FailureCategory] = {
    "transient_api_error": PROVIDER_BUSY,
    "all_attempts_exhausted": PROVIDER_BUSY,
    "transient": PROVIDER_BUSY,
    "usage_limit": PROVIDER_LIMIT,
    "auth_expired": SERVICE_REJECTED,
    "invalid_credential": SERVICE_REJECTED,
    "policy_denied": REQUEST_DECLINED,
    "model_unavailable": MODEL_UNAVAILABLE,
    "entitlement_required": NOT_IN_PLAN,
    "idle_timeout": TOO_LONG,
    "max_turns_exhausted": STEP_LIMIT,
    "max_budget_exhausted": STEP_LIMIT,
    "circuit_breaker_empty_tool_calls": STEP_LIMIT,
    "empty_completion": EMPTY_REPLY,
}

# BotStreamError.error_kind values raised by the bot itself.
_BY_KIND: dict[str, FailureCategory] = {
    "stream_timeout": TOO_LONG,
    "subscribe_failed": START_FAILED,
}


def categorize(exc: BaseException) -> FailureCategory:
    if isinstance(exc, NotFoundError):
        return START_FAILED
    if not isinstance(exc, BotStreamError):
        return INTERNAL
    if exc.error_kind in _BY_KIND:
        return _BY_KIND[exc.error_kind]
    if exc.code in _BY_CODE:
        return _BY_CODE[exc.code]
    if str(exc) == CANCELLED_MESSAGE:
        return STOPPED
    return INTERNAL


def new_reference() -> str:
    return uuid4().hex[:8]


def failure_reply(category: FailureCategory, reference: str) -> str:
    return (
        f"AutoGPT couldn't finish that: {category.reason}. {category.advice} "
        f"(ref {reference})"
    )


def report_failure(
    exc: BaseException,
    category: FailureCategory,
    reference: str,
    *,
    platform: str,
) -> None:
    """Log the failure once, tagged so the chat reference finds it in Sentry."""
    tags = {
        "bot_error_ref": reference,
        "bot_error_category": category.key,
        "bot_platform": platform,
    }
    # Expected stream failures carry no useful traceback, only the backend's
    # own text; anything else is a bug in the bot and needs its stack.
    exc_info = None if isinstance(exc, BotStreamError) else exc
    with sentry_sdk.new_scope() as scope:
        for tag, value in tags.items():
            scope.set_tag(tag, value)
        # %-args, not an f-string: Sentry groups on the template, and a
        # formatted reference would open a new issue for every failure.
        logger.error(
            "Bot reply failed: category=%s ref=%s platform=%s error=%s",
            category.key,
            reference,
            platform,
            exc,
            exc_info=exc_info,
            extra={"json_fields": {**tags, "error_type": type(exc).__name__}},
        )
