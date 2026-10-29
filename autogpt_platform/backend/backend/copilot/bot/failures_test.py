"""The bot's failure reply: a plain cause, a reference, and nothing internal."""

import logging
import re

import pytest
import sentry_sdk
from autogpt_libs.auth.dependencies import ACCOUNT_BANNED_DETAIL
from sentry_sdk.integrations.logging import LoggingIntegration

from backend.copilot.executor.processor import RATE_LIMIT_UNAVAILABLE_CODE
from backend.copilot.provider_failure import ProviderFailureKind
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.util.exceptions import NotFoundError
from backend.util.llm.provider_billing import PROVIDER_UNAVAILABLE_CODE

from . import failures
from .bot_backend import BotStreamError


@pytest.mark.parametrize(
    "exc, category",
    [
        (
            BotStreamError(
                "backend_stream_error", "x", code="circuit_breaker_empty_tool_calls"
            ),
            failures.TOO_LARGE,
        ),
        (
            BotStreamError("backend_stream_error", "x", code="transient_api_error"),
            failures.PROVIDER_BUSY,
        ),
        (
            BotStreamError("backend_stream_error", "x", code="all_attempts_exhausted"),
            failures.CONVERSATION_TOO_LONG,
        ),
        (
            BotStreamError("backend_stream_error", "x", code="auth_expired"),
            failures.SERVICE_REJECTED,
        ),
        (
            BotStreamError("backend_stream_error", "x", code="usage_limit"),
            failures.PROVIDER_LIMIT,
        ),
        (
            BotStreamError("backend_stream_error", "x", code="policy_denied"),
            failures.REQUEST_DECLINED,
        ),
        # Our own provider account is out of credits: not a bug in the turn.
        pytest.param(
            BotStreamError("backend_stream_error", "x", code=PROVIDER_UNAVAILABLE_CODE),
            failures.MODEL_UNAVAILABLE,
        ),
        # The executor refuses a turn whose allowance it can't read (#15396).
        (
            BotStreamError(
                "backend_stream_error", "x", code=RATE_LIMIT_UNAVAILABLE_CODE
            ),
            failures.LIMITS_UNCHECKED,
        ),
        # Banned between the bot's own check and the executor's (#15701).
        (
            BotStreamError(
                "backend_stream_error", "x", code=ACCOUNT_BANNED_DETAIL["code"]
            ),
            failures.ACCOUNT_BANNED,
        ),
        (BotStreamError("stream_timeout", "response timed out"), failures.TOO_LONG),
        (BotStreamError("subscribe_failed", "no stream"), failures.START_FAILED),
        (
            BotStreamError("backend_stream_error", CANCELLED_MESSAGE),
            failures.STOPPED,
        ),
        (
            BotStreamError("backend_stream_error", "boom", code="sdk_error"),
            failures.INTERNAL,
        ),
        (BotStreamError("backend_stream_error", "boom"), failures.INTERNAL),
        (NotFoundError("no such session"), failures.START_FAILED),
        (RuntimeError("anything else"), failures.INTERNAL),
    ],
)
def test_categorize(exc: BaseException, category: failures.FailureCategory) -> None:
    assert failures.categorize(exc) == category


@pytest.mark.parametrize("kind", list(ProviderFailureKind))
def test_every_provider_failure_kind_names_its_cause(
    kind: ProviderFailureKind,
) -> None:
    exc = BotStreamError("backend_stream_error", "x", code=kind.value)
    assert failures.categorize(exc) != failures.INTERNAL


def test_reply_names_the_cause_and_reference() -> None:
    reply = failures.failure_reply(failures.PROVIDER_BUSY, "3f9a2c1d")

    assert reply == (
        "Otto couldn't finish that: the model provider is busy. "
        "Try again in a moment. (ref 3f9a2c1d)"
    )


def test_a_reply_with_nothing_to_try_ends_at_the_cause() -> None:
    reply = failures.failure_reply(failures.ACCOUNT_BANNED, "3f9a2c1d")

    assert reply == (
        "Otto couldn't finish that: this account has been banned. (ref 3f9a2c1d)"
    )


def test_references_are_short_and_distinct() -> None:
    references = {failures.new_reference() for _ in range(50)}

    assert len(references) == 50
    assert all(re.fullmatch(r"[0-9a-f]{8}", ref) for ref in references)


def test_report_logs_the_reference_and_the_raw_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exc = BotStreamError("backend_stream_error", "provider said 529", code="x")

    with caplog.at_level(logging.ERROR, logger=failures.__name__):
        failures.report_failure(exc, failures.INTERNAL, "3f9a2c1d", platform="discord")

    [record] = caplog.records
    assert "ref=3f9a2c1d" in record.getMessage()
    assert "provider said 529" in record.getMessage()
    assert record.__dict__["json_fields"]["bot_error_ref"] == "3f9a2c1d"
    assert record.exc_info is None


def test_report_tags_the_sentry_event_with_the_reference() -> None:
    captured: list[dict] = []

    class Recorder(sentry_sdk.transport.Transport):
        def capture_envelope(self, envelope):
            for item in envelope.items:
                if item.type == "event":
                    captured.append(item.payload.json)

    # Restore the original client: init(dsn=None) would leave one bound and
    # active for the rest of the session.
    original_client = sentry_sdk.get_client()
    sentry_sdk.init(
        dsn="https://public@example.invalid/1",
        transport=Recorder(),
        integrations=[LoggingIntegration()],
        default_integrations=False,
    )
    try:
        failures.report_failure(
            BotStreamError("backend_stream_error", "x", code="usage_limit"),
            failures.PROVIDER_LIMIT,
            "3f9a2c1d",
            platform="telegram",
        )
    finally:
        sentry_sdk.get_global_scope().set_client(original_client)

    [event] = captured
    assert event["tags"] == {
        "bot_error_ref": "3f9a2c1d",
        "bot_error_category": "provider_limit",
        "bot_platform": "telegram",
    }
