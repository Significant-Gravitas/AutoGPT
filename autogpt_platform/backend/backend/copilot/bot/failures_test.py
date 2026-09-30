"""The bot's failure reply: a plain cause, a reference, and nothing internal."""

import logging
import re

import pytest

from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.util.exceptions import NotFoundError

from . import failures
from .bot_backend import BotStreamError


@pytest.mark.parametrize(
    "exc, category",
    [
        # 2026-09-15 12:29 UTC in Discord: the turn hit its LLM-call cap and
        # the user was told only "AutoGPT ran into an error".
        (
            BotStreamError("backend_stream_error", "cap", code="max_turns_exhausted"),
            failures.STEP_LIMIT,
        ),
        (
            BotStreamError("backend_stream_error", "x", code="transient_api_error"),
            failures.PROVIDER_BUSY,
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


def test_reply_names_the_cause_and_reference() -> None:
    reply = failures.failure_reply(failures.PROVIDER_BUSY, "3f9a2c1d")

    assert reply == (
        "AutoGPT couldn't finish that: the model provider is busy. "
        "Try again in a moment. (ref 3f9a2c1d)"
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
