import logging
from unittest.mock import patch

import anthropic
import httpx
import openai
import pytest

from backend.util.exceptions import InsufficientBalanceError
from backend.util.llm.provider_billing import (
    PROVIDER_OUT_OF_CREDITS_ALERT,
    is_provider_out_of_credits,
    report_provider_out_of_credits,
)


def _openai_error(
    status: int, body: dict | None, message: str
) -> openai.APIStatusError:
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat")
    )
    cls = openai.RateLimitError if status == 429 else openai.APIStatusError
    return cls(message, response=response, body=body)


def _openrouter_upstream_error(
    status: int, upstream_raw: str, provider_name: str
) -> openai.APIStatusError:
    """How the OpenAI SDK raises an error OpenRouter relays from an upstream."""
    body = {
        "message": "Provider returned error",
        "code": status,
        "metadata": {"raw": upstream_raw, "provider_name": provider_name},
    }
    return _openai_error(status, body, f"Error code: {status} - {{'error': {body}}}")


_GEMINI_QUOTA_429 = (
    '{"error": {"code": 429, "message": "You exceeded your current quota, please '
    "check your plan and billing details. For more information on this error, "
    'head to: https://ai.google.dev/gemini-api/docs/rate-limits.", '
    '"status": "RESOURCE_EXHAUSTED"}}'
)


def _wrapped(cause: BaseException) -> RuntimeError:
    error = RuntimeError("LLM call failed")
    error.__cause__ = cause
    return error


def _anthropic_error(status: int, body: dict, message: str) -> anthropic.APIStatusError:
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    return anthropic.APIStatusError(message, response=response, body=body)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            _openai_error(402, {"message": "Insufficient credits", "code": 402}, "x"),
            id="openrouter-402",
        ),
        pytest.param(
            _openai_error(
                429,
                {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                },
                "Error code: 429 - insufficient_quota",
            ),
            id="openai-insufficient-quota",
        ),
        pytest.param(
            _anthropic_error(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Your credit balance is too low to access the "
                        "Anthropic API.",
                    },
                },
                "Your credit balance is too low to access the Anthropic API.",
            ),
            id="anthropic-credit-balance",
        ),
        pytest.param(
            _anthropic_error(
                402,
                {
                    "type": "error",
                    "error": {"type": "billing_error", "message": "Billing issue"},
                },
                "Billing issue",
            ),
            id="anthropic-billing-error-type",
        ),
        pytest.param(
            # Mid-stream, the OpenAI SDK raises a bare APIError with the
            # error object as its body, so there is no status to read.
            openai.APIError(
                "Insufficient credits",
                httpx.Request("POST", "https://openrouter.ai/api/v1/chat"),
                body={"message": "Insufficient credits", "code": 402},
            ),
            id="openrouter-402-mid-stream",
        ),
        pytest.param(
            _wrapped(_openai_error(402, {"message": "x", "code": 402}, "x")),
            id="wrapped-openrouter-402",
        ),
        pytest.param(
            'API Error: 402 {"error":{"message":"This request requires more '
            'credits, or fewer max_tokens."}}',
            id="claude-cli-402-text",
        ),
        pytest.param("billing_error Credit balance is too low", id="sdk-billing"),
    ],
)
def test_provider_billing_refusals_are_recognised(error):
    assert is_provider_out_of_credits(error)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            InsufficientBalanceError(
                "Insufficient balance of $0.10, where this will cost $0.50",
                user_id="u",
                balance=10,
                amount=50,
            ),
            id="platform-insufficient-balance",
        ),
        pytest.param(
            "You have no credits left to run an agent.", id="platform-no-credits"
        ),
        pytest.param(
            "Organization has 0 credits but needs 5", id="platform-org-credits"
        ),
        pytest.param(
            "You've reached your usage limit. Please try again later.",
            id="copilot-usage-limit",
        ),
        pytest.param(
            _openai_error(
                429,
                {"message": "Rate limit reached", "code": "rate_limit_exceeded"},
                "Error code: 429 - rate limit",
            ),
            id="plain-rate-limit",
        ),
        pytest.param(_openai_error(500, None, "Error code: 500"), id="server-error"),
        pytest.param(
            # Gemini's ordinary rate limit uses OpenAI's out-of-quota wording.
            _openrouter_upstream_error(429, _GEMINI_QUOTA_429, "Google AI Studio"),
            id="gemini-rate-limit-via-openrouter",
        ),
        pytest.param(
            _wrapped(
                _openrouter_upstream_error(429, _GEMINI_QUOTA_429, "Google AI Studio")
            ),
            id="wrapped-gemini-rate-limit",
        ),
        pytest.param(
            f"429 RESOURCE_EXHAUSTED. {_GEMINI_QUOTA_429}",
            id="gemini-rate-limit-text",
        ),
        pytest.param(
            _openai_error(
                400,
                {
                    "message": "Invalid schema for function 'report': "
                    "'billing_error' is not of type 'object'",
                    "code": "invalid_request_error",
                },
                "Error code: 400 - Invalid schema for function 'report': "
                "'billing_error' is not of type 'object'",
            ),
            id="openai-400-mentioning-billing-error",
        ),
        pytest.param(
            _anthropic_error(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "tools.0.name: 'billing_error' is reserved",
                    },
                },
                "tools.0.name: 'billing_error' is reserved",
            ),
            id="anthropic-400-mentioning-billing-error",
        ),
        pytest.param(
            "Tool output: at this price you can only afford two seats",
            id="unrelated-can-only-afford",
        ),
        pytest.param("Error code: 4029 - unknown", id="longer-number-than-402"),
        pytest.param(
            # Untyped: its text is whatever the raiser put there, such as a
            # user's message or a fetched page quoting OpenRouter's wording.
            ValueError(
                "Could not parse tool input: 'see openrouter.ai/settings/credits, "
                "this request requires more credits'"
            ),
            id="untyped-exception-quoting-billing-words",
        ),
        pytest.param(
            RuntimeError('API Error: 402 {"error":{"message":"x","code":402}}'),
            id="untyped-exception-quoting-a-402",
        ),
        pytest.param(None, id="none"),
    ],
)
def test_other_failures_are_not_mistaken_for_billing(error):
    assert not is_provider_out_of_credits(error)


def test_report_logs_error_under_its_own_fingerprint(caplog):
    with (
        patch("backend.util.llm.provider_billing.sentry_sdk.new_scope") as new_scope,
        caplog.at_level(logging.ERROR),
    ):
        report_provider_out_of_credits(
            provider="open_router",
            model="openai/gpt-oss-120b",
            surface="block",
            error="Error code: 402",
            user_id="u1",
        )

    scope = new_scope.return_value.__enter__.return_value
    assert scope.fingerprint == ["llm-provider-out-of-credits", "open_router"]
    [record] = caplog.records
    assert record.levelno == logging.ERROR
    # A stable title keeps every user's failure in one Sentry issue.
    assert record.getMessage() == PROVIDER_OUT_OF_CREDITS_ALERT
    assert record.__dict__["json_fields"] == {
        "provider": "open_router",
        "model": "openai/gpt-oss-120b",
        "surface": "block",
        "error": "Error code: 402",
        "user_id": "u1",
    }
