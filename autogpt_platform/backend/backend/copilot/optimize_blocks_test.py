"""Unit tests for optimize_blocks._call_llm_with_retry."""

from unittest.mock import MagicMock, patch

import openai
import pytest

from backend.copilot.optimize_blocks import _MAX_RETRIES, _call_llm_with_retry


def _make_client_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestCallLlmWithRetry:
    """Tests for _call_llm_with_retry exponential-backoff behaviour."""

    def test_success_on_first_attempt(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_client_response(
            "Short desc."
        )

        result = _call_llm_with_retry(client, "gpt-4o-mini", "MyBlock", "A block.")

        assert result == "Short desc."
        assert client.chat.completions.create.call_count == 1

    def test_retries_on_rate_limit_then_succeeds(self):
        client = MagicMock()
        # First call raises RateLimitError, second succeeds
        rate_limit_error = openai.RateLimitError(
            "rate limited", response=MagicMock(), body={}
        )
        client.chat.completions.create.side_effect = [
            rate_limit_error,
            _make_client_response("Retried desc."),
        ]

        with patch("backend.copilot.optimize_blocks.time.sleep") as mock_sleep:
            result = _call_llm_with_retry(client, "gpt-4o-mini", "MyBlock", "A block.")

        assert result == "Retried desc."
        assert client.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once()  # slept once between attempts

    def test_raises_after_max_retries(self):
        client = MagicMock()
        rate_limit_error = openai.RateLimitError(
            "rate limited", response=MagicMock(), body={}
        )
        client.chat.completions.create.side_effect = rate_limit_error

        with patch("backend.copilot.optimize_blocks.time.sleep"):
            with pytest.raises(openai.RateLimitError):
                _call_llm_with_retry(client, "gpt-4o-mini", "MyBlock", "A block.")

        assert client.chat.completions.create.call_count == _MAX_RETRIES

    def test_backoff_doubles_between_retries(self):
        client = MagicMock()
        rate_limit_error = openai.RateLimitError(
            "rate limited", response=MagicMock(), body={}
        )
        # Fail twice, succeed on third (last allowed) attempt
        client.chat.completions.create.side_effect = [
            rate_limit_error,
            rate_limit_error,
            _make_client_response("ok"),
        ]

        sleep_calls: list[float] = []
        with patch(
            "backend.copilot.optimize_blocks.time.sleep",
            side_effect=lambda t: sleep_calls.append(t),
        ):
            _call_llm_with_retry(client, "gpt-4o-mini", "MyBlock", "desc")

        # Backoff should double: [initial, initial*2]
        assert len(sleep_calls) == 2
        assert sleep_calls[1] == sleep_calls[0] * 2
