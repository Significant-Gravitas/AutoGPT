"""Unit tests for optimize_blocks._optimize_descriptions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.copilot.optimize_blocks import _RATE_LIMIT_DELAY, _optimize_descriptions


def _make_client_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestOptimizeDescriptions:
    """Tests for _optimize_descriptions async function."""

    def test_returns_empty_when_no_client(self):
        with patch(
            "backend.copilot.optimize_blocks.get_openai_client", return_value=None
        ):
            result = _run(
                _optimize_descriptions([{"id": "b1", "name": "B", "description": "d"}])
            )
        assert result == {}

    def test_success_single_block(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_client_response("Short desc.")
        )
        blocks = [{"id": "b1", "name": "MyBlock", "description": "A block."}]

        with (
            patch(
                "backend.copilot.optimize_blocks.get_openai_client", return_value=client
            ),
            patch(
                "backend.copilot.optimize_blocks.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            result = _run(_optimize_descriptions(blocks))

        assert result == {"b1": "Short desc."}
        client.chat.completions.create.assert_called_once()

    def test_skips_block_on_exception(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        blocks = [{"id": "b1", "name": "MyBlock", "description": "A block."}]

        with (
            patch(
                "backend.copilot.optimize_blocks.get_openai_client", return_value=client
            ),
            patch(
                "backend.copilot.optimize_blocks.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            result = _run(_optimize_descriptions(blocks))

        assert result == {}

    def test_sleeps_between_blocks(self):
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_make_client_response("desc")
        )
        blocks = [
            {"id": "b1", "name": "B1", "description": "d1"},
            {"id": "b2", "name": "B2", "description": "d2"},
        ]
        sleep_mock = AsyncMock()

        with (
            patch(
                "backend.copilot.optimize_blocks.get_openai_client", return_value=client
            ),
            patch("backend.copilot.optimize_blocks.asyncio.sleep", sleep_mock),
        ):
            _run(_optimize_descriptions(blocks))

        assert sleep_mock.call_count == 2
        sleep_mock.assert_called_with(_RATE_LIMIT_DELAY)
