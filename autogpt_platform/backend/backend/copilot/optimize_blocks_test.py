"""Unit tests for optimize_blocks._optimize_descriptions.

Tests are written as ``async def`` so ``asyncio_mode="auto"`` (set in
``pyproject.toml``) wires them through pytest-asyncio's session-scoped
runner. Previously these tests used a ``_run`` helper around
``asyncio.get_event_loop().run_until_complete`` — that pattern raises
``RuntimeError: There is no current event loop in thread 'MainThread'``
under Python 3.12 when the thread doesn't have a current loop (the
previous test in the suite can close one and never set a new one). The
auto-mode runner gives each test its own loop deterministically.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from backend.copilot.optimize_blocks import _RATE_LIMIT_DELAY, _optimize_descriptions


def _make_client_response(text: str) -> MagicMock:
    """Build a minimal mock that looks like an OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


class TestOptimizeDescriptions:
    """Tests for _optimize_descriptions async function."""

    async def test_returns_empty_when_no_client(self):
        with patch(
            "backend.copilot.optimize_blocks.get_openai_client", return_value=None
        ):
            result = await _optimize_descriptions(
                [{"id": "b1", "name": "B", "description": "d"}]
            )
        assert result == {}

    async def test_success_single_block(self):
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
            result = await _optimize_descriptions(blocks)

        assert result == {"b1": "Short desc."}
        client.chat.completions.create.assert_called_once()

    async def test_skips_block_on_exception(self):
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
            result = await _optimize_descriptions(blocks)

        assert result == {}

    async def test_sleeps_between_blocks(self):
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
            await _optimize_descriptions(blocks)

        assert sleep_mock.call_count == 2
        sleep_mock.assert_called_with(_RATE_LIMIT_DELAY)
