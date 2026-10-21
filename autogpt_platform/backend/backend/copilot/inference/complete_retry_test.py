"""The forced-tool self-heal in ``structured_complete``: a model missing
from the provider's forced-tool list answers the forced ``tool_choice`` with
Anthropic's 400, and the call goes once more under ``auto``. Only that 400
retries, and only once.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.dream.structured_output import output_tool_name
from backend.util.llm.tool_use import auto_tool_choice, force_tool_choice

from ._test_data import (
    ANTHROPIC,
    CALL_PROVIDER,
    FORCED_TOOL_REJECTION,
    ROUTING,
    SampleOutput,
    bad_request,
    server_error,
    sync_ctx,
    tool_response,
)
from .complete import structured_complete
from .context import InferenceError


class TestForcedToolRetry:
    """One retry under ``auto``, on Anthropic's 400 naming ``tool_choice``
    and nothing else."""

    @pytest.mark.asyncio
    async def test_forced_tool_rejection_retries_once_with_auto(self, caplog):
        """A model missing from the forced-tool list answers the forced
        choice with a 400. The call goes once more under ``auto``, the
        prompt asking for the tool, and the swap is logged as a warning."""
        healed = tool_response('{"facts": [{"content": "healed", "confidence": 0.7}]}')
        call_provider = AsyncMock(
            side_effect=[bad_request(FORCED_TOOL_REJECTION), healed]
        )
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ), caplog.at_level(
            logging.WARNING, logger="backend.copilot.inference.complete"
        ):
            result = await structured_complete(
                sync_ctx("anthropic", "claude-sonnet-5"),
                [{"role": "user", "content": "hi"}],
                SampleOutput,
            )

        assert result.value.facts[0].content == "healed"
        tool_name = output_tool_name(SampleOutput)
        first, second = call_provider.call_args_list
        assert first.kwargs["tool_choice"] == force_tool_choice(tool_name)
        assert first.kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert second.kwargs["tool_choice"] == auto_tool_choice()
        assert second.kwargs["model"] == "claude-sonnet-5"
        assert tool_name in second.kwargs["messages"][-1]["content"]
        assert "retrying once with tool_choice=auto" in caplog.text

    @pytest.mark.asyncio
    async def test_forced_tool_rejection_is_retried_only_once(self):
        call_provider = AsyncMock(
            side_effect=[
                bad_request(FORCED_TOOL_REJECTION),
                bad_request(FORCED_TOOL_REJECTION),
            ]
        )
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="tool_choice") as exc_info:
                await structured_complete(
                    sync_ctx("anthropic", "claude-sonnet-5"),
                    [{"role": "user", "content": "hi"}],
                    SampleOutput,
                )
        assert call_provider.await_count == 2
        # Neither call came back with a response, so nothing was billed.
        assert exc_info.value.usage is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [server_error(FORCED_TOOL_REJECTION), RuntimeError(FORCED_TOOL_REJECTION)],
        ids=["500", "runtime-error"],
    )
    async def test_only_a_400_rejection_is_retried(self, error: Exception):
        """The documented text is not enough: a 5xx or an exception of our
        own quoting it keeps the forced tool and fails once."""
        call_provider = AsyncMock(side_effect=error)
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="tool_choice"):
                await structured_complete(
                    sync_ctx("anthropic", "claude-sonnet-5"),
                    [{"role": "user", "content": "hi"}],
                    SampleOutput,
                )
        assert call_provider.await_count == 1

    @pytest.mark.asyncio
    async def test_other_bad_requests_are_not_retried(self):
        call_provider = AsyncMock(
            side_effect=bad_request("messages: at least one message is required")
        )
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="at least one message"):
                await structured_complete(
                    sync_ctx("anthropic", "claude-sonnet-5"),
                    [{"role": "user", "content": "hi"}],
                    SampleOutput,
                )
        assert call_provider.await_count == 1
