"""Tests for Slack blocks – API layer, block execution, and error handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.slack._api import (
    SlackAPIException,
    SlackMessageResult,
    call_slack_api,
    post_message,
)
from backend.blocks.slack._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.slack.blocks import SendSlackMessageBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ok": True,
        "ts": "1234567890.123456",
        "channel": "C1234567890",
        "message": {"text": "hello"},
    }
    if extra:
        base.update(extra)
    return base


def _error_response(error: str = "channel_not_found") -> dict[str, Any]:
    return {"ok": False, "error": error}


def _mock_http_response(json_data: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    return resp


async def _collect_outputs(
    block: SendSlackMessageBlock, input_data, credentials
) -> dict[str, object]:
    outputs: dict[str, object] = {}
    async for name, value in block.run(input_data, credentials=credentials):
        outputs[name] = value
    return outputs


# ---------------------------------------------------------------------------
# SlackAPIException
# ---------------------------------------------------------------------------


class TestSlackAPIException:
    def test_inherits_value_error(self):
        exc = SlackAPIException("not_authed")
        assert isinstance(exc, ValueError)

    def test_stores_error_code(self):
        exc = SlackAPIException("invalid_auth")
        assert exc.error == "invalid_auth"

    def test_message_format(self):
        exc = SlackAPIException("channel_not_found")
        assert str(exc) == "Slack API error: channel_not_found"


# ---------------------------------------------------------------------------
# SlackMessageResult
# ---------------------------------------------------------------------------


class TestSlackMessageResult:
    def test_basic_fields(self):
        r = SlackMessageResult(ts="123.456", channel="C999")
        assert r.ts == "123.456"
        assert r.channel == "C999"
        assert r.message == {}

    def test_extra_fields_allowed(self):
        r = SlackMessageResult(ts="1", channel="C1", extra_field="hi")
        assert r.model_extra is not None
        assert r.model_extra["extra_field"] == "hi"


# ---------------------------------------------------------------------------
# call_slack_api
# ---------------------------------------------------------------------------


class TestCallSlackApi:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_resp = _mock_http_response(_ok_response())
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            result = await call_slack_api(
                TEST_CREDENTIALS, "chat.postMessage", {"channel": "C1", "text": "hi"}
            )
            assert result["ok"] is True
            assert result["ts"] == "1234567890.123456"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "slack.com/api/chat.postMessage" in call_args.args[0]
            assert call_args.kwargs["headers"]["Authorization"] == (
                "Bearer mock-slack-bot-token"
            )

    @pytest.mark.asyncio
    async def test_api_error_raises(self):
        mock_resp = _mock_http_response(_error_response("channel_not_found"))
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            with pytest.raises(SlackAPIException, match="channel_not_found"):
                await call_slack_api(
                    TEST_CREDENTIALS, "chat.postMessage", {"channel": "BAD"}
                )

    @pytest.mark.asyncio
    async def test_unknown_error(self):
        mock_resp = _mock_http_response({"ok": False})
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            with pytest.raises(SlackAPIException, match="unknown_error"):
                await call_slack_api(TEST_CREDENTIALS, "chat.postMessage")

    @pytest.mark.asyncio
    async def test_empty_data_sends_empty_dict(self):
        mock_resp = _mock_http_response(_ok_response())
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await call_slack_api(TEST_CREDENTIALS, "auth.test")
            call_args = mock_post.call_args
            assert call_args.kwargs["json"] == {}


# ---------------------------------------------------------------------------
# post_message
# ---------------------------------------------------------------------------


class TestPostMessage:
    @pytest.mark.asyncio
    async def test_basic_message(self):
        mock_resp = _mock_http_response(_ok_response())
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await post_message(
                credentials=TEST_CREDENTIALS,
                channel="C1234567890",
                text="Hello!",
            )
            assert isinstance(result, SlackMessageResult)
            assert result.ts == "1234567890.123456"
            assert result.channel == "C1234567890"

    @pytest.mark.asyncio
    async def test_optional_params_included(self):
        mock_resp = _mock_http_response(_ok_response())
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await post_message(
                credentials=TEST_CREDENTIALS,
                channel="C1",
                text="threaded",
                thread_ts="111.222",
                username="TestBot",
                icon_emoji=":robot_face:",
                unfurl_links=False,
                mrkdwn=False,
            )
            payload = mock_post.call_args.kwargs["json"]
            assert payload["thread_ts"] == "111.222"
            assert payload["username"] == "TestBot"
            assert payload["icon_emoji"] == ":robot_face:"
            assert payload["unfurl_links"] is False
            assert payload["mrkdwn"] is False

    @pytest.mark.asyncio
    async def test_optional_params_omitted_when_none(self):
        mock_resp = _mock_http_response(_ok_response())
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_post:
            await post_message(
                credentials=TEST_CREDENTIALS,
                channel="C1",
                text="basic",
            )
            payload = mock_post.call_args.kwargs["json"]
            assert "thread_ts" not in payload
            assert "username" not in payload
            assert "icon_emoji" not in payload

    @pytest.mark.asyncio
    async def test_error_propagates(self):
        mock_resp = _mock_http_response(_error_response("not_in_channel"))
        with patch(
            "backend.blocks.slack._api.Requests.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            with pytest.raises(SlackAPIException, match="not_in_channel"):
                await post_message(
                    credentials=TEST_CREDENTIALS,
                    channel="C1",
                    text="oops",
                )


# ---------------------------------------------------------------------------
# SendSlackMessageBlock
# ---------------------------------------------------------------------------


class TestSendSlackMessageBlock:
    def setup_method(self):
        self.block = SendSlackMessageBlock()

    def test_block_id_is_valid_uuid(self):
        import uuid

        uuid.UUID(self.block.id, version=4)

    def test_block_category(self):
        from backend.blocks._base import BlockCategory

        assert BlockCategory.SOCIAL in self.block.categories

    def test_input_schema_has_required_fields(self):
        fields = SendSlackMessageBlock.Input.model_fields
        assert "credentials" in fields
        assert "channel" in fields
        assert "text" in fields

    def test_output_schema_has_required_fields(self):
        fields = SendSlackMessageBlock.Output.model_fields
        assert "ts" in fields
        assert "channel" in fields

    @pytest.mark.asyncio
    async def test_run_yields_ts_and_channel(self):
        input_data = SendSlackMessageBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            channel="C1234567890",
            text="Hello from test!",
        )
        mock_result = SlackMessageResult(ts="9999.8888", channel="C1234567890")
        with patch.object(
            self.block,
            "_post_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            outputs = await _collect_outputs(
                self.block, input_data, credentials=TEST_CREDENTIALS
            )
            assert outputs["ts"] == "9999.8888"
            assert outputs["channel"] == "C1234567890"

    @pytest.mark.asyncio
    async def test_run_passes_all_optional_params(self):
        input_data = SendSlackMessageBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            channel="C1",
            text="threaded msg",
            thread_ts="111.222",
            username="Bot",
            icon_emoji=":wave:",
            unfurl_links=False,
            mrkdwn=False,
        )
        mock_result = SlackMessageResult(ts="1.1", channel="C1")
        with patch.object(
            self.block,
            "_post_message",
            new_callable=AsyncMock,
            return_value=mock_result,
        ) as mock_pm:
            await _collect_outputs(self.block, input_data, credentials=TEST_CREDENTIALS)
            mock_pm.assert_called_once_with(
                credentials=TEST_CREDENTIALS,
                channel="C1",
                text="threaded msg",
                thread_ts="111.222",
                username="Bot",
                icon_emoji=":wave:",
                unfurl_links=False,
                mrkdwn=False,
            )

    @pytest.mark.asyncio
    async def test_run_propagates_api_error(self):
        """SlackAPIException (a ValueError subclass) propagates directly from run()."""
        input_data = SendSlackMessageBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            channel="C1",
            text="fail",
        )
        with patch.object(
            self.block,
            "_post_message",
            new_callable=AsyncMock,
            side_effect=SlackAPIException("invalid_auth"),
        ):
            with pytest.raises(SlackAPIException, match="invalid_auth"):
                await _collect_outputs(
                    self.block, input_data, credentials=TEST_CREDENTIALS
                )

    @pytest.mark.asyncio
    async def test_run_propagates_generic_exception(self):
        """Non-ValueError exceptions propagate without wrapping — the executor
        framework handles them at a higher level."""
        input_data = SendSlackMessageBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT,
            channel="C1",
            text="boom",
        )
        with patch.object(
            self.block,
            "_post_message",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection lost"),
        ):
            with pytest.raises(RuntimeError, match="connection lost"):
                await _collect_outputs(
                    self.block, input_data, credentials=TEST_CREDENTIALS
                )

    @pytest.mark.asyncio
    async def test_framework_test_mock_works(self):
        """Verify the test_mock fixture from __init__ works with execute_block_test."""
        from backend.util.test import execute_block_test

        await execute_block_test(self.block)
