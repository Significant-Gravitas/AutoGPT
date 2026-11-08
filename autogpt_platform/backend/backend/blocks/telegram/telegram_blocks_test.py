"""Telegram block tests that mock only HTTP, so the real API helpers run."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.telegram._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.telegram.blocks import (
    DeleteTelegramMessageBlock,
    EditTelegramMessageBlock,
)
from backend.util.request import Requests


def _http_200(payload: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.status = 200
    response.json.return_value = payload
    return AsyncMock(return_value=response)


async def _run(block, inputs: dict[str, Any], payload: dict[str, Any]):
    request = _http_200(payload)
    with patch.object(Requests, "request", request):
        outputs = [
            o
            async for o in block.run(
                block.input_schema(**inputs, credentials=TEST_CREDENTIALS_INPUT),
                credentials=TEST_CREDENTIALS,
            )
        ]
    return outputs, request


async def test_delete_message_true_result_succeeds():
    outputs, request = await _run(
        DeleteTelegramMessageBlock(),
        {"chat_id": 12345678, "message_id": 42},
        {"ok": True, "result": True},
    )
    assert outputs == [("status", "Message deleted")]
    assert request.await_args.args[1].endswith("/deleteMessage")
    assert request.await_args.kwargs["json"] == {"chat_id": 12345678, "message_id": 42}


async def test_delete_message_error_reply_fails():
    with pytest.raises(ValueError, match="message to delete not found"):
        await _run(
            DeleteTelegramMessageBlock(),
            {"chat_id": 12345678, "message_id": 42},
            {
                "ok": False,
                "error_code": 400,
                "description": "message to delete not found",
            },
        )


async def test_delete_message_unconfirmed_result_fails():
    with pytest.raises(ValueError, match="did not confirm"):
        await _run(
            DeleteTelegramMessageBlock(),
            {"chat_id": 12345678, "message_id": 42},
            {"ok": True, "result": False},
        )


async def test_edit_message_parses_message_result():
    message = {"message_id": 42, "chat": {"id": 12345678}, "date": 0, "text": "new"}
    outputs, _ = await _run(
        EditTelegramMessageBlock(),
        {"chat_id": 12345678, "message_id": 42, "text": "new"},
        {"ok": True, "result": message},
    )
    assert ("message_id", 42) in outputs
