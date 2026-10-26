"""Tests for the Telegram Bot API helpers against mocked HTTP replies."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.telegram._api import (
    TelegramAPIException,
    TelegramFileResult,
    _as_message,
    call_telegram_api,
    call_telegram_api_with_file,
    get_file_info,
)
from backend.blocks.telegram._auth import TEST_CREDENTIALS
from backend.util.request import Requests

MESSAGE = {"message_id": 42, "chat": {"id": 1}, "date": 0, "text": "hi"}
ERROR = {"ok": False, "error_code": 400, "description": "message to delete not found"}


def _http_200(payload: dict[str, Any]) -> AsyncMock:
    response = MagicMock()
    response.status = 200
    response.json.return_value = payload
    return AsyncMock(return_value=response)


async def _call(helper: str, payload: dict[str, Any]) -> Any:
    with patch.object(Requests, "request", _http_200(payload)):
        if helper == "json":
            return await call_telegram_api(TEST_CREDENTIALS, "deleteMessage", {})
        return await call_telegram_api_with_file(
            TEST_CREDENTIALS,
            "sendPhoto",
            file_field="photo",
            file_data=b"x",
            filename="x.jpg",
            content_type="image/jpeg",
        )


@pytest.mark.parametrize("helper", ["json", "file"])
async def test_true_result_is_returned(helper: str):
    assert await _call(helper, {"ok": True, "result": True}) is True


@pytest.mark.parametrize("helper", ["json", "file"])
async def test_message_result_is_returned(helper: str):
    result = await _call(helper, {"ok": True, "result": MESSAGE})
    assert _as_message(result).message_id == 42


@pytest.mark.parametrize("helper", ["json", "file"])
async def test_error_reply_raises(helper: str):
    with pytest.raises(TelegramAPIException) as exc:
        await _call(helper, ERROR)
    assert exc.value.error_code == 400
    assert "message to delete not found" in str(exc.value)


@pytest.mark.parametrize("result", [True, None, [1], "x"])
def test_as_message_rejects_non_dict(result: Any):
    with pytest.raises(TelegramAPIException):
        _as_message(result)


async def test_get_file_info_parses_file():
    file = {
        "file_id": "abc",
        "file_unique_id": "u",
        "file_size": 3,
        "file_path": "photos/a.jpg",
    }
    with patch.object(Requests, "request", _http_200({"ok": True, "result": file})):
        info = await get_file_info(TEST_CREDENTIALS, "abc")
    assert info == TelegramFileResult(**file)
