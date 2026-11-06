from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.telegram._api import (
    TelegramAPIException,
    TelegramMessageResult,
    call_telegram_api,
)


def _credentials() -> MagicMock:
    credentials = MagicMock()
    credentials.api_key.get_secret_value.return_value = "token"
    return credentials


async def test_boolean_result_is_treated_as_success():
    """`deleteMessage` answers `{"ok": true, "result": true}`."""
    response = MagicMock()
    response.json.return_value = {"ok": True, "result": True}

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "backend.blocks.telegram._api.Requests",
            lambda: MagicMock(post=AsyncMock(return_value=response)),
        )
        result = await call_telegram_api(
            _credentials(), "deleteMessage", {"chat_id": 1, "message_id": 2}
        )

    assert isinstance(result, TelegramMessageResult)


async def test_message_result_is_parsed_as_before():
    response = MagicMock()
    response.json.return_value = {
        "ok": True,
        "result": {"message_id": 7, "chat": {"id": 1}, "date": 1, "text": "hi"},
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "backend.blocks.telegram._api.Requests",
            lambda: MagicMock(post=AsyncMock(return_value=response)),
        )
        result = await call_telegram_api(_credentials(), "sendMessage", {})

    assert result.message_id == 7
    assert result.text == "hi"


async def test_error_response_still_raises():
    response = MagicMock()
    response.json.return_value = {
        "ok": False,
        "error_code": 400,
        "description": "message to delete not found",
    }

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "backend.blocks.telegram._api.Requests",
            lambda: MagicMock(post=AsyncMock(return_value=response)),
        )
        with pytest.raises(TelegramAPIException, match="message to delete not found"):
            await call_telegram_api(_credentials(), "deleteMessage", {})
