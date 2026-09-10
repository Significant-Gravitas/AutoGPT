"""Tests for Telegram bot-command parsing + handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.command_core import CommandReply

from . import commands

_CMD = "backend.copilot.bot.adapters.telegram.commands"


def _message(text: str, chat_type: str = "supergroup") -> dict:
    return {
        "message_id": 5,
        "text": text,
        "entities": [
            {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
        ],
        "chat": {"id": -100123456, "type": chat_type, "title": "Builders"},
        "from": {"id": 42, "username": "bently"},
    }


def test_parse_command_bare_and_addressed_forms():
    assert commands.parse_command(_message("/setup"), "OurBot") == "setup"
    assert commands.parse_command(_message("/setup@OurBot"), "OurBot") == "setup"
    assert commands.parse_command(_message("/SETUP@ourbot"), "OurBot") == "setup"


def test_parse_command_ignores_other_bots_and_non_commands():
    assert commands.parse_command(_message("/setup@OtherBot"), "OurBot") is None
    plain = _message("hello /setup")
    plain["entities"] = []
    assert commands.parse_command(plain, "OurBot") is None


@pytest.mark.asyncio
async def test_setup_in_group_mints_link_and_sends_button():
    client = MagicMock()
    client.call = AsyncMock()
    reply = CommandReply(
        text="**Set up AutoGPT for Builders**",
        button_label="Link Group",
        button_url="https://x/l",
    )
    with patch(f"{_CMD}.setup_reply", new=AsyncMock(return_value=reply)) as sr:
        await commands.handle(MagicMock(), client, _message("/setup"), "setup")
    kwargs = sr.call_args.kwargs
    assert kwargs["platform"] == "telegram"
    assert kwargs["server_noun"] == "group"
    assert kwargs["platform_server_id"] == "-100123456"
    assert kwargs["server_name"] == "Builders"
    sent = client.call.call_args.kwargs
    assert sent["reply_markup"]["inline_keyboard"][0][0]["url"] == "https://x/l"
    assert "<b>Set up AutoGPT for Builders</b>" in sent["text"]


@pytest.mark.asyncio
async def test_setup_in_private_chat_redirects_to_dm_linking():
    client = MagicMock()
    client.call = AsyncMock()
    with patch(f"{_CMD}.setup_reply", new=AsyncMock()) as sr:
        await commands.handle(
            MagicMock(), client, _message("/setup", chat_type="private"), "setup"
        )
    sr.assert_not_called()
    assert "send me a message" in client.call.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_help_sends_usage_text():
    client = MagicMock()
    client.call = AsyncMock()
    await commands.handle(MagicMock(), client, _message("/help"), "help")
    text = client.call.call_args.kwargs["text"]
    # Rendered as real HTML — an escaped-entity regression would show the
    # user literal &lt;b&gt; junk.
    assert "<b>AutoGPT for Telegram</b>" in text
    assert "&lt;" not in text


@pytest.mark.asyncio
async def test_new_clears_the_target_session():
    client = MagicMock()
    client.call = AsyncMock()
    msg = _message("/new")
    msg["message_thread_id"] = 7
    with patch(f"{_CMD}.sessions.clear_session", new=AsyncMock()) as clear:
        await commands.handle(MagicMock(), client, msg, "new")
    clear.assert_awaited_once_with("telegram", "-100123456|7")
    assert "fresh conversation" in client.call.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_commands_track_command_used_with_group_scope():
    client = MagicMock()
    client.call = AsyncMock()
    api = MagicMock()
    await commands.handle(api, client, _message("/help"), "help")
    api.track_event.assert_called_once_with(
        platform="telegram",
        event_type="command_used",
        server_id="-100123456",
        command_name="help",
    )


@pytest.mark.asyncio
async def test_dm_commands_track_without_server_id():
    client = MagicMock()
    client.call = AsyncMock()
    api = MagicMock()
    await commands.handle(api, client, _message("/help", chat_type="private"), "help")
    assert api.track_event.call_args.kwargs["server_id"] is None
