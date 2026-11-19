"""Tests for Telegram bot-command parsing + handling."""

import html
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.bot.command_core import AdminCheck, CommandReply

from . import commands
from .api_client import TelegramAPIError

_CMD = "backend.copilot.bot.adapters.telegram.commands"
_LINK = "https://x/link/group-tok?platform=TELEGRAM"


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


def _bot_api(
    *,
    status: str = "administrator",
    member_error: Exception | None = None,
    refuse_login_button: bool = False,
) -> MagicMock:
    """A Bot API client whose getChatMember answers ``status`` for the sender
    and whose sendMessage can refuse login_url buttons, as Telegram does for
    a domain not registered with @BotFather."""
    client = MagicMock()

    async def call(method: str, **params: Any) -> Any:
        if method == "getChatMember":
            if member_error is not None:
                raise member_error
            return {"status": status, "user": {"id": params["user_id"]}}
        markup = params.get("reply_markup") or {}
        buttons = [b for row in markup.get("inline_keyboard", []) for b in row]
        if refuse_login_button and any("login_url" in b for b in buttons):
            raise TelegramAPIError("sendMessage failed: BOT_DOMAIN_INVALID")
        return {"message_id": 99}

    client.call = AsyncMock(side_effect=call)
    return client


def _sent(client: MagicMock) -> list[dict]:
    return [c.kwargs for c in client.call.await_args_list if c.args[0] == "sendMessage"]


def _link_api() -> MagicMock:
    api = MagicMock()
    api.create_link_token = AsyncMock(return_value=MagicMock(link_url=_LINK))
    return api


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
@pytest.mark.parametrize("status", ["creator", "administrator"])
async def test_setup_by_a_group_admin_posts_a_sign_in_button(status):
    api = _link_api()
    client = _bot_api(status=status)
    await commands.handle(api, client, _message("/setup"), "setup", AsyncMock())

    client.call.assert_any_await("getChatMember", chat_id="-100123456", user_id=42)
    kwargs = api.create_link_token.await_args.kwargs
    assert kwargs["platform"] == "telegram"
    assert kwargs["platform_server_id"] == "-100123456"
    assert kwargs["platform_user_id"] == "42"
    assert kwargs["server_name"] == "Builders"
    [sent] = _sent(client)
    assert sent["chat_id"] == "-100123456"
    assert "<b>Set up AutoGPT for Builders</b>" in sent["text"]
    assert "Only the admin who ran /setup can use this button." in sent["text"]
    # A login_url button: Telegram signs who taps it, so the link page can
    # refuse anyone but this admin. Never a plain url button.
    [[button]] = sent["reply_markup"]["inline_keyboard"]
    assert button == {"text": "Link Group", "login_url": {"url": _LINK}}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["member", "restricted", "left", "kicked"])
async def test_setup_by_a_group_member_is_refused_without_a_link(status):
    api = _link_api()
    client = _bot_api(status=status)
    await commands.handle(api, client, _message("/setup"), "setup", AsyncMock())

    api.create_link_token.assert_not_awaited()
    [sent] = _sent(client)
    assert "reply_markup" not in sent
    assert sent["text"].startswith("Only a group admin can link this group")


@pytest.mark.asyncio
async def test_setup_is_refused_when_telegram_cannot_say_who_ran_it():
    api = _link_api()
    client = _bot_api(member_error=TelegramAPIError("getChatMember failed"))
    await commands.handle(api, client, _message("/setup"), "setup", AsyncMock())

    api.create_link_token.assert_not_awaited()
    [sent] = _sent(client)
    assert "reply_markup" not in sent
    assert "couldn't confirm that you're a group admin" in html.unescape(sent["text"])


@pytest.mark.asyncio
async def test_a_group_link_is_never_posted_as_a_plain_url():
    # Without the login_url button nothing could tell the admin's tap from any
    # other member's, so a refused button is not retried as a plain link.
    api = _link_api()
    client = _bot_api(refuse_login_button=True)
    await commands.handle(api, client, _message("/setup"), "setup", AsyncMock())

    refused, notice = _sent(client)
    assert "login_url" in refused["reply_markup"]["inline_keyboard"][0][0]
    assert "reply_markup" not in notice
    assert "can't link this group right now" in html.unescape(notice["text"])
    assert "group-tok" not in notice["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["/start", "/start connect", "/setup"])
async def test_start_in_private_chat_sends_the_dm_link_button(text):
    # The Bots page's "Message bot" link opens t.me/<bot>?start=connect, which
    # Telegram delivers as "/start connect" in the private chat.
    client = MagicMock()
    client.call = AsyncMock()
    send_link = AsyncMock()
    message = _message(text, chat_type="private")
    message["chat"] = {"id": 42, "type": "private"}
    reply = CommandReply(
        text="**Welcome to AutoGPT**",
        button_label="Link Account",
        button_url="https://x/link/dm",
    )
    command = commands.parse_command(message, "OurBot")
    with (
        patch(f"{_CMD}.setup_reply", new=AsyncMock()) as sr,
        patch(f"{_CMD}.dm_link_reply", new=AsyncMock(return_value=reply)) as dr,
    ):
        await commands.handle(MagicMock(), client, message, command, send_link)
    sr.assert_not_called()
    assert dr.call_args.kwargs["platform"] == "telegram"
    assert dr.call_args.kwargs["platform_user_id"] == "42"
    assert dr.call_args.kwargs["platform_username"] == "bently"
    send_link.assert_awaited_once_with(
        "42", "**Welcome to AutoGPT**", "Link Account", "https://x/link/dm"
    )
    client.call.assert_not_called()


@pytest.mark.asyncio
async def test_start_in_private_chat_when_already_linked_sends_plain_text():
    client = MagicMock()
    client.call = AsyncMock()
    send_link = AsyncMock()
    reply = CommandReply(text="Your Telegram DMs are linked to AutoGPT.")
    with patch(f"{_CMD}.dm_link_reply", new=AsyncMock(return_value=reply)):
        await commands.handle(
            MagicMock(),
            client,
            _message("/start", chat_type="private"),
            "start",
            send_link,
        )
    send_link.assert_not_called()
    assert "DMs are linked" in client.call.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_start_when_already_linked_names_the_account_in_the_sent_text():
    # Runs the real dm_link_reply + HTML conversion, so the masked email
    # reaches Telegram literally (its *** must not turn into emphasis).
    client = MagicMock()
    client.call = AsyncMock()
    api = MagicMock()
    api.resolve_user = AsyncMock(
        return_value=MagicMock(linked=True, account_hint="b***@agpt.co")
    )
    message = _message("/start", chat_type="private")
    message["chat"] = {"id": 42, "type": "private"}
    await commands.handle(api, client, message, "start", AsyncMock())
    api.resolve_user.assert_awaited_once_with("telegram", "42", include_account=True)
    sent = client.call.call_args.kwargs
    assert "reply_markup" not in sent
    assert "linked to the AutoGPT account b***@agpt.co." in sent["text"]
    assert "Send /unlink, sign in as b***@agpt.co" in sent["text"]


@pytest.mark.asyncio
async def test_unlink_in_private_chat_points_at_bots_settings():
    client = MagicMock()
    client.call = AsyncMock()
    message = _message("/unlink", chat_type="private")
    message["chat"] = {"id": 42, "type": "private"}
    with patch(
        f"{_CMD}.unlink_reply",
        return_value=CommandReply(
            text="Unlinking requires authentication",
            button_label="Open Settings",
            button_url="https://x/settings/bots",
        ),
    ):
        await commands.handle(MagicMock(), client, message, "unlink", AsyncMock())
    sent = client.call.call_args.kwargs
    assert sent["chat_id"] == "42"
    button = sent["reply_markup"]["inline_keyboard"][0][0]
    assert button["url"] == "https://x/settings/bots"


@pytest.mark.asyncio
async def test_start_in_group_still_links_the_group():
    client = _bot_api(status="member")
    reply = CommandReply(
        text="**Set up AutoGPT for Builders**",
        button_label="Link Group",
        button_url=_LINK,
    )
    with (
        patch(f"{_CMD}.setup_reply", new=AsyncMock(return_value=reply)) as sr,
        patch(f"{_CMD}.dm_link_reply", new=AsyncMock()) as dr,
    ):
        await commands.handle(
            MagicMock(), client, _message("/start"), "start", AsyncMock()
        )
    dr.assert_not_called()
    kwargs = sr.call_args.kwargs
    assert kwargs["platform_server_id"] == "-100123456"
    # /start in a group goes through the same admin check as /setup.
    assert kwargs["admin_check"] is AdminCheck.NOT_ADMIN
    assert kwargs["admin_role"] == "group admin"
    [sent] = _sent(client)
    assert sent["reply_markup"]["inline_keyboard"][0][0]["login_url"] == {"url": _LINK}


@pytest.mark.asyncio
async def test_help_sends_usage_text():
    client = MagicMock()
    client.call = AsyncMock()
    await commands.handle(MagicMock(), client, _message("/help"), "help", AsyncMock())
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
        await commands.handle(MagicMock(), client, msg, "new", AsyncMock())
    clear.assert_awaited_once_with("telegram", "-100123456|7")
    assert "fresh conversation" in client.call.call_args.kwargs["text"]


@pytest.mark.asyncio
async def test_commands_track_command_used_with_group_scope():
    client = MagicMock()
    client.call = AsyncMock()
    api = MagicMock()
    await commands.handle(api, client, _message("/help"), "help", AsyncMock())
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
    await commands.handle(
        api, client, _message("/help", chat_type="private"), "help", AsyncMock()
    )
    assert api.track_event.call_args.kwargs["server_id"] is None
