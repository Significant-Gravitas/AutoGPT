"""Tests for DiscordAdapter helpers that don't need a live gateway."""

from typing import cast
from unittest.mock import MagicMock

import discord
import pytest

from backend.copilot.bot.adapters.discord.adapter import DiscordAdapter


def _adapter_with_bot_user(bot_id: int | None) -> DiscordAdapter:
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    client = MagicMock()
    client.user = MagicMock(id=bot_id) if bot_id is not None else None
    adapter._client = cast(discord.Client, client)
    return adapter


def _mention(user_id: int, display_name: str) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.display_name = display_name
    return user


def _message(content: str, mentions: list[MagicMock]) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.mentions = mentions
    return msg


class TestStripMentions:
    def test_strips_only_bot_mention(self):
        adapter = _adapter_with_bot_user(bot_id=1000)
        bot = _mention(1000, "AutoPilot")
        alice = _mention(2000, "Alice")
        msg = _message(
            "<@1000> please summarise what <@2000> said",
            mentions=[bot, alice],
        )

        assert adapter._strip_mentions(msg) == "please summarise what @Alice said"

    def test_handles_nickname_style_tokens(self):
        adapter = _adapter_with_bot_user(bot_id=1000)
        bot = _mention(1000, "AutoPilot")
        alice = _mention(2000, "Alice")
        msg = _message("<@!1000> ping <@!2000>", mentions=[bot, alice])

        assert adapter._strip_mentions(msg) == "ping @Alice"

    def test_no_bot_user_leaves_all_mentions_as_names(self):
        adapter = _adapter_with_bot_user(bot_id=None)
        alice = _mention(2000, "Alice")
        msg = _message("hi <@2000>", mentions=[alice])

        assert adapter._strip_mentions(msg) == "hi @Alice"

    def test_message_without_mentions_is_trimmed(self):
        adapter = _adapter_with_bot_user(bot_id=1000)
        msg = _message("  hello world  ", mentions=[])

        assert adapter._strip_mentions(msg) == "hello world"

    @pytest.mark.parametrize(
        "content,expected",
        [
            ("<@1000>", ""),
            ("<@!1000>", ""),
            ("<@1000> hi", "hi"),
            ("hi <@1000>", "hi"),
        ],
    )
    def test_bot_only_variants(self, content: str, expected: str):
        adapter = _adapter_with_bot_user(bot_id=1000)
        bot = _mention(1000, "AutoPilot")
        msg = _message(content, mentions=[bot])

        assert adapter._strip_mentions(msg) == expected
