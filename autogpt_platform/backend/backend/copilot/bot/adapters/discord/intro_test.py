"""Tests for the on-add intro helpers."""

from unittest.mock import MagicMock

import discord

from .intro import intro_message, pick_intro_channel


def _channel(name: str, sendable: bool) -> MagicMock:
    ch = MagicMock(spec=discord.TextChannel)
    ch.name = name
    perms = MagicMock()
    perms.send_messages = sendable
    ch.permissions_for = MagicMock(return_value=perms)
    return ch


def _guild(
    *,
    system_channel: MagicMock | None,
    text_channels: list[MagicMock],
    has_me: bool = True,
) -> MagicMock:
    guild = MagicMock()
    guild.me = MagicMock() if has_me else None
    guild.system_channel = system_channel
    guild.text_channels = text_channels
    return guild


def test_prefers_system_channel_when_sendable():
    sys_ch = _channel("system", sendable=True)
    other = _channel("other", sendable=True)
    guild = _guild(system_channel=sys_ch, text_channels=[other])

    assert pick_intro_channel(guild) is sys_ch


def test_falls_back_to_first_sendable_text_channel():
    sys_ch = _channel("system", sendable=False)
    a = _channel("a", sendable=False)
    b = _channel("b", sendable=True)
    c = _channel("c", sendable=True)
    guild = _guild(system_channel=sys_ch, text_channels=[a, b, c])

    assert pick_intro_channel(guild) is b


def test_returns_none_when_nothing_sendable():
    sys_ch = _channel("system", sendable=False)
    a = _channel("a", sendable=False)
    guild = _guild(system_channel=sys_ch, text_channels=[a])

    assert pick_intro_channel(guild) is None


def test_no_system_channel_uses_first_sendable_text_channel():
    a = _channel("a", sendable=True)
    guild = _guild(system_channel=None, text_channels=[a])

    assert pick_intro_channel(guild) is a


def test_no_member_returns_none():
    guild = _guild(system_channel=None, text_channels=[], has_me=False)

    assert pick_intro_channel(guild) is None


def test_intro_message_mentions_setup():
    msg = intro_message()
    assert "/setup" in msg
    assert "AutoPilot" in msg
