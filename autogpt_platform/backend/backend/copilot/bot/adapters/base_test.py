"""Tests for the adapter base-class hierarchy."""

from .base import PlatformAdapter, SocketAdapter, WebhookAdapter
from .discord.adapter import DiscordAdapter


def test_socket_and_webhook_adapters_share_the_platform_contract():
    assert issubclass(SocketAdapter, PlatformAdapter)
    assert issubclass(WebhookAdapter, PlatformAdapter)


def test_discord_is_a_socket_adapter():
    assert issubclass(DiscordAdapter, SocketAdapter)
