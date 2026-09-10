"""Tests for the adapter base-class hierarchy."""

from unittest.mock import MagicMock

from .base import PlatformAdapter, SocketAdapter, WebhookAdapter
from .discord.adapter import DiscordAdapter


def test_socket_and_webhook_adapters_share_the_platform_contract():
    assert issubclass(SocketAdapter, PlatformAdapter)
    assert issubclass(WebhookAdapter, PlatformAdapter)


def test_discord_is_a_socket_adapter():
    assert issubclass(DiscordAdapter, SocketAdapter)


def test_localize_markup_defaults_to_identity():
    # Discord doesn't override the base seam, so it must leave canonical
    # CommonMark untouched (the default `return text`).
    adapter = DiscordAdapter(MagicMock())
    assert adapter.localize_markup("**bold** [x](y)") == "**bold** [x](y)"
