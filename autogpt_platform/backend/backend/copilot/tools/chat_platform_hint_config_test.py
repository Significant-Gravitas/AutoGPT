from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import chat_platform
from .chat_platform_hint import bot_link_hint


@pytest.fixture
def bot_secrets():
    secrets = SimpleNamespace(
        autopilot_bot_discord_token="",
        autopilot_bot_slack_token="",
        autopilot_bot_slack_signing_secret="",
        autopilot_bot_slack_client_id="",
        autopilot_bot_slack_client_secret="",
        autopilot_bot_telegram_token="",
        autopilot_bot_telegram_webhook_secret="",
        microsoft_client_id="",
        microsoft_client_secret="",
        microsoft_tenant_id="",
    )
    chat_platform._configured_chat_platforms.cache_clear()
    with (
        patch.object(
            chat_platform, "Settings", return_value=SimpleNamespace(secrets=secrets)
        ),
        patch(
            "backend.copilot.tools.chat_platform_hint.get_current_permissions",
            return_value=None,
        ),
    ):
        yield secrets
    chat_platform._configured_chat_platforms.cache_clear()


@pytest.mark.parametrize(
    "platform,overrides",
    [
        ("discord", {"autopilot_bot_discord_token": "test-token"}),
        (
            "slack",
            {
                "autopilot_bot_slack_token": "test-token",
                "autopilot_bot_slack_signing_secret": "test-secret",
            },
        ),
        (
            "telegram",
            {
                "autopilot_bot_telegram_token": "test-token",
                "autopilot_bot_telegram_webhook_secret": "test-secret",
            },
        ),
        (
            "teams",
            {
                "microsoft_client_id": "test-id",
                "microsoft_client_secret": "test-secret",
                "microsoft_tenant_id": "test-tenant",
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "block_name,block_platform",
    [
        ("SendDiscordMessageBlock", "discord"),
        ("CreateDiscordThreadBlock", "discord"),
        ("SendSlackMessageBlock", "slack"),
        ("SendTelegramMessageBlock", "telegram"),
    ],
)
def test_hint_requires_its_own_platform(
    bot_secrets, platform, overrides, block_name, block_platform
):
    for name, value in overrides.items():
        setattr(bot_secrets, name, value)

    assert bool(bot_link_hint(block_name, {})) is (block_platform == platform)
