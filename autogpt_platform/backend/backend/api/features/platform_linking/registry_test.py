"""Tests for the bot-platforms registry."""

from unittest.mock import patch

from backend.api.features.platform_linking.registry import enabled_platforms


def test_discord_excluded_when_bot_token_not_set():
    with patch(
        "backend.api.features.platform_linking.registry.discord_config.get_bot_token",
        return_value="",
    ):
        assert enabled_platforms() == []


def test_discord_appears_with_invite_url_when_client_id_set():
    with (
        patch(
            "backend.api.features.platform_linking.registry.discord_config.get_bot_token",
            return_value="token",
        ),
        patch(
            "backend.api.features.platform_linking.registry.discord_config.get_client_id",
            return_value="my-client-id",
        ),
        patch(
            "backend.api.features.platform_linking.registry.discord_config.get_invite_permissions",
            return_value="123",
        ),
    ):
        platforms = enabled_platforms()

    assert len(platforms) == 1
    discord = platforms[0]
    assert discord.platform == "DISCORD"
    assert discord.enabled is True
    assert discord.add_bot_url is not None
    assert "client_id=my-client-id" in discord.add_bot_url
    assert "permissions=123" in discord.add_bot_url


def test_discord_appears_without_invite_url_when_client_id_missing():
    with (
        patch(
            "backend.api.features.platform_linking.registry.discord_config.get_bot_token",
            return_value="token",
        ),
        patch(
            "backend.api.features.platform_linking.registry.discord_config.get_client_id",
            return_value="",
        ),
    ):
        platforms = enabled_platforms()

    assert len(platforms) == 1
    assert platforms[0].enabled is True
    assert platforms[0].add_bot_url is None
