"""Tests for the bot-platforms registry."""

from unittest.mock import patch

from backend.api.features.platform_linking.registry import enabled_platforms

_REG = "backend.api.features.platform_linking.registry"


def _slack_off():
    # Slack disabled unless a test opts in — enabled always requires the signing
    # secret, so clearing it hides Slack regardless of whatever token / OAuth
    # creds happen to be in the environment.
    return patch(f"{_REG}.slack_config.get_signing_secret", return_value="")


def _slack_oauth_off():
    # Force single-workspace mode: no "Add to Slack" URL even when Slack is on.
    return (
        patch(f"{_REG}.slack_config.get_client_id", return_value=""),
        patch(f"{_REG}.slack_config.get_client_secret", return_value=""),
    )


def _discord_off():
    return patch(f"{_REG}.discord_config.get_bot_token", return_value="")


def test_no_platforms_when_none_configured():
    with _discord_off(), _slack_off():
        assert enabled_platforms() == []


def test_discord_appears_with_invite_url_when_client_id_set():
    with (
        _slack_off(),
        patch(f"{_REG}.discord_config.get_bot_token", return_value="token"),
        patch(f"{_REG}.discord_config.get_client_id", return_value="my-client-id"),
        patch(f"{_REG}.discord_config.get_invite_permissions", return_value="123"),
    ):
        platforms = enabled_platforms()

    assert [p.platform for p in platforms] == ["DISCORD"]
    discord = platforms[0]
    assert discord.enabled is True
    assert discord.add_bot_url is not None
    assert "client_id=my-client-id" in discord.add_bot_url
    assert "permissions=123" in discord.add_bot_url


def test_discord_appears_without_invite_url_when_client_id_missing():
    with (
        _slack_off(),
        patch(f"{_REG}.discord_config.get_bot_token", return_value="token"),
        patch(f"{_REG}.discord_config.get_client_id", return_value=""),
    ):
        platforms = enabled_platforms()

    assert [p.platform for p in platforms] == ["DISCORD"]
    assert platforms[0].add_bot_url is None


def test_slack_appears_in_single_workspace_mode_without_install_url():
    # Static token + signing secret only (no OAuth app creds) → Slack is enabled
    # but there's no multi-workspace "Add to Slack" button.
    oauth_id_off, oauth_secret_off = _slack_oauth_off()
    with (
        _discord_off(),
        patch(f"{_REG}.slack_config.get_bot_token", return_value="xoxb-x"),
        patch(f"{_REG}.slack_config.get_signing_secret", return_value="secret"),
        oauth_id_off,
        oauth_secret_off,
    ):
        platforms = enabled_platforms()

    assert [p.platform for p in platforms] == ["SLACK"]
    slack = platforms[0]
    assert slack.enabled is True
    assert slack.display_name == "Slack"
    assert slack.icon == "slack.png"
    assert slack.add_bot_url is None


def test_slack_add_to_slack_url_when_oauth_configured():
    with (
        _discord_off(),
        patch(f"{_REG}.slack_config.get_bot_token", return_value=""),
        patch(f"{_REG}.slack_config.get_signing_secret", return_value="secret"),
        patch(f"{_REG}.slack_config.get_client_id", return_value="cid"),
        patch(f"{_REG}.slack_config.get_client_secret", return_value="csecret"),
        patch(f"{_REG}.Settings") as settings,
    ):
        settings.return_value.config.platform_base_url = "https://backend.example"
        platforms = enabled_platforms()

    assert [p.platform for p in platforms] == ["SLACK"]
    slack = platforms[0]
    assert slack.enabled is True
    assert (
        slack.add_bot_url
        == "https://backend.example/api/copilot-webhooks/slack/install"
    )


def test_slack_hidden_when_signing_secret_missing():
    with (
        _discord_off(),
        patch(f"{_REG}.slack_config.get_bot_token", return_value="xoxb-x"),
        patch(f"{_REG}.slack_config.get_signing_secret", return_value=""),
    ):
        assert enabled_platforms() == []


def test_both_platforms_when_both_configured():
    oauth_id_off, oauth_secret_off = _slack_oauth_off()
    with (
        patch(f"{_REG}.discord_config.get_bot_token", return_value="token"),
        patch(f"{_REG}.discord_config.get_client_id", return_value=""),
        patch(f"{_REG}.slack_config.get_bot_token", return_value="xoxb-x"),
        patch(f"{_REG}.slack_config.get_signing_secret", return_value="secret"),
        oauth_id_off,
        oauth_secret_off,
    ):
        platforms = enabled_platforms()

    assert {p.platform for p in platforms} == {"DISCORD", "SLACK"}
