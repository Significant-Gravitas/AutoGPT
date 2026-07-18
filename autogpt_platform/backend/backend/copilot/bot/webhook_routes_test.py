"""Tests for the webhook adapter route-mounting helper."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI

from .adapters.base import WebhookAdapter
from .webhook_routes import _build_webhook_adapters, register_webhook_adapters

_SLACK_CFG = "backend.copilot.bot.webhook_routes.slack_config"
_TELEGRAM_CFG = "backend.copilot.bot.webhook_routes.telegram_config"


def test_build_webhook_adapters_empty_without_creds():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value=""),
    ):
        assert _build_webhook_adapters(MagicMock()) == []


def test_build_webhook_adapters_includes_slack_when_configured():
    with (
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_bot_token", return_value="xoxb-x"),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value="secret"),
        patch("backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"),
    ):
        adapters = _build_webhook_adapters(MagicMock())
    assert len(adapters) == 1
    assert adapters[0].platform_name == "slack"


def test_build_webhook_adapters_includes_telegram_when_configured():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value="123:abc"),
        patch(f"{_TELEGRAM_CFG}.get_webhook_secret", return_value="s3cret"),
    ):
        adapters = _build_webhook_adapters(MagicMock())
    assert len(adapters) == 1
    assert adapters[0].platform_name == "telegram"


def test_telegram_needs_both_token_and_webhook_secret():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value="123:abc"),
        patch(f"{_TELEGRAM_CFG}.get_webhook_secret", return_value=""),
    ):
        assert _build_webhook_adapters(MagicMock()) == []


def test_register_webhook_adapters_wires_each_adapter():
    app = FastAPI()
    adapter = MagicMock(spec=WebhookAdapter)
    api = MagicMock()

    with patch(
        "backend.copilot.bot.webhook_routes._build_webhook_adapters",
        return_value=[adapter],
    ):
        register_webhook_adapters(app, api)

    adapter.on_message.assert_called_once()
    adapter.register_routes.assert_called_once_with(app)
