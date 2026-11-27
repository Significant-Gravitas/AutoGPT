"""Tests for the webhook adapter route-mounting helper."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from .adapters.base import WebhookAdapter
from .webhook_routes import (
    build_webhook_adapters,
    register_webhook_adapters,
    run_webhook_adapter_startup,
)

_SLACK_CFG = "backend.copilot.bot.webhook_routes.slack_config"
_TELEGRAM_CFG = "backend.copilot.bot.webhook_routes.telegram_config"
_TEAMS_CFG = "backend.copilot.bot.webhook_routes.teams_config"


def test_build_webhook_adapters_empty_without_creds():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value=""),
        patch(f"{_TEAMS_CFG}.is_configured", return_value=False),
    ):
        assert build_webhook_adapters(MagicMock()) == []


def test_build_webhook_adapters_includes_slack_when_configured():
    with (
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_bot_token", return_value="xoxb-x"),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value="secret"),
        patch("backend.copilot.bot.adapters.slack.adapter.AsyncWebClient"),
        patch(f"{_TEAMS_CFG}.is_configured", return_value=False),
    ):
        adapters = build_webhook_adapters(MagicMock())
    assert len(adapters) == 1
    assert adapters[0].platform_name == "slack"


def test_build_webhook_adapters_includes_telegram_when_configured():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value="123:abc"),
        patch(f"{_TELEGRAM_CFG}.get_webhook_secret", return_value="s3cret"),
        patch(f"{_TEAMS_CFG}.is_configured", return_value=False),
    ):
        adapters = build_webhook_adapters(MagicMock())
    assert len(adapters) == 1
    assert adapters[0].platform_name == "telegram"


def test_telegram_needs_both_token_and_webhook_secret():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value="123:abc"),
        patch(f"{_TELEGRAM_CFG}.get_webhook_secret", return_value=""),
        patch(f"{_TEAMS_CFG}.is_configured", return_value=False),
    ):
        assert build_webhook_adapters(MagicMock()) == []


def test_build_webhook_adapters_includes_teams_when_configured():
    with (
        patch(f"{_SLACK_CFG}.get_bot_token", return_value=""),
        patch(f"{_SLACK_CFG}.get_signing_secret", return_value=""),
        patch(f"{_TELEGRAM_CFG}.get_bot_token", return_value=""),
        patch(f"{_TEAMS_CFG}.is_configured", return_value=True),
    ):
        adapters = build_webhook_adapters(MagicMock())
    assert len(adapters) == 1
    assert adapters[0].platform_name == "teams"


def test_register_webhook_adapters_wires_each_adapter():
    app = FastAPI()
    adapter = MagicMock(spec=WebhookAdapter)
    api = MagicMock()

    with patch(
        "backend.copilot.bot.webhook_routes.build_webhook_adapters",
        return_value=[adapter],
    ):
        mounted = register_webhook_adapters(app, api)

    adapter.on_message.assert_called_once()
    adapter.register_routes.assert_called_once_with(app)
    assert mounted == [adapter]


async def test_one_failing_adapter_startup_never_stops_the_others(caplog):
    failing = MagicMock(spec=WebhookAdapter, platform_name="telegram")
    failing.on_startup = AsyncMock(side_effect=RuntimeError("down"))
    healthy = MagicMock(spec=WebhookAdapter, platform_name="teams")
    healthy.on_startup = AsyncMock()

    with caplog.at_level(logging.WARNING):
        await run_webhook_adapter_startup([failing, healthy])

    healthy.on_startup.assert_awaited_once()
    assert [r.getMessage() for r in caplog.records] == [
        "Startup of the telegram webhook adapter failed"
    ]
