"""Tests for the webhook adapter route-mounting helper."""

from unittest.mock import MagicMock, patch

from fastapi import FastAPI

from .adapters.base import WebhookAdapter
from .webhook_routes import _build_webhook_adapters, register_webhook_adapters


def test_build_webhook_adapters_starts_empty():
    assert _build_webhook_adapters(MagicMock()) == []


def test_register_webhook_adapters_wires_each_adapter():
    app = FastAPI()
    adapter = MagicMock(spec=WebhookAdapter)

    with (
        patch("backend.copilot.bot.bot_backend.get_platform_linking_manager_client"),
        patch(
            "backend.copilot.bot.webhook_routes._build_webhook_adapters",
            return_value=[adapter],
        ),
    ):
        register_webhook_adapters(app)

    adapter.on_message.assert_called_once()
    adapter.register_routes.assert_called_once_with(app)
