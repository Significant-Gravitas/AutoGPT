"""Mount webhook adapter routes onto the main backend API."""

import logging

from fastapi import FastAPI

from .adapters.base import WebhookAdapter
from .adapters.slack import config as slack_config
from .adapters.slack.adapter import SlackAdapter
from .bot_backend import BotBackend
from .handler import MessageHandler

logger = logging.getLogger(__name__)


def register_webhook_adapters(app: FastAPI) -> None:
    """Wire every configured webhook adapter's routes onto `app`."""
    api = BotBackend()
    handler = MessageHandler(api)
    adapters = _build_webhook_adapters(api)
    for adapter in adapters:
        adapter.on_message(handler.handle)
        adapter.register_routes(app)
    logger.info("Mounted %d webhook adapter(s) on the main backend API", len(adapters))


def _build_webhook_adapters(api: BotBackend) -> list[WebhookAdapter]:
    """Instantiate webhook adapters from configured platform credentials."""
    adapters: list[WebhookAdapter] = []
    if slack_config.get_bot_token() and slack_config.get_signing_secret():
        adapters.append(SlackAdapter(api))
        logger.info("Slack adapter enabled")
    # Telegram / Teams / WhatsApp adapters slot in here as they land.
    return adapters
