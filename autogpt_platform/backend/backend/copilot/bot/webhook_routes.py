"""Mount webhook adapter routes onto the main backend API.

Webhook-driven chat adapters (Slack Events API, Telegram, Teams, WhatsApp) are
stateless HTTPS receivers, so they ride the existing N-replica main backend API
rather than the single-connection ``copilot-bot`` pod. This helper wires each
configured webhook adapter's routes onto the app, mirroring how ``app.py`` wires
the socket adapters' ``on_message`` callback to the shared ``MessageHandler``.
"""

import logging

from fastapi import FastAPI

from .adapters.base import WebhookAdapter
from .adapters.slack import config as slack_config
from .adapters.slack.adapter import SlackAdapter
from .adapters.telegram import config as telegram_config
from .adapters.telegram.adapter import TelegramAdapter
from .bot_backend import BotBackend
from .handler import MessageHandler

logger = logging.getLogger(__name__)


def register_webhook_adapters(app: FastAPI, api: BotBackend) -> None:
    """Wire every configured webhook adapter's routes onto ``app``.

    The caller owns the ``api`` lifecycle (close it in the app's lifespan
    cleanup) — matching the pattern used by the socket bridge in ``app.py``.
    """
    handler = MessageHandler(api)
    adapters = _build_webhook_adapters(api)
    for adapter in adapters:
        adapter.on_message(handler.handle)
        adapter.register_routes(app)
    logger.info(f"Mounted {len(adapters)} webhook adapter(s) on the main backend API")


def _build_webhook_adapters(api: BotBackend) -> list[WebhookAdapter]:
    """Instantiate webhook adapters from configured platform credentials.

    Slack / Telegram / Teams / WhatsApp adapters slot in here as they land —
    each gated on its own credentials being set, so an unconfigured platform
    mounts nothing.
    """
    adapters: list[WebhookAdapter] = []
    # Signing secret is always required (inbound verification, one per app). Then
    # either OAuth app creds (multi-workspace "Add to Slack") or a single static
    # token (single-workspace back-compat) is enough to mount the adapter.
    has_credentials = (
        slack_config.get_client_id() and slack_config.get_client_secret()
    ) or slack_config.get_bot_token()
    if slack_config.get_signing_secret() and has_credentials:
        adapters.append(SlackAdapter(api))
        logger.info("Slack adapter enabled")
    # Telegram: one BotFather token for every chat; the webhook secret is what
    # authenticates inbound updates, so both are required.
    if telegram_config.get_bot_token() and telegram_config.get_webhook_secret():
        adapters.append(TelegramAdapter(api))
        logger.info("Telegram adapter enabled")
    return adapters
