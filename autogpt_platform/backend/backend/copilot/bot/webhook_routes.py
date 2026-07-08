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
    return adapters
