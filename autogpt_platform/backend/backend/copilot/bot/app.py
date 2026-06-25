"""CoPilot Chat Bridge — AppService that runs the configured chat-platform
adapters (Discord, Telegram, Slack) and exposes outbound message RPC for
other services to push messages into chat platforms.
"""

import asyncio
import logging
from concurrent.futures import Future

from backend.platform_linking.models import Platform
from backend.util.service import (
    AppService,
    AppServiceClient,
    UnhealthyServiceError,
    endpoint_to_async,
    expose,
)
from backend.util.settings import Settings

from . import outbound
from .adapters.base import ChannelInfo, PlatformAdapter
from .adapters.discord import config as discord_config
from .adapters.discord.adapter import DiscordAdapter
from .bot_backend import BotBackend
from .handler import MessageHandler
from .outbound import DeliveryResult

logger = logging.getLogger(__name__)

# Stay up for health-checks and runtime reconfiguration when no adapter is
# configured (e.g. deployed without a Discord token).
_NO_ADAPTER_SLEEP_SECONDS = 3600


class CoPilotChatBridge(AppService):
    """Bridges AutoPilot to external chat platforms via per-platform adapters."""

    def __init__(self):
        super().__init__()
        # Flipped to True once `_run_adapters` reaches its blocking gather
        # (or the no-adapter idle loop), and back to False if the task exits
        # for any reason. Consumed by `health_check` so orchestrators can
        # restart the pod when the bridge is dead-but-listening.
        self._adapters_healthy = False
        # Populated by `_run_adapters` so the outbound RPC handlers can reach
        # the live adapter (and its gateway connection) by platform name. The
        # RPC handlers run on the same shared event loop as the adapters, so
        # they can await adapter methods directly.
        self._api: BotBackend | None = None
        self._adapters_by_platform: dict[str, PlatformAdapter] = {}

    @classmethod
    def get_port(cls) -> int:
        return Settings().config.copilot_chat_bridge_port

    def run_service(self) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._run_adapters(), self.shared_event_loop
        )
        future.add_done_callback(self._on_adapters_exit)
        super().run_service()

    async def _run_adapters(self) -> None:
        api = BotBackend()
        self._api = api
        adapters = _build_adapters(api)
        self._adapters_by_platform = {a.platform_name: a for a in adapters}

        if not adapters:
            logger.info(
                "CoPilotChatBridge: no platform adapters configured — idling. "
                "Set AUTOPILOT_BOT_DISCORD_TOKEN (or another platform token) to "
                "enable an adapter."
            )
            self._adapters_healthy = True
            try:
                while True:
                    await asyncio.sleep(_NO_ADAPTER_SLEEP_SECONDS)
            finally:
                await api.close()

        handler = MessageHandler(api)
        for adapter in adapters:
            adapter.on_message(handler.handle)

        self._adapters_healthy = True
        try:
            await asyncio.gather(*(a.start() for a in adapters))
        finally:
            await asyncio.gather(*(a.stop() for a in adapters), return_exceptions=True)
            await api.close()

    def _on_adapters_exit(self, future: "Future[None]") -> None:
        """Surface exceptions from `_run_adapters` and flip the health flag.

        `run_coroutine_threadsafe` would otherwise swallow the exception
        into the returned future, leaving the FastAPI health endpoint
        cheerfully reporting OK on a dead bridge.
        """
        self._adapters_healthy = False
        # Drop the adapter/api handles so outbound RPCs raise UnhealthyService
        # instead of dispatching into an adapter whose gateway connection has
        # already torn down.
        self._adapters_by_platform = {}
        self._api = None
        exc = future.exception()
        if exc is not None:
            logger.error("CoPilotChatBridge adapters crashed: %r", exc, exc_info=exc)
        else:
            logger.warning("CoPilotChatBridge adapters exited without error")

    async def health_check(self) -> str:
        if not self._adapters_healthy:
            raise UnhealthyServiceError("CoPilotChatBridge adapter task is not running")
        return await super().health_check()

    def _require(self, platform: Platform) -> tuple[PlatformAdapter, BotBackend]:
        """Resolve the live adapter + backend for ``platform`` or raise.

        Raises ``UnhealthyServiceError`` (a transient-shaped error) when the
        bridge hasn't finished starting or the platform's adapter isn't
        configured, so a retrying caller backs off instead of treating it as
        a permanent failure.
        """
        if not self._adapters_healthy:
            raise UnhealthyServiceError("CoPilotChatBridge adapter task is not running")
        adapter = self._adapters_by_platform.get(platform.value.lower())
        if adapter is None or self._api is None:
            raise UnhealthyServiceError(
                f"No running adapter for platform {platform.value}"
            )
        return adapter, self._api

    @expose
    async def list_channels(
        self,
        platform: Platform,
        user_id: str,
    ) -> list[ChannelInfo]:
        """List channels ``user_id`` may post to via the bot on ``platform``.

        Backs channel-name resolution and the picker in the copilot tool.
        """
        adapter, api = self._require(platform)
        return await outbound.list_channels(adapter, api, platform.value, user_id)

    @expose
    async def send_message_to_channel(
        self,
        platform: Platform,
        user_id: str,
        channel: str,
        content: str,
    ) -> DeliveryResult:
        """Post ``content`` to ``channel`` (name or ID) as a standalone message.

        ``user_id`` is the AutoGPT user the post acts on behalf of; delivery
        is authorized against the servers that user has linked.
        """
        adapter, api = self._require(platform)
        return await outbound.deliver_message(
            adapter, api, platform.value, user_id, channel, content
        )

    @expose
    async def create_thread_in_channel(
        self,
        platform: Platform,
        user_id: str,
        channel: str,
        thread_name: str,
        content: str,
    ) -> DeliveryResult:
        """Create a standalone thread in ``channel`` and post ``content`` in it."""
        adapter, api = self._require(platform)
        return await outbound.create_thread(
            adapter, api, platform.value, user_id, channel, thread_name, content
        )


class CoPilotChatBridgeClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return CoPilotChatBridge

    list_channels = endpoint_to_async(CoPilotChatBridge.list_channels)
    send_message_to_channel = endpoint_to_async(
        CoPilotChatBridge.send_message_to_channel
    )
    create_thread_in_channel = endpoint_to_async(
        CoPilotChatBridge.create_thread_in_channel
    )


def _build_adapters(api: BotBackend) -> list[PlatformAdapter]:
    """Instantiate adapters based on which platform tokens are configured."""
    adapters: list[PlatformAdapter] = []
    if discord_config.get_bot_token():
        adapters.append(DiscordAdapter(api))
        logger.info("Discord adapter enabled")
    # Future:
    # if telegram_config.get_bot_token():
    #     adapters.append(TelegramAdapter(api))
    # if slack_config.get_bot_token():
    #     adapters.append(SlackAdapter(api))
    return adapters
