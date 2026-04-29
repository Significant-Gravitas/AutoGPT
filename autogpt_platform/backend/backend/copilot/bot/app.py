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

from .adapters.base import PlatformAdapter
from .adapters.discord import config as discord_config
from .adapters.discord.adapter import DiscordAdapter
from .bot_backend import BotBackend
from .handler import MessageHandler

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
        adapters = _build_adapters(api)

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
        exc = future.exception()
        if exc is not None:
            logger.error("CoPilotChatBridge adapters crashed: %r", exc, exc_info=exc)
        else:
            logger.warning("CoPilotChatBridge adapters exited without error")

    async def health_check(self) -> str:
        if not self._adapters_healthy:
            raise UnhealthyServiceError("CoPilotChatBridge adapter task is not running")
        return await super().health_check()

    @expose
    async def send_message_to_channel(
        self,
        platform: Platform,
        channel_id: str,
        content: str,
    ) -> bool:
        """Deliver a message to a channel on the given platform.

        Stub — scaffolding for the inbound-RPC pattern (backend → chat
        platform). Not yet wired to a concrete adapter. Callers must not use
        ``request_retry=True`` on the client until this is implemented, since
        ``ValueError`` crosses the RPC boundary as a client-side 4xx-ish error
        rather than a transient 5xx.
        """
        raise ValueError(f"send_message_to_channel not yet wired for {platform.value}")

    @expose
    async def send_dm(
        self,
        platform: Platform,
        platform_user_id: str,
        content: str,
    ) -> bool:
        """Deliver a DM to a user on the given platform.

        Stub — scaffolding for the inbound-RPC pattern. See
        :meth:`send_message_to_channel` for the retry caveat.
        """
        raise ValueError(f"send_dm not yet wired for {platform.value}")


class CoPilotChatBridgeClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return CoPilotChatBridge

    send_message_to_channel = endpoint_to_async(
        CoPilotChatBridge.send_message_to_channel
    )
    send_dm = endpoint_to_async(CoPilotChatBridge.send_dm)


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
