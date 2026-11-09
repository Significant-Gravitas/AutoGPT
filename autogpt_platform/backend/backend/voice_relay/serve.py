"""Run the voice relay: ``poetry run voice-relay``.

Stock uvicorn answers SIGTERM by closing every open socket at once, which
would drop every call on each deploy. The relay drains instead: its first
SIGTERM fails readiness, refuses new calls with 1013, and lets live calls run
to their end (the call cap bounds each) and their post-call jobs finish, then
exits. Calls still open after ``VOICE_RELAY_DRAIN_MAX_S`` end with reason
``server_restart``; a post-call job still running at exit is left to another
pod's sweep. A second SIGTERM, or a SIGINT, exits at once.
"""

import asyncio
import logging
import signal
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import FrameType

import uvicorn
from google import genai

from backend.api.utils.cors import build_cors_params
from backend.app import run_processes
from backend.copilot.onboarding_interview_voice import (
    VOICE_INTERVIEW_END_TOOL,
    VOICE_INTERVIEW_END_TOOL_PARAMETERS,
    VOICE_INTERVIEW_PROMPT,
)
from backend.data import db, redis_client
from backend.data.onboarding import record_onboarding_voice_call
from backend.data.user import get_user_by_id
from backend.monitoring.instrumentation import start_metrics_server
from backend.util.process import AppProcess
from backend.util.settings import Settings
from backend.voice_relay import tokens
from backend.voice_relay.app import Relay, RelayDeps, create_app, first_name_of
from backend.voice_relay.config import TRANSCRIBE_MODEL, WS_MAX_MESSAGE_BYTES, price_of
from backend.voice_relay.metering import drain_cost_logs
from backend.voice_relay.postcall import GeminiTranscriber, PostCall
from backend.voice_relay.record import CloudRecordings
from backend.voice_relay.scope import GeminiScopeClassifier
from backend.voice_relay.vendor import GeminiLive

logger = logging.getLogger(__name__)

# Once the drain limit has ended the calls left, how long their endings get.
SETTLE_S = 20.0
POLL_S = 0.2

END_TOOL_DESCRIPTION = (
    "End the call once they've agreed to the first thing their team will start "
    "on, or corrected it. Call it, then say a short goodbye."
)


class DrainingServer(uvicorn.Server):
    """uvicorn, with SIGTERM draining live calls instead of dropping them."""

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        relay: Relay,
        drain_max_s: float,
        settle_s: float = SETTLE_S,
    ) -> None:
        super().__init__(config)
        self.relay = relay
        self.drain_max_s = drain_max_s
        self.settle_s = settle_s
        self.drain_task: asyncio.Future | None = None

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        relay = self.relay
        if sig == signal.SIGTERM and not relay.draining and relay.loop is not None:
            relay.start_drain()
            logger.warning(f"SIGTERM: draining {len(relay.sessions)} live call(s)")
            relay.loop.call_soon_threadsafe(self._start_drain)
            return
        super().handle_exit(sig, frame)

    def _start_drain(self) -> None:
        self.drain_task = asyncio.ensure_future(self.drain())

    async def drain(self) -> None:
        relay = self.relay
        started = relay.drain_started or time.monotonic()
        while relay.busy and time.monotonic() - started < self.drain_max_s:
            await asyncio.sleep(POLL_S)
        for session in list(relay.sessions.values()):
            session.end("server_restart")
        settling = time.monotonic()
        while relay.busy and time.monotonic() - settling < self.settle_s:
            await asyncio.sleep(POLL_S)
        logger.warning(f"Drain done after {time.monotonic() - started:.1f}s")
        self.should_exit = True


@asynccontextmanager
async def production_deps() -> AsyncIterator[RelayDeps]:
    """The relay's services: Gemini, the database and Redis."""
    settings = Settings()
    model = settings.config.voice_interview_model
    # A call is priced as it runs, so a model with no price cannot take one.
    price_of(model)
    price_of(TRANSCRIBE_MODEL)
    if not settings.secrets.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if not tokens.is_configured():
        raise RuntimeError(
            f"VOICE_RELAY_SIGNING_KEY is not set, or shorter than "
            f"{tokens.MIN_KEY_CHARS} characters"
        )
    await db.connect()
    await redis_client.get_redis_async()
    client = genai.Client(api_key=settings.secrets.gemini_api_key)
    try:
        yield RelayDeps(
            vendor=GeminiLive(
                client,
                model=model,
                voice=settings.config.voice_interview_voice,
                system_prompt=VOICE_INTERVIEW_PROMPT,
                tool_name=VOICE_INTERVIEW_END_TOOL,
                tool_description=END_TOOL_DESCRIPTION,
                tool_parameters=VOICE_INTERVIEW_END_TOOL_PARAMETERS,
            ),
            classifier=GeminiScopeClassifier(client),
            redis=redis_client.get_redis_async,
            account_name=_account_name,
            record_call=record_onboarding_voice_call,
            live_model=model,
            postcall=PostCall(
                redis=redis_client.get_redis_async,
                recordings=CloudRecordings(),
                transcriber=GeminiTranscriber(client),
                first_name=_first_name,
                live_model=model,
                pod=socket.gethostname(),
            ),
            spool_dir=settings.config.voice_interview_spool_dir,
        )
    finally:
        await drain_cost_logs()
        try:
            await redis_client.disconnect_async()
        except Exception:
            logger.warning("Redis did not disconnect cleanly", exc_info=True)
        try:
            await db.disconnect()
        except Exception:
            logger.warning("The database did not disconnect cleanly", exc_info=True)


async def _account_name(user_id: str) -> str | None:
    # The relay connects the database itself, as the REST API does.
    user = await get_user_by_id(user_id)
    return user.name


async def _first_name(user_id: str) -> str | None:
    """The first name the transcription of a call is told the speaker has."""
    try:
        return first_name_of(await _account_name(user_id))
    except Exception:
        logger.info("Voice post-call: the account's name could not be read")
        return None


class VoiceRelay(AppProcess):
    # The start route decides who may call; the relay reads no flags.
    reads_feature_flags = False

    def run(self) -> None:
        config = Settings().config
        relay = Relay(
            max_calls=config.voice_relay_max_calls,
            origins=build_cors_params(
                config.backend_cors_allow_origins, config.app_env
            ),
        )
        app = create_app(relay, setup=production_deps)
        start_metrics_server(config.voice_relay_metrics_port)
        server = DrainingServer(
            uvicorn.Config(
                app,
                host=config.voice_relay_host,
                port=config.voice_relay_port,
                ws="websockets-sansio",
                ws_max_size=WS_MAX_MESSAGE_BYTES,
                ws_per_message_deflate=False,
                log_config=None,
            ),
            relay=relay,
            drain_max_s=config.voice_relay_drain_max_s,
        )
        server.run()


def main() -> None:
    """Run the voice relay process."""
    run_processes(VoiceRelay())


if __name__ == "__main__":
    main()
