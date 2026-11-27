"""The REST API lifespan runs the chat webhook adapters' startup hooks.

The API has its own ``lifespan``, so Starlette never fires ``on_startup``
handlers on it: anything an adapter needs once at boot (Telegram's command
menu) has to run from the lifespan itself.
"""

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import fastapi
import pytest

from backend.api import rest_api
from backend.copilot.bot.adapters.telegram import commands
from backend.copilot.bot.adapters.telegram.adapter import TelegramAdapter

_TIMEOUT_SECONDS = 10


@pytest.fixture
def boot_steps_stubbed(mocker):
    """Stub every service the lifespan connects to, so it runs without them."""
    mocker.patch.object(rest_api, "verify_auth_settings")
    mocker.patch.object(rest_api.backend.data.db, "connect", AsyncMock())
    mocker.patch.object(rest_api.backend.data.db, "disconnect", AsyncMock())
    mocker.patch.object(
        rest_api.backend.data.redis_client, "get_redis_async", AsyncMock()
    )
    mocker.patch.object(
        rest_api.backend.data.redis_client, "disconnect_async", AsyncMock()
    )
    mocker.patch("backend.sdk.registry.AutoRegistry.patch_integrations")
    mocker.patch("backend.integrations.managed_providers.register_all")
    mocker.patch.object(rest_api.backend.data.block, "initialize_blocks", AsyncMock())
    mocker.patch.object(rest_api, "_run_data_migrations", AsyncMock())
    mocker.patch.object(rest_api.backend.data.llm_registry, "load_catalog")
    mocker.patch.object(rest_api, "freeze_startup_heap")
    mocker.patch.object(rest_api.mcp_mount, "running", lambda: contextlib.nullcontext())
    mocker.patch(
        "backend.api.features.integrations.codex.codex_login_coordinator.shutdown",
        AsyncMock(),
    )
    mocker.patch.object(rest_api, "shutdown_cloud_storage_handler", AsyncMock())
    mocker.patch.object(rest_api, "shutdown_workspace_storage", AsyncMock())
    mocker.patch.object(rest_api._webhook_bot_backend, "close", AsyncMock())


def _telegram_adapter(mocker, call: AsyncMock) -> TelegramAdapter:
    """A real Telegram adapter, mounted on an app that boots with the real
    REST API lifespan, with only its Bot API client faked."""
    with patch(
        "backend.copilot.bot.adapters.telegram.adapter.config.get_bot_token",
        return_value="123:abc",
    ):
        adapter = TelegramAdapter(MagicMock())
    adapter._client = MagicMock(call=call)
    mocker.patch.object(rest_api, "_webhook_adapters", [adapter])
    return adapter


def _api_app(adapter: TelegramAdapter) -> fastapi.FastAPI:
    app = fastapi.FastAPI(lifespan=rest_api.lifespan_context)
    adapter.register_routes(app)
    return app


async def test_lifespan_publishes_the_telegram_command_menu(
    mocker, boot_steps_stubbed
) -> None:
    published = asyncio.Event()
    call = AsyncMock(side_effect=lambda *a, **kw: published.set())
    app = _api_app(_telegram_adapter(mocker, call))

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(published.wait(), _TIMEOUT_SECONDS)

    call.assert_awaited_once_with("setMyCommands", commands=commands.COMMAND_MENU)


async def test_a_failing_menu_publish_is_a_warning_and_the_api_still_serves(
    mocker, boot_steps_stubbed, caplog
) -> None:
    attempted = asyncio.Event()

    def fail(*args, **kwargs):
        attempted.set()
        raise RuntimeError("telegram is down")

    app = _api_app(_telegram_adapter(mocker, AsyncMock(side_effect=fail)))

    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(attempted.wait(), _TIMEOUT_SECONDS)
            await asyncio.sleep(0)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("telegram" in r.getMessage().lower() for r in warnings)


def _hanging_call() -> tuple[AsyncMock, asyncio.Event, asyncio.Event]:
    """A Bot API call that never answers, and events for its start and cancel."""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    return AsyncMock(side_effect=hang), started, cancelled


async def test_a_hanging_menu_publish_never_holds_up_the_boot(
    mocker, boot_steps_stubbed
) -> None:
    call, started, cancelled = _hanging_call()
    app = _api_app(_telegram_adapter(mocker, call))

    async def serve_then_shut_down() -> None:
        async with app.router.lifespan_context(app):
            await started.wait()

    await asyncio.wait_for(serve_then_shut_down(), _TIMEOUT_SECONDS)
    assert cancelled.is_set(), "shutdown must not leave the publish running"


async def test_a_crash_while_serving_still_stops_a_pending_menu_publish(
    mocker, boot_steps_stubbed
) -> None:
    call, started, cancelled = _hanging_call()
    app = _api_app(_telegram_adapter(mocker, call))

    async def serve_then_crash() -> None:
        async with app.router.lifespan_context(app):
            await started.wait()
            raise RuntimeError("serving crashed")

    with pytest.raises(RuntimeError, match="serving crashed"):
        await asyncio.wait_for(serve_then_crash(), _TIMEOUT_SECONDS)
    assert cancelled.is_set(), "a crashed lifespan must not leave the publish running"
