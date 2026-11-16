"""The database manager must load the block registry before it serves any RPC."""

from unittest.mock import AsyncMock

from backend.data import db_manager
from backend.data.db_manager import DatabaseManager


async def test_startup_loads_blocks_before_freezing_the_heap_and_serving(mocker):
    """Otherwise the first RPC that needs a block imports them all on the
    service's one event loop, and every other RPC waits seconds behind it."""
    order: list[str] = []

    def load_blocks() -> dict:
        order.append("load_blocks")
        return {}

    mocker.patch("backend.util.service.redis_client.get_redis_async", AsyncMock())
    mocker.patch("backend.util.service.redis_client.disconnect_async", AsyncMock())
    mocker.patch.object(db_manager.db, "connect", AsyncMock())
    mocker.patch.object(db_manager.db, "disconnect", AsyncMock())
    mocker.patch.object(db_manager, "get_blocks", side_effect=load_blocks)
    mocker.patch(
        "backend.util.service.freeze_startup_heap",
        lambda: order.append("freeze_heap"),
    )
    app = DatabaseManager().build_fastapi_app()

    async with app.router.lifespan_context(app):
        order.append("serving")

    assert order == ["load_blocks", "freeze_heap", "serving"]
