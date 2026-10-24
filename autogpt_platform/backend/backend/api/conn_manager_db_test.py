"""A subscribe against a seeded run, after the real Prisma query engine
process has been killed: the client still reports itself connected, so only
a reconnect brings the lookup back (AUTOGPT-SERVER-8VJ)."""

import uuid
from unittest.mock import AsyncMock

import httpx
import prisma.models
import pytest
from fastapi import WebSocket

from backend.api.conn_manager import ConnectionManager, graph_exec_channel_key
from backend.data import db


@pytest.mark.asyncio(loop_scope="session")
async def test_subscribe_reconnects_after_the_query_engine_dies(server, monkeypatch):
    user_id, graph_id, graph_exec_id = await _seed_execution()
    manager = ConnectionManager()
    monkeypatch.setattr(manager, "_open_subscription", AsyncMock())
    try:
        _kill_query_engine()
        assert db.is_connected()

        channel_key = await manager.subscribe_graph_exec(
            user_id=user_id,
            graph_exec_id=graph_exec_id,
            websocket=AsyncMock(spec=WebSocket),
        )

        assert channel_key == graph_exec_channel_key(
            user_id, graph_exec_id=graph_exec_id
        )
    finally:
        await _revive_engine()
        await prisma.models.AgentGraph.prisma().delete_many(where={"id": graph_id})
        await prisma.models.User.prisma().delete_many(where={"id": user_id})


async def _seed_execution() -> tuple[str, str, str]:
    user_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"dead-engine-{user_id}@example.com"}
    )
    await prisma.models.AgentGraph.prisma().create(
        data={"id": graph_id, "version": 1, "name": "dead-engine", "userId": user_id}
    )
    execution = await prisma.models.AgentGraphExecution.prisma().create(
        data={"agentGraphId": graph_id, "agentGraphVersion": 1, "userId": user_id}
    )
    return user_id, graph_id, execution.id


def _kill_query_engine() -> None:
    engine = db.prisma._internal_engine
    assert engine is not None and engine.process is not None
    engine.process.kill()
    engine.process.wait(timeout=10)


async def _revive_engine() -> None:
    # Later tests share this client; without the fix under test it stays dead.
    try:
        await db.query_raw_with_schema("SELECT 1")
    except httpx.ConnectError:
        await db.disconnect()
        await db.connect()
