"""Reach the tasks store from any process.

The REST server holds a Prisma connection and calls ``tasks.db`` directly.
The copilot executor (where task tools and the turn-finish hook run) has
none, so the same calls go through the DatabaseManager RPC service, which
exposes every function below as ``tasks_<name>``.
"""

from datetime import datetime
from typing import Any

import prisma.models
from pydantic import BaseModel

from backend.data import db as prisma_db
from backend.util.clients import get_database_manager_async_client


class TurnUsage(BaseModel):
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


async def get_turn_usage(session_id: str, since: datetime) -> TurnUsage:
    """Sum the platform cost log of one chat since *since* (one turn)."""
    rows = await prisma.models.PlatformCostLog.prisma().find_many(
        where={"chatSessionId": session_id, "createdAt": {"gte": since}}
    )
    return TurnUsage(
        cost_usd=sum(int(r.costMicrodollars or 0) for r in rows) / 1_000_000,
        input_tokens=sum(
            (r.inputTokens or 0)
            + (r.cacheReadTokens or 0)
            + (r.cacheCreationTokens or 0)
            for r in rows
        ),
        output_tokens=sum(r.outputTokens or 0 for r in rows),
    )


class _RemoteTasksStore:
    """``tasks.db``-shaped facade over the DatabaseManager RPC client."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_database_manager_async_client(), f"tasks_{name}")


class _LocalTasksStore:
    def __getattr__(self, name: str) -> Any:
        from backend.api.features.tasks import db as tasks_db

        if name == "get_turn_usage":
            return get_turn_usage
        return getattr(tasks_db, name)


_local = _LocalTasksStore()
_remote = _RemoteTasksStore()


def tasks_store() -> Any:
    """``tasks.db`` when this thread owns a Prisma connection, else the RPC."""
    return _local if prisma_db.is_connected() else _remote
