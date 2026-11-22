from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import graph


@pytest.mark.asyncio
async def test_handing_over_moves_only_the_rows_the_old_owner_still_has(monkeypatch):
    execute = AsyncMock(return_value=2)
    monkeypatch.setattr(graph, "execute_raw_with_schema", execute)
    tx = MagicMock(name="tx")

    moved = await graph.hand_over_graphs(
        ["g1", "g2"], from_user_id="old", to_user_id="new", tx=tx
    )

    assert moved == 2
    sql, *args = execute.await_args.args
    assert 'SET "userId" = $1' in sql and 'WHERE "userId" = $2' in sql
    assert args == ["new", "old", ["g1", "g2"]]
    assert execute.await_args.kwargs["client"] is tx
