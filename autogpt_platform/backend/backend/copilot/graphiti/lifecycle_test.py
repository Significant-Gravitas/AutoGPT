from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from . import lifecycle


@pytest.mark.asyncio
async def test_shared_retrieval_returns_only_active_graph_state() -> None:
    active_edge = SimpleNamespace(uuid="edge-active")
    tentative_edge = SimpleNamespace(uuid="edge-tentative")
    driver = SimpleNamespace(
        execute_query=AsyncMock(return_value=([{"uuid": "edge-active"}], None, None)),
        close=AsyncMock(),
    )

    with patch.object(lifecycle, "_open_driver", return_value=driver):
        edges = await lifecycle.filter_active_shared_edges(
            "org_1",
            [active_edge, tentative_edge],
        )

    assert edges == [active_edge]
    assert "e.status = 'active'" in driver.execute_query.await_args.args[0]
    assert "e.expired_at IS NULL" in driver.execute_query.await_args.args[0]
    driver.close.assert_awaited_once()


def test_shared_search_filter_excludes_expired_edges() -> None:
    from graphiti_core.search.search_filters import ComparisonOperator

    search_filter = lifecycle.active_shared_search_filter()

    assert search_filter.expired_at is not None
    assert search_filter.expired_at[0][0].comparison_operator == (
        ComparisonOperator.is_null
    )


@pytest.mark.asyncio
async def test_shared_retrieval_fails_closed_when_lifecycle_query_fails() -> None:
    driver = SimpleNamespace(
        execute_query=AsyncMock(side_effect=RuntimeError("unavailable")),
        close=AsyncMock(),
    )

    with (
        patch.object(lifecycle, "_open_driver", return_value=driver),
        pytest.raises(RuntimeError, match="unavailable"),
    ):
        await lifecycle.filter_active_shared_edges(
            "org_1", [SimpleNamespace(uuid="edge")]
        )

    driver.close.assert_awaited_once()
