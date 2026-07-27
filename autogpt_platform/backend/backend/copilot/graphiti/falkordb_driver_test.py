from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import falkordb_driver as fdb
from .falkordb_driver import AutoGPTFalkorDriver


class _FakeResult:
    """Minimal stand-in for a falkordb query result (header + rows)."""

    def __init__(self, header, result_set):
        self.header = header
        self.result_set = result_set


def _set_query(driver: AutoGPTFalkorDriver, side_effect) -> AsyncMock:
    """Wire ``graph.query`` (reached via ``_get_graph``) to ``side_effect``."""
    query = AsyncMock(side_effect=side_effect)
    driver.client.select_graph.return_value.query = query
    return query


def _overflow() -> Exception:
    return Exception("Max pending queries exceeded")


@pytest.fixture
def driver() -> AutoGPTFalkorDriver:
    # ``build_fulltext_query`` is a pure string-builder that never touches
    # the FalkorDB client; injecting a mock avoids the eager Redis probe
    # that the upstream ``FalkorDriver.__init__`` runs against
    # ``localhost:6379``.
    return AutoGPTFalkorDriver(falkor_db=MagicMock())


def test_build_fulltext_query_uses_unquoted_group_ids_for_falkordb(
    driver: AutoGPTFalkorDriver,
) -> None:
    query = driver.build_fulltext_query(
        "Sarah",
        group_ids=["user_883cc9da-fe37-4863-839b-acba022bf3ef"],
    )

    assert query == "(@group_id:user_883cc9da-fe37-4863-839b-acba022bf3ef) (Sarah)"
    assert '"user_883cc9da-fe37-4863-839b-acba022bf3ef"' not in query


def test_build_fulltext_query_joins_multiple_group_ids_with_or(
    driver: AutoGPTFalkorDriver,
) -> None:
    query = driver.build_fulltext_query("Sarah", group_ids=["user_a", "user_b"])

    assert query == "(@group_id:user_a|user_b) (Sarah)"


def test_stopwords_only_query_returns_group_filter_only(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Line 25: sanitized_query is empty (all stopwords) but group_ids present."""
    # "the" is a common stopword — the query should reduce to just the group filter.
    query = driver.build_fulltext_query(
        "the",
        group_ids=["user_abc"],
    )

    assert query == "(@group_id:user_abc)"


def test_query_without_group_ids_returns_parenthesized_query(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Line 27: sanitized_query has content but no group_ids provided."""
    query = driver.build_fulltext_query("Sarah", group_ids=None)

    assert query == "(Sarah)"


# ---------------------------------------------------------------------------
# build_indices opt-out — pins the contract that suppresses graphiti-core's
# per-driver background indexing task on read-only / per-request paths.
# Regression coverage for the "Buffer is closed" log spam on admin viz loads.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_indices_false_skips_super_call() -> None:
    """``build_indices=False`` → our override returns early and never
    delegates to ``FalkorDriver.build_indices_and_constraints``."""
    with patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
        return_value=None,
    ), patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
        new=AsyncMock(),
    ) as super_build:
        driver = AutoGPTFalkorDriver(build_indices=False)
        await driver.build_indices_and_constraints()
    super_build.assert_not_called()


@pytest.mark.asyncio
async def test_build_indices_true_delegates_to_super() -> None:
    """Default ``build_indices=True`` preserves upstream behaviour —
    the long-lived chat-write client still gets its indices built."""
    with patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
        return_value=None,
    ), patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
        new=AsyncMock(),
    ) as super_build:
        driver = AutoGPTFalkorDriver(build_indices=True)
        await driver.build_indices_and_constraints()
    super_build.assert_awaited_once()


@pytest.mark.asyncio
async def test_default_build_indices_is_upstream_compat() -> None:
    """Omitting the kwarg keeps the upstream-default behaviour so
    existing call sites (long-lived chat client) don't silently lose
    their index-build path."""
    with patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
        return_value=None,
    ), patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
        new=AsyncMock(),
    ) as super_build:
        driver = AutoGPTFalkorDriver()
        await driver.build_indices_and_constraints()
    super_build.assert_awaited_once()


@pytest.mark.asyncio
async def test_build_indices_false_persists_across_repeated_calls() -> None:
    """The override doesn't flip after the first call — every invocation
    against a ``build_indices=False`` driver stays a no-op."""
    with patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.__init__",
        return_value=None,
    ), patch(
        "graphiti_core.driver.falkordb_driver.FalkorDriver.build_indices_and_constraints",
        new=AsyncMock(),
    ) as super_build:
        driver = AutoGPTFalkorDriver(build_indices=False)
        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()
        await driver.build_indices_and_constraints()
    super_build.assert_not_called()


# ---------------------------------------------------------------------------
# execute_query retry — pins the bounded backoff on FalkorDB's transient
# "Max pending queries exceeded" backpressure. (SENTRY-1384.)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_query_success_returns_upstream_shaped_records(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Happy path: no retry, and results are shaped like upstream
    (list-of-dicts, header, None)."""
    _set_query(driver, [_FakeResult([("n.count", "count")], [[5]])])

    records, header, meta = await driver.execute_query("MATCH (n) RETURN count(n)")

    assert records == [{"count": 5}]
    assert header == ["count"]
    assert meta is None


@pytest.mark.asyncio
async def test_execute_query_retries_pending_queue_overflow_then_succeeds(
    driver: AutoGPTFalkorDriver,
) -> None:
    """Two transient overflows are retried; the third attempt succeeds and its
    result is returned (memory op recovers instead of being dropped)."""
    query = _set_query(
        driver,
        [_overflow(), _overflow(), _FakeResult([("x", "count")], [[1]])],
    )

    with patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep, patch(
        "backend.copilot.graphiti.config.graphiti_config.falkordb_query_max_attempts",
        3,
    ):
        records, _, _ = await driver.execute_query("MATCH (n) RETURN 1")

    assert records == [{"count": 1}]
    assert query.await_count == 3
    assert sleep.await_count == 2  # backoff between the three attempts


@pytest.mark.asyncio
async def test_execute_query_raises_after_exhausting_retries(
    driver: AutoGPTFalkorDriver,
) -> None:
    """A sustained overflow exhausts the budget, then raises AND logs exactly
    one terminal error under the upstream logger (so Sentry sees one event)."""
    query = _set_query(driver, _overflow())

    with patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep, patch.object(
        fdb, "_UPSTREAM_QUERY_LOGGER"
    ) as upstream_logger, patch(
        "backend.copilot.graphiti.config.graphiti_config.falkordb_query_max_attempts",
        3,
    ):
        with pytest.raises(Exception, match="Max pending queries exceeded"):
            await driver.execute_query("MATCH (n) RETURN 1")

    assert query.await_count == 3
    assert sleep.await_count == 2
    upstream_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_execute_query_non_overflow_error_fails_fast(
    driver: AutoGPTFalkorDriver,
) -> None:
    """A genuine query error (Cypher typo, missing graph, teardown) is not
    retried — it raises on the first attempt and logs once."""
    query = _set_query(driver, ValueError("syntax error near RETRN"))

    with patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep, patch.object(
        fdb, "_UPSTREAM_QUERY_LOGGER"
    ) as upstream_logger:
        with pytest.raises(ValueError):
            await driver.execute_query("MATCH (n) RETRN 1")

    assert query.await_count == 1
    sleep.assert_not_awaited()
    upstream_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_execute_query_already_indexed_returns_none_without_retry(
    driver: AutoGPTFalkorDriver,
) -> None:
    """The upstream 'already indexed' short-circuit is preserved: returns None,
    no retry, no terminal error."""
    query = _set_query(driver, Exception("Index already indexed"))

    with patch.object(fdb.asyncio, "sleep", new=AsyncMock()) as sleep, patch.object(
        fdb, "_UPSTREAM_QUERY_LOGGER"
    ) as upstream_logger:
        result = await driver.execute_query("CREATE INDEX ...")

    assert result is None
    assert query.await_count == 1
    sleep.assert_not_awaited()
    upstream_logger.error.assert_not_called()
