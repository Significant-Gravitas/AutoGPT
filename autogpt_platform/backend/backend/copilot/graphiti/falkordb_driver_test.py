from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .falkordb_driver import AutoGPTFalkorDriver


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
