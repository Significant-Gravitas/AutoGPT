from unittest.mock import MagicMock

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
