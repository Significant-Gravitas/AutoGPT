"""Unit tests for ``fact_list.py``; the paging itself runs against a live
FalkorDB in ``fact_list_integration_test.py``."""

from unittest.mock import AsyncMock, patch

import pytest
from redis.exceptions import ResponseError

from .fact_list import (
    InvalidFactCursorError,
    _decode_cursor,
    _encode_cursor,
    list_live_facts,
)
from .recall import live_fact_predicate
from .scope import MemoryScope

_SCOPE = MemoryScope.for_user("user-1")
_OTHER_SCOPE = MemoryScope.for_user("user-2")


def _row(uuid: str, created_at: str = "2026-10-23T12:00:00+00:00") -> dict:
    return {
        "uuid": uuid,
        "fact": f"fact {uuid}",
        "name": "knows",
        "source": "User",
        "target": "Thing",
        "created_at": created_at,
        "sort_key": created_at,
    }


def _driver(*results) -> AsyncMock:
    driver = AsyncMock()
    driver.execute_query.side_effect = [
        r if isinstance(r, Exception) else (r, None, None) for r in results
    ]
    return driver


async def _list(driver: AsyncMock, **kwargs):
    with patch("backend.copilot.graphiti.fact_list.open_driver", return_value=driver):
        return await list_live_facts(_SCOPE, **kwargs)


async def test_first_page_lists_live_facts_newest_first_with_the_total() -> None:
    driver = _driver([_row("a"), _row("b")], [{"c": 2}])

    page = await _list(driver, limit=6)

    assert [fact.uuid for fact in page.items] == ["a", "b"]
    assert page.total == 2
    assert page.next_cursor is None
    query, params = driver.execute_query.await_args_list[0].args[0], (
        driver.execute_query.await_args_list[0].kwargs
    )
    assert live_fact_predicate("e") in query
    assert "ORDER BY sort_key DESC, e.uuid DESC" in query
    assert "$after_key" not in query
    assert params == {"g": _SCOPE.group_id, "fetch": 7}
    total_query = driver.execute_query.await_args_list[1].args[0]
    assert live_fact_predicate("e") in total_query
    driver.close.assert_awaited_once()


async def test_a_page_with_more_after_it_hands_out_the_last_facts_key() -> None:
    rows = [_row(u, f"2026-10-23T12:0{i}:00+00:00") for i, u in enumerate("abc")]
    driver = _driver(rows, [{"c": 19}])

    page = await _list(driver, limit=2)

    assert [fact.uuid for fact in page.items] == ["a", "b"]
    assert page.next_cursor is not None
    assert _decode_cursor(page.next_cursor, _SCOPE) == (
        "2026-10-23T12:01:00+00:00",
        "b",
    )


async def test_the_cursor_starts_the_page_after_its_fact() -> None:
    driver = _driver([_row("c")], [{"c": 3}])
    cursor = _encode_cursor(_SCOPE, "2026-10-23T12:01:00+00:00", "b")

    await _list(driver, limit=2, cursor=cursor)

    query = driver.execute_query.await_args_list[0].args[0]
    params = driver.execute_query.await_args_list[0].kwargs
    assert "sort_key < $after_key" in query
    assert "e.uuid < $after_uuid" in query
    assert params["after_key"] == "2026-10-23T12:01:00+00:00"
    assert params["after_uuid"] == "b"


@pytest.mark.parametrize(
    "cursor",
    [
        "not base64!",
        "bm90IGpzb24",
        _encode_cursor(_SCOPE, "x", "b")[:-2],
        "W10",
        _encode_cursor(_OTHER_SCOPE, "2026-10-23", "b"),
    ],
)
async def test_a_cursor_it_did_not_write_is_rejected(cursor: str) -> None:
    driver = _driver()
    with pytest.raises(InvalidFactCursorError):
        await _list(driver, limit=2, cursor=cursor)
    driver.execute_query.assert_not_awaited()


def test_a_cursor_from_another_scope_is_rejected() -> None:
    """Another scope's cursor would only move the caller's own page
    boundary (the query is always on the caller's graph), but it is not a
    cursor for this list, so it is refused rather than misapplied."""
    cursor = _encode_cursor(_OTHER_SCOPE, "2026-10-23", "b")
    with pytest.raises(InvalidFactCursorError):
        _decode_cursor(cursor, _SCOPE)
    assert _decode_cursor(cursor, _OTHER_SCOPE) == ("2026-10-23", "b")


def test_a_cursor_with_no_fact_uuid_is_rejected() -> None:
    with pytest.raises(InvalidFactCursorError):
        _decode_cursor(_encode_cursor(_SCOPE, "2026-10-23", ""), _SCOPE)


async def test_a_scope_without_a_graph_has_no_facts() -> None:
    driver = _driver(ResponseError("no such graph"), ResponseError("no such graph"))

    page = await _list(driver, limit=6)

    assert page.items == []
    assert page.total == 0
    assert page.next_cursor is None


async def test_any_other_graph_error_is_raised() -> None:
    driver = _driver(ResponseError("connection reset"))

    with pytest.raises(ResponseError):
        await _list(driver, limit=6)
    driver.close.assert_awaited_once()
