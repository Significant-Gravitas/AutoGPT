"""Page through a scope's live facts, newest first, with their total.

Backs the Settings → Memory fact list (``api/features/memory/routes.py``).
A fact is live by recall's own test (``recall.live_fact_predicate``), so the
list and its total hold exactly the facts the assistant can still recall.

Pages are keyset pages: the cursor is the sort key of the last fact a page
returned, so a fact forgotten while someone pages through the list does not
shift the facts after it out of the next page, as an offset would. A cursor
also names its scope, by digest, and is refused in any other: it could only
move the caller's own page boundary there (the query always runs on the
caller's graph), so it needs no signature, but it is not that list's cursor.
"""

import base64
import hashlib
import json
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, TypeAdapter
from redis.exceptions import ResponseError

from .falkordb_driver import open_driver
from .recall import live_fact_predicate
from .scope import MemoryScope

_MISSING_GRAPH_MARKERS = ("no such graph", "does not exist", "invalid graph")

# Every live fact the list can show; the cursor clause, when there is one,
# goes between this and the ``RETURN``. A fact stored without ``created_at``
# sorts last instead of breaking the comparison.
_LIVE_FACTS = f"""
MATCH (src:Entity)-[e:RELATES_TO]->(tgt:Entity)
WHERE e.group_id = $g AND {live_fact_predicate("e")}
WITH src, e, tgt, coalesce(toString(e.created_at), '') AS sort_key
"""

_AFTER_CURSOR = """
WHERE sort_key < $after_key OR (sort_key = $after_key AND e.uuid < $after_uuid)
"""

_PAGE = """
RETURN e.uuid AS uuid,
       e.fact AS fact,
       e.name AS name,
       src.name AS source,
       tgt.name AS target,
       toString(e.created_at) AS created_at,
       sort_key
ORDER BY sort_key DESC, e.uuid DESC
LIMIT $fetch
"""

_TOTAL = _LIVE_FACTS + "RETURN count(e) AS c"

_CURSOR = TypeAdapter(tuple[str, str, str])


class InvalidFactCursorError(ValueError):
    """The cursor is not one this module wrote."""


class LiveFact(BaseModel):
    uuid: str
    fact: str | None = None
    name: str | None = None
    source: str
    target: str
    created_at: str | None = None


class LiveFactPage(BaseModel):
    """One page of live facts. ``total`` counts every live fact in the scope,
    and ``next_cursor`` is ``None`` on the last page."""

    items: list[LiveFact]
    total: int
    next_cursor: str | None = None


async def list_live_facts(
    scope: MemoryScope, *, limit: int, cursor: str | None = None
) -> LiveFactPage:
    """The ``limit`` newest live facts after ``cursor``, and the scope's total.

    Raises ``InvalidFactCursorError`` for a cursor this module did not write.
    A scope whose graph does not exist yet has no facts.
    """
    after = _decode_cursor(cursor, scope) if cursor else None
    query = _LIVE_FACTS + (_AFTER_CURSOR if after else "") + _PAGE
    params: dict[str, object] = {"g": scope.group_id, "fetch": limit + 1}
    if after:
        params["after_key"], params["after_uuid"] = after

    driver = open_driver(scope)
    try:
        rows = await _rows(driver.execute_query(query, **params))
        total_rows = await _rows(driver.execute_query(_TOTAL, g=scope.group_id))
    finally:
        await driver.close()

    page, more = rows[:limit], len(rows) > limit
    last = page[-1] if page else None
    return LiveFactPage(
        items=[_to_fact(row) for row in page],
        total=int(total_rows[0]["c"]) if total_rows else 0,
        next_cursor=(
            _encode_cursor(scope, str(last["sort_key"]), str(last["uuid"]))
            if more and last
            else None
        ),
    )


async def _rows(pending: Awaitable[Any]) -> list[dict]:
    try:
        result = await pending
    except ResponseError as exc:
        message = str(exc).lower()
        if any(marker in message for marker in _MISSING_GRAPH_MARKERS):
            return []
        raise
    return result[0] if result else []


def _to_fact(row: dict) -> LiveFact:
    return LiveFact(
        uuid=str(row.get("uuid", "")),
        fact=row.get("fact"),
        name=row.get("name"),
        source=str(row.get("source") or ""),
        target=str(row.get("target") or ""),
        created_at=row.get("created_at"),
    )


def _encode_cursor(scope: MemoryScope, sort_key: str, uuid: str) -> str:
    raw = json.dumps([_scope_tag(scope), sort_key, uuid], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, scope: MemoryScope) -> tuple[str, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        tag, sort_key, uuid = _CURSOR.validate_json(raw)
    except ValueError as exc:  # bad base64 or JSON, a ValidationError too
        raise InvalidFactCursorError("Invalid cursor") from exc
    if tag != _scope_tag(scope) or not uuid:
        raise InvalidFactCursorError("Invalid cursor")
    return sort_key, uuid


def _scope_tag(scope: MemoryScope) -> str:
    return hashlib.sha256(scope.group_id.encode()).hexdigest()[:16]
