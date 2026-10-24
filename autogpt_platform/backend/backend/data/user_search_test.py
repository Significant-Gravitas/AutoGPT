"""DB-backed tests for admin user search matching text literally (#15660)."""

import logging
from uuid import uuid4

import pytest
from prisma.models import User

from backend.data.user import search_users
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _seed(tag: str) -> dict[str, str]:
    users = {
        "first_last": f"first_last.{tag}@example.com",
        "firstxlast": f"firstxlast.{tag}@example.com",
        "dana": f"dana.{tag}@example.com",
    }
    ids: dict[str, str] = {}
    for key, email in users.items():
        user_id = f"search-{key}-{tag}"
        await User.prisma().create(
            data={
                "id": user_id,
                "email": email,
                "name": f"DANA SMITH {tag}" if key == "dana" else None,
            }
        )
        ids[key] = user_id
    return ids


async def _cleanup(tag: str) -> None:
    try:
        await User.prisma().delete_many(where={"id": {"endswith": f"-{tag}"}})
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        logger.warning("cleanup for %s failed: %s", tag, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_users_matches_underscore_and_percent_literally(
    server: SpinTestServer,
):
    tag = uuid4().hex[:12]
    ids = await _seed(tag)
    try:
        hits = {uid for uid, _ in await search_users(f"first_last.{tag}")}
        assert hits == {ids["first_last"]}

        # Still case-insensitive.
        hits = {uid for uid, _ in await search_users(f"FIRST_LAST.{tag}")}
        assert hits == {ids["first_last"]}

        # `%` is literal: no name contains "d%h <tag>".
        assert await search_users(f"d%h {tag}") == []
        assert await search_users(f"dana%{tag}") == []
    finally:
        await _cleanup(tag)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_users_wildcard_only_query_is_not_match_all(
    server: SpinTestServer,
):
    tag = uuid4().hex[:12]
    ids = await _seed(tag)
    try:
        for query in ("___", "%%%"):
            results = await search_users(query, limit=50)
            assert not {uid for uid, _ in results} & set(ids.values())
            # Anything returned must contain the text literally.
            if results:
                rows = await User.prisma().find_many(
                    where={"id": {"in": [uid for uid, _ in results]}}
                )
                for row in rows:
                    haystack = f"{row.email} {row.name or ''}"
                    assert query in haystack, (query, row.id)
    finally:
        await _cleanup(tag)
