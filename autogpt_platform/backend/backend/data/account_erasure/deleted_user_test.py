"""The placeholder's handle: anyone can pick `deleted-user` and its
fallbacks in their profile, so the first free one is used, never one taken
(a unique violation would roll back the hand-over and stop every erasure
that needs it)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_erasure import deleted_user
from backend.data.account_erasure.deleted_user import (
    DELETED_USER_ID,
    DELETED_USER_USERNAME,
)

FALLBACK = f"{DELETED_USER_USERNAME}-{DELETED_USER_ID[:8]}"


def _taken(monkeypatch, *usernames: str) -> AsyncMock:
    find_many = AsyncMock(
        return_value=[SimpleNamespace(username=name) for name in usernames]
    )
    profiles = MagicMock()
    profiles.prisma.return_value.find_many = find_many
    monkeypatch.setattr(deleted_user, "Profile", profiles)
    return find_many


@pytest.mark.asyncio
async def test_the_handle_is_deleted_user_while_it_is_free(monkeypatch):
    _taken(monkeypatch)

    assert await deleted_user._free_username(MagicMock()) == DELETED_USER_USERNAME


@pytest.mark.asyncio
async def test_a_taken_handle_falls_back_to_the_id_one(monkeypatch):
    _taken(monkeypatch, DELETED_USER_USERNAME)

    assert await deleted_user._free_username(MagicMock()) == FALLBACK


@pytest.mark.asyncio
async def test_both_taken_handles_fall_back_to_the_next_free_one(monkeypatch):
    find_many = _taken(monkeypatch, DELETED_USER_USERNAME, FALLBACK)

    username = await deleted_user._free_username(MagicMock())

    assert username == f"{DELETED_USER_USERNAME}-2"
    # One query for every candidate, not a lookup per handle.
    find_many.assert_awaited_once()
    candidates = find_many.await_args.kwargs["where"]["username"]["in"]
    assert candidates[:3] == [DELETED_USER_USERNAME, FALLBACK, username]


@pytest.mark.asyncio
async def test_every_candidate_taken_still_gives_a_new_handle(monkeypatch):
    find_many = _taken(monkeypatch)
    candidates: list[str] = []

    async def every_one_taken(*, where):
        candidates.extend(where["username"]["in"])
        return [SimpleNamespace(username=name) for name in candidates]

    find_many.side_effect = every_one_taken

    username = await deleted_user._free_username(MagicMock())

    assert username.startswith(f"{DELETED_USER_USERNAME}-")
    assert username not in candidates
