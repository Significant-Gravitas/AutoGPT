"""The gate for background work: a banned or a closed account gets none."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.data import user as user_module
from backend.data.user import forget_refusal, is_user_refused


@pytest.fixture
def lookups(mocker: MockerFixture) -> MagicMock:
    parent = MagicMock()
    parent.attach_mock(
        mocker.patch.object(user_module, "is_user_banned", AsyncMock()), "banned"
    )
    parent.attach_mock(
        mocker.patch.object(user_module, "is_user_account_closed", AsyncMock()),
        "closed",
    )
    return parent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "banned, closed, refused",
    [(True, False, True), (False, True, True), (False, False, False)],
)
async def test_a_banned_or_closed_account_is_refused(
    lookups: MagicMock, banned: bool, closed: bool, refused: bool
):
    lookups.banned.return_value = banned
    lookups.closed.return_value = closed

    assert await is_user_refused("user-1") is refused


@pytest.mark.asyncio
async def test_a_ban_answers_without_the_closed_lookup(lookups: MagicMock):
    lookups.banned.return_value = True

    assert await is_user_refused("user-1") is True
    lookups.closed.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_lookup_propagates(lookups: MagicMock):
    """Each gate decides what an unknown means."""
    lookups.banned.return_value = False
    lookups.closed.side_effect = ConnectionError("db manager down")

    with pytest.raises(ConnectionError):
        await is_user_refused("user-1")


def test_forget_refusal_drops_the_cached_closed_answer(mocker: MockerFixture):
    drop = mocker.patch.object(user_module.is_user_account_closed, "cache_delete")

    forget_refusal("user-1")

    drop.assert_called_once_with("user-1")


def test_forget_refusal_never_raises(mocker: MockerFixture):
    mocker.patch.object(
        user_module.is_user_account_closed,
        "cache_delete",
        side_effect=ConnectionError("redis down"),
    )

    forget_refusal("user-1")
