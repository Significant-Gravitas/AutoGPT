"""Which engine a load-test account's chat turns run on."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import load_test
from backend.copilot.load_test import engine_for_email, load_test_engine

DOMAIN = "previews.agpt.co"


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("loadtest-dummy-ab12-3@previews.agpt.co", "dummy"),
        ("loadtest-scripted-ab12-0@previews.agpt.co", "scripted"),
        ("LoadTest-Dummy-AB12-3@Previews.AGPT.co", "dummy"),
        # A load-test account that measures the real model.
        ("loadtest-ab12-3@previews.agpt.co", None),
        ("loadtest-dummy-ab12-3@agpt.co", None),
        ("loadtest-dummy-ab12-3@evil-previews.agpt.co", None),
        ("someone@previews.agpt.co", None),
        ("loadtest-dummy-ab12-3", None),
    ],
)
def test_the_address_picks_the_engine(email: str, expected: str | None) -> None:
    assert engine_for_email(email, DOMAIN) == expected


def test_without_a_domain_no_address_is_routed() -> None:
    assert engine_for_email("loadtest-dummy-ab12-3@previews.agpt.co", "") is None


def _config(domain: str) -> MagicMock:
    return MagicMock(load_test_email_domain=domain)


@pytest.mark.asyncio
async def test_with_the_routing_off_no_email_is_looked_up() -> None:
    lookup = AsyncMock()
    with patch.object(load_test, "_email_for", lookup):
        assert await load_test_engine("user-1", _config("")) is None
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_load_test_account_gets_the_engine_its_address_names() -> None:
    lookup = AsyncMock(return_value="loadtest-scripted-ab12-0@previews.agpt.co")
    with patch.object(load_test, "_email_for", lookup):
        assert await load_test_engine("user-1", _config(DOMAIN)) == "scripted"
    lookup.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_a_person_on_the_same_preview_keeps_their_engine() -> None:
    lookup = AsyncMock(return_value="someone@previews.agpt.co")
    with patch.object(load_test, "_email_for", lookup):
        assert await load_test_engine("user-1", _config(DOMAIN)) is None


@pytest.mark.asyncio
async def test_an_account_whose_email_cannot_be_found_keeps_its_engine() -> None:
    lookup = AsyncMock(side_effect=LookupError("No email for user user-1"))
    with patch.object(load_test, "_email_for", lookup):
        assert await load_test_engine("user-1", _config(DOMAIN)) is None


@pytest.mark.asyncio
async def test_a_failed_email_lookup_fails_the_turn_rather_than_route_it() -> None:
    # Returning None here would hand a load-test account to the real model.
    lookup = AsyncMock(side_effect=ConnectionError("user DB unavailable"))
    with patch.object(load_test, "_email_for", lookup):
        with pytest.raises(ConnectionError):
            await load_test_engine("user-1", _config(DOMAIN))


@pytest.mark.asyncio
async def test_a_turn_without_a_user_keeps_its_engine() -> None:
    lookup = AsyncMock()
    with patch.object(load_test, "_email_for", lookup):
        assert await load_test_engine(None, _config(DOMAIN)) is None
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_email_lookup_that_finds_nothing_is_tried_again() -> None:
    load_test._email_for.cache_clear()
    db = MagicMock()
    db.get_auth_user_flag_fields = AsyncMock(
        side_effect=[None, MagicMock(email="loadtest-dummy-ab12-3@previews.agpt.co")]
    )
    try:
        with patch.object(load_test, "user_db", return_value=db):
            with pytest.raises(LookupError):
                await load_test._email_for("user-2")
            assert (
                await load_test._email_for("user-2")
                == "loadtest-dummy-ab12-3@previews.agpt.co"
            )
            # Found once, then cached.
            await load_test._email_for("user-2")
        assert db.get_auth_user_flag_fields.await_count == 2
    finally:
        load_test._email_for.cache_clear()
