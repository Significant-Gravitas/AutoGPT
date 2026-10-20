"""The volume knob's daily ceiling.

`daily_limit` was persisted and returned by the API but nothing consumed it,
so the number the settings page showed had no effect — and the `0` that a
one-click unsubscribe writes did nothing at all.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.notifications.dedupe import claim_daily_send

DAY = date(2026, 8, 3)


def _redis(counts=None):
    """A Redis stand-in whose INCR actually counts."""
    state = {"n": 0}
    counts = counts if counts is not None else state

    async def incr(_key):
        counts["n"] += 1
        return counts["n"]

    return SimpleNamespace(incr=incr, expire=AsyncMock())


@pytest.mark.asyncio
async def test_sends_are_allowed_up_to_the_limit_then_refused():
    client = _redis()
    with patch(
        "backend.notifications.dedupe.get_redis_async", AsyncMock(return_value=client)
    ):
        allowed = [await claim_daily_send("u1", 3, DAY) for _ in range(5)]

    assert allowed == [True, True, True, False, False]


@pytest.mark.asyncio
async def test_a_limit_of_zero_sends_nothing():
    """This is what one-click unsubscribe writes, so it has to mean it."""
    client = _redis()
    with patch(
        "backend.notifications.dedupe.get_redis_async", AsyncMock(return_value=client)
    ):
        assert await claim_daily_send("u1", 0, DAY) is False


@pytest.mark.asyncio
async def test_each_day_gets_its_own_allowance():
    keys = []

    async def incr(key):
        keys.append(key)
        return 1

    client = SimpleNamespace(incr=incr, expire=AsyncMock())
    with patch(
        "backend.notifications.dedupe.get_redis_async", AsyncMock(return_value=client)
    ):
        await claim_daily_send("u1", 3, DAY)
        await claim_daily_send("u1", 3, date(2026, 8, 4))

    assert len(set(keys)) == 2


@pytest.mark.asyncio
async def test_the_counter_expires_so_it_never_needs_resetting():
    client = _redis()
    with patch(
        "backend.notifications.dedupe.get_redis_async", AsyncMock(return_value=client)
    ):
        await claim_daily_send("u1", 3, DAY)

    client.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_redis_outage_lets_the_email_through():
    """Fails open, like `claim_once`: a rare extra email beats silently
    swallowing someone's briefing because a counter was unreachable."""
    with patch(
        "backend.notifications.dedupe.get_redis_async",
        AsyncMock(side_effect=RuntimeError("redis down")),
    ):
        assert await claim_daily_send("u1", 3, DAY) is True


@pytest.mark.asyncio
async def test_service_mail_is_never_capped():
    """Billing messages ignore the volume knob for the same reason they ignore
    every other preference: a payment failure has to arrive."""
    from prisma.enums import NotificationType

    from backend.notifications.preferences import SERVICE_MESSAGES

    assert NotificationType.PAYMENT_FAILED in SERVICE_MESSAGES
    assert NotificationType.SUBSCRIPTION_ENDED in SERVICE_MESSAGES
    # ...and the product families are not exempt.
    assert NotificationType.BRIEFING not in SERVICE_MESSAGES
    assert NotificationType.ALERT not in SERVICE_MESSAGES
    assert NotificationType.VERDICT not in SERVICE_MESSAGES


def test_zero_survives_the_read_back_from_the_database():
    """`maxEmailsPerDay or 3` turned the unsubscribe's 0 into 3.

    The column is non-nullable with its own default, so the coalesce could only
    ever catch the one value that carries meaning.
    """
    import inspect

    from backend.data import user as user_db

    source = inspect.getsource(user_db)
    assert "maxEmailsPerDay or 3" not in source
