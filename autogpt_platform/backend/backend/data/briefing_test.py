"""Integration tests for the briefing data layer."""

import logging
from datetime import date, timedelta
from uuid import uuid4

import pytest
from prisma.errors import UniqueViolationError
from prisma.models import User, UserBriefing

from backend.data import briefing
from backend.util.test import SpinTestServer

logger = logging.getLogger(__name__)


async def _create_user(user_id: str) -> None:
    try:
        await User.prisma().create(
            data={
                "id": user_id,
                "email": f"briefing-{user_id}@example.com",
                "name": "Briefing Test",
            }
        )
    except UniqueViolationError:
        pass


async def _cleanup(user_id: str) -> None:
    try:
        await UserBriefing.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete_many(where={"id": user_id})
    except Exception as exc:
        logger.warning("cleanup for %s failed: %s", user_id, exc)


@pytest.mark.asyncio(loop_scope="session")
async def test_create_and_get_briefing(server: SpinTestServer):
    user_id = f"briefing-{uuid4()}"
    await _create_user(user_id)
    try:
        content = {"run_items": [], "decision_items": [{"title": "Approve email"}]}

        record = await briefing.create_briefing(user_id, date(2026, 8, 7), content)
        assert record.user_id == user_id
        assert record.content["decision_items"][0]["title"] == "Approve email"

        again = await briefing.create_briefing(
            user_id, date(2026, 8, 7), {"other": True}
        )
        assert again.id == record.id  # idempotent per (user, date)

        assert (
            await briefing.get_briefing_for_date(user_id, date(2026, 8, 7))
        ).id == record.id
        assert (await briefing.get_briefing_for_date(user_id, date(2026, 8, 8))) is None
        assert (await briefing.get_latest_briefings(user_id))[0].id == record.id
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_briefings_order_by_briefing_date_not_created_at(
    server: SpinTestServer,
):
    """A backfill/retry can write an earlier date's briefing after a later
    date's already exists; "latest" must mean the latest date covered, not
    insertion order."""
    user_id = f"briefing-order-{uuid4()}"
    await _create_user(user_id)
    try:
        later = await briefing.create_briefing(
            user_id, date(2026, 8, 10), {"which": "later-date"}
        )
        earlier = await briefing.create_briefing(
            user_id, date(2026, 8, 5), {"which": "earlier-date-inserted-second"}
        )
        # Force the out-of-order insertion the test depends on: createdAt has
        # millisecond precision, so back-to-back inserts can tie and make an
        # `earlier.created_at > later.created_at` assertion flaky.
        await UserBriefing.prisma().update(
            where={"id": earlier.id},
            data={"createdAt": later.created_at + timedelta(seconds=1)},
        )

        latest = await briefing.get_latest_briefings(user_id)
        assert [record.id for record in latest] == [later.id, earlier.id]
        assert latest[0].briefing_date == date(2026, 8, 10)
    finally:
        await _cleanup(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_mark_briefing_delivered_is_user_scoped(server: SpinTestServer):
    user_id = f"briefing-delivered-{uuid4()}"
    await _create_user(user_id)
    try:
        record = await briefing.create_briefing(user_id, date(2026, 8, 7), {"k": "v"})
        assert record.delivered_at is None

        await briefing.mark_briefing_delivered(f"not-{user_id}", record.id)
        undelivered = await briefing.get_briefing_for_date(user_id, date(2026, 8, 7))
        assert undelivered is not None
        assert undelivered.delivered_at is None  # foreign user can't mark it

        await briefing.mark_briefing_delivered(user_id, record.id)
        delivered = await briefing.get_briefing_for_date(user_id, date(2026, 8, 7))
        assert delivered is not None
        assert delivered.delivered_at is not None
    finally:
        await _cleanup(user_id)
