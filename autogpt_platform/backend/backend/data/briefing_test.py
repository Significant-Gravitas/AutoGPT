"""Integration tests for the briefing data layer."""

import logging
from datetime import date
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
        assert (await briefing.get_latest_briefing(user_id)).id == record.id
    finally:
        await _cleanup(user_id)
