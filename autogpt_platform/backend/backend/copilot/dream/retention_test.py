"""The DreamPass retention job over the in-memory store: closed rows past
their retention go, a batch at a time, open rows never; and a run is bounded
and logged."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from prisma.enums import DreamPassRoute, DreamPassStatus, DreamPassTrigger

from backend.data.dream_pass_models import DreamPassDraft

from . import retention as retention_mod
from .retention import delete_expired_records

_NOW = datetime.now(timezone.utc)


class TestRetention:
    async def test_deletes_closed_passes_past_their_retention_and_keeps_the_rest(
        self, fake_dream_db, caplog
    ):
        _seed(fake_dream_db, "old-complete", DreamPassStatus.COMPLETE, days=91)
        _seed(fake_dream_db, "old-cancelled", DreamPassStatus.CANCELLED, days=120)
        _seed(fake_dream_db, "old-but-open", DreamPassStatus.SUBMITTED, days=200)
        _seed(fake_dream_db, "recent", DreamPassStatus.COMPLETE, days=10)

        with caplog.at_level(logging.INFO):
            deleted = await delete_expired_records(90, now=_NOW)

        assert deleted == 2
        assert sorted(fake_dream_db.rows) == ["old-but-open", "recent"]
        assert "Dream pass retention: deleted 2 closed pass(es) created before" in (
            caplog.text
        )

    async def test_goes_a_batch_at_a_time_until_a_batch_comes_back_short(
        self, mocker, fake_dream_db
    ):
        mocker.patch.object(retention_mod, "RETENTION_BATCH_SIZE", 2)
        delete = mocker.spy(fake_dream_db, "delete_old_dream_passes")
        for n in range(5):
            _seed(fake_dream_db, f"p{n}", DreamPassStatus.ERRORED, days=100)

        assert await delete_expired_records(90, now=_NOW) == 5

        assert [call.kwargs["limit"] for call in delete.call_args_list] == [2, 2, 2]
        assert fake_dream_db.rows == {}

    async def test_stops_after_its_batch_cap_and_leaves_the_rest_for_next_week(
        self, mocker, fake_dream_db
    ):
        mocker.patch.object(retention_mod, "RETENTION_BATCH_SIZE", 1)
        mocker.patch.object(retention_mod, "RETENTION_MAX_BATCHES", 2)
        for n in range(3):
            _seed(fake_dream_db, f"p{n}", DreamPassStatus.SKIPPED, days=100)

        assert await delete_expired_records(90, now=_NOW) == 2

        assert len(fake_dream_db.rows) == 1

    async def test_a_store_that_fails_ends_the_run_logged_not_raised(
        self, fake_dream_db, caplog
    ):
        fake_dream_db.fail = True

        with caplog.at_level(logging.WARNING):
            assert await delete_expired_records(90, now=_NOW) == 0

        assert "Dream pass retention stopped after 0 deleted" in caplog.text

    async def test_a_run_never_outlasts_its_budget(self, mocker):
        async def stalled(*_args, **_kwargs) -> int:
            await asyncio.Event().wait()
            return 0

        mocker.patch.object(retention_mod, "RETENTION_BUDGET_SECONDS", 0.2)
        mocker.patch.object(
            retention_mod, "delete_old_passes", AsyncMock(side_effect=stalled)
        )
        loop = asyncio.get_running_loop()

        started = loop.time()
        assert await asyncio.wait_for(delete_expired_records(90, now=_NOW), 10) == 0

        assert loop.time() - started < 5


def _seed(fake_dream_db, pass_id: str, status: DreamPassStatus, *, days: int) -> None:
    fake_dream_db.seed(
        DreamPassDraft(
            id=pass_id,
            user_id="u1",
            scope_key="u1",
            route=DreamPassRoute.SYNC,
            trigger=DreamPassTrigger.CRON,
            status=status,
        ),
        created_at=_NOW - timedelta(days=days),
    )
