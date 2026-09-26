"""The guard against overlapping dream passes, with the store mocked the way
``store_test.py`` mocks it: which open row skips a pass, which one it expires
and goes on past, and that neither the store nor the flag service can hold a
pass up."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import (
    DreamPassOperations,
    DreamPassRecord,
    DreamPhaseOutputs,
)
from backend.util.feature_flag import Flag

from . import guard, store
from .locks import DEFAULT_LOCK_TTL_SECONDS
from .pass_run import DreamPassRun, PassEnded

_NOW = datetime.now(timezone.utc)
_SCOPE = MemoryScope.for_user("u1")
_STALE = _NOW - timedelta(seconds=DEFAULT_LOCK_TTL_SECONDS + 60)


@pytest.fixture
def db(mocker) -> MagicMock:
    """The accessor's target: no open pass, and every write landing."""
    database = MagicMock()
    database.list_open_dream_passes = AsyncMock(return_value=[])
    database.update_dream_pass = AsyncMock(return_value=True)
    database.get_dream_pass = AsyncMock(return_value=None)
    mocker.patch.object(store, "dream_db", return_value=database)
    return database


async def _hang(*_args) -> None:
    await asyncio.Event().wait()


def _run(*, force: bool = False) -> DreamPassRun:
    return DreamPassRun.begin("u1", "sync_baseline", force=force)


def _row(pass_id: str = "other", **overrides) -> DreamPassRecord:
    """Another pass of the scope, open and written a minute ago."""
    fields = {
        "id": pass_id,
        "user_id": "u1",
        "expert_id": None,
        "scope_key": _SCOPE.scope_key,
        "route": DreamPassRoute.SYNC,
        "trigger": DreamPassTrigger.CRON,
        "phase": DreamPassPhase.RECOMBINE,
        "status": DreamPassStatus.RUNNING,
        "skip_reason": None,
        "cancel_generation": 0,
        "provider_batch_id": None,
        "lease_token": None,
        "lease_expires_at": None,
        "input_bundle": None,
        "phase_outputs": DreamPhaseOutputs(),
        "operations": DreamPassOperations(),
        "usage": None,
        "window_start": None,
        "window_end": None,
        "error": None,
        "created_at": _NOW - timedelta(minutes=5),
        "started_at": _NOW - timedelta(minutes=5),
        "submitted_at": None,
        "applied_at": None,
        "completed_at": None,
        "updated_at": _NOW - timedelta(minutes=1),
    }
    return DreamPassRecord(**{**fields, **overrides})


async def _skip_reason(run: DreamPassRun) -> str | None:
    with pytest.raises(PassEnded) as ended:
        await guard.guard_dream_pass(run, _SCOPE)
    assert ended.value.result.skipped is True
    assert ended.value.result.pass_id == run.pass_id
    return ended.value.result.skip_reason


class TestOpenPasses:
    async def test_a_scope_with_only_the_pass_own_row_goes_on(self, db):
        run = _run()
        db.list_open_dream_passes.return_value = [_row(run.pass_id)]

        await guard.guard_dream_pass(run, _SCOPE)

        db.list_open_dream_passes.assert_awaited_once_with(_SCOPE.scope_key)
        db.update_dream_pass.assert_not_awaited()

    async def test_a_fresh_open_pass_skips_this_one(self, db, caplog):
        run = _run()
        db.list_open_dream_passes.return_value = [_row(run.pass_id), _row()]

        with caplog.at_level(logging.INFO, logger=guard.logger.name):
            assert await _skip_reason(run) == "pass_in_progress"

        db.update_dream_pass.assert_not_awaited()
        [record] = [r for r in caplog.records if "in progress" in r.getMessage()]
        assert record.levelno == logging.INFO
        assert run.pass_id in record.getMessage() and "other" in record.getMessage()

    async def test_a_leased_pass_is_fresh_until_its_lease_lapses(self, db):
        """Written a day ago, but its lease still runs: a batch pass waiting
        on its provider."""
        db.list_open_dream_passes.return_value = [
            _row(
                status=DreamPassStatus.SUBMITTED,
                updated_at=_NOW - timedelta(days=1),
                lease_expires_at=_NOW + timedelta(hours=1),
            )
        ]

        assert await _skip_reason(_run()) == "pass_in_progress"
        db.update_dream_pass.assert_not_awaited()

    async def test_a_stale_pass_is_expired_and_this_one_goes_on(self, db, caplog):
        run = _run()
        stale = _row(updated_at=_STALE)
        db.list_open_dream_passes.return_value = [stale]

        with caplog.at_level(logging.WARNING, logger=guard.logger.name):
            await guard.guard_dream_pass(run, _SCOPE)

        pass_id, update = db.update_dream_pass.await_args.args
        assert pass_id == "other"
        assert update.status is DreamPassStatus.EXPIRED
        assert update.bump_cancel_generation is True
        assert update.not_updated_since == stale.updated_at
        assert update.owner_user_id is None
        assert update.completed_at is not None
        assert update.error is not None
        assert "no progress since" in update.error and run.pass_id in update.error
        [record] = [r for r in caplog.records if "expired pass other" in r.getMessage()]
        assert record.levelno == logging.WARNING

    async def test_a_lapsed_lease_is_stale_however_recently_written(self, db):
        leased = _row(
            status=DreamPassStatus.SUBMITTED,
            updated_at=_NOW - timedelta(minutes=1),
            lease_expires_at=_NOW - timedelta(minutes=1),
        )
        db.list_open_dream_passes.return_value = [leased]

        await guard.guard_dream_pass(_run(), _SCOPE)

        _, update = db.update_dream_pass.await_args.args
        assert update.status is DreamPassStatus.EXPIRED
        assert update.not_updated_since == leased.updated_at
        assert update.error is not None and "lease lapsed" in update.error

    async def test_a_stale_pass_that_moved_before_its_expiry_still_blocks(self, db):
        """The expiry is refused because the row was written after the guard
        read it: that pass is alive."""
        stale = _row(updated_at=_STALE)
        db.list_open_dream_passes.return_value = [stale]
        db.update_dream_pass.return_value = False
        db.get_dream_pass.return_value = stale.model_copy(
            update={"updated_at": _NOW, "phase": DreamPassPhase.SANITIZE}
        )

        assert await _skip_reason(_run()) == "pass_in_progress"
        db.get_dream_pass.assert_awaited_once_with("other")

    async def test_a_stale_pass_that_ended_meanwhile_does_not_block(self, db):
        stale = _row(updated_at=_STALE)
        db.list_open_dream_passes.return_value = [stale]
        db.update_dream_pass.return_value = False
        db.get_dream_pass.return_value = stale.model_copy(
            update={"status": DreamPassStatus.COMPLETE}
        )

        await guard.guard_dream_pass(_run(), _SCOPE)

    async def test_force_expires_a_fresh_pass_whatever_wrote_it_last(self, db):
        run = _run(force=True)
        db.list_open_dream_passes.return_value = [_row()]

        await guard.guard_dream_pass(run, _SCOPE)

        pass_id, update = db.update_dream_pass.await_args.args
        assert pass_id == "other"
        assert update.status is DreamPassStatus.EXPIRED
        assert update.bump_cancel_generation is True
        assert update.not_updated_since is None
        assert update.error == f"forced by an admin-triggered dream pass {run.pass_id}"

    async def test_stale_passes_are_expired_before_a_fresh_one_skips(self, db):
        db.list_open_dream_passes.return_value = [
            _row("stale", updated_at=_STALE),
            _row("fresh"),
        ]

        assert await _skip_reason(_run()) == "pass_in_progress"

        [call] = db.update_dream_pass.await_args_list
        assert call.args[0] == "stale"


class TestTheStoreNeverHoldsThePassUp:
    async def test_a_read_that_runs_out_of_time_lets_the_pass_go_on(
        self, db, monkeypatch, caplog
    ):
        monkeypatch.setattr(store, "RECORD_WRITE_TIMEOUT_SECONDS", 0.05)
        db.list_open_dream_passes.side_effect = _hang

        with caplog.at_level(logging.WARNING, logger=guard.logger.name):
            await asyncio.wait_for(guard.guard_dream_pass(_run(), _SCOPE), 5)

        [record] = [r for r in caplog.records if "open passes" in r.getMessage()]
        assert record.levelno == logging.WARNING
        db.update_dream_pass.assert_not_awaited()

    async def test_a_read_that_fails_lets_the_pass_go_on(self, db):
        db.list_open_dream_passes.side_effect = ConnectionError("db down")

        await guard.guard_dream_pass(_run(), _SCOPE)

    async def test_an_expiry_that_runs_out_of_time_lets_the_pass_go_on(
        self, db, monkeypatch, caplog
    ):
        monkeypatch.setattr(store, "RECORD_WRITE_TIMEOUT_SECONDS", 0.05)
        db.list_open_dream_passes.return_value = [_row(updated_at=_STALE)]
        db.update_dream_pass.side_effect = _hang

        with caplog.at_level(logging.WARNING, logger=guard.logger.name):
            await asyncio.wait_for(guard.guard_dream_pass(_run(), _SCOPE), 5)

        assert "could not expire pass other" in caplog.text


class TestTheMasterFlag:
    async def test_the_flag_off_skips_the_pass_as_disabled(self, db, dream_pass_flag):
        dream_pass_flag.return_value = (False, True)
        run = _run()

        assert await _skip_reason(run) == "disabled"

        dream_pass_flag.assert_awaited_once_with(Flag.DREAM_PASS_ENABLED, "u1")
        db.list_open_dream_passes.assert_not_awaited()

    async def test_force_does_not_get_past_the_flag(self, db, dream_pass_flag):
        dream_pass_flag.return_value = (False, True)

        assert await _skip_reason(_run(force=True)) == "disabled"

    async def test_a_flag_read_without_an_answer_lets_the_pass_go_on(
        self, db, dream_pass_flag, caplog
    ):
        """The vendor fell back to the default: a blip, not an "off"."""
        dream_pass_flag.return_value = (False, False)

        with caplog.at_level(logging.WARNING, logger=guard.logger.name):
            await guard.guard_dream_pass(_run(), _SCOPE)

        assert "unanswered" in caplog.text
        db.list_open_dream_passes.assert_awaited_once()

    async def test_a_flag_read_that_runs_out_of_time_lets_the_pass_go_on(
        self, db, dream_pass_flag, monkeypatch, caplog
    ):
        monkeypatch.setattr(guard, "FLAG_READ_TIMEOUT_SECONDS", 0.05)
        dream_pass_flag.side_effect = _hang

        with caplog.at_level(logging.WARNING, logger=guard.logger.name):
            await asyncio.wait_for(guard.guard_dream_pass(_run(), _SCOPE), 5)

        assert "could not read dream-pass-enabled" in caplog.text
