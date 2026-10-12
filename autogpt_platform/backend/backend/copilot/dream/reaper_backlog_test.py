"""The reaper behind rows it cannot finish, over the in-memory store and
Redis: a row whose attempt hangs is cut after its own ``ROW_ATTEMPT_SECONDS``,
so the rows behind it get their turn in the same run, and a healthy cleanup
behind five hanging ones is charged in the first run; the open rows are
scanned from where the last run stopped, so one the reaper cannot close waits
for every other row before it is tried again, its row and lease untouched,
and a healthy row behind more stuck rows than a run can try, or than fit its
list, is closed; and a pass whose batch state lapsed before its landed phases
were charged is reported as lost usage, an error, rather than taken for one
with nothing to charge."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassRoute, DreamPassStatus

from backend.util.llm.providers import BatchResultRow

from . import cleanup as cleanup_mod
from . import locks as locks_mod
from . import provider_batch as provider_batch_mod
from . import reaper as reaper_mod
from . import store as store_mod
from .batch_state import state_key, write_phase_to_state
from .pass_record import cleanup_unfinished
from .reaper import (
    OPEN_CURSOR_KEY,
    OPEN_CURSOR_TTL_SECONDS,
    REAPER_ROW_LIMIT,
    reap_expired_passes,
)
from .reaper_cleanup_test import _scale_the_budget
from .reaper_test import _charged, _seed
from .schemas import ConsolidationOutput, RecombinationOutput

_MODELS = {p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")}


@pytest.fixture(autouse=True)
def phase_models(mocker) -> None:
    mocker.patch.object(reaper_mod, "phase_models_for_config", return_value=_MODELS)


@pytest.fixture(autouse=True)
def charges(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )


class TestARowWhoseAttemptHangs:
    async def test_leaves_the_rows_behind_it_their_turn_in_the_same_run(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Five pending cleanups whose state read never answers, written
        before a healthy one with a landed phase. Each hanging attempt is cut
        after its own few seconds, not the rest of the run's time, so the
        healthy cleanup is reached in the first run and charged once."""
        _scale_the_budget(mocker, budget=1.0, release=0.02, reserve=0.1)
        mocker.patch.object(reaper_mod, "ROW_ATTEMPT_SECONDS", 0.1)
        stuck = [f"stuck{index}" for index in range(5)]
        now = datetime.now(timezone.utc)
        for index, pass_id in enumerate([*stuck, "healthy"]):
            _seed(
                fake_dream_db,
                pass_id,
                route=DreamPassRoute.ANTHROPIC_BATCH,
                status=DreamPassStatus.EXPIRED,
                cleanup_pending_at=now - timedelta(hours=1),
            )
            fake_dream_db.rows[pass_id]["updated_at"] = now - timedelta(
                minutes=60 - index
            )
        await _consolidate_landed("healthy")
        tried = _hang_the_state_read_of(mocker, *stuck)

        run = await asyncio.wait_for(reap_expired_passes(), 5)

        assert tried == [*stuck, "healthy"]
        assert run.outcomes == {"out_of_budget": 5, "cleaned": 1}
        assert _charged(charges) == ["consolidate"]
        assert state_key("healthy") not in fake_dream_redis.hashes
        assert fake_dream_db.rows["healthy"]["cleanup_pending_at"] is None
        assert all(fake_dream_db.rows[p]["cleanup_pending_at"] for p in stuck)


class TestASlowCleanup:
    async def test_is_not_cut_for_the_time_its_move_to_the_back_took(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Every bound scaled to a tenth: the write that puts the cleanup
        behind the others takes 9 s of its 10, and the state read 4 s of its
        5. Neither runs over, and the write has a bound of its own, outside
        the 12 s attempt, so the attempt reaches the charge: one charge, the
        state gone, the mark cleared, in the first run."""
        _scale_every_bound(mocker, 0.1)
        _seed(
            fake_dream_db,
            "slow",
            route=DreamPassRoute.ANTHROPIC_BATCH,
            status=DreamPassStatus.EXPIRED,
            cleanup_pending_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await _consolidate_landed("slow")
        write = fake_dream_db.update_dream_pass
        read = cleanup_mod.read_state

        async def slow_write(pass_id, update) -> bool:
            await asyncio.sleep(0.9)
            return await write(pass_id, update)

        async def slow_read(pass_id: str) -> dict:
            await asyncio.sleep(0.4)
            return await read(pass_id)

        mocker.patch.object(fake_dream_db, "update_dream_pass", slow_write)
        mocker.patch.object(cleanup_mod, "read_state", slow_read)

        run = await asyncio.wait_for(reap_expired_passes(), 10)

        assert run.outcomes == {"cleaned": 1}
        assert _charged(charges) == ["consolidate"]
        assert state_key("slow") not in fake_dream_redis.hashes
        assert fake_dream_db.rows["slow"]["cleanup_pending_at"] is None

    async def test_that_cannot_be_moved_to_the_back_is_left_for_the_next_run(
        self, mocker, fake_dream_db, fake_dream_redis, charges, caplog
    ):
        """The write that would put it behind the others fails: the cleanup is
        not tried this run (it would head the next list unmoved), and the
        others still are."""
        now = datetime.now(timezone.utc)
        for index, pass_id in enumerate(("unmovable", "next")):
            _seed(
                fake_dream_db,
                pass_id,
                route=DreamPassRoute.SYNC,
                status=DreamPassStatus.EXPIRED,
                cleanup_pending_at=now - timedelta(hours=1),
            )
            fake_dream_db.rows[pass_id]["updated_at"] = now - timedelta(
                minutes=60 - index
            )
        write = fake_dream_db.update_dream_pass

        async def unmovable(pass_id, update) -> bool:
            if pass_id == "unmovable" and update == cleanup_unfinished():
                raise TimeoutError("this row's write stalls")
            return await write(pass_id, update)

        mocker.patch.object(fake_dream_db, "update_dream_pass", unmovable)

        with caplog.at_level(logging.WARNING):
            run = await reap_expired_passes()

        assert run.outcomes == {"failed": 1, "cleaned": 1}
        assert fake_dream_db.rows["unmovable"]["cleanup_pending_at"] is not None
        assert fake_dream_db.rows["next"]["cleanup_pending_at"] is None
        assert "could not move pass unmovable behind the other cleanups" in (
            caplog.text
        )


class TestAnOpenRowItCannotClose:
    async def test_waits_for_every_other_row_its_row_untouched(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        """Nine open rows whose stop write never answers, the oldest lapses,
        ahead of a healthy one: more than one run can try. The scan goes on
        from where the last run stopped, so no row is tried twice before
        every row was tried once, whatever the gap between runs (the order
        keeps no clock), and the healthy row is closed. Every other write to
        the stuck rows lands, and none is made: their lease and their last
        write are their pass's alone."""
        _scale_the_budget(mocker, budget=0.6, release=0.02, reserve=0.1)
        mocker.patch.object(reaper_mod, "ROW_ATTEMPT_SECONDS", 0.1)
        stuck = [f"stuck{index}" for index in range(9)]
        rows = [*stuck, "healthy"]
        for index, pass_id in enumerate(rows):
            _seed(
                fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=4000 - index
            )
        before = {pass_id: dict(fake_dream_db.rows[pass_id]) for pass_id in stuck}
        write = fake_dream_db.update_dream_pass
        tried: list[str] = []

        async def stop_never_answers_for_the_stuck(pass_id, update) -> bool:
            if update.status is DreamPassStatus.EXPIRED:
                tried.append(pass_id)
                if pass_id in stuck:
                    await asyncio.Event().wait()
            return await write(pass_id, update)

        mocker.patch.object(
            fake_dream_db, "update_dream_pass", stop_never_answers_for_the_stuck
        )

        for _ in range(3):
            await asyncio.wait_for(reap_expired_passes(), 5)

        assert len(tried) > len(rows), "every row tried, and some again"
        assert tried[: len(rows)] == rows, "none tried twice before all once"
        assert fake_dream_db.rows["healthy"]["status"] is DreamPassStatus.EXPIRED
        assert {pass_id: fake_dream_db.rows[pass_id] for pass_id in stuck} == before

    async def test_past_the_row_limit_still_leaves_the_rows_behind_it_their_turn(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        """A hundred open rows whose close fails at once fill a run's list; the
        healthy row lapsed after them is the hundred and first. The next run
        lists from after the last row tried, and closes it."""
        stuck = [f"stuck{index:03}" for index in range(REAPER_ROW_LIMIT)]
        for index, pass_id in enumerate([*stuck, "healthy"]):
            _seed(
                fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=4000 - index
            )
        write = fake_dream_db.update_dream_pass

        async def fails_for_the_stuck(pass_id, update) -> bool:
            if pass_id in stuck and update.status is DreamPassStatus.EXPIRED:
                raise TimeoutError("this row's close keeps failing")
            return await write(pass_id, update)

        mocker.patch.object(fake_dream_db, "update_dream_pass", fails_for_the_stuck)

        first = await reap_expired_passes()

        assert first.outcomes == {"failed": REAPER_ROW_LIMIT}
        assert fake_dream_db.rows["healthy"]["status"] is DreamPassStatus.RUNNING

        second = await reap_expired_passes()

        assert second.outcomes == {"expired": 1, "failed": REAPER_ROW_LIMIT - 1}
        assert fake_dream_db.rows["healthy"]["status"] is DreamPassStatus.EXPIRED

    async def test_the_scan_goes_back_to_the_first_past_the_last(
        self, fake_dream_db, fake_dream_redis
    ):
        """In lapse order, then by id: after the cursor, then from the first
        again, each row once."""
        lapse = datetime.now(timezone.utc) - timedelta(hours=2)
        for pass_id in ("a", "b", "c", "d"):
            _seed(fake_dream_db, pass_id, route=DreamPassRoute.SYNC)
            fake_dream_db.rows[pass_id]["lease_expires_at"] = lapse
        now = datetime.now(timezone.utc)

        lapsed, _ = await reaper_mod._list(now)
        assert [row.id for row in lapsed] == ["a", "b", "c", "d"]

        await reaper_mod._advance_cursor(lapsed[1])
        lapsed, _ = await reaper_mod._list(now)
        assert [row.id for row in lapsed] == ["c", "d", "a", "b"]
        assert fake_dream_redis.ttls[OPEN_CURSOR_KEY] == OPEN_CURSOR_TTL_SECONDS

    async def test_without_redis_to_say_the_scan_starts_from_the_first(
        self, monkeypatch, fake_dream_db, fake_dream_redis, caplog
    ):
        for index, pass_id in enumerate(("s1", "s2")):
            _seed(
                fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=4000 - index
            )
        read, write = fake_dream_redis.get, fake_dream_redis.set

        async def get(key: str):
            if key == OPEN_CURSOR_KEY:
                raise ConnectionError("redis down")
            return await read(key)

        async def set_(key: str, value, **kwargs):
            if key == OPEN_CURSOR_KEY:
                raise ConnectionError("redis down")
            return await write(key, value, **kwargs)

        monkeypatch.setattr(fake_dream_redis, "get", get)
        monkeypatch.setattr(fake_dream_redis, "set", set_)

        with caplog.at_level(logging.WARNING):
            run = await reap_expired_passes()

        assert run.outcomes == {"expired": 2}
        assert "could not read where its scan of open passes stopped" in caplog.text
        assert "could not note pass s1 as tried" in caplog.text


class TestABatchStateThatLapsed:
    @pytest.mark.parametrize(
        ("claims", "lost"),
        [
            ({}, "consolidate, recombine"),
            (
                {
                    "dream:batch:costs_logged:p1": "per_phase",
                    "dream:batch:charged:p1:consolidate": "1",
                },
                "recombine",
            ),
        ],
        ids=["neither charged", "one charged"],
    )
    async def test_before_its_landed_phases_were_charged_is_reported_lost(
        self, fake_dream_db, fake_dream_redis, charges, caplog, claims, lost
    ):
        """The row says consolidate and recombine landed, and their state is
        gone: an error names the pass, its owner and the phases nobody
        charged. Nothing more can be done for them, so the cleanup ends."""
        _closed_with_two_landed(fake_dream_db)
        fake_dream_redis.store.update(claims)

        with caplog.at_level(logging.ERROR):
            run = await reap_expired_passes()

        assert run.outcomes == {"cleaned": 1}
        charges.assert_not_awaited()
        assert _errors(caplog) == [
            f"Dream pass p1 of user u1 (expert none): the batch state of landed "
            f"phase(s) {lost} lapsed before they were charged; their usage "
            "cannot be recovered"
        ]

    @pytest.mark.parametrize(
        "claims",
        [
            {
                "dream:batch:costs_logged:p1": "per_phase",
                "dream:batch:charged:p1:consolidate": "1",
                "dream:batch:charged:p1:recombine": "1",
            },
            {"dream:batch:costs_logged:p1": "1"},
        ],
        ids=["charged phase by phase", "charged whole by an earlier build"],
    )
    async def test_after_its_landed_phases_were_charged_reports_nothing(
        self, fake_dream_db, fake_dream_redis, caplog, claims
    ):
        """The pass's own end charged them, then lost the write that clears
        the mark: their state is gone because it was done with."""
        _closed_with_two_landed(fake_dream_db)
        fake_dream_redis.store.update(claims)

        with caplog.at_level(logging.ERROR):
            run = await reap_expired_passes()

        assert run.outcomes == {"cleaned": 1}
        assert _errors(caplog) == []


def _hang_the_state_read_of(mocker, *hanging: str) -> list[str]:
    """Make the cleanup's state read never answer for the *hanging* passes,
    and read the rest's; returns the passes it was asked for, in order."""
    tried: list[str] = []
    read = cleanup_mod.read_state

    async def read_state(pass_id: str) -> dict:
        tried.append(pass_id)
        if pass_id in hanging:
            await asyncio.Event().wait()
        return await read(pass_id)

    mocker.patch.object(cleanup_mod, "read_state", read_state)
    return tried


async def _consolidate_landed(pass_id: str) -> None:
    await write_phase_to_state(
        pass_id=pass_id,
        phase="consolidate",
        row=BatchResultRow(
            custom_id=f"{pass_id}_consolidate",
            content=ConsolidationOutput().model_dump_json(),
            input_tokens=10,
            output_tokens=20,
        ),
        model="claude-sonnet-5",
    )


def _closed_with_two_landed(fake_dream_db) -> None:
    """Batch pass p1, closed EXPIRED and marked for its cleanup an hour ago,
    its row recording consolidate's and recombine's output; no batch state."""
    _seed(
        fake_dream_db,
        "p1",
        route=DreamPassRoute.ANTHROPIC_BATCH,
        status=DreamPassStatus.EXPIRED,
        cleanup_pending_at=datetime.now(timezone.utc) - timedelta(hours=1),
        phase_outputs={
            "consolidate": ConsolidationOutput(),
            "recombine": RecombinationOutput(),
        },
    )


def _errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _scale_every_bound(mocker, scale: float) -> None:
    """Every bound on the reaper's path, scaled by *scale*."""
    for module, name, seconds in (
        (reaper_mod, "REAPER_BUDGET_SECONDS", 60),
        (reaper_mod, "RELEASE_TIMEOUT_SECONDS", 2),
        (reaper_mod, "ROW_RESERVE_SECONDS", 12),
        (reaper_mod, "ROW_ATTEMPT_SECONDS", 12),
        (reaper_mod, "LOCK_CHECK_TIMEOUT_SECONDS", 2),
        (store_mod, "RECORD_WRITE_TIMEOUT_SECONDS", 10),
        (cleanup_mod, "STATE_STEP_TIMEOUT_SECONDS", 5),
        (cleanup_mod, "CHARGE_STEP_TIMEOUT_SECONDS", 10),
        (cleanup_mod, "LOCK_CHECK_TIMEOUT_SECONDS", 2),
        (locks_mod, "LOCK_CHECK_TIMEOUT_SECONDS", 2),
        (provider_batch_mod, "PROVIDER_CANCEL_TIMEOUT_SECONDS", 10),
    ):
        mocker.patch.object(module, name, seconds * scale)
