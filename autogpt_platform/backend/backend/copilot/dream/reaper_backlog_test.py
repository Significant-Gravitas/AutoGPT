"""The reaper behind rows it cannot finish, over the in-memory store and
Redis: a row whose attempt hangs is cut after its own ``ROW_ATTEMPT_SECONDS``,
so the rows behind it get their turn in the same run, and a healthy cleanup
behind five hanging ones is charged in the first run; slow but bounded work
reaches the charge, and a slow write of one kind does not cost the other its
share; each kind is scanned in cycles, so a row that keeps failing waits for
every other row present when the cycle started before it is tried again,
however many rows arrive meanwhile, with no row written to make room; and a
pass whose batch state lapsed before its landed phases were charged is
reported as lost usage, an error, rather than taken for one with nothing to
charge."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassRoute, DreamPassStatus

from backend.util.llm.providers import BatchResultRow

from . import batch_costs as batch_costs_mod
from . import cleanup as cleanup_mod
from . import locks as locks_mod
from . import provider_batch as provider_batch_mod
from . import reaper as reaper_mod
from . import store as store_mod
from .batch_state import claim_per_phase_charging, state_key, write_phase_to_state
from .reaper import (
    REAP_GRACE_SECONDS,
    REAPER_ROW_LIMIT,
    SCAN_TTL_SECONDS,
    reap_expired_passes,
)
from .reaper_cleanup_test import _scale_the_budget
from .reaper_test import _charged, _seed
from .schemas import ConsolidationOutput, RecombinationOutput

_MODELS = {p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")}
_OPEN_SCAN = "dream:reaper:open_scan"


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
        """Five pending cleanups whose state read never answers, marked before
        a healthy one with a landed phase. Each hanging attempt is cut after
        its own few seconds, not the rest of the run's time, so the healthy
        cleanup is reached in the first run and charged once."""
        _scale_the_budget(mocker, budget=1.0, release=0.02, reserve=0.1)
        mocker.patch.object(reaper_mod, "ROW_ATTEMPT_SECONDS", 0.1)
        stuck = [f"stuck{index}" for index in range(5)]
        _pending(fake_dream_db, [*stuck, "healthy"])
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
    async def test_reaches_its_charge_and_clears_its_mark(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Every bound scaled to a tenth: the state read takes 4 s of its 5,
        and the write that clears the mark 9 s of its 10. Neither runs over,
        and the write comes after the 12 s attempt, on a bound of its own: one
        charge, the state gone, the mark cleared, in the first run."""
        _scale_every_bound(mocker, 0.1)
        _pending(fake_dream_db, ["slow"])
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

    async def test_whose_provider_answers_slowly_finishes_over_two_runs(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        """Every bound scaled to a tenth: Anthropic refuses the cancel in 8 s
        of its 10 (the batch has ended) and answers the status read in 8 s
        too. Both do not fit one 12 s attempt; the cancel is noted as sent,
        so the next run only reads the status, and the cleanup finishes."""
        _scale_every_bound(mocker, 0.1)
        mocker.patch.object(provider_batch_mod, "anthropic_api_key", return_value="k")
        _pending(fake_dream_db, ["slow-provider"], provider_batch_id="b-ended")
        calls: list[str] = []

        async def refused(**_kwargs) -> bool:
            await asyncio.sleep(0.8)
            calls.append("cancel")
            return False

        async def ended(**_kwargs) -> str:
            await asyncio.sleep(0.8)
            calls.append("status")
            return "ended"

        mocker.patch.object(provider_batch_mod, "cancel_batch", refused)
        mocker.patch.object(provider_batch_mod, "poll_batch", ended)

        first = await asyncio.wait_for(reap_expired_passes(), 10)
        second = await asyncio.wait_for(reap_expired_passes(), 10)

        assert (first.outcomes, second.outcomes) == (
            {"out_of_budget": 1},
            {"cleaned": 1},
        )
        assert calls == ["cancel", "status"]
        assert fake_dream_db.rows["slow-provider"]["cleanup_pending_at"] is None

    async def test_whose_billing_gate_answers_slowly_charges_once_and_finishes(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Codex's probe, every bound scaled to a tenth: an earlier cleanup
        set the pass's billing gate and stopped before its phase's claim; the
        state read takes 4 s, and each Redis command on the gate 4.5 s. Read
        before it is set, the gate costs the retry one command, not two: one
        charge and the mark cleared in the first run, nothing left after."""
        _scale_every_bound(mocker, 0.1)
        _pending(fake_dream_db, ["gated"])
        await _consolidate_landed("gated")
        assert await claim_per_phase_charging("gated")
        _slow_the_gate(mocker, fake_dream_redis, "gated", seconds=0.45)
        _slow_the_state_read(mocker, seconds=0.4)

        runs = [await asyncio.wait_for(reap_expired_passes(), 10) for _ in range(3)]

        assert [run.outcomes for run in runs] == [{"cleaned": 1}, {}, {}]
        assert _charged(charges) == ["consolidate"]
        assert fake_dream_db.rows["gated"]["cleanup_pending_at"] is None

    async def test_whose_billing_gate_outlasts_the_attempt_is_charged_anyway(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Every bound scaled to a tenth: the state read takes 4 s and the
        gate's read 9 s, past the 12 s attempt. The charge runs on after the
        cut and charges the phase; the next run, cut the same way, charges
        nothing again, and the mark waits for a run the gate answers in."""
        _scale_every_bound(mocker, 0.1)
        _pending(fake_dream_db, ["gated"])
        await _consolidate_landed("gated")
        assert await claim_per_phase_charging("gated")
        _slow_the_gate(mocker, fake_dream_redis, "gated", seconds=0.9)
        _slow_the_state_read(mocker, seconds=0.4)

        first = await asyncio.wait_for(reap_expired_passes(), 10)
        await asyncio.wait_for(asyncio.gather(*batch_costs_mod._CHARGES_IN_FLIGHT), 5)

        assert first.outcomes == {"out_of_budget": 1}
        assert _charged(charges) == ["consolidate"]

        second = await asyncio.wait_for(reap_expired_passes(), 10)
        await asyncio.wait_for(asyncio.gather(*batch_costs_mod._CHARGES_IN_FLIGHT), 5)

        assert second.outcomes == {"out_of_budget": 1}
        assert _charged(charges) == ["consolidate"]
        assert fake_dream_db.rows["gated"]["cleanup_pending_at"] is not None

    async def test_of_one_kind_leaves_the_other_its_share(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Every bound scaled to a tenth, and every listing, and every write
        to a newly lapsed row, takes 9.5 s of its 10; the open rows' cycle
        has just ended, so they are listed twice. The open row's close and the
        write clearing its mark both end with the open rows' share, the write
        cut short there (the next run clears the mark), so the pending
        cleanup still starts within its own, and is charged in the first
        run."""
        _scale_every_bound(mocker, 0.1)
        _pending(fake_dream_db, ["healthy"])
        await _consolidate_landed("healthy")
        _seed(fake_dream_db, "new", route=DreamPassRoute.SYNC)
        ended = datetime.now(timezone.utc) - timedelta(days=1)
        fake_dream_redis.hashes[_OPEN_SCAN] = {"end": ended.isoformat(), "last": ""}
        list_open = fake_dream_db.list_expired_dream_passes
        list_pending = fake_dream_db.list_dream_pass_cleanups
        write = fake_dream_db.update_dream_pass

        async def slow_open(*args, **kwargs):
            await asyncio.sleep(0.95)
            return await list_open(*args, **kwargs)

        async def slow_pending(*args, **kwargs):
            await asyncio.sleep(0.95)
            return await list_pending(*args, **kwargs)

        async def slow_for_the_new_row(pass_id, update) -> bool:
            if pass_id == "new":
                await asyncio.sleep(0.95)
            return await write(pass_id, update)

        mocker.patch.object(fake_dream_db, "list_expired_dream_passes", slow_open)
        mocker.patch.object(fake_dream_db, "list_dream_pass_cleanups", slow_pending)
        mocker.patch.object(fake_dream_db, "update_dream_pass", slow_for_the_new_row)

        run = await asyncio.wait_for(reap_expired_passes(), 10)

        assert run.outcomes.get("cleaned") == 1
        assert _charged(charges) == ["consolidate"]
        assert fake_dream_db.rows["healthy"]["cleanup_pending_at"] is None


class TestAScanCycle:
    async def test_tries_every_open_row_once_before_any_twice(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        """Nine open rows whose stop write never answers, the oldest lapses,
        ahead of a healthy one: more than one run can try. No row is tried
        twice before every row was tried once, and the healthy row is closed.
        Every other write to the stuck rows lands, and none is made: their
        lease and their last write are their pass's alone."""
        _scale_the_budget(mocker, budget=0.6, release=0.02, reserve=0.1)
        mocker.patch.object(reaper_mod, "ROW_ATTEMPT_SECONDS", 0.1)
        stuck = [f"stuck{index}" for index in range(9)]
        rows = [*stuck, "healthy"]
        for index, pass_id in enumerate(rows):
            _seed(
                fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=4000 - index
            )
        before = {pass_id: dict(fake_dream_db.rows[pass_id]) for pass_id in stuck}
        tried = _hang_the_stop_of(mocker, fake_dream_db, *stuck)

        for _ in range(3):
            await asyncio.wait_for(reap_expired_passes(), 5)

        assert len(tried) > len(rows), "every row tried, and some again"
        assert tried[: len(rows)] == rows, "none tried twice before all once"
        assert fake_dream_db.rows["healthy"]["status"] is DreamPassStatus.EXPIRED
        assert {pass_id: fake_dream_db.rows[pass_id] for pass_id in stuck} == before

    async def test_reaches_open_rows_past_the_row_limit(
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

        assert second.outcomes == {"expired": 1}
        assert fake_dream_db.rows["healthy"]["status"] is DreamPassStatus.EXPIRED

    async def test_reaches_a_cleanup_behind_ones_that_keep_failing(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """Every bound scaled to a tenth: six cleanups whose write clearing the
        mark fails after 9 s (a row another transaction keeps locked), marked
        before a healthy one, more than a run can try. The scan goes on past
        them whatever their writes do, and the healthy cleanup is charged in
        the second run."""
        _scale_every_bound(mocker, 0.1)
        locked = [f"locked{index}" for index in range(6)]
        _pending(fake_dream_db, [*locked, "healthy"])
        await _consolidate_landed("healthy")
        write = fake_dream_db.update_dream_pass
        tried: list[str] = []

        async def locked_rows_time_out(pass_id, update) -> bool:
            tried.append(pass_id)
            if pass_id in locked:
                await asyncio.sleep(0.9)
                raise TimeoutError("canceling statement due to lock timeout")
            return await write(pass_id, update)

        mocker.patch.object(fake_dream_db, "update_dream_pass", locked_rows_time_out)

        first = await asyncio.wait_for(reap_expired_passes(), 10)
        assert _charged(charges) == []
        second = await asyncio.wait_for(reap_expired_passes(), 10)

        assert first.outcomes["failed"] >= 4
        assert second.outcomes.get("cleaned") == 1
        assert _charged(charges) == ["consolidate"]
        assert tried[: len(locked) + 1] == [*locked, "healthy"]

    async def test_lets_no_arrival_delay_a_row_it_started_with(
        self, mocker, fake_dream_db, fake_dream_redis
    ):
        """Five open rows whose stop write never answers and a healthy one are
        there when the cycle starts; two more rows lapse before every run.
        The arrivals have keys past the cycle's end and wait for the next
        cycle, so the healthy row is closed once the five have had their
        turn, and every row the cycle started with is tried before any
        arrival."""
        _scale_the_budget(mocker, budget=0.6, release=0.02, reserve=0.1)
        mocker.patch.object(reaper_mod, "ROW_ATTEMPT_SECONDS", 0.1)
        stuck = [f"stuck{index}" for index in range(5)]
        started_with = [*stuck, "healthy"]
        for index, pass_id in enumerate(started_with):
            _seed(
                fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=4000 - index
            )
        tried = _hang_the_stop_of(mocker, fake_dream_db, *stuck)
        base = datetime.now(timezone.utc)
        arrivals: list[str] = []

        for run in range(3):
            now = base + timedelta(seconds=601 * run)
            for index in range(2):
                pass_id = f"arrival{run}-{index}"
                _seed(fake_dream_db, pass_id, route=DreamPassRoute.SYNC)
                fake_dream_db.rows[pass_id]["lease_expires_at"] = now - timedelta(
                    seconds=REAP_GRACE_SECONDS + 2 - index
                )
                arrivals.append(pass_id)
            await asyncio.wait_for(reap_expired_passes(now=now), 5)

        assert fake_dream_db.rows["healthy"]["status"] is DreamPassStatus.EXPIRED
        first_cycle = started_with + arrivals[:2]
        assert tried[: len(first_cycle)] == first_cycle
        assert not set(arrivals[2:]) & set(tried[: tried.index("healthy")])

    async def test_covers_the_rows_there_when_it_starts(
        self, fake_dream_db, fake_dream_redis
    ):
        """The cycle lists its rows after the last one tried, up to its end; a
        row lapsing later waits, and once the cycle's rows are done, the next
        one starts from the first row."""
        now = datetime.now(timezone.utc)
        for pass_id in ("a", "b"):
            _seed(fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=7200)

        lapsed, _ = await reaper_mod._list(now)
        assert [row.id for row in lapsed] == ["a", "b"]
        assert fake_dream_redis.ttls[_OPEN_SCAN] == SCAN_TTL_SECONDS

        await reaper_mod._note_tried(fake_dream_db.record("a"))
        _seed(fake_dream_db, "later", route=DreamPassRoute.SYNC)
        fake_dream_db.rows["later"]["lease_expires_at"] = now - timedelta(
            seconds=REAP_GRACE_SECONDS - 60
        )
        lapsed, _ = await reaper_mod._list(now + timedelta(minutes=10))
        assert [row.id for row in lapsed] == ["b"]

        await reaper_mod._note_tried(fake_dream_db.record("b"))
        lapsed, _ = await reaper_mod._list(now + timedelta(minutes=20))
        assert [row.id for row in lapsed] == ["a", "b", "later"]

    async def test_without_redis_to_say_starts_from_the_first_row(
        self, monkeypatch, fake_dream_db, fake_dream_redis, caplog
    ):
        for index, pass_id in enumerate(("s1", "s2")):
            _seed(
                fake_dream_db, pass_id, route=DreamPassRoute.SYNC, lapsed=4000 - index
            )

        read, write = fake_dream_redis.hgetall, fake_dream_redis.hset

        async def hgetall(name: str):
            if name.startswith("dream:reaper:"):
                raise ConnectionError("redis down")
            return await read(name)

        async def hset(name: str, *args, **kwargs):
            if name.startswith("dream:reaper:"):
                raise ConnectionError("redis down")
            return await write(name, *args, **kwargs)

        monkeypatch.setattr(fake_dream_redis, "hgetall", hgetall)
        monkeypatch.setattr(fake_dream_redis, "hset", hset)

        with caplog.at_level(logging.WARNING):
            run = await reap_expired_passes()

        assert run.outcomes == {"expired": 2}
        assert "could not read where its scan of open passes stands" in caplog.text
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

    async def test_of_a_sync_pass_reports_nothing(
        self, fake_dream_db, fake_dream_redis, charges, caplog
    ):
        """A sync pass charges each phase as it lands and keeps no batch
        state: its recorded outputs, with no state and no batch claims, are
        how an already charged pass looks, not lost usage."""
        _seed(
            fake_dream_db,
            "p1",
            route=DreamPassRoute.SYNC,
            status=DreamPassStatus.EXPIRED,
            cleanup_pending_at=datetime.now(timezone.utc) - timedelta(hours=1),
            phase_outputs={"consolidate": ConsolidationOutput()},
        )

        with caplog.at_level(logging.ERROR):
            run = await reap_expired_passes()

        assert run.outcomes == {"cleaned": 1}
        charges.assert_not_awaited()
        assert _errors(caplog) == []

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


def _pending(fake_dream_db, pass_ids: list[str], **columns) -> None:
    """Closed batch rows marked for their cleanup, the first marked first, an
    hour or so ago."""
    now = datetime.now(timezone.utc)
    for index, pass_id in enumerate(pass_ids):
        _seed(
            fake_dream_db,
            pass_id,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            status=DreamPassStatus.EXPIRED,
            cleanup_pending_at=now - timedelta(minutes=60 - index),
            **columns,
        )


def _hang_the_stop_of(mocker, fake_dream_db, *hanging: str) -> list[str]:
    """Make the stop write of the *hanging* rows never answer; returns the
    rows whose stop was tried, in order."""
    write = fake_dream_db.update_dream_pass
    tried: list[str] = []

    async def stop(pass_id, update) -> bool:
        if update.status is DreamPassStatus.EXPIRED:
            tried.append(pass_id)
            if pass_id in hanging:
                await asyncio.Event().wait()
        return await write(pass_id, update)

    mocker.patch.object(fake_dream_db, "update_dream_pass", stop)
    return tried


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


def _slow_the_gate(mocker, fake_dream_redis, pass_id: str, *, seconds: float) -> None:
    """Every Redis read and write of the pass's billing gate takes *seconds*."""
    gate = f"dream:batch:costs_logged:{pass_id}"
    get, set_ = fake_dream_redis.get, fake_dream_redis.set

    async def slow_get(key: str):
        if key == gate:
            await asyncio.sleep(seconds)
        return await get(key)

    async def slow_set(key: str, *args, **kwargs):
        if key == gate:
            await asyncio.sleep(seconds)
        return await set_(key, *args, **kwargs)

    mocker.patch.object(fake_dream_redis, "get", slow_get)
    mocker.patch.object(fake_dream_redis, "set", slow_set)


def _slow_the_state_read(mocker, *, seconds: float) -> None:
    read = cleanup_mod.read_state

    async def slow_read(pass_id: str) -> dict:
        await asyncio.sleep(seconds)
        return await read(pass_id)

    mocker.patch.object(cleanup_mod, "read_state", slow_read)


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
        (batch_costs_mod, "CLAIM_WINDOW_SECONDS", 30),
        (batch_costs_mod, "CLAIM_TIMEOUT_SECONDS", 5),
        (cleanup_mod, "LOCK_CHECK_TIMEOUT_SECONDS", 2),
        (locks_mod, "LOCK_CHECK_TIMEOUT_SECONDS", 2),
        (provider_batch_mod, "LOCK_CHECK_TIMEOUT_SECONDS", 2),
        (provider_batch_mod, "PROVIDER_CANCEL_TIMEOUT_SECONDS", 10),
    ):
        mocker.patch.object(module, name, seconds * scale)
