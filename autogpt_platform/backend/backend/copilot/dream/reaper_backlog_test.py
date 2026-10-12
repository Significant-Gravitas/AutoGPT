"""The reaper behind rows it cannot finish, over the in-memory store and
Redis: a row whose attempt hangs is cut after its own ``ROW_ATTEMPT_SECONDS``,
so the rows behind it get their turn in the same run, and a healthy cleanup
behind five hanging ones is charged in the first run; and a pass whose batch
state lapsed before its landed phases were charged is reported as lost
usage, an error, rather than taken for one with nothing to charge."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassRoute, DreamPassStatus

from backend.util.llm.providers import BatchResultRow

from . import cleanup as cleanup_mod
from . import reaper as reaper_mod
from .batch_state import state_key, write_phase_to_state
from .reaper import reap_expired_passes
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
