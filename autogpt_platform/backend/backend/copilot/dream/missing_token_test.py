"""A batch pass whose input bundle lost its lock token (a bundle an earlier
build wrote, or a corrupted one), over the in-memory store and Redis. A
missing token is unknown ownership, never "no lock": the pass goes by the
token its row keeps, fenced and released under it as under the bundle's.
With no token to be had it does not apply, and its row stays marked with the
unlock unfinished, its lock left alone, until the reaper releases the lock by
the row's token or, for a row that kept none, finds it gone. The provider and
the charges are stubbed at their edges; the lock, the fence, the claims and
the row transitions are the real ones."""

import json
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassStatus

from . import reaper as reaper_mod
from .batch_callbacks import handle_dream_batch_result
from .batch_deliveries_test import _LOCK_KEY, _MODELS, _entry, _in_flight, _row
from .batch_state import state_key
from .batch_submit import input_bundle_key
from .conftest import FakeAsyncRedis, FakeDreamDb
from .locks import BATCH_LOCK_TTL_SECONDS
from .reaper import REAP_GRACE_SECONDS, reap_expired_passes
from .reaper_test import _charged
from .schemas import DreamOperations

_ALL_THREE = ["consolidate", "recombine", "sanitize"]


@pytest.fixture(autouse=True)
def provider(mocker) -> AsyncMock:
    mocker.patch(
        "backend.copilot.dream.provider_batch.anthropic_api_key", return_value="k"
    )
    mocker.patch.object(reaper_mod, "phase_models_for_config", return_value=_MODELS)
    return mocker.patch(
        "backend.copilot.dream.provider_batch.cancel_batch",
        AsyncMock(return_value=True),
    )


@pytest.fixture(autouse=True)
def charges(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )


@pytest.fixture
def apply(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.apply.apply_operations",
        AsyncMock(return_value={"writes": 0}),
    )


class TestABundleThatLostItsToken:
    async def test_goes_by_the_token_its_row_keeps(
        self, fake_dream_db, fake_dream_redis, apply, charges
    ):
        """Codex's probe: only the token is gone from an otherwise valid
        bundle. The fence renews under the row's token, apply runs once, and
        the cleanup releases the lock under that token: no lock left, no
        mark, every landed phase charged once."""
        await _last_phase_in_flight_without_a_bundle_token(
            fake_dream_db, fake_dream_redis
        )

        await _deliver_sanitize()

        apply.assert_awaited_once()
        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.COMPLETE
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)
        assert _charged(charges) == _ALL_THREE
        assert _LOCK_KEY not in fake_dream_redis.store
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store

    async def test_whose_row_cannot_say_does_not_apply_and_the_reaper_unlocks(
        self, monkeypatch, fake_dream_db, fake_dream_redis, apply, charges
    ):
        """The row's token cannot be read either (the store down for reads):
        ownership is unknown, so apply does not run; the landed phases are
        charged once and the row closes ERRORED, marked, its token kept, the
        lock left held. The reaper then releases the lock by that token."""
        await _last_phase_in_flight_without_a_bundle_token(
            fake_dream_db, fake_dream_redis
        )
        read_row = fake_dream_db.get_dream_pass
        monkeypatch.setattr(
            fake_dream_db,
            "get_dream_pass",
            AsyncMock(side_effect=ConnectionError("reads down")),
        )

        await _deliver_sanitize()

        monkeypatch.setattr(fake_dream_db, "get_dream_pass", read_row)
        apply.assert_not_awaited()
        _assert_errored_and_marked(fake_dream_db, token="tok")
        assert _charged(charges) == _ALL_THREE
        assert fake_dream_redis.store[_LOCK_KEY] == "tok"
        assert fake_dream_redis.ttls[_LOCK_KEY] == BATCH_LOCK_TTL_SECONDS

        run = await reap_expired_passes(now=_past_the_grace())

        assert run.outcomes == {"cleaned": 1}
        assert _LOCK_KEY not in fake_dream_redis.store
        row = fake_dream_db.rows["p1"]
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)
        assert _charged(charges) == _ALL_THREE


class TestAPassWithNoTokenAnywhere:
    async def test_does_not_apply_and_its_lock_is_left_to_lapse(
        self, fake_dream_db, fake_dream_redis, apply, charges, caplog
    ):
        """Neither the bundle nor the row keeps a token (a row written before
        leases): apply does not run, the landed phases are charged once, and
        the row closes ERRORED and marked with its unlock unfinished. The
        reaper retries while the lock is held under a token no open pass
        keeps, never deleting it, and cleans the row up once it has lapsed."""
        await _last_phase_in_flight_without_a_bundle_token(
            fake_dream_db, fake_dream_redis
        )
        fake_dream_db.rows["p1"]["lease_token"] = None

        with caplog.at_level(logging.WARNING):
            await _deliver_sanitize()
            waiting = await reap_expired_passes(now=_past_the_grace())

        apply.assert_not_awaited()
        _assert_errored_and_marked(fake_dream_db, token=None)
        assert _charged(charges) == _ALL_THREE
        assert "cleanup unfinished at unlock; left marked" in caplog.text
        assert waiting.outcomes == {"retry": 1}
        assert "held under a token no open pass keeps" in caplog.text
        assert fake_dream_redis.store[_LOCK_KEY] == "tok"

        del fake_dream_redis.store[_LOCK_KEY]
        lapsed = await reap_expired_passes(now=_past_the_grace())

        assert lapsed.outcomes == {"cleaned": 1}
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is None
        assert _charged(charges) == _ALL_THREE


async def _last_phase_in_flight_without_a_bundle_token(
    fake_dream_db: FakeDreamDb, fake_dream_redis: FakeAsyncRedis
) -> None:
    """Pass p1 waiting on sanitize's batch, holding its lock under ``tok``
    for a batch lifetime, its row keeping that token, and its input bundle
    valid but for the token it lost."""
    await _in_flight(fake_dream_db, fake_dream_redis, landed=2)
    await fake_dream_redis.expire(_LOCK_KEY, BATCH_LOCK_TTL_SECONDS)
    bundle = json.loads(fake_dream_redis.store[input_bundle_key("p1")])
    del bundle["lock_token"]
    fake_dream_redis.store[input_bundle_key("p1")] = json.dumps(bundle)


async def _deliver_sanitize() -> None:
    await handle_dream_batch_result(
        _entry("sanitize"), [_row("sanitize", DreamOperations())]
    )


def _assert_errored_and_marked(fake_dream_db: FakeDreamDb, *, token: str | None):
    row = fake_dream_db.rows["p1"]
    assert row["status"] is DreamPassStatus.ERRORED
    assert row["cleanup_pending_at"] is not None
    assert row["lease_token"] == token


def _past_the_grace() -> datetime:
    """An instant by which the cleanup of a row closed now is due."""
    return datetime.now(timezone.utc) + timedelta(seconds=REAP_GRACE_SECONDS + 60)
