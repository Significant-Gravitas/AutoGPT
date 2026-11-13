"""A cleanup step that fails without raising, over the in-memory store and
Redis: a Redis error at the state read, at a charge claim, at the unlock or
at a delete; a provider that cannot confirm the batch stopped; no phase
models to price with. Each leaves the row marked and its lease token kept,
and its state too while the charge is unfinished; the run says ``retry`` at
warning, naming the steps left; and the next run finishes the cleanup once:
one charge, the lock, the state and the bundle gone, the mark and the token
cleared. The provider and the charges are stubbed at their edges; the lock,
the claims, the batch state and the row transitions are the real ones."""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassStatus

from . import reaper as reaper_mod
from .batch_state import state_key
from .batch_submit import input_bundle_key
from .conftest import FakeAsyncRedis, FakeDreamDb
from .reaper import reap_expired_passes
from .reaper_test import _LOCK_KEY, _charged, _dead_batch_pass

_MODELS = {p: "claude-sonnet-5" for p in ("consolidate", "recombine", "sanitize")}

# Each Redis call a cleanup step makes, the key it fails on, and the steps
# its failure leaves unfinished.
_REDIS_FAILURES: dict[str, tuple[str, str, str]] = {
    "state read": ("hgetall", state_key("p1"), "charge, delete"),
    "pass claim": ("set", "dream:batch:costs_logged:p1", "charge, delete"),
    "phase claim": ("set", "dream:batch:charged:p1:consolidate", "charge, delete"),
    "unlock": ("eval", _LOCK_KEY, "unlock"),
    "state delete": ("delete", state_key("p1"), "delete"),
    "bundle delete": ("delete", input_bundle_key("p1"), "delete"),
}


@pytest.fixture(autouse=True)
def provider(mocker) -> AsyncMock:
    """The provider's cancel, acknowledged unless a test says otherwise."""
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


class TestARedisErrorInAStep:
    @pytest.mark.parametrize("failure", list(_REDIS_FAILURES))
    async def test_keeps_the_mark_and_the_next_run_finishes_once(
        self, monkeypatch, fake_dream_db, fake_dream_redis, charges, caplog, failure
    ):
        """Codex's three (the state read, the state delete, the unlock) and
        every other Redis call a step makes: the failure is swallowed, as
        before, but no longer taken for success."""
        await _dead_batch_pass(fake_dream_db)
        fake_dream_redis.store[_LOCK_KEY] = "dead-token"
        method, key, unfinished = _REDIS_FAILURES[failure]
        restore = _fail_on(monkeypatch, fake_dream_redis, method, key)

        with caplog.at_level(logging.WARNING):
            first = await reap_expired_passes()

        assert first.outcomes == {"retry": 1}
        _assert_marked(fake_dream_db)
        assert f"cleanup unfinished at {unfinished}; listed again" in caplog.text
        if unfinished.startswith("charge"):
            charges.assert_not_awaited()
            assert state_key("p1") in fake_dream_redis.hashes

        restore()
        await _assert_the_next_run_finishes_once(
            fake_dream_db, fake_dream_redis, charges
        )


class TestAProviderThatCannotConfirmTheStop:
    @pytest.mark.parametrize(
        "status",
        [ConnectionError("anthropic unreachable"), "processing"],
        ids=["status unreadable", "still processing"],
    )
    async def test_keeps_the_mark_until_the_batch_has_stopped(
        self, fake_dream_db, fake_dream_redis, provider, batch_status, charges, status
    ):
        """The cancel is refused and the batch is not found ended: the other
        steps finish (one charge), the provider one is retried."""
        await _dead_batch_pass(fake_dream_db)
        provider.return_value = False
        if isinstance(status, Exception):
            batch_status.side_effect = status
        else:
            batch_status.return_value = status

        first = await reap_expired_passes()

        assert first.outcomes == {"retry": 1}
        _assert_marked(fake_dream_db)
        assert _charged(charges) == ["consolidate"]
        assert state_key("p1") not in fake_dream_redis.hashes

        provider.return_value = True
        await _assert_the_next_run_finishes_once(
            fake_dream_db, fake_dream_redis, charges
        )

    async def test_a_refused_cancel_of_a_batch_that_ended_is_no_failure(
        self, fake_dream_db, provider, batch_status
    ):
        await _dead_batch_pass(fake_dream_db)
        provider.return_value = False
        batch_status.return_value = "ended"

        run = await reap_expired_passes()

        assert run.outcomes == {"expired": 1}
        batch_status.assert_awaited_once()
        assert fake_dream_db.rows["p1"]["cleanup_pending_at"] is None

    @pytest.mark.parametrize(
        ("marked_hours_ago", "outcome"), [(1, "retry"), (26, "cleaned")]
    )
    async def test_without_a_key_a_batch_is_waited_on_only_its_window(
        self, mocker, fake_dream_db, marked_hours_ago, outcome
    ):
        """No key to cancel or read the batch with: the step stays
        unfinished until the row was marked longer ago than a batch can run,
        when the batch has ended whatever the provider could say."""
        mocker.patch(
            "backend.copilot.dream.provider_batch.anthropic_api_key",
            return_value=None,
        )
        await _dead_batch_pass(fake_dream_db)
        fake_dream_db.rows["p1"].update(
            status=DreamPassStatus.EXPIRED,
            cleanup_pending_at=datetime.now(timezone.utc)
            - timedelta(hours=marked_hours_ago),
        )

        run = await reap_expired_passes()

        assert run.outcomes == {outcome: 1}
        cleared = fake_dream_db.rows["p1"]["cleanup_pending_at"] is None
        assert cleared is (outcome == "cleaned")


class TestNoPhaseModelsToPriceWith:
    async def test_keep_the_state_until_the_charge_can_be_made(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        await _dead_batch_pass(fake_dream_db)
        models = mocker.patch.object(
            reaper_mod, "phase_models_for_config", side_effect=KeyError("model")
        )

        first = await reap_expired_passes()

        assert first.outcomes == {"retry": 1}
        _assert_marked(fake_dream_db)
        charges.assert_not_awaited()
        assert state_key("p1") in fake_dream_redis.hashes

        models.side_effect = None
        models.return_value = _MODELS
        await _assert_the_next_run_finishes_once(
            fake_dream_db, fake_dream_redis, charges
        )


def _fail_on(
    monkeypatch, redis: FakeAsyncRedis, method: str, key: str
) -> Callable[[], None]:
    """Make Redis's *method* raise for *key* until the returned undo runs."""
    real = getattr(redis, method)

    async def failing(*args: Any, **kwargs: Any) -> Any:
        if key in args:
            raise ConnectionError(f"redis down at {method} {key}")
        return await real(*args, **kwargs)

    monkeypatch.setattr(redis, method, failing)
    return lambda: monkeypatch.setattr(redis, method, real)


def _assert_marked(fake_dream_db: FakeDreamDb) -> None:
    """The row closed EXPIRED, still marked, the dead pass's token kept."""
    row = fake_dream_db.rows["p1"]
    assert row["status"] is DreamPassStatus.EXPIRED
    assert row["cleanup_pending_at"] is not None
    assert row["lease_token"] == "dead-token"


async def _assert_the_next_run_finishes_once(
    fake_dream_db: FakeDreamDb, fake_dream_redis: FakeAsyncRedis, charges: AsyncMock
) -> None:
    second = await reap_expired_passes()

    assert (second.listed, second.outcomes) == (1, {"cleaned": 1})
    assert _charged(charges) == ["consolidate"]
    assert state_key("p1") not in fake_dream_redis.hashes
    assert input_bundle_key("p1") not in fake_dream_redis.store
    assert _LOCK_KEY not in fake_dream_redis.store
    row = fake_dream_db.rows["p1"]
    assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)
    assert (await reap_expired_passes()).listed == 0
