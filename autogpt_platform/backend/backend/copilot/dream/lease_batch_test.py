"""A batch pass's lease through the real callbacks, over the in-memory Redis
(its TTLs run down by hand here, the way Redis would expire the keys) and
the pass's row in the in-memory store: every callback renews the lock and the
row's lease, so a chain of three batches that runs past a day keeps its lock
and applies; a callback whose lock a newer pass took ends the pass without
chaining. Only the provider, the charges and apply are stubbed."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import DreamPassDraft
from backend.executor.batch_executor import PendingEntry
from backend.util.llm.providers import BatchResultRow

from . import batch_callbacks as batch_callbacks_mod
from . import lease as lease_mod
from .batch_callbacks import handle_dream_batch_result
from .batch_state import state_key
from .batch_submit import (
    input_bundle_key,
    persist_input_bundle,
    refresh_input_bundle_ttl,
)
from .fetch import DreamInput
from .locks import BATCH_LOCK_TTL_SECONDS
from .schemas import ConsolidationOutput, DreamOperations, RecombinationOutput

_SCOPE = MemoryScope.for_user("u1")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")
_TOKEN = "our-token"
_HOURS_20 = 20 * 60 * 60
_OUTPUTS: dict[str, BaseModel] = {
    "consolidate": ConsolidationOutput(),
    "recombine": RecombinationOutput(),
    "sanitize": DreamOperations(summary_for_user="ok"),
}


@pytest.fixture(autouse=True)
def provider(mocker) -> AsyncMock:
    """The next phase's submit, which re-arms the bundle's TTL as the real
    one does; the provider key; the charges."""

    async def submit(**kwargs: Any) -> MagicMock:
        await refresh_input_bundle_ttl(kwargs["pass_id"])
        return MagicMock(provider_batch_id=f"msgbatch_{kwargs['phase']}")

    mocker.patch.object(batch_callbacks_mod, "anthropic_api_key", return_value="k")
    mocker.patch("backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock())
    return mocker.patch.object(
        batch_callbacks_mod, "submit_phase", AsyncMock(side_effect=submit)
    )


@pytest.fixture
def apply(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.apply.apply_operations",
        AsyncMock(return_value={"session_id": "s", "consolidated_count": 0}),
    )


class TestTheBatchLease:
    async def test_each_callback_renews_the_lock_and_the_rows_lease(
        self, fake_dream_db, fake_dream_redis
    ):
        await _submitted(fake_dream_db, fake_dream_redis)
        fake_dream_redis.ttls[_LOCK_KEY] = 60
        before = datetime.now(timezone.utc)

        await _deliver("consolidate")

        assert fake_dream_redis.ttls[_LOCK_KEY] == BATCH_LOCK_TTL_SECONDS
        row = fake_dream_db.rows["p1"]
        assert row["lease_token"] == _TOKEN
        assert row["lease_expires_at"] >= before + timedelta(
            seconds=BATCH_LOCK_TTL_SECONDS
        )
        assert row["status"] is DreamPassStatus.SUBMITTED

    async def test_a_chain_of_three_batches_past_a_day_keeps_its_lock_and_applies(
        self, fake_dream_db, fake_dream_redis, apply, provider
    ):
        """Twenty hours per batch, sixty in all: past the lock's TTL from
        its submit twice over, yet each callback renewed it, so the last
        one still holds the scope and applies."""
        await _submitted(fake_dream_db, fake_dream_redis)
        for phase in ("consolidate", "recombine", "sanitize"):
            _elapse(fake_dream_redis, _HOURS_20)
            await _deliver(phase)

        apply.assert_awaited_once()
        assert provider.await_count == 2
        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.COMPLETE
        assert (row["lease_token"], row["lease_expires_at"]) == (None, None)
        assert _LOCK_KEY not in fake_dream_redis.store

    async def test_a_callback_whose_lock_was_taken_ends_the_pass_without_chaining(
        self, fake_dream_db, fake_dream_redis, provider, caplog
    ):
        await _submitted(fake_dream_db, fake_dream_redis)
        fake_dream_redis.store[_LOCK_KEY] = "newer-token"

        with caplog.at_level(logging.WARNING):
            await _deliver("consolidate")

        provider.assert_not_awaited()
        row = fake_dream_db.rows["p1"]
        error = "recombine: dream lock lost before recombine"
        assert (row["status"], row["error"]) == (DreamPassStatus.ERRORED, error)
        assert [p.phase for p in row["usage"].phases] == ["consolidate"]
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"
        assert state_key("p1") not in fake_dream_redis.hashes
        assert input_bundle_key("p1") not in fake_dream_redis.store
        assert f"Dream batch pass p1 stops: {error}" in caplog.text

    async def test_a_renewal_redis_cannot_answer_goes_on_down_the_chain(
        self, mocker, fake_dream_db, fake_dream_redis, provider, caplog
    ):
        await _submitted(fake_dream_db, fake_dream_redis)
        mocker.patch.object(
            lease_mod,
            "extend_dream_lock",
            AsyncMock(side_effect=ConnectionError("redis down")),
        )

        with caplog.at_level(logging.WARNING):
            await _deliver("consolidate")

        provider.assert_awaited_once()
        assert "Dream pass p1: could not renew its lease" in caplog.text
        assert fake_dream_db.rows["p1"]["status"] is DreamPassStatus.SUBMITTED


async def _submitted(fake_dream_db, fake_dream_redis) -> None:
    """Pass p1 as its handoff leaves it: consolidate submitted, the lock
    extended to a batch lifetime under the token its bundle carries."""
    now = datetime.now(timezone.utc)
    bundle = DreamInput(
        user_id="u1", group_id=_SCOPE.group_id, window_start=now, window_end=now
    )
    await persist_input_bundle("p1", bundle, lock_token=_TOKEN)
    fake_dream_redis.store[_LOCK_KEY] = _TOKEN
    fake_dream_redis.ttls[_LOCK_KEY] = BATCH_LOCK_TTL_SECONDS
    fake_dream_db.seed(
        DreamPassDraft(
            id="p1",
            user_id="u1",
            scope_key=_SCOPE.scope_key,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.CONSOLIDATE,
            lease_token=_TOKEN,
            lease_expires_at=now + timedelta(seconds=BATCH_LOCK_TTL_SECONDS),
        ),
        provider_batch_id="msgbatch_consolidate",
    )


async def _deliver(phase: str) -> None:
    now = datetime.now(timezone.utc)
    entry = PendingEntry(
        provider="anthropic",
        provider_batch_id=f"msgbatch_{phase}",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u1",
            "pass_id": "p1",
            "job_id": "",
            "phase": phase,
            "phase_models": {p: "claude-sonnet-5" for p in _OUTPUTS},
        },
    )
    row = BatchResultRow(
        custom_id=f"p1_{phase}",
        content=_OUTPUTS[phase].model_dump_json(),
        input_tokens=10,
        output_tokens=20,
    )
    await handle_dream_batch_result(entry, [row])


def _elapse(fake_dream_redis, seconds: int) -> None:
    """Let *seconds* pass on the in-memory Redis: every TTL runs down, and a
    key whose TTL runs out goes, as Redis would expire it."""
    for key, ttl in list(fake_dream_redis.ttls.items()):
        if ttl > seconds:
            fake_dream_redis.ttls[key] = ttl - seconds
            continue
        fake_dream_redis.store.pop(key, None)
        fake_dream_redis.hashes.pop(key, None)
        del fake_dream_redis.ttls[key]
