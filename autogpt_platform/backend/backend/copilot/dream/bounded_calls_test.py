"""A cut that lands as a bounded call answers is kept. On Python 3.11,
``asyncio.wait_for`` runs its call in a task of its own and, cancelled from
outside once that task is done, returns the call's result: the cut is lost,
and a reaper row cut at its deadline runs on into the time of the rows behind
it. Every bounded call on the reaper's path is a timeout around a direct
await instead, which keeps it. Each test cancels the task making the call
from inside the call, as the call answers (the moment ``wait_for`` lost the
cut), and checks that the task still ends cancelled."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassRoute

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import DreamPassUpdate

from . import cleanup as cleanup_mod
from . import provider_batch as provider_batch_mod
from . import reaper as reaper_mod
from . import store as store_mod
from .batch_costs import PhaseCharges
from .reaper_test import _seed

_SCOPE = MemoryScope.build("u1")


class _Cut:
    """Stands in for a call: cancels the task making it, then answers, both
    in the same step."""

    def __init__(self) -> None:
        self.task: asyncio.Task[Any] | None = None

    def answering(self, answer: object) -> Callable[..., Awaitable[object]]:
        async def call(*_args: object, **_kwargs: object) -> object:
            assert self.task is not None
            self.task.cancel()
            return answer

        return call

    async def run(self, call: Awaitable[object]) -> asyncio.Task[Any]:
        self.task = asyncio.ensure_future(call)
        await asyncio.wait([self.task])
        return self.task


@pytest.fixture
def cut() -> _Cut:
    return _Cut()


@pytest.fixture
def provider_key(mocker) -> None:
    mocker.patch.object(provider_batch_mod, "anthropic_api_key", return_value="k")


async def test_the_provider_cancel(mocker, cut, provider_key):
    mocker.patch.object(provider_batch_mod, "cancel_batch", cut.answering(True))

    task = await cut.run(provider_batch_mod.cancel_provider_batch("b-1"))

    assert task.cancelled()


async def test_the_provider_status_read(mocker, cut, provider_key):
    mocker.patch.object(
        provider_batch_mod, "cancel_batch", AsyncMock(return_value=False)
    )
    mocker.patch.object(provider_batch_mod, "poll_batch", cut.answering("ended"))

    task = await cut.run(provider_batch_mod.provider_batch_stopped("b-1"))

    assert task.cancelled()


async def test_a_record_write(mocker, cut, fake_dream_db):
    mocker.patch.object(fake_dream_db, "update_dream_pass", cut.answering(True))

    task = await cut.run(store_mod.write_stop("p1", DreamPassUpdate()))

    assert task.cancelled()


async def test_the_reapers_take_of_a_scope(monkeypatch, cut, fake_dream_redis):
    monkeypatch.setattr(fake_dream_redis, "set", cut.answering(True))

    task = await cut.run(reaper_mod._take_scope(_SCOPE))

    assert task.cancelled()


async def test_the_reapers_note_of_the_row_it_tries(
    monkeypatch, cut, fake_dream_db, fake_dream_redis
):
    _seed(fake_dream_db, "p1", route=DreamPassRoute.SYNC)
    monkeypatch.setattr(fake_dream_redis, "hset", cut.answering(1))

    task = await cut.run(reaper_mod._note_tried(fake_dream_db.record("p1")))

    assert task.cancelled()


async def test_the_reapers_read_of_where_its_scan_stands(
    monkeypatch, cut, fake_dream_redis
):
    monkeypatch.setattr(fake_dream_redis, "hgetall", cut.answering({}))

    task = await cut.run(reaper_mod._read_scan("open"))

    assert task.cancelled()


async def test_the_reapers_release_of_its_hold(mocker, cut):
    mocker.patch.object(reaper_mod, "release_dream_lock", cut.answering(True))

    task = await cut.run(reaper_mod._release_within_bound(_SCOPE, "reaper:x"))

    assert task.cancelled()


async def test_the_cleanups_state_read(mocker, cut):
    mocker.patch.object(cleanup_mod, "read_state", cut.answering({}))

    task = await cut.run(cleanup_mod._charge("p1", _SCOPE, None, ()))

    assert task.cancelled()


async def test_the_cleanups_charges(mocker, cut):
    mocker.patch.object(
        cleanup_mod,
        "read_state",
        AsyncMock(return_value={"consolidate": {"content": "{}"}}),
    )
    mocker.patch.object(
        cleanup_mod, "charge_landed_phases", cut.answering(PhaseCharges(settled=True))
    )

    task = await cut.run(
        cleanup_mod._charge("p1", _SCOPE, {"consolidate": "claude-sonnet-5"}, ())
    )

    assert task.cancelled()


async def test_the_cleanups_unlock(mocker, cut):
    mocker.patch.object(cleanup_mod, "release_dream_lock", cut.answering(True))

    task = await cut.run(cleanup_mod._unlock("p1", _SCOPE, "token", attribute=False))

    assert task.cancelled()


async def test_the_cleanups_read_of_a_tokenless_rows_lock(mocker, cut):
    mocker.patch.object(cleanup_mod, "read_dream_lock_token", cut.answering(None))

    task = await cut.run(cleanup_mod._lock_is_not_the_passes("p1", _SCOPE))

    assert task.cancelled()


async def test_the_cleanups_delete(mocker, cut):
    mocker.patch.object(cleanup_mod, "best_effort_cleanup", cut.answering(True))

    task = await cut.run(cleanup_mod._deleted("p1"))

    assert task.cancelled()
