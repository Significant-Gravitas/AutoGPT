"""The dream pass reaper also sweeps the memory graphs where a dream write's
derivation record failed (``graphiti/provenance_pending.py``), with what its
budget leaves after the rows. The sweep itself is pinned in
``graphiti/recall_reconcile_test.py`` and run live in
``graphiti/recall_provenance_integration_test.py``.
"""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.provenance_pending import Swept

from . import reaper as reaper_mod
from .reaper import reap_expired_passes


@pytest.fixture
def sweep(mocker) -> AsyncMock:
    return mocker.patch.object(
        reaper_mod, "sweep_pending", AsyncMock(return_value=Swept(completed=2))
    )


async def test_a_run_sweeps_the_pending_dream_records(
    fake_dream_db, fake_dream_redis, sweep: AsyncMock
) -> None:
    run = await reap_expired_passes()

    sweep.assert_awaited_once()
    assert (run.listed, run.reconciled) == (0, 2)


async def test_a_run_short_of_budget_leaves_the_sweep_to_the_next(
    fake_dream_db, fake_dream_redis, sweep: AsyncMock, mocker
) -> None:
    mocker.patch.object(reaper_mod, "REAPER_BUDGET_SECONDS", 10.0)

    run = await reap_expired_passes()

    sweep.assert_not_awaited()
    assert run.reconciled == 0


async def test_a_sweep_that_raises_does_not_fail_the_run(
    fake_dream_db, fake_dream_redis, sweep: AsyncMock
) -> None:
    sweep.side_effect = RuntimeError("down")

    run = await reap_expired_passes()

    assert (run.listed, run.reconciled) == (0, 0)
