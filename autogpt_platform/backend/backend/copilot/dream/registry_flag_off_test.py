"""Tests for a dream-system cron whose flag goes off: the registry leaves
the scope's recorded crons in place, and pause or wipe removes them. Their
bodies check the flag at run time and do nothing; the scheduler's tests pin
that (``TestSkillLearningScopeGate::test_the_flag_off_still_short_circuits``,
``TestExecuteNightlyBatchSyncRuntimeGate``). Cleaning them up on flag-off
needs generation-aware intent, which is left for later.

Fakes come from ``registry_fakes_test.py``.
"""

from __future__ import annotations

import pytest

from backend.util.feature_flag import Flag

from . import registry
from .registry_fakes_test import (
    ACCOUNT,
    EXPERT,
    PARIS,
    PAUSED,
    USER,
    WIPED,
    RegistryEnv,
    install_registry_env,
    job_ids,
    learning_job_id,
)

LEARNING = learning_job_id(ACCOUNT)
NIGHTLY = f"dream_nightly_batch_{USER}"
LEARN = Flag.DREAM_SKILL_LEARNING_ENABLED
DREAM = Flag.DREAM_PASS_ENABLED


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    env = install_registry_env(monkeypatch)
    env.flags = {LEARN: True}
    return env


@pytest.mark.asyncio
async def test_turning_learning_off_leaves_its_cron_recorded(env):
    """Community maintenance does not erase the recorded learning cron."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["skill_learning_nightly"] == {
        "skipped": True,
        "reason": "skill_learning_disabled",
    }
    assert env.db.rows[USER].learning_job_id == LEARNING
    reports = [await env.reconcile() for _ in range(3)]
    assert env.scheduler.jobs == {f"community_rebuild_{USER}": PARIS, LEARNING: PARIS}
    assert all(
        not (report.dropped or report.removed or report.restored) for report in reports
    )


@pytest.mark.asyncio
async def test_turning_learning_off_with_the_dream_on_leaves_both_recorded(env):
    env.flags[DREAM] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False

    await registry.ensure_scope_scheduled(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.nightly_job_id, row.learning_job_id) == (NIGHTLY, LEARNING)
    assert env.scheduler.jobs == {
        f"community_rebuild_{USER}": PARIS,
        NIGHTLY: PARIS,
        LEARNING: PARIS,
    }


@pytest.mark.asyncio
async def test_turning_the_dream_off_leaves_an_experts_crons_recorded(env):
    env.flags = {DREAM: True}
    await registry.ensure_scope_scheduled(EXPERT)
    env.flags = {}

    await registry.ensure_scope_scheduled(EXPERT)

    row = env.db.rows[EXPERT.scope_key]
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [PAUSED, WIPED])
async def test_a_stop_removes_the_crons_of_a_flag_turned_off(env, state):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False
    await registry.ensure_scope_scheduled(ACCOUNT)

    stop = registry.pause_scope if state == PAUSED else registry.mark_wiped
    assert await stop(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id) == (state, None)
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_other_flags_off_preserves_learning_when_registry_is_unavailable(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags = {}
    env.db.fail = True

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {
        "registry_unavailable",
        "dream_pass_disabled",
        "skill_learning_disabled",
    }
    env.db.fail = False
    assert env.db.rows[USER].learning_job_id == LEARNING
