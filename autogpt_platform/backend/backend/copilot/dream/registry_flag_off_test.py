"""Tests for a dream-system cron whose flag goes off: the registry reads the
owner's flags only at an ensure, and there a cron whose flag is
authoritatively off loses its record and then its job, whatever else is on
and whatever the ensure's mode, while one whose flag could not be read keeps
both. The reconcile reads no flag.

The probes are an independent review's: a learning-only account kept its
recorded cron through every ensure once learning went off, and an ordinary
ensure with the dream on kept the learning cron's id. Fakes come from
``registry_fakes_test.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.util.feature_flag import Flag

from . import registry, scope_jobs
from .registry_fakes_test import (
    ACCOUNT,
    EXPERT,
    PARIS,
    TOKYO,
    USER,
    RegistryEnv,
    install_registry_env,
    learning_job_id,
)

LEARNING = learning_job_id(ACCOUNT)
NIGHTLY = f"dream_nightly_batch_{USER}"
LEARN = Flag.DREAM_SKILL_LEARNING_ENABLED
DREAM = Flag.DREAM_PASS_ENABLED
COMMUNITIES = Flag.GRAPHITI_COMMUNITIES_ENABLED


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    env = install_registry_env(monkeypatch)
    env.flags = {LEARN: True}
    return env


def _lose(env: RegistryEnv, job_id: str) -> None:
    del env.scheduler.jobs[job_id]
    del env.scheduler.scopes[job_id]


def _record_goes_first(env: RegistryEnv) -> list[str | None]:
    """Make the removal RPC note what the row recorded for the cron it
    removes, at the moment it removes it."""
    seen: list[str | None] = []
    remove = env.scheduler.remove_scope_memory_job.side_effect

    async def noting(**kwargs: Any) -> bool:
        seen.append(env.db.rows[USER].learning_job_id)
        return await remove(**kwargs)

    env.scheduler.remove_scope_memory_job.side_effect = noting
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("force", [False, True], ids=["ordinary", "forced"])
async def test_learning_off_on_a_learning_only_account_removes_its_cron(env, force):
    """The review's probe: the only enabled flag goes off after the reconcile
    restored the cron with its tag. The ensure forgets the cron, then removes
    it, and no later pass brings it back."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    _lose(env, LEARNING)
    await env.reconcile()
    assert LEARNING in env.scheduler.tagged
    seen = _record_goes_first(env)
    env.flags[LEARN] = False

    result = await registry.ensure_scope_scheduled(ACCOUNT, force_refresh=force)

    assert result["skill_learning_nightly"] == {
        "skipped": True,
        "reason": "skill_learning_disabled",
    }
    assert seen == [None]
    assert env.db.rows[USER].learning_job_id is None
    assert env.scheduler.jobs == {}
    reports = [await env.reconcile() for _ in range(3)]
    assert env.scheduler.jobs == {}
    assert all(not report.restored for report in reports)


@pytest.mark.asyncio
async def test_learning_off_with_the_dream_on_removes_only_learning(env):
    """The review's second probe: an ordinary ensure, the dream still on,
    the row not stale."""
    env.flags[DREAM] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["dream_nightly_batch"] is None
    row = env.db.rows[USER]
    assert (row.nightly_job_id, row.learning_job_id) == (NIGHTLY, None)
    assert env.scheduler.jobs == {NIGHTLY: PARIS}


@pytest.mark.asyncio
async def test_the_last_enabled_flag_going_off_removes_its_cron(env):
    """The dream goes off first and its cron goes; then learning, the last
    flag on, goes off, and its cron goes too, though no cron is left to
    register."""
    env.flags[DREAM] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[DREAM] = False

    await registry.ensure_scope_scheduled(ACCOUNT)

    assert env.db.rows[USER].nightly_job_id is None
    assert env.scheduler.jobs == {LEARNING: PARIS}

    env.flags[LEARN] = False
    await registry.ensure_scope_scheduled(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.nightly_job_id, row.learning_job_id) == (None, None)
    assert env.scheduler.jobs == {}
    report = await env.reconcile()
    assert (report.restored, env.scheduler.jobs) == ([], {})


@pytest.mark.asyncio
async def test_an_experts_crons_go_when_their_flags_go_off(env):
    env.flags = {DREAM: True, COMMUNITIES: True}
    await registry.ensure_scope_scheduled(EXPERT)
    env.flags = {DREAM: False, COMMUNITIES: False}

    await registry.ensure_scope_scheduled(EXPERT)

    row = env.db.rows[EXPERT.scope_key]
    assert (row.community_job_id, row.nightly_job_id) == (None, None)
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["unanswered", "raises"])
@pytest.mark.parametrize("force", [False, True], ids=["ordinary", "forced"])
async def test_a_flag_that_cannot_be_read_keeps_the_cron_and_its_record(
    env, monkeypatch, lookup, force
):
    """A lookup that fails, or answers only with its default, is not an
    off: the cron and its record stay, and with no flag read at all the
    ensure makes no database call."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    if lookup == "raises":

        async def failing(flag: Flag, user_id: str) -> tuple[bool, bool]:
            raise ConnectionError("flags down")

        monkeypatch.setattr(scope_jobs, "evaluate_feature_flag", failing)
    else:
        env.flags = {flag: None for flag in Flag}
    env.db.fail = True

    result = await registry.ensure_scope_scheduled(ACCOUNT, force_refresh=force)

    assert {r["reason"] for r in result.values()} == {"flag_unavailable"}
    env.db.fail = False
    assert env.db.rows[USER].learning_job_id == LEARNING
    assert env.scheduler.jobs == {LEARNING: PARIS}
    env.scheduler.remove_scope_memory_job.assert_not_awaited()
    report = await env.reconcile()
    assert env.scheduler.jobs == {LEARNING: PARIS} and not report.dropped


@pytest.mark.asyncio
async def test_an_unread_flag_keeps_its_record_through_a_timezone_change(env):
    """Another cron re-registers in the new timezone; the one whose flag
    could not be read keeps its record, so the reconcile moves it there too
    rather than leaving it unrecorded in the old one."""
    env.flags[DREAM] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = None
    env.timezone = TOKYO

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["dream_nightly_batch"]["id"] == NIGHTLY
    assert result["skill_learning_nightly"]["reason"] == "flag_unavailable"
    row = env.db.rows[USER]
    assert (row.timezone, row.learning_job_id) == (TOKYO, LEARNING)
    await env.reconcile()
    assert env.scheduler.jobs == {NIGHTLY: TOKYO, LEARNING: TOKYO}


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["forget", "remove"])
async def test_a_removal_not_confirmed_keeps_the_record_and_is_tried_again(
    env, monkeypatch, failing
):
    """The record goes first, then the job. When either step fails the
    record stays (written back if the forget had landed), so the next
    ensure tries again rather than leave an unrecorded cron running."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False
    if failing == "forget":
        forget = env.db.forget_scope_job

        async def refused(*args: Any) -> bool:
            raise ConnectionError("db down")

        monkeypatch.setattr(env.db, "forget_scope_job", refused)
    else:
        env.scheduler.remove_scope_memory_job.side_effect = ConnectionError("down")

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["skill_learning_nightly"] == {
        "skipped": True,
        "reason": "removal_failed",
    }
    assert env.db.rows[USER].learning_job_id == LEARNING
    assert env.scheduler.jobs == {LEARNING: PARIS}

    if failing == "forget":
        monkeypatch.setattr(env.db, "forget_scope_job", forget)
    else:
        env.scheduler.remove_scope_memory_job.side_effect = env.scheduler._remove_one
    await registry.ensure_scope_scheduled(ACCOUNT)

    assert env.db.rows[USER].learning_job_id is None
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_every_flag_off_and_no_row_creates_nothing(env):
    """An owner with every flag off and no row: one row read, no row
    created and no scheduler call."""
    env.flags = {}

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {
        "graphiti_communities_disabled",
        "dream_pass_disabled",
        "skill_learning_disabled",
    }
    assert env.db.rows == {}
    assert env.adds() == 0
    env.scheduler.remove_scope_memory_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_timezone_change_removes_the_crons_of_an_owner_with_every_flag_off(
    env,
):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags = {}
    env.timezone = TOKYO

    await registry.reregister_user(USER)

    assert env.db.rows[USER].learning_job_id is None
    assert env.scheduler.jobs == {}
