"""Tests for the scheduler's reconcile pass over the memory-scope crons
(``scope_reconcile.py``), against the registry fakes: it restores what the
event-driven paths lose, removes what they leave behind, and converges.

The race probes are an independent review's. Round 1's left an ACTIVE row
with its job ids recorded and no crons; each now ends with the crons back
after one pass. Round 2's: a restore that lands after a timezone change,
set right by the next pass, and passes cut off at their ceiling, each now
carrying on where the last one stopped. Round 3's: a pass cut off inside a
scope's work, and a place Redis would not keep, each now moving on; and a
restore whose cron the row stopped recording meanwhile, taken back.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.data import redis_client

from . import registry, scope_crons, scope_jobs, scope_reconcile
from .conftest import FakeAsyncRedis
from .registry_fakes_test import (
    ACCOUNT,
    ACTIVE,
    EXPERT,
    OTHER_EXPERT,
    PARIS,
    PAUSED,
    TOKYO,
    USER,
    WIPED,
    RegistryEnv,
    install_registry_env,
    job_ids,
)

TIMEOUT = 2


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


def _lose_every_cron(env: RegistryEnv) -> None:
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()


async def _stale_takedown(env: RegistryEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """One interleaving from the review: an ensure parks before recording
    while a pause lands, its record is refused, and its takedown parks while
    a resume registers and records the crons; then the stale takedown lands
    and takes them."""
    record, remove = env.db.record_scope_jobs, env.scheduler._remove
    record_reached, record_go = asyncio.Event(), asyncio.Event()
    remove_reached, remove_go = asyncio.Event(), asyncio.Event()
    ensure: asyncio.Task | None = None

    async def parked_record(*args: Any, **kwargs: Any) -> bool:
        if asyncio.current_task() is ensure and not record_reached.is_set():
            record_reached.set()
            await record_go.wait()
        return await record(*args, **kwargs)

    async def parked_remove(*, scope):
        if asyncio.current_task() is ensure and not remove_reached.is_set():
            remove_reached.set()
            await remove_go.wait()
        return await remove(scope=scope)

    monkeypatch.setattr(env.db, "record_scope_jobs", parked_record)
    env.scheduler.remove_scope_memory_jobs.side_effect = parked_remove
    ensure = asyncio.create_task(
        registry.ensure_scope_scheduled(EXPERT, force_refresh=True)
    )
    await asyncio.wait_for(record_reached.wait(), TIMEOUT)
    assert await registry.pause_scope(EXPERT)
    record_go.set()
    await asyncio.wait_for(remove_reached.wait(), TIMEOUT)
    await registry.resume_scope(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    remove_go.set()
    await asyncio.wait_for(ensure, TIMEOUT)
    monkeypatch.setattr(env.db, "record_scope_jobs", record)
    env.scheduler.remove_scope_memory_jobs.side_effect = remove


@pytest.mark.asyncio
@pytest.mark.parametrize("cycles", [1, 2])
async def test_every_pause_resume_interleaving_is_restored_by_one_pass(
    env, monkeypatch, cycles
):
    """After ``cycles`` stale takedowns the row is ACTIVE with both ids
    recorded and no crons, and an ordinary ensure, trusting the ids,
    restores nothing. One pass restores them."""
    await registry.ensure_scope_scheduled(EXPERT)
    for _ in range(cycles):
        await _stale_takedown(env, monkeypatch)
    row = env.db.rows[EXPERT.scope_key]
    assert row.state == ACTIVE
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert env.scheduler.jobs == {}
    await registry.ensure_scope_scheduled(EXPERT)
    assert env.scheduler.jobs == {}, "an ensure trusts the recorded ids"

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert sorted(report.restored) == sorted(job_ids(EXPERT))


@pytest.mark.asyncio
async def test_a_pause_whose_re_read_fails_reports_false_and_is_restored(
    env, monkeypatch
):
    """The review's probe through the pause: a resume registers its crons just
    before the pause's removal takes them, and the pause's one re-read
    fails. The pause no longer reports success, and one pass restores the
    crons."""
    await registry.ensure_scope_scheduled(EXPERT)
    remove = env.scheduler._remove
    get = env.db.get_scope_schedule

    async def resume_before_remove(*, scope):
        await registry.resume_scope(scope)
        removed = await remove(scope=scope)
        monkeypatch.setattr(
            env.db, "get_scope_schedule", AsyncMock(side_effect=ConnectionError)
        )
        return removed

    env.scheduler.remove_scope_memory_jobs.side_effect = resume_before_remove

    assert not await registry.pause_scope(EXPERT), "the re-read failed"
    assert env.db.rows[EXPERT.scope_key].state == ACTIVE
    assert env.scheduler.jobs == {}

    monkeypatch.setattr(env.db, "get_scope_schedule", get)
    env.scheduler.remove_scope_memory_jobs.side_effect = remove
    await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
async def test_a_gate_fire_under_a_paused_row_never_takes_crons_down(env):
    """The review's probe through the gate needed the gate to remove a
    straggler's crons, racing a resume. It no longer does: a fire under a
    PAUSED row only skips, so a resume after it keeps the crons, and the
    pass after leaves them."""
    await registry.ensure_scope_scheduled(EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)

    assert not await scope_crons.memory_scope_may_fire(
        EXPERT, AsyncMock(return_value="active")
    )
    env.scheduler.remove_scope_memory_jobs.assert_not_awaited()
    await registry.resume_scope(EXPERT)
    await env.reconcile()

    assert env.db.rows[EXPERT.scope_key].state == ACTIVE
    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [PAUSED, WIPED])
async def test_a_pass_removes_the_crons_of_a_scope_that_left_active(env, state):
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, state)

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(OTHER_EXPERT)
    assert report.removed == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_a_pass_removes_a_deleted_experts_crons_and_keeps_a_legacy_accounts(
    env,
):
    """An expert's crons with no row are an expert deleted; an account's are
    crons from before the registry, which the cron gate lets run."""
    for scope in (EXPERT, ACCOUNT):
        await env.scheduler.add_scope_nightly_batch_schedule(
            scope=scope, user_timezone="UTC"
        )

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == {f"dream_nightly_batch_{ACCOUNT.scope_key}"}
    assert report.removed == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_a_pass_registers_what_the_row_records_and_reads_no_flag(
    env, monkeypatch
):
    """An ACTIVE row's crons are the ones it records: the owner's flags, off
    by now, are not read, a cron the row no longer records stays missing,
    and a row recording none is left to the next ensure."""
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    env.db.update(EXPERT.scope_key, community_job_id=None)
    env.db.update(OTHER_EXPERT.scope_key, community_job_id=None, nightly_job_id=None)
    _lose_every_cron(env)
    flags = AsyncMock(return_value=False)
    monkeypatch.setattr(scope_jobs, "is_feature_enabled", flags)

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == {f"dream_nightly_batch_{EXPERT.scope_key}"}
    assert report.restored == [f"dream_nightly_batch_{EXPERT.scope_key}"]
    flags.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_cron_the_row_does_not_record_is_left_where_it_is(env):
    """A cron an ACTIVE row does not record (its record was refused, or it
    predates the registry) is the registry's to decide: the pass neither
    removes it nor moves it to the row's timezone."""
    await registry.ensure_scope_scheduled(EXPERT)
    env.db.update(EXPERT.scope_key, community_job_id=None, timezone=TOKYO)
    community = f"community_rebuild_{EXPERT.scope_key}"
    env.scheduler.jobs[f"dream_nightly_batch_{EXPERT.scope_key}"] = TOKYO

    report = await env.reconcile()

    assert env.scheduler.jobs[community] == PARIS
    assert report.restored == report.removed == []


@pytest.mark.asyncio
async def test_a_pass_restores_a_cron_in_the_rows_timezone(env):
    env.timezone = TOKYO
    await registry.ensure_scope_scheduled(EXPERT)
    env.timezone = PARIS  # the owner moved; the row still says Tokyo
    _lose_every_cron(env)

    await env.reconcile()

    assert set(env.scheduler.jobs.values()) == {TOKYO}


@pytest.mark.asyncio
async def test_a_cron_in_another_timezone_is_registered_again_in_the_rows(env):
    await registry.ensure_scope_scheduled(EXPERT)
    for job_id in job_ids(EXPERT):
        env.scheduler.jobs[job_id] = TOKYO

    report = await env.reconcile()

    assert set(env.scheduler.jobs.values()) == {PARIS}
    assert sorted(report.restored) == sorted(job_ids(EXPERT))
    assert (await env.reconcile()).restored == []


@pytest.mark.asyncio
async def test_a_restore_landing_after_a_timezone_change_is_undone_by_the_next_pass(
    env, monkeypatch
):
    """The review's probe: a restore reads the row (Tokyo) and parks while
    the owner moves to Paris and ``reregister_user`` registers and records
    the crons there; released, it registers them in Tokyo again. The row
    says Paris with both ids recorded, so the next pass, comparing
    timezones, puts them back in Paris."""
    env.timezone = TOKYO
    await registry.ensure_scope_scheduled(EXPERT)
    _lose_every_cron(env)
    add = env.crons.add
    reached, go = asyncio.Event(), asyncio.Event()

    async def parked_add(job, scope, user_timezone):
        if not reached.is_set():
            reached.set()
            await go.wait()
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "add", parked_add)
    stale = asyncio.create_task(env.reconcile())
    await asyncio.wait_for(reached.wait(), TIMEOUT)
    env.timezone = PARIS
    await registry.reregister_user(USER)
    assert _zones(env, EXPERT) == {PARIS}
    go.set()
    await asyncio.wait_for(stale, TIMEOUT)
    assert _zones(env, EXPERT) == {TOKYO}, "the stale restore"
    assert env.db.rows[EXPERT.scope_key].timezone == PARIS

    report = await env.reconcile()

    assert _zones(env, EXPERT) == {PARIS}
    assert sorted(report.restored) == sorted(job_ids(EXPERT))


def _zones(env: RegistryEnv, scope) -> set[str]:
    """The timezones the scheduler runs ``scope``'s two crons in."""
    return {env.scheduler.jobs[job_id] for job_id in job_ids(scope)}


@pytest.mark.asyncio
async def test_a_pause_landing_during_a_pass_wins(env, monkeypatch):
    """The listing saw the row ACTIVE and its crons gone; a pause lands before
    the pass registers them: the pass reads the row again and registers
    nothing."""
    await registry.ensure_scope_scheduled(EXPERT)
    _lose_every_cron(env)
    listing = env.db.list_scope_schedules

    async def paused_after_listing(**kwargs: Any):
        page = await listing(**kwargs)
        await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)
        return page

    monkeypatch.setattr(env.db, "list_scope_schedules", paused_after_listing)

    report = await env.reconcile()

    assert env.scheduler.jobs == {}
    assert report.restored == []


@pytest.mark.asyncio
async def test_a_resume_landing_during_a_pass_wins(env, monkeypatch):
    """The listing saw the row PAUSED with crons behind it; a resume lands
    before the pass removes them: the pass reads the row again and removes
    nothing."""
    await registry.ensure_scope_scheduled(EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)
    listing = env.db.list_scope_schedules

    async def resumed_after_listing(**kwargs: Any):
        page = await listing(**kwargs)
        await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, ACTIVE)
        return page

    monkeypatch.setattr(env.db, "list_scope_schedules", resumed_after_listing)

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert report.removed == []


@pytest.mark.asyncio
async def test_two_passes_at_once_converge(env, monkeypatch):
    """Two schedulers reconciling at once, as in a rolling deploy, each
    listing before the other changes anything: the same end state, and
    nothing failed."""
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    _lose_every_cron(env)
    await registry.ensure_scope_scheduled(ACCOUNT)
    await env.db.set_scope_state(ACCOUNT.owner_user_id, ACCOUNT.scope_key, PAUSED)
    listing = env.crons.listing
    both_listed = asyncio.Barrier(2)

    async def list_together():
        listed = await listing()
        await both_listed.wait()
        return listed

    monkeypatch.setattr(env.crons, "listing", list_together)

    reports = await asyncio.gather(env.reconcile(), env.reconcile())

    assert set(env.scheduler.jobs) == job_ids(EXPERT) | job_ids(OTHER_EXPERT)
    assert all(report.failed == [] for report in reports)
    monkeypatch.setattr(env.crons, "listing", listing)
    again = await env.reconcile()
    assert (again.restored, again.removed) == ([], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("down", ["registry", "scheduler"])
async def test_a_pass_that_cannot_list_changes_nothing(env, monkeypatch, down):
    await registry.ensure_scope_scheduled(EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)
    if down == "registry":
        env.db.fail = True
    else:
        monkeypatch.setattr(
            env.crons, "listing", AsyncMock(side_effect=TimeoutError("jobstore"))
        )

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert report == scope_reconcile.ReconcileReport()


@pytest.mark.asyncio
async def test_a_failed_registration_is_reported_and_the_pass_goes_on(env):
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    _lose_every_cron(env)
    add = env.scheduler.add_scope_nightly_batch_schedule.side_effect

    async def refuse_expert_one(*, scope, user_timezone):
        if scope == EXPERT:
            raise ConnectionError("jobstore down")
        return await add(scope=scope, user_timezone=user_timezone)

    env.scheduler.add_scope_nightly_batch_schedule.side_effect = refuse_expert_one

    report = await env.reconcile()

    assert report.failed == [EXPERT.scope_key]
    assert job_ids(OTHER_EXPERT) <= set(env.scheduler.jobs)


@pytest.mark.asyncio
async def test_a_pass_reads_every_page_of_the_registry(env, monkeypatch):
    monkeypatch.setattr(scope_reconcile, "PAGE_SIZE", 2)
    scopes = [registry.MemoryScope.for_expert(USER, f"e-{i}") for i in range(5)]
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    _lose_every_cron(env)

    await env.reconcile()

    assert set(env.scheduler.jobs) == set().union(*(job_ids(s) for s in scopes))


@pytest.mark.asyncio
async def test_a_pass_settles_each_scope_in_key_order(env, monkeypatch):
    """Removal and restore go scope by scope, in key order: a straggler's
    crons go before a later scope's are restored, not after every restore."""
    scopes = sorted(
        (registry.MemoryScope.for_expert(USER, f"e-{i}") for i in range(4)),
        key=lambda scope: scope.scope_key,
    )
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    for scope in scopes[1:]:
        for job_id in job_ids(scope):
            env.scheduler.jobs.pop(job_id)
            env.scheduler.scopes.pop(job_id)
    await env.db.set_scope_state(USER, scopes[0].scope_key, PAUSED)
    done: list[str] = []
    remove, add = env.crons.remove, env.crons.add

    async def noted_remove(scope):
        done.append(f"remove {scope.scope_key}")
        await remove(scope)

    async def noted_add(job, scope, user_timezone):
        done.append(f"add {scope.scope_key}")
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "remove", noted_remove)
    monkeypatch.setattr(env.crons, "add", noted_add)

    await env.reconcile()

    assert done == [f"remove {scopes[0].scope_key}"] + [
        f"add {scope.scope_key}" for scope in scopes[1:] for _ in range(2)
    ]


@pytest.mark.asyncio
async def test_passes_cut_off_at_their_ceiling_carry_on_where_the_last_stopped(
    env, monkeypatch, fake_dream_redis: FakeAsyncRedis
):
    """The review's starvation probe. Eight scopes lost their crons, the
    first three of them on a jobstore that stalls every registration until
    it fails, and the last scope's row was paused with its crons left
    behind; every pass is cut off at its third registration attempt.
    Starting over each time, every pass would spend itself on the same three
    scopes. Each keeps its place before it works a scope and the next
    carries on after it, so every pass gets further, a scope a pass was cut
    in is retried the next cycle, and the passes get through every scope,
    the straggler's removal included."""
    scopes = sorted(
        (registry.MemoryScope.for_expert(USER, f"e-{i}") for i in range(9)),
        key=lambda scope: scope.scope_key,
    )
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    for scope in scopes[:-1]:
        for job_id in job_ids(scope):
            env.scheduler.jobs.pop(job_id)
            env.scheduler.scopes.pop(job_id)
    straggler = scopes[-1]
    await env.db.set_scope_state(USER, straggler.scope_key, PAUSED)
    stalled = {scope.scope_key for scope in scopes[:3]}
    add = env.crons.add
    budget = {"left": 0}

    async def ceiling_add(job, scope, user_timezone):
        if budget["left"] == 0:
            raise asyncio.CancelledError("the pass's ceiling")
        budget["left"] -= 1
        if scope.scope_key in stalled:
            raise ConnectionError("the jobstore stalled")
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "add", ceiling_add)
    healthy = set().union(*(job_ids(s) for s in scopes[3:-1]))
    cycle: list[str | None] = []
    removed: list[str] = []
    for _ in range(20):
        budget["left"] = 2
        try:
            removed += (await env.reconcile()).removed
        except asyncio.CancelledError:
            cycle.append(fake_dream_redis.store.get(scope_reconcile.CURSOR_KEY))
            continue
        if set(env.scheduler.jobs) == healthy:
            break
        assert cycle == sorted(cycle) and len(set(cycle)) == len(cycle), cycle
        cycle = []
    else:
        pytest.fail(f"the passes never got through; jobs {sorted(env.scheduler.jobs)}")

    assert removed == [straggler.scope_key]
    assert scope_reconcile.CURSOR_KEY not in fake_dream_redis.store
    assert scope_reconcile._Place.local is None


@pytest.mark.asyncio
async def test_a_pass_cut_off_in_a_scope_still_moves_the_next_one_on(
    env, monkeypatch, fake_dream_redis: FakeAsyncRedis
):
    """The review's hung write: a registration that never returns until the
    ceiling cuts its pass. The pass kept its place before it began that
    scope, so the next one carries on after it and repairs the rest; the
    scope it was cut in is tried again the next cycle."""
    scopes = [registry.MemoryScope.for_user(f"cut-{i}") for i in range(3)]
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    _lose_every_cron(env)
    entered = asyncio.Event()
    tried: list[str] = []
    add = env.crons.add

    async def hung_add(job, scope, user_timezone):
        tried.append(scope.scope_key)
        if scope == scopes[0]:
            entered.set()
            await asyncio.Event().wait()
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "add", hung_add)
    cut = asyncio.create_task(env.reconcile())
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    cut.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cut

    await asyncio.wait_for(env.reconcile(), TIMEOUT)

    assert set(env.scheduler.jobs) == job_ids(scopes[1]) | job_ids(scopes[2])
    assert tried.count(scopes[0].scope_key) == 1, "not tried again this cycle"
    entered.clear()
    again = asyncio.create_task(env.reconcile())
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    again.cancel()
    with pytest.raises(asyncio.CancelledError):
        await again


@pytest.mark.asyncio
async def test_a_place_redis_would_not_keep_is_kept_in_the_process(
    env, monkeypatch, fake_dream_redis: FakeAsyncRedis
):
    """The review's flapping Redis: every pass's first write of its place
    fails, Redis answering again by the next pass but holding no place, and
    each pass is cut off after three registration attempts, the first three
    scopes failing. The place this process kept carries each pass on, so
    they reach the last scope."""
    scopes = [registry.MemoryScope.for_user(f"flap-{i}") for i in range(5)]
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    _lose_every_cron(env)
    tried: list[str] = []
    budget = {"left": 0}
    add = env.crons.add

    async def flap(key, value, **kwargs):
        raise ConnectionError("redis flapped")

    async def bounded_add(job, scope, user_timezone):
        if budget["left"] == 0:
            raise asyncio.CancelledError("the pass's ceiling")
        budget["left"] -= 1
        tried.append(scope.scope_key)
        if scope in scopes[:3]:
            raise ConnectionError("this scope's jobstore write failed")
        await add(job, scope, user_timezone)

    monkeypatch.setattr(fake_dream_redis, "set", flap)
    monkeypatch.setattr(env.crons, "add", bounded_add)
    for _ in range(3):
        budget["left"] = 3
        try:
            await env.reconcile()
        except asyncio.CancelledError:
            pass

    assert scopes[-1].scope_key in tried, tried
    assert job_ids(scopes[-1]) <= set(env.scheduler.jobs)


@pytest.mark.asyncio
async def test_a_place_a_finished_cycle_cleared_starts_the_next_from_the_first(
    env, fake_dream_redis: FakeAsyncRedis
):
    """A pass that settles the last scope clears both places, Redis's and
    the process's, so the next pass looks at every scope again."""
    await registry.ensure_scope_scheduled(EXPERT)
    scope_reconcile._Place.local = EXPERT.scope_key
    _lose_every_cron(env)

    await env.reconcile()
    assert env.scheduler.jobs == {}, "after the place, nothing to do"
    await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert scope_reconcile._Place.local is None


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "pause", "unreadable"])
async def test_a_restore_the_row_stopped_recording_meanwhile_is_taken_back(
    env, monkeypatch, change: str
):
    """The review's stale restore: a cron is deleted (its record forgotten
    first, then the job), or the scope paused, between the restore's read
    and its write, which puts the cron back. A second read takes back what
    the row no longer records; when that read fails the restore takes back
    all it added, and the next pass restores what the row still records."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    _lose_every_cron(env)
    add = env.crons.add
    get = env.db.get_scope_schedule
    nightly = f"dream_nightly_batch_{ACCOUNT.scope_key}"

    async def interleaved_add(job, scope, user_timezone):
        if job.job_id(scope) == nightly:
            if change == "delete":
                await env.db.forget_scope_job(USER, ACCOUNT.scope_key, nightly)
                env.scheduler.jobs.pop(nightly, None)
            elif change == "pause":
                await env.db.set_scope_state(USER, ACCOUNT.scope_key, PAUSED)
            else:
                monkeypatch.setattr(
                    env.db, "get_scope_schedule", AsyncMock(side_effect=ConnectionError)
                )
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "add", interleaved_add)

    report = await env.reconcile()

    community = f"community_rebuild_{ACCOUNT.scope_key}"
    if change == "delete":
        assert set(env.scheduler.jobs) == {community}
        assert report.restored == [community]
    elif change == "pause":
        assert env.scheduler.jobs == {}
    else:
        assert env.scheduler.jobs == {} and report.restored == []
        monkeypatch.setattr(env.db, "get_scope_schedule", get)
        monkeypatch.setattr(env.crons, "add", add)
        await env.reconcile()
        assert set(env.scheduler.jobs) == job_ids(ACCOUNT)


@pytest.mark.asyncio
async def test_a_pass_that_cannot_keep_its_place_starts_from_the_first_scope(
    env, monkeypatch, fake_dream_redis: FakeAsyncRedis
):
    """With Redis out of reach and no place kept in the process a pass
    starts from the first scope, still settles every one, and stops calling
    Redis after the first failure, so an outage costs it one timeout, not
    one per scope it settles."""
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    fake_dream_redis.store[scope_reconcile.CURSOR_KEY] = max(
        EXPERT.scope_key, OTHER_EXPERT.scope_key
    )
    down = AsyncMock(side_effect=ConnectionError)
    monkeypatch.setattr(redis_client, "get_redis_async", down)
    _lose_every_cron(env)

    await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT) | job_ids(OTHER_EXPERT)
    assert down.await_count == 1
    assert scope_reconcile._Place.local is None, "the cycle finished"
