"""Tests for the scheduler's reconcile pass over the memory-scope crons
(``scope_reconcile.py``), against the registry fakes: it restores what the
event-driven paths lose, removes what they leave behind, and converges.

The race probes are an independent review's. Round 1's left an ACTIVE row
with its job ids recorded and no crons; each now ends with the crons back
after one pass. Round 2's: a restore that lands after a timezone change,
set right by the next pass, and passes cut off at their ceiling, each now
walking on round the ring. Round 3's: a pass cut off inside a scope's work,
now moving on, and a restore whose cron the row stopped recording
meanwhile. Round 4's: that restore cut short by a failure or a
cancellation, and a stale place. A stale restore leaves a cron tagged as
the reconcile's, which the next pass takes back.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from . import registry, scope_crons, scope_jobs, scope_reconcile
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


async def _refused_then_resumed(
    env: RegistryEnv, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """The interleaving from the round-2 review: an ensure parks before
    recording while a pause lands, so its record is refused, and a resume
    registers and records the crons before the ensure goes on. Returns the
    ensure's outcomes."""
    record = env.db.record_scope_jobs
    reached, go = asyncio.Event(), asyncio.Event()
    ensure: asyncio.Task | None = None

    async def parked_record(*args: Any, **kwargs: Any) -> bool:
        if asyncio.current_task() is not ensure or reached.is_set():
            return await record(*args, **kwargs)
        reached.set()
        await go.wait()
        refused = await record(*args, **kwargs)
        await registry.resume_scope(EXPERT)
        return refused

    monkeypatch.setattr(env.db, "record_scope_jobs", parked_record)
    ensure = asyncio.create_task(
        registry.ensure_scope_scheduled(EXPERT, force_refresh=True)
    )
    await asyncio.wait_for(reached.wait(), TIMEOUT)
    assert await registry.pause_scope(EXPERT)
    go.set()
    outcomes = await asyncio.wait_for(ensure, TIMEOUT)
    monkeypatch.setattr(env.db, "record_scope_jobs", record)
    return outcomes


@pytest.mark.asyncio
@pytest.mark.parametrize("cycles", [1, 2])
async def test_a_registration_a_pause_refused_takes_no_later_resumes_crons(
    env, monkeypatch, cycles
):
    """After ``cycles`` of the interleaving the refused registrations report
    ``scope_changed`` and take nothing down: the row is ACTIVE with both ids
    recorded and the resume's crons in place, and a pass has nothing to
    change. (A takedown there once took the resume's crons, for the pass
    to restore.)"""
    await registry.ensure_scope_scheduled(EXPERT)
    for _ in range(cycles):
        outcomes = await _refused_then_resumed(env, monkeypatch)
        assert {r["reason"] for r in outcomes.values()} == {"scope_changed"}
    row = env.db.rows[EXPERT.scope_key]
    assert row.state == ACTIVE
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT)

    report = await env.reconcile()

    assert (report.restored, report.removed, report.dropped) == ([], [], [])
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
    flags = AsyncMock(return_value=(False, True))
    monkeypatch.setattr(scope_jobs, "evaluate_feature_flag", flags)
    monkeypatch.setattr(scope_reconcile, "evaluate_feature_flag", flags, raising=False)

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
async def test_passes_cut_off_at_their_ceiling_walk_on_round_the_ring(env, monkeypatch):
    """The review's starvation probe, in one process. Eight scopes lost their
    crons, the first three of them on a jobstore that fails every
    registration, and the last scope's row was paused with its crons left
    behind; every pass is cut off at its third registration attempt. Each
    pass takes a scope as its place before it works it and the next walks on
    after it, round the ring, so the passes get through every scope, the
    straggler's removal included, however often the failing ones come
    round."""
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
    for _ in range(12):
        budget["left"] = 2
        try:
            await env.reconcile()
        except asyncio.CancelledError:
            pass
        if set(env.scheduler.jobs) == healthy:
            break
    else:
        pytest.fail(f"the passes never got through; jobs {sorted(env.scheduler.jobs)}")

    assert not job_ids(straggler) & env.scheduler.jobs.keys()


@pytest.mark.asyncio
async def test_no_scope_starves_whatever_place_a_pass_starts_from(env, monkeypatch):
    """The review's stale-place probe without Redis: a place left near the
    start, every scope failing, and every pass cut off at its third attempt
    (the scope it was cut in taken, its work not done). Started over, six
    passes visited the same two scopes; each pass walks on from the last
    scope taken, so the six reach the last scope and come round again."""
    scopes = sorted(
        (registry.MemoryScope.for_user(f"stale-{i}") for i in range(7)),
        key=lambda scope: scope.scope_key,
    )
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    _lose_every_cron(env)
    monkeypatch.setattr(scope_reconcile._Place, "last", scopes[0].scope_key)
    tried: list[str] = []
    budget = {"left": 0}

    async def failing_add(job, scope, user_timezone):
        if budget["left"] == 0:
            raise asyncio.CancelledError("the pass's ceiling")
        budget["left"] -= 1
        tried.append(scope.scope_key)
        raise ConnectionError("this scope fails")

    monkeypatch.setattr(env.crons, "add", failing_add)
    for _ in range(6):
        budget["left"] = 2
        with pytest.raises(asyncio.CancelledError):
            await env.reconcile()

    assert scopes[-1].scope_key in tried, tried
    assert scopes[0].scope_key in tried, "and round again"


@pytest.mark.asyncio
async def test_a_pass_cut_off_in_a_scope_still_moves_the_next_one_on(env, monkeypatch):
    """The review's hung write: a registration that never returns until the
    ceiling cuts its pass. The pass took the scope as its place before it
    began it, so the next one walks on after it and repairs the rest; the
    scope it was cut in comes round again at the end of the ring."""
    scopes = [registry.MemoryScope.for_user(f"cut-{i}") for i in range(3)]
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    _lose_every_cron(env)
    entered = asyncio.Event()
    tried: list[str] = []
    add = env.crons.add

    async def hung_add(job, scope, user_timezone):
        tried.append(scope.scope_key)
        if scope == scopes[0] and not entered.is_set():
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

    assert set(env.scheduler.jobs) == set().union(*(job_ids(s) for s in scopes))
    assert tried.index(scopes[1].scope_key) < len(tried) - 2, "the rest went first"
    assert tried[-2:] == [scopes[0].scope_key] * 2, "the cut scope last"


@pytest.mark.asyncio
async def test_a_process_starts_its_walk_at_a_random_scope(env, monkeypatch):
    """So a scheduler that restarts often can't keep repairing the same
    first scopes."""
    scopes = sorted(
        (registry.MemoryScope.for_user(f"start-{i}") for i in range(4)),
        key=lambda scope: scope.scope_key,
    )
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    _lose_every_cron(env)
    monkeypatch.setattr(scope_reconcile._Place, "last", None)
    monkeypatch.setattr(scope_reconcile.random, "randrange", lambda n: 2)
    order: list[str] = []
    add = env.crons.add

    async def noted_add(job, scope, user_timezone):
        if scope.scope_key not in order:
            order.append(scope.scope_key)
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "add", noted_add)

    await env.reconcile()

    keys = [scope.scope_key for scope in scopes]
    assert order == keys[2:] + keys[:2]
    assert scope_reconcile._Place.last == keys[1], "the last scope it took"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["delete", "pause"])
async def test_a_restore_the_row_stopped_recording_meanwhile_is_taken_back_next_pass(
    env, monkeypatch, change: str
):
    """The review's stale restore: a cron is deleted (its record forgotten
    first, then the job), or the scope paused, between the restore's read
    and its write, which puts the cron back, tagged as the reconcile's. The
    next pass takes back the tagged cron the row no longer records (the
    pause's: every cron of the scope)."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    _lose_every_cron(env)
    add = env.crons.add
    nightly = f"dream_nightly_batch_{ACCOUNT.scope_key}"
    community = f"community_rebuild_{ACCOUNT.scope_key}"

    async def interleaved_add(job, scope, user_timezone):
        if job.job_id(scope) == nightly:
            if change == "delete":
                await env.db.forget_scope_job(USER, ACCOUNT.scope_key, nightly)
                env.scheduler.jobs.pop(nightly, None)
            else:
                await env.db.set_scope_state(USER, ACCOUNT.scope_key, PAUSED)
        await add(job, scope, user_timezone)

    monkeypatch.setattr(env.crons, "add", interleaved_add)
    await env.reconcile()
    assert nightly in env.scheduler.tagged, "the stale restore landed"
    monkeypatch.setattr(env.crons, "add", add)

    report = await env.reconcile()

    if change == "delete":
        assert set(env.scheduler.jobs) == {community}
        assert report.dropped == [nightly]
    else:
        assert env.scheduler.jobs == {}
        assert report.removed == [ACCOUNT.scope_key]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["error", "cancel"])
async def test_a_restore_cut_short_after_a_stale_add_leaves_nothing_next_pass(
    env, monkeypatch, ending: str
):
    """The round-4 review's partial restore: the community cron is deleted
    between the restore's read and its add, which puts it back, then the
    nightly add fails or the pass is cancelled. The tagged community cron
    the row no longer records goes on the next pass."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    _lose_every_cron(env)
    add = env.crons.add
    community = f"community_rebuild_{ACCOUNT.scope_key}"

    async def partial_add(job, scope, user_timezone):
        if job.job_id(scope) == community:
            await env.db.forget_scope_job(USER, ACCOUNT.scope_key, community)
            await add(job, scope, user_timezone)
            return
        if ending == "cancel":
            raise asyncio.CancelledError("the pass's ceiling")
        raise ConnectionError("the second add failed")

    monkeypatch.setattr(env.crons, "add", partial_add)
    if ending == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await env.reconcile()
    else:
        assert (await env.reconcile()).failed == [ACCOUNT.scope_key]
    assert community in env.scheduler.jobs, "the stale add landed"
    monkeypatch.setattr(env.crons, "add", add)

    await env.reconcile()

    assert set(env.scheduler.jobs) == {f"dream_nightly_batch_{ACCOUNT.scope_key}"}
    assert env.db.rows[ACCOUNT.scope_key].community_job_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["error", "cancel"])
async def test_a_stray_goes_before_a_restore_that_fails_in_its_scope(
    env, monkeypatch, ending: str
):
    """The round-5 review's ordering: the scope's restore fails (or the pass
    is cancelled there) on every pass; its tagged stray goes first all the
    same, so a repair that keeps failing never keeps one running."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    community, nightly = scope_reconcile.DREAM_SYSTEM_JOBS[:2]
    env.scheduler.tagged.add(community.job_id(ACCOUNT))
    await env.db.forget_scope_job(USER, ACCOUNT.scope_key, community.job_id(ACCOUNT))
    env.scheduler.jobs[nightly.job_id(ACCOUNT)] = TOKYO

    async def failing(job, scope, user_timezone):
        if ending == "cancel":
            raise asyncio.CancelledError
        raise ConnectionError("the nightly's repair is unavailable")

    monkeypatch.setattr(env.crons, "add", failing)

    if ending == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await env.reconcile()
    else:
        report = await env.reconcile()
        assert report.dropped == [community.job_id(ACCOUNT)]
        assert report.failed == [ACCOUNT.scope_key]

    assert community.job_id(ACCOUNT) not in env.scheduler.jobs
    assert env.scheduler.jobs[nightly.job_id(ACCOUNT)] == TOKYO


@pytest.mark.asyncio
async def test_a_take_back_that_fails_does_not_hold_up_the_restore(env, monkeypatch):
    """Each step stands on its own: the scope's stray cannot be taken back
    (its drop fails), and the row's cron running in the wrong timezone is
    repaired all the same; the scope is reported failed once."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    community, nightly = scope_reconcile.DREAM_SYSTEM_JOBS[:2]
    env.scheduler.tagged.add(community.job_id(ACCOUNT))
    await env.db.forget_scope_job(USER, ACCOUNT.scope_key, community.job_id(ACCOUNT))
    env.scheduler.jobs[nightly.job_id(ACCOUNT)] = TOKYO

    async def failing(job, scope):
        raise ConnectionError("the take-back is unavailable")

    monkeypatch.setattr(env.crons, "drop", failing)

    report = await env.reconcile()

    assert report.failed == [ACCOUNT.scope_key]
    assert report.restored == [nightly.job_id(ACCOUNT)]
    assert env.scheduler.jobs[nightly.job_id(ACCOUNT)] == PARIS


@pytest.mark.asyncio
async def test_a_tagged_cron_with_no_row_goes_and_an_untagged_one_stays(env):
    """An account's crons with no row predate the registry and stay; one the
    reconcile tagged had a row once, and goes with it."""
    legacy = registry.MemoryScope.for_user("legacy")
    gone = registry.MemoryScope.for_user("gone")
    for job in scope_reconcile.DREAM_SYSTEM_JOBS:
        await job.register(env.scheduler, legacy, PARIS)
        await env.crons.add(job, gone, PARIS)

    report = await env.reconcile()

    every = scope_reconcile.DREAM_SYSTEM_JOBS
    assert set(env.scheduler.jobs) == {job.job_id(legacy) for job in every}
    assert sorted(report.dropped) == sorted(job.job_id(gone) for job in every)
