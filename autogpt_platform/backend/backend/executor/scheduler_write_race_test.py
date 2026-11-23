"""Forced-order races between a schedule's writers (#14577).

Pause, resume and the two edits each read a job and write back state built
from that read. Each test stops one of them at a seam after its read, lets a
pause in, then lets the first go on, and checks the job ends as if the two had
run one after the other. A write built from the stale read would undo the
pause: the archive's mark back on a schedule the user paused, a running
schedule's mark lost, or a paused schedule set running by an edit.

The order comes from events, never from timing: the competing pause either
finishes while the first call is stopped, or is seen waiting for the jobstore
lock (``_ReportingLock``) before the first goes on. Against a real APScheduler
on a SQL job store, whose reads are copies, so a stale read really is stale.
"""

import threading
from collections.abc import Callable
from typing import Any, Literal
from unittest.mock import patch

import pytest

from backend.executor.scheduler import Scheduler
from backend.executor.scheduler_jobstore_test_helpers import (
    STORE,
    USER,
    add_follow_up,
    add_graph_job,
    pause,
    resume,
    sqlite_scheduler,
    stored,
)

# Only stops a hung test: a passing run never waits this long.
_HANG_SECONDS = 10
_NEW_INPUTS = {"topic": "new"}
_NEW_CRON = "0 7 * * *"
_ORDERS = pytest.mark.parametrize(
    "raced", [True, False], ids=["pause-mid-edit", "edit-then-pause"]
)


class _Race:
    def __init__(self) -> None:
        self.first_stopped = threading.Event()
        self.first_may_go = threading.Event()
        # The competitor has finished, or is waiting for the jobstore lock.
        self.settled = threading.Event()
        self.competitor_waited = False
        self.errors: list[BaseException] = []


class _ReportingLock:
    """APScheduler's jobstore lock, reporting the competitor waiting for it."""

    def __init__(self, lock: Any, race: _Race) -> None:
        self._lock = lock
        self._race = race

    def __enter__(self) -> bool:
        if self._lock.acquire(blocking=False):
            return True
        if threading.current_thread().name == "competitor":
            self._race.competitor_waited = True
            self._race.settled.set()
        return self._lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


def _race(
    scheduler: Scheduler,
    first: Callable[[], object],
    competitor: Callable[[], object],
    *,
    seam: Literal["read", "validation"],
) -> bool:
    """Stop *first* at *seam*: just after it reads the job, or while it checks
    new inputs against the graph. Run *competitor* until it finishes or waits
    for the jobstore lock, then let *first* go on. Returns whether the
    competitor had to wait."""
    race = _Race()
    aps = scheduler.scheduler
    lock = aps._jobstores_lock
    aps._jobstores_lock = _ReportingLock(lock, race)
    authorized_job = scheduler._authorized_job

    def stop_first(at: str) -> None:
        if at != seam or threading.current_thread().name != "first":
            return
        if not race.first_stopped.is_set():
            race.first_stopped.set()
            race.first_may_go.wait(_HANG_SECONDS)

    def read(schedule_id: str, user_id: str, *, action: str):
        found = authorized_job(schedule_id, user_id, action=action)
        stop_first("read")
        return found

    def check(inputs_check) -> None:
        inputs_check.close()
        stop_first("validation")

    threads: list[threading.Thread] = []
    try:
        with (
            patch.object(scheduler, "_authorized_job", side_effect=read),
            patch("backend.executor.scheduler.run_async", side_effect=check),
        ):
            try:
                threads.append(_start(first, "first", race))
                assert race.first_stopped.wait(_HANG_SECONDS), race.errors
                threads.append(_start(competitor, "competitor", race))
                assert race.settled.wait(_HANG_SECONDS), "the competitor hung"
            finally:
                # On every path, so no worker outlives the patches, the lock
                # or the scheduler, even when an assertion above fails.
                race.first_may_go.set()
                for thread in threads:
                    thread.join(_HANG_SECONDS)
    finally:
        aps._jobstores_lock = lock
    hung = [thread.name for thread in threads if thread.is_alive()]
    assert not hung, f"hung: {hung}"
    if race.errors:
        raise race.errors[0]
    return race.competitor_waited


def _start(target: Callable[[], object], name: str, race: _Race) -> threading.Thread:
    def run() -> None:
        try:
            target()
        except BaseException as e:
            race.errors.append(e)
        finally:
            if name == "competitor":
                race.settled.set()

    thread = threading.Thread(target=run, name=name)
    thread.start()
    return thread


def _edit_and(
    scheduler: Scheduler,
    raced: bool,
    fields: dict[str, Any],
    competitor: Callable[[], object],
    seam: Literal["read", "validation"],
) -> bool | None:
    """Edit the graph schedule with *fields* and run *competitor*: mid-edit
    when *raced*, after the edit commits otherwise. Returns whether a raced
    competitor had to wait for the jobstore lock."""

    def edit() -> None:
        scheduler.update_graph_execution_schedule("job-1", USER, **fields)

    if raced:
        return _race(scheduler, edit, competitor, seam=seam)
    with patch("backend.executor.scheduler.run_async", side_effect=_skip_check):
        edit()
    competitor()
    return None


def _skip_check(inputs_check) -> None:
    inputs_check.close()


@pytest.mark.parametrize(
    ("archived", "flow"),
    [
        pytest.param(
            False,
            lambda scheduler: pause(scheduler, for_expert_archive=True),
            id="archive-pauses",
        ),
        pytest.param(
            True,
            lambda scheduler: resume(scheduler, only_if_paused_by_expert_archive=True),
            id="rehire-resumes",
        ),
    ],
)
def test_a_users_pause_during_the_archive_flow_lands_after_it(
    tmp_path, archived: bool, flow: Callable[[Scheduler], bool]
):
    """The archive flow reads the job and writes it back under the lock, so
    the user's pause waits and lands after it. Landing in between, it would
    be overwritten: the archive would mark a schedule the user had just
    paused, or re-hire resume it."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        if archived:
            pause(scheduler, for_expert_archive=True)

        waited = _race(
            scheduler, lambda: flow(scheduler), lambda: pause(scheduler), seam="read"
        )

        after = stored(scheduler)
        assert after.next_run_time == "" and not after.paused_by_expert_archive
        assert waited


@_ORDERS
def test_an_edit_keeps_a_users_pause_of_a_schedule_the_archive_paused(
    tmp_path, raced: bool
):
    """The archive paused it, and the user pauses it while an edit of its
    inputs is checked, which clears the archive's mark. The edit keeps the
    mark cleared, so re-hire leaves the schedule paused."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        pause(scheduler, for_expert_archive=True)

        waited = _edit_and(
            scheduler,
            raced,
            {"input_data": _NEW_INPUTS},
            lambda: pause(scheduler),
            seam="validation",
        )

        after = stored(scheduler)
        assert after.input_data == _NEW_INPUTS
        assert after.next_run_time == "" and not after.paused_by_expert_archive
        assert resume(scheduler, only_if_paused_by_expert_archive=True) is False
        # The check runs without the lock: the pause goes straight through.
        assert waited is (False if raced else None)


@_ORDERS
def test_an_edit_keeps_the_mark_of_an_archive_pause(tmp_path, raced: bool):
    """The archive pauses the schedule while an edit of its inputs is checked.
    The edit keeps the mark, so re-hire resumes the schedule, edited."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)

        waited = _edit_and(
            scheduler,
            raced,
            {"input_data": _NEW_INPUTS},
            lambda: pause(scheduler, for_expert_archive=True),
            seam="validation",
        )

        after = stored(scheduler)
        assert after.next_run_time == "" and after.paused_by_expert_archive
        assert resume(scheduler, only_if_paused_by_expert_archive=True) is True
        resumed = stored(scheduler)
        assert resumed.next_run_time and resumed.input_data == _NEW_INPUTS
        assert waited is (False if raced else None)


@_ORDERS
@pytest.mark.parametrize(
    ("fields", "seam", "waits"),
    [
        pytest.param({"cron": _NEW_CRON}, "read", True, id="cron"),
        pytest.param(
            {"cron": _NEW_CRON, "input_data": _NEW_INPUTS},
            "validation",
            False,
            id="cron-and-inputs",
        ),
    ],
)
def test_a_cron_edit_keeps_a_pause(
    tmp_path,
    raced: bool,
    fields: dict[str, Any],
    seam: Literal["read", "validation"],
    waits: bool,
):
    """The user pauses the schedule mid-edit of its cadence. The edit moves
    the trigger and leaves the schedule paused: a next run time worked out
    from its earlier read would set it running again."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)

        waited = _edit_and(scheduler, raced, fields, lambda: pause(scheduler), seam)

        after = stored(scheduler)
        assert after.cron == _NEW_CRON and after.next_run_time == ""
        assert waited is (waits if raced else None)


@_ORDERS
def test_a_cadence_change_keeps_a_pause_of_a_follow_up(tmp_path, raced: bool):
    """The same for a repeating follow-up's cadence, which is read and
    written under the lock with nothing slow in between: the pause waits."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_follow_up(scheduler)

        def change_cadence() -> None:
            scheduler.update_copilot_turn_cron("job-1", USER, _NEW_CRON)

        waited = None
        if raced:
            waited = _race(
                scheduler, change_cadence, lambda: pause(scheduler), seam="read"
            )
        else:
            change_cadence()
            pause(scheduler)

        job = scheduler.scheduler.get_job("job-1", jobstore=STORE)
        assert job.kwargs["cron"] == _NEW_CRON and job.next_run_time is None
        assert waited is (True if raced else None)
