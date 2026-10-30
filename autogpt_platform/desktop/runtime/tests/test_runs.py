import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from autogpt_desktop import ports, rabbitmq, runs, valkey
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import wait_until

DESKTOP = Path(__file__).resolve().parents[2]


class Cursor:
    def __init__(self, stale: list[str]) -> None:
        self.stale = stale
        self.executed: list[tuple[str, dict | None]] = []
        self.rowcount = 0

    def execute(self, sql: str, arguments: dict | None = None) -> None:
        self.executed.append((sql, arguments))
        self.rowcount = len((arguments or {}).get("ids", []))

    def fetchall(self) -> list[tuple[str]]:
        return [(run,) for run in self.stale]


class Connection:
    def __init__(self, stale: list[str]) -> None:
        self.recorded = Cursor(stale)
        self.committed = self.closed = False

    def cursor(self) -> Cursor:
        return self.recorded

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


def ended(connection: Connection) -> list[tuple[str, list[str]]]:
    """(status, run ids) of each batch of runs that was ended."""
    return [
        (arguments["status"], arguments["ids"])
        for sql, arguments in connection.recorded.executed
        if sql == runs.END_RUNS and arguments
    ]


def test_a_run_the_user_stopped_is_ended_and_an_old_one_failed(tmp_path: Path, caplog):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    stopped.write_text(json.dumps(["run-the-user-stopped"]))
    connection = Connection(stale=["run-from-last-week"])
    caplog.set_level("INFO")

    runs.reconcile(lambda: connection, stopped)

    assert ended(connection) == [
        ("TERMINATED", ["run-the-user-stopped"]),
        ("FAILED", ["run-from-last-week"]),
    ]
    # Their running nodes first, as the platform does it.
    statements = [sql for sql, _ in connection.recorded.executed]
    assert statements.index(runs.END_NODES) < statements.index(runs.END_RUNS)
    assert connection.committed and connection.closed
    assert not stopped.exists()
    assert "marked 1 agent run(s) TERMINATED" in caplog.text


def test_with_nothing_interrupted_nothing_is_written(tmp_path: Path):
    connection = Connection(stale=[])
    runs.reconcile(lambda: connection, tmp_path / runs.STOPPED_RUNS_FILE)
    assert ended(connection) == []


def test_only_runs_older_than_a_day_are_failed_the_rest_resume():
    assert "interval '24 hours'" in runs.UNFINISHED_AND_OLD
    assert "IN ('RUNNING', 'QUEUED')" in runs.UNFINISHED_AND_OLD
    # A run that has already ended, or was deleted, is left alone.
    assert "IN ('RUNNING', 'QUEUED') AND NOT \"isDeleted\"" in runs.END_RUNS


@pytest.mark.parametrize("content", ["not json", '{"a": 1}', "[1, null]"])
def test_a_damaged_record_of_stopped_runs_is_ignored(tmp_path: Path, content: str):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    stopped.write_text(content)
    connection = Connection(stale=[])

    runs.reconcile(lambda: connection, stopped)

    assert ended(connection) == []
    assert not stopped.exists()


def test_a_database_that_changed_shape_does_not_stop_the_start(tmp_path: Path, caplog):
    """Upstream renamed a table: runs stay as they are, the app starts, and
    the record of stopped runs is kept for a version that can apply it."""
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    stopped.write_text(json.dumps(["run"]))

    def connect():
        raise RuntimeError('relation "AgentGraphExecution" does not exist')

    runs.reconcile(connect, stopped)

    assert "could not settle interrupted agent runs" in caplog.text
    assert stopped.exists()


def test_a_cache_that_is_not_there_does_not_stop_the_start(caplog):
    pytest.importorskip("redis")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    runs.Cache(port, "password").clear_stale_locks(["executor"])

    assert "could not clear stale run locks" in caplog.text


def test_services_without_locks_do_not_touch_the_cache():
    runs.Cache(1, "password").clear_stale_locks(
        ["scheduler", "rest"]
    )  # nothing to connect to


def bundled_valkey() -> Bundle | None:
    configured = os.environ.get("AUTOGPT_DESKTOP_RUNTIME")
    bundle = Bundle(Path(configured) if configured else DESKTOP / "build" / "runtime")
    return bundle if bundle.valkey_server.is_file() else None


@pytest.fixture
def cache(tmp_path: Path):
    """The bundle's Valkey, in cluster mode as the runtime runs it."""
    redis = pytest.importorskip("redis")
    bundle = bundled_valkey()
    if bundle is None:
        pytest.skip("no assembled bundle (build/runtime) to take Valkey from")
    data = DataDir(tmp_path / "data")
    data.prepare()
    port = ports._free_port(exclude=set())
    valkey.write_config(data, port, ports._free_port(exclude={port}), "password")
    process = valkey.process(bundle, data, port, "password")
    process.start()
    try:
        assert wait_until(lambda: valkey.is_ready(port, "password"), 30, 0.2)
        valkey.ensure_cluster(port, "password")
        yield runs.Cache(port, "password"), redis.Redis(port=port, password="password")
    finally:
        process.stop()


def test_stale_locks_in_every_slot_are_cleared_and_nothing_else(cache):
    """The node is in cluster mode, where one command cannot name keys from
    different slots; the locks of many runs are spread over all of them."""
    locks, client = cache
    run_locks = [f"exec_lock:run-{number}" for number in range(40)]
    session_locks = [f"copilot:session:session-{number}:lock" for number in range(40)]
    for key in [*run_locks, *session_locks, "copilot:session:s:meta", "some:other:key"]:
        client.set(key, "an executor that is gone", ex=300)
    assert (
        len({client.execute_command("CLUSTER", "KEYSLOT", key) for key in run_locks})
        > 20
    )
    with pytest.raises(redis_errors().ClusterCrossSlotError):
        client.delete(*run_locks)  # why the locks are deleted one per command

    locks.clear_stale_locks(["scheduler", "executor"])
    assert not any(client.exists(key) for key in run_locks)
    assert all(client.exists(key) for key in session_locks)

    locks.clear_stale_locks(["executor", "copilot-executor"])
    assert not any(client.exists(key) for key in session_locks)
    assert client.exists("copilot:session:s:meta") and client.exists("some:other:key")


def redis_errors():
    import redis

    return redis.exceptions


def test_a_record_of_stopped_runs_that_cannot_be_removed_does_not_stop_the_app(
    tmp_path: Path, monkeypatch, caplog
):
    """Read-only, or held open by a scanner on Windows. It is applied again
    on the next start, to runs that have all ended by then."""
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    stopped.write_text("[]", encoding="utf-8")

    def refuse(self, missing_ok=False):
        raise PermissionError(13, "in use by another process")

    monkeypatch.setattr(Path, "unlink", refuse)

    runs.reconcile(lambda: NothingToSettle(), stopped)

    assert f"could not remove {runs.STOPPED_RUNS_FILE}" in caplog.text


class NothingToSettle:
    """As much of a database connection as settling uses when no run needs it."""

    def cursor(self):
        return self

    def execute(self, statement, arguments=None) -> None:
        pass

    def fetchall(self) -> list:
        return []

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


# --- what is running, and what the user chose to stop -------------------------


def test_what_is_running_is_said_in_words():
    assert runs.Activity(("a",), ()).describe() == "1 agent run"
    assert (
        runs.Activity(("a", "b"), ("s",)).describe()
        == "2 agent runs and 1 AutoPilot reply"
    )
    assert runs.Activity((), ("s", "t")).describe() == "2 AutoPilot replies"
    assert not runs.Activity()
    assert runs.Activity((), ("s",))


def test_what_is_still_running_of_what_was_stopped():
    stopped = runs.Activity(("a", "b"), ("s",))
    now = runs.Activity(("b", "started-since"), ())
    assert stopped.still(now) == runs.Activity(("b",), ())
    assert not stopped.still(runs.Activity(("started-since",), ("another",)))


def test_the_locks_say_what_is_running_and_stopped_runs_lose_theirs(cache):
    locks, client = cache
    assert locks.active() == runs.Activity()
    for key in (
        "exec_lock:run-b",
        "exec_lock:run-a",
        "copilot:session:chat-1:lock",
        "copilot:session:chat-1:meta",
    ):
        client.set(key, "an executor", ex=300)

    assert locks.active() == runs.Activity(("run-a", "run-b"), ("chat-1",))

    locks.release(runs.Activity(("run-a",), ("chat-1",)))
    assert locks.active() == runs.Activity(("run-b",), ())
    assert client.exists("copilot:session:chat-1:meta")
    locks.close()


def test_a_cache_that_does_not_answer_says_nothing_about_what_is_running():
    pytest.importorskip("redis")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert runs.Cache(port, "password").active() is None


def test_the_choice_to_stop_is_written_whole_and_added_to_one_not_yet_applied(
    tmp_path: Path,
):
    stopped = tmp_path / "run" / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity(("a",), ("s",)))
    assert json.loads(stopped.read_text()) == {
        "graph_runs": ["a"],
        "copilot_turns": ["s"],
    }

    runs.record_stopped(stopped, runs.Activity(("a", "b"), ()))

    assert runs.read_stopped(stopped) == runs.Activity(("a", "b"), ("s",))
    assert [entry.name for entry in stopped.parent.iterdir()] == [
        runs.STOPPED_RUNS_FILE
    ]


def test_a_record_written_before_turns_were_recorded_is_still_read(tmp_path: Path):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    stopped.write_text(json.dumps(["run"]))
    assert runs.read_stopped(stopped) == runs.Activity(("run",), ())


class Broker:
    """As much of runs.Queue as settling uses. `finds` is what each question
    finds a message for, the last answer from then on: True for every
    session asked about, a set for those, None for a broker that does not
    answer."""

    def __init__(self, *finds: bool | set[str] | None) -> None:
        self.finds = list(finds) or [True]
        self.dropped: list[list[str]] = []

    def drop_turns(self, sessions, turn_ids=None) -> set[str] | None:
        self.dropped.append(sorted(sessions))
        found = self.finds.pop(0) if len(self.finds) > 1 else self.finds[0]
        return set(sessions) if found is True else found


@pytest.fixture(autouse=True)
def no_waiting_between_tries(monkeypatch):
    monkeypatch.setattr(runs, "DROP_RETRY_SECONDS", 0)


class Locks:
    """As much of runs.Cache as settling uses. `held` is what it answers,
    one answer per question, the last one from then on."""

    def __init__(self, *held: runs.Activity | None) -> None:
        self.held = list(held) or [runs.Activity()]
        self.released: list[runs.Activity] = []

    def active(self) -> runs.Activity | None:
        return self.held.pop(0) if len(self.held) > 1 else self.held[0]

    def release(self, activity: runs.Activity) -> None:
        self.released.append(activity)


def test_the_next_start_applies_a_choice_the_last_one_did_not_get_to(tmp_path: Path):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity(("run",), ("chat",)))
    connection, broker = Connection(stale=[]), Broker()

    runs.reconcile(lambda: connection, stopped, broker)

    assert ended(connection) == [("TERMINATED", ["run"])]
    assert broker.dropped == [["chat"]]
    assert not stopped.exists()
    # Its chat is still marked busy: left for the copilot executor's host.
    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == ["chat"]


def test_a_turn_that_could_not_be_taken_off_the_queue_keeps_the_record(tmp_path: Path):
    """And its chat is not closed: the executor is about to be let at that
    message, and the turn would run in a chat closed under it."""
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("chat",)))
    broker = Broker(None)

    runs.reconcile(lambda: Connection(stale=[]), stopped, broker)

    assert runs.read_stopped(stopped) == runs.Activity((), ("chat",))
    assert len(broker.dropped) == runs.DROP_TRIES
    assert not (tmp_path / runs.ENDED_TURNS_FILE).exists()


def test_a_broker_that_is_late_to_answer_at_boot_is_asked_again(tmp_path: Path):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("chat", "ended-by-itself")))
    broker = Broker(None, None, {"chat"})

    runs.reconcile(lambda: Connection(stale=[]), stopped, broker)

    # Nothing consumes yet, so what was not found is not there; every chat
    # is closed (closing one that is not busy changes nothing).
    assert len(broker.dropped) == 3
    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == [
        "chat",
        "ended-by-itself",
    ]
    assert not stopped.exists()


def test_stopped_runs_are_ended_as_the_app_closes_and_then_forgotten(
    tmp_path: Path, caplog
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    chosen = runs.Activity(("run-a", "run-b"), ("chat",))
    runs.record_stopped(stopped, chosen)
    connection, broker, locks = Connection(stale=[]), Broker(), Locks()
    caplog.set_level("INFO")

    runs.end_stopped(lambda: connection, stopped, locks, broker)

    assert ended(connection) == [("TERMINATED", ["run-a", "run-b"])]
    assert connection.committed and connection.closed
    # Only what is still unfinished is touched: one that ended by itself
    # keeps the status its executor gave it.
    assert "IN ('RUNNING', 'QUEUED')" in runs.END_RUNS
    assert broker.dropped == [["chat"]]
    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == ["chat"]
    assert locks.released == [chosen]
    assert not stopped.exists()
    assert "stopped as AutoGPT closed" in caplog.text


@pytest.mark.parametrize(
    "held", [runs.Activity((), ("stuck",)), None], ids=["still locked", "cache gone"]
)
def test_a_turn_that_did_not_end_when_asked_loses_its_message_and_has_its_chat_listed(
    tmp_path: Path, held: runs.Activity | None
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("stuck",)))
    (tmp_path / runs.ENDED_TURNS_FILE).write_text(json.dumps(["from-an-earlier-stop"]))
    broker = Broker()

    runs.end_stopped(lambda: Connection(stale=[]), stopped, Locks(held), broker)

    assert broker.dropped == [["stuck"]]
    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == [
        "from-an-earlier-stop",
        "stuck",
    ]
    assert not stopped.exists()


def test_a_message_that_could_not_be_dropped_keeps_the_choice_and_leaves_the_chat(
    tmp_path: Path,
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("stuck",)))
    locks = Locks(None)

    runs.end_stopped(lambda: Connection(stale=[]), stopped, locks, Broker(None))

    assert runs.read_stopped(stopped) == runs.Activity((), ("stuck",))
    assert not (tmp_path / runs.ENDED_TURNS_FILE).exists()
    assert locks.released == []


def test_a_message_the_broker_has_not_handed_back_yet_is_waited_for(tmp_path: Path):
    """The executor was killed holding it; the broker returns it to the queue
    a moment after, which can be after the first look."""
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    held = runs.Activity((), ("stuck", "stuck-too"))
    runs.record_stopped(stopped, held)
    broker = Broker(set(), {"stuck"}, set(), {"stuck-too"})

    runs.end_stopped(lambda: Connection(stale=[]), stopped, Locks(held), broker)

    assert broker.dropped == [
        ["stuck", "stuck-too"],
        ["stuck", "stuck-too"],
        ["stuck-too"],
        ["stuck-too"],
    ]
    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == [
        "stuck",
        "stuck-too",
    ]
    assert not stopped.exists()


def test_a_message_that_never_shows_keeps_the_record_for_the_next_start(tmp_path: Path):
    """Forgotten now, the turn would run again when its message does show:
    nothing would be left to drop it. Only the chat whose message is gone is
    closed."""
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    held = runs.Activity((), ("found", "missing"))
    runs.record_stopped(stopped, held)
    broker, locks = Broker({"found"}, set()), Locks(held)

    runs.end_stopped(lambda: Connection(stale=[]), stopped, locks, broker)

    assert len(broker.dropped) == runs.DROP_TRIES
    assert runs.read_stopped(stopped) == held
    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == ["found"]
    assert locks.released == []


def test_a_choice_the_disk_did_not_take_is_applied_all_the_same(
    tmp_path: Path, monkeypatch, caplog
):
    """A full disk: the record could not be written. The process that was
    told still knows, and ends what it was told to."""

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    chosen = runs.Activity(("run",), ("stuck",))
    with monkeypatch.context() as patched:
        patched.setattr(runs.os, "replace", full)
        runs.record_stopped(stopped, chosen)
    assert f"could not write {runs.STOPPED_RUNS_FILE}" in caplog.text
    assert not stopped.exists()
    connection, broker, locks = (
        Connection(stale=[]),
        Broker(),
        Locks(runs.Activity((), ("stuck",))),
    )

    runs.end_stopped(lambda: connection, stopped, locks, broker, chosen)

    assert ended(connection) == [("TERMINATED", ["run"])]
    assert broker.dropped == [["stuck"]]
    assert locks.released == [chosen]


def test_what_was_told_and_what_is_on_disk_are_both_ended(tmp_path: Path):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity(("from-a-stop-that-died",), ()))
    connection = Connection(stale=[])

    runs.end_stopped(
        lambda: connection, stopped, Locks(), Broker(), runs.Activity(("told-now",), ())
    )

    assert ended(connection) == [("TERMINATED", ["from-a-stop-that-died", "told-now"])]


def test_the_record_is_on_the_disk_before_it_takes_its_name(
    tmp_path: Path, monkeypatch
):
    """A power cut after the rename must not find an empty file."""
    order: list[str] = []
    fsync, replace = os.fsync, os.replace
    monkeypatch.setattr(
        runs.os,
        "fsync",
        lambda descriptor: (order.append("synced"), fsync(descriptor))[1],
    )
    monkeypatch.setattr(
        runs.os,
        "replace",
        lambda old, new: (order.append("named"), replace(old, new))[1],
    )

    runs.record_stopped(tmp_path / runs.STOPPED_RUNS_FILE, runs.Activity(("run",), ()))

    assert order == ["synced", "named"]


# --- runs that go on without their lock ---------------------------------------


class Database:
    """What `runs.running` is asked, and its answers: one per question, the
    last one from then on. None: it does not say."""

    def __init__(self, *answers: tuple[str, ...] | None) -> None:
        self.answers = list(answers)
        self.asked: list[tuple[str, ...]] = []

    def __call__(self, among: tuple[str, ...]) -> tuple[str, ...] | None:
        self.asked.append(tuple(among))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def test_a_run_whose_lock_ran_out_counts_for_as_long_as_the_database_has_it_running():
    """The computer slept for longer than a lock lasts. Upstream never takes
    a lock again once it is gone, and the run goes on."""
    clock, database = Clock(), Database(("slept-through",), ("slept-through",), ())
    watch = runs.Watch(database, clock)
    both = runs.Activity(("slept-through", "finishes"), ("chat",))
    assert watch.look(both) == both

    woke = watch.look(runs.Activity())  # every lock has run out

    assert woke == runs.Activity(("slept-through",), ())
    assert database.asked == [("slept-through", "finishes")]
    # Not asked about again at every look, only every so often.
    assert watch.look(runs.Activity()) == woke
    assert len(database.asked) == 1
    clock.now += runs.UNLOCKED_RUNS_EVERY_SECONDS
    assert watch.look(runs.Activity()) == woke
    clock.now += runs.UNLOCKED_RUNS_EVERY_SECONDS
    assert watch.look(runs.Activity()) == runs.Activity()
    assert database.asked[1:] == [("slept-through",), ("slept-through",)]


def test_a_run_that_finished_costs_one_question_and_no_more():
    database = Database(())
    watch = runs.Watch(database, Clock())
    watch.look(runs.Activity(("run",), ()))
    assert watch.look(runs.Activity()) == runs.Activity()
    assert watch.look(runs.Activity()) == runs.Activity()
    assert database.asked == [("run",)]


def test_with_every_run_locked_the_database_is_not_asked():
    database = Database(())
    watch = runs.Watch(database, Clock())
    for _ in range(3):
        assert watch.look(runs.Activity(("run",), ("chat",))) == runs.Activity(
            ("run",), ("chat",)
        )
    assert database.asked == []


def test_a_database_that_does_not_say_leaves_the_run_counted():
    clock, database = Clock(), Database(None, ())
    watch = runs.Watch(database, clock)
    watch.look(runs.Activity(("run",), ()))
    assert watch.look(runs.Activity()) == runs.Activity(("run",), ())
    clock.now += runs.UNLOCKED_RUNS_EVERY_SECONDS
    assert watch.look(runs.Activity()) == runs.Activity()


def test_a_run_that_is_locked_again_is_counted_once():
    """Its host was restarted: the lock was cleared and the run resumed."""
    watch = runs.Watch(Database(("run",)), Clock())
    watch.look(runs.Activity(("run",), ()))
    assert watch.look(runs.Activity()).counts == (1, 0)
    assert watch.look(runs.Activity(("run",), ())).counts == (1, 0)
    assert watch.unlocked == ()


def test_a_run_whose_executor_is_gone_is_not_counted_until_it_is_locked_again():
    database = Database(("run",))
    watch = runs.Watch(database, Clock())
    watch.look(runs.Activity(("run",), ("chat",), (("chat", "turn"),)))

    watch.executor_gone()
    assert watch.seen.copilot_turn_ids == (("chat", "turn"),)

    # The turn is another executor's.
    assert watch.look(runs.Activity((), ("chat",))) == runs.Activity((), ("chat",))
    assert database.asked == []
    assert watch.look(runs.Activity(("run",), ("chat",))).counts == (1, 1)


def test_what_the_database_has_running_is_asked_for_all_or_for_some():
    connection = Connection(stale=["a", "b"])
    assert runs.running(lambda: connection) == ("a", "b")
    assert runs.running(lambda: connection, ["a", "x"]) == (
        "a",
        "b",
    )  # the stand-in answers the same
    assert runs.running(lambda: connection, []) == ()
    asked = connection.recorded.executed
    assert asked == [(runs.RUNNING, None), (runs.RUNNING_AMONG, {"ids": ["a", "x"]})]
    assert connection.closed
    assert '"executionStatus" = \'RUNNING\' AND NOT "isDeleted"' in runs.RUNNING
    assert (
        runs.RUNNING_AMONG.startswith(runs.RUNNING)
        and "id = ANY(%(ids)s)" in runs.RUNNING_AMONG
    )


def test_a_database_that_refuses_does_not_say_what_is_running():
    def connect():
        raise RuntimeError("the database is shutting down")

    assert runs.running(connect) is None
    assert runs.running(connect, ["a"]) is None


def test_with_nothing_chosen_the_database_is_not_even_asked(tmp_path: Path):
    def connect():
        raise AssertionError("nothing to end")

    runs.end_stopped(connect, tmp_path / runs.STOPPED_RUNS_FILE, Locks(), Broker())


def test_a_database_that_refuses_keeps_the_choice_for_the_next_start(
    tmp_path: Path, caplog
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity(("run",), ()))
    locks = Locks()

    def connect():
        raise RuntimeError("the database is shutting down")

    runs.end_stopped(connect, stopped, locks, Broker())

    assert "could not end the stopped agent runs" in caplog.text
    assert runs.read_stopped(stopped) == runs.Activity(("run",), ())
    assert locks.released == []


def test_a_stopped_turn_is_removed_after_its_lock_was_released_before_ack(
    tmp_path: Path,
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("chat",)))
    broker = Broker()

    runs.end_stopped(lambda: Connection([]), stopped, Locks(), broker)

    assert broker.dropped == [["chat"]]


def test_a_failed_chat_closure_record_preserves_the_stop_intent(
    tmp_path: Path, monkeypatch
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("chat",)))
    replace = runs.os.replace

    def refuse_closure_record(old, new):
        if Path(new).name == runs.ENDED_TURNS_FILE:
            raise OSError(28, "No space left on device")
        return replace(old, new)

    monkeypatch.setattr(runs.os, "replace", refuse_closure_record)
    runs.end_stopped(
        lambda: Connection([]), stopped, Locks(runs.Activity((), ("chat",))), Broker()
    )

    assert stopped.exists()
    assert not stopped.with_name(runs.ENDED_TURNS_FILE).exists()


@pytest.mark.parametrize("prefix", [runs.SESSION_META_PREFIX, "custom:meta:"])
def test_copilot_metadata_retains_active_work_after_its_lock_expires(prefix):
    class Client:
        def scan_iter(self, match, count):
            return [f"{prefix}{{chat}}".encode()] if match == f"{prefix}*" else []

        def hgetall(self, key):
            return {b"status": b"running", b"session_id": b"chat", b"turn_id": b"turn"}

    cache = runs.Cache(0, "", session_meta_prefix=prefix)
    cache._kept.append(Client())

    assert cache.active() == runs.Activity((), ("chat",), (("chat", "turn"),))


def test_lost_copilot_metadata_is_not_a_completion_of_known_work():
    class Client:
        def __init__(self):
            self.metadata = {b"status": b"running", b"turn_id": b"turn"}

        def scan_iter(self, match, count):
            return (
                [b"chat:task:meta:{chat}"]
                if match == "chat:task:meta:*" and self.metadata
                else []
            )

        def hgetall(self, key):
            return self.metadata

    client = Client()
    cache = runs.Cache(0, "")
    cache._kept.append(client)
    expected = cache.active()
    client.metadata = {}

    assert cache.active() == expected
    client.metadata = {b"status": b"completed", b"turn_id": b"turn"}
    assert cache.active() == runs.Activity()


def test_turn_identity_survives_the_durable_stop_record(tmp_path: Path):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    chosen = runs.Activity((), ("chat",), (("chat", "old-turn"),))
    runs.record_stopped(stopped, chosen)

    assert runs.read_stopped(stopped) == chosen


def test_recovered_chat_closure_keeps_the_original_turn_guard(tmp_path: Path):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    chosen = runs.Activity((), ("chat",), (("chat", "old-turn"),))
    runs.record_stopped(stopped, chosen)

    assert runs.reconcile(lambda: Connection([]), stopped, Broker()) is True

    assert json.loads((tmp_path / runs.ENDED_TURNS_FILE).read_text()) == [
        {"session_id": "chat", "turn_id": "old-turn"}
    ]


def test_queue_drain_does_not_remove_a_later_turn_in_the_same_chat():
    messages = [
        json.dumps({"session_id": "chat", "turn_id": turn}).encode()
        for turn in ("stopped", "later")
    ]
    broker = FakeBroker(set(), {runs.COPILOT_TURN_QUEUE: list(messages)})

    found = runs._drop(
        broker.channel(),
        runs.COPILOT_TURN_QUEUE,
        "session_id",
        {"chat"},
        {"chat": "stopped"},
    )

    assert found == {"chat"}
    assert broker.queues[runs.COPILOT_TURN_QUEUE] == [messages[1]]


def test_reconcile_refuses_to_resume_when_the_broker_cannot_settle_stop_intent(
    tmp_path: Path,
):
    stopped = tmp_path / runs.STOPPED_RUNS_FILE
    runs.record_stopped(stopped, runs.Activity((), ("chat",)))

    assert runs.reconcile(lambda: Connection([]), stopped, Broker(None)) is False
    assert stopped.exists()


# --- asking the executors to cancel -------------------------------------------


class Channel:
    def __init__(self, broker: "FakeBroker") -> None:
        self.broker = broker
        self.unsettled: dict[int, bytes] = {}

    def basic_publish(self, exchange: str, routing_key: str, body: bytes) -> None:
        import pika.exceptions

        if exchange not in self.broker.exchanges:
            raise pika.exceptions.ChannelClosedByBroker(
                404, f"NOT_FOUND - no exchange '{exchange}'"
            )
        self.broker.published.append((exchange, routing_key, json.loads(body)))

    def close(self) -> None:
        pass

    def basic_get(self, queue: str, auto_ack: bool):
        import pika.exceptions

        assert auto_ack is False
        if queue not in self.broker.queues:
            raise pika.exceptions.ChannelClosedByBroker(
                404, f"NOT_FOUND - no queue '{queue}'"
            )
        if not self.broker.queues[queue]:
            return None, None, None
        body = self.broker.queues[queue].pop(0)
        tag = len(self.unsettled) + 1
        self.unsettled[tag] = body
        return type("Method", (), {"delivery_tag": tag}), None, body

    def basic_ack(self, tag: int) -> None:
        del self.unsettled[tag]

    def basic_nack(self, tag: int, requeue: bool) -> None:
        assert requeue
        self.broker.queues[runs.COPILOT_TURN_QUEUE].append(self.unsettled.pop(tag))


class FakeBroker:
    def __init__(
        self, exchanges: set[str], queues: dict[str, list[bytes]] | None = None
    ) -> None:
        self.exchanges = exchanges
        self.queues = queues or {}
        self.published: list[tuple[str, str, dict]] = []
        self.closed = False

    def channel(self) -> Channel:
        return Channel(self)

    def close(self) -> None:
        self.closed = True


def queue_on(monkeypatch, broker: FakeBroker) -> runs.Queue:
    pytest.importorskip("pika")
    monkeypatch.setattr(runs.Queue, "_connect", lambda self: broker)
    return runs.Queue(5672, "user", "password")


def test_cancelling_is_asked_the_way_the_platforms_own_stop_asks(monkeypatch):
    broker = FakeBroker({runs.GRAPH_CANCEL_EXCHANGE, runs.COPILOT_CANCEL_EXCHANGE})

    queue_on(monkeypatch, broker).cancel(runs.Activity(("run-a", "run-b"), ("chat",)))

    assert broker.published == [
        ("graph_execution_cancel", "", {"graph_exec_id": "run-a"}),
        ("graph_execution_cancel", "", {"graph_exec_id": "run-b"}),
        ("copilot_cancel", "", {"session_id": "chat"}),
    ]
    assert broker.closed


def test_an_exchange_that_is_not_there_costs_only_its_own_requests(monkeypatch, caplog):
    """The one for agent runs exists only while an executor listens on it."""
    broker = FakeBroker({runs.COPILOT_CANCEL_EXCHANGE})
    caplog.set_level("INFO")

    queue_on(monkeypatch, broker).cancel(runs.Activity(("run",), ("chat",)))

    assert broker.published == [("copilot_cancel", "", {"session_id": "chat"})]
    assert "no executor listens on graph_execution_cancel" in caplog.text


def test_a_broker_that_is_gone_does_not_stop_the_stop(caplog):
    pytest.importorskip("pika")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    queue = runs.Queue(port, "user", "password")

    queue.cancel(runs.Activity(("run",), ()))
    assert "could not ask the executors to cancel" in caplog.text
    assert queue.drop_turns(["chat"]) is None


def turn(session: str, message: str) -> bytes:
    return json.dumps({"session_id": session, "message": message}).encode()


def test_only_the_stopped_sessions_turns_are_taken_off_the_queue(monkeypatch):
    waiting = [
        turn("stopped", "one"),
        turn("another", "two"),
        b"not json",
        turn("stopped", "three"),
    ]
    broker = FakeBroker(set(), {runs.COPILOT_TURN_QUEUE: list(waiting)})

    assert queue_on(monkeypatch, broker).drop_turns(["stopped", "not-queued"]) == {
        "stopped"
    }

    assert broker.queues[runs.COPILOT_TURN_QUEUE] == [
        turn("another", "two"),
        b"not json",
    ]


def test_a_queue_no_turn_was_ever_put_on_has_nothing_to_drop(monkeypatch):
    queue = queue_on(monkeypatch, FakeBroker(set(), {}))
    assert queue.drop_turns(["stopped"]) == set()
    assert queue.drop_turns([]) == set()


# --- runs from before this start that nothing will finish ---------------------

BOOTED = datetime(2026, 10, 3, 12, 0, 0)
AT_REST = {"messages_ready": 0, "messages_unacknowledged": 0, "consumers": 1}


def swept(
    connection: Connection, locks: Locks, counts: dict | None
) -> tuple[bool, list[int]]:
    asked = []

    def queue_counts():
        asked.append(1)
        return counts

    return runs.sweep_orphans(lambda: connection, locks, queue_counts, BOOTED), asked


def test_a_run_from_before_this_start_that_nobody_holds_and_no_message_stands_for_is_failed(
    caplog,
):
    connection = Connection(stale=["lost", "resumed"])
    locks = Locks(runs.Activity(("resumed",), ()))
    counts = {"messages_ready": 0, "messages_unacknowledged": 1, "consumers": 1}
    caplog.set_level("INFO")

    done, _ = swept(connection, locks, counts)

    assert done
    assert ended(connection) == [("FAILED", ["lost"])]
    assert connection.recorded.executed[0] == (
        runs.UNFINISHED_BEFORE,
        {"before": BOOTED},
    )
    assert connection.committed
    assert "marked 1 agent run(s) FAILED" in caplog.text


def test_only_runs_from_before_this_start_are_looked_at():
    assert '"createdAt" < %(before)s' in runs.UNFINISHED_BEFORE
    assert "IN ('RUNNING', 'QUEUED') AND NOT \"isDeleted\"" in runs.UNFINISHED_BEFORE


def test_with_every_unfinished_run_held_the_broker_is_not_asked():
    connection = Connection(stale=["resumed"])

    done, asked = swept(connection, Locks(runs.Activity(("resumed",), ())), AT_REST)

    assert done and not asked
    assert ended(connection) == []


@pytest.mark.parametrize(
    "counts",
    [
        {
            "messages_ready": 1,
            "messages_unacknowledged": 0,
            "consumers": 1,
        },  # one waits: maybe its
        {
            "messages_ready": 0,
            "messages_unacknowledged": 1,
            "consumers": 1,
        },  # one is being handed about
        {
            "messages_ready": 0,
            "messages_unacknowledged": 0,
            "consumers": 0,
        },  # no executor yet
        None,  # the broker did not say
        {},
    ],
)
def test_while_a_message_could_be_the_runs_it_is_left_and_asked_about_again(counts):
    connection = Connection(stale=["maybe-lost"])

    done, asked = swept(connection, Locks(runs.Activity()), counts)

    assert not done and asked
    assert ended(connection) == []


def test_locks_that_changed_while_the_broker_was_asked_put_the_question_off():
    connection = Connection(stale=["maybe-lost"])
    locks = Locks(runs.Activity(), runs.Activity(("maybe-lost",), ()))

    done, _ = swept(connection, locks, AT_REST)

    assert not done
    assert ended(connection) == []


def test_a_cache_or_a_database_that_does_not_answer_puts_the_question_off(caplog):
    done, asked = swept(Connection(stale=["lost"]), Locks(None), AT_REST)
    assert not done and not asked

    def connect():
        raise RuntimeError('relation "AgentGraphExecution" does not exist')

    assert runs.sweep_orphans(connect, Locks(), lambda: AT_REST, BOOTED) is False
    assert "could not look for agent runs that nothing will finish" in caplog.text


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_sweep_waits_a_minute_after_the_app_is_up_and_runs_until_done_once():
    clock, answers, calls = Clock(), [False, False, True], []

    def sweep() -> bool:
        calls.append(clock.now)
        return answers.pop(0)

    orphans = runs.OrphanSweep(sweep, clock)
    orphans.tend()  # the app is not up yet
    orphans.app_is_up()
    clock.now += runs.ORPHAN_SWEEP_AFTER_SECONDS - 1
    orphans.tend()
    assert calls == []
    for _ in range(600):
        clock.now += 1
        orphans.tend()
        orphans.wait()

    first = 1000.0 + runs.ORPHAN_SWEEP_AFTER_SECONDS
    every = runs.ORPHAN_SWEEP_EVERY_SECONDS
    assert calls == [first, first + every, first + 2 * every]


def test_a_queue_that_never_comes_to_rest_is_given_up_on_until_the_next_start(caplog):
    clock, calls = Clock(), []
    orphans = runs.OrphanSweep(lambda: bool(calls.append(1)), clock)
    orphans.app_is_up()
    for _ in range(runs.ORPHAN_SWEEP_TRIES * 4):
        clock.now += runs.ORPHAN_SWEEP_EVERY_SECONDS
        orphans.tend()
        orphans.wait()

    assert len(calls) == runs.ORPHAN_SWEEP_TRIES
    assert "gave up looking for agent runs that nothing will finish" in caplog.text


def test_a_broker_that_is_slow_to_say_does_not_hold_up_whoever_tends_the_sweep():
    """That is the loop that restarts processes and hears a stop."""
    clock, asked, answer = Clock(), threading.Event(), threading.Event()
    calls = []

    def sweep() -> bool:
        calls.append(1)
        asked.set()
        return answer.wait(30)

    orphans = runs.OrphanSweep(sweep, clock)
    orphans.app_is_up()
    clock.now += runs.ORPHAN_SWEEP_AFTER_SECONDS
    started = time.monotonic()
    orphans.tend()
    assert asked.wait(10)
    clock.now += 10 * runs.ORPHAN_SWEEP_EVERY_SECONDS
    orphans.tend()  # still asking: not asked a second time meanwhile
    orphans.wait(0.2)  # a stop waits for it only so long
    assert time.monotonic() - started < 5
    assert calls == [1]

    answer.set()
    orphans.wait()
    clock.now += 10 * runs.ORPHAN_SWEEP_EVERY_SECONDS
    orphans.tend()
    assert calls == [1] and orphans.due is None  # done once


def test_a_sweep_that_raises_is_tried_again(caplog):
    clock, calls = Clock(), []

    def sweep() -> bool:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("no such tool")
        return True

    orphans = runs.OrphanSweep(sweep, clock)
    orphans.app_is_up()
    for _ in range(4):
        clock.now += runs.ORPHAN_SWEEP_AFTER_SECONDS
        orphans.tend()
        orphans.wait()
    assert calls == [1, 1]
    assert (
        "could not look for agent runs that nothing will finish: no such tool"
        in caplog.text
    )


# --- asking the broker how many messages there are ----------------------------


def asked_of_rabbitmqctl(monkeypatch, tmp_path: Path, answer):
    """rabbitmq.queue_counts with rabbitmqctl's answer made up: its output,
    or what running it raised."""
    asked: list[list[str]] = []

    def run_tool(argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        asked.append(argv)
        if isinstance(answer, BaseException):
            raise answer
        return subprocess.CompletedProcess(argv, 0, answer, b"")

    monkeypatch.setattr(rabbitmq, "run_tool", run_tool)
    monkeypatch.setattr(rabbitmq, "_script", lambda bundle, name: name)
    return rabbitmq.queue_counts(Bundle(tmp_path), {}, runs.GRAPH_RUN_QUEUE), asked


def test_the_broker_is_asked_for_what_waits_what_is_unsettled_and_who_consumes(
    monkeypatch, tmp_path: Path
):
    listed = [
        {
            "name": "copilot_execution_queue_v2",
            "messages_ready": 4,
            "messages_unacknowledged": 0,
            "consumers": 1,
        },
        {
            "name": "graph_execution_queue_v2",
            "messages_ready": 0,
            "messages_unacknowledged": 2,
            "consumers": 1,
        },
    ]

    counts, asked = asked_of_rabbitmqctl(
        monkeypatch, tmp_path, json.dumps(listed).encode()
    )

    assert counts == {"messages_ready": 0, "messages_unacknowledged": 2, "consumers": 1}
    assert asked == [
        # With a limit of its own: the one on running it does not end the
        # Erlang VM on Windows.
        [
            "rabbitmqctl",
            "-n",
            rabbitmq.NODE_NAME,
            "--timeout",
            "10",
            "--quiet",
            "--formatter",
            "json",
        ]
        + [
            "list_queues",
            "name",
            "messages_ready",
            "messages_unacknowledged",
            "consumers",
        ]
    ]


@pytest.mark.parametrize(
    "answer",
    [
        b"Error: unable to perform an operation on node 'rabbit@localhost'",
        b"[]",
        b'[{"name": "graph_execution_queue_v2", "messages_ready": 0}]',
        b'{"name": "graph_execution_queue_v2"}',
        OSError("no such file"),
        subprocess.TimeoutExpired("rabbitmqctl", 20),
    ],
    ids=[
        "an error",
        "no queues",
        "a column short",
        "not a list",
        "cannot run",
        "no answer",
    ],
)
def test_a_broker_that_does_not_say_is_not_guessed_at(
    monkeypatch, tmp_path: Path, answer
):
    counts, _ = asked_of_rabbitmqctl(monkeypatch, tmp_path, answer)
    assert counts is None
