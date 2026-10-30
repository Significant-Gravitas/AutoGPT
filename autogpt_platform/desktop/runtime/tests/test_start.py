"""How a start is laid out in time: what runs beside what, what may never be
cut short, what a slow or missing piece costs, and the numbers it reports.

The pieces are stand-ins here. build/smoke_test.py times the real ones and
holds a restart to a budget.
"""

import importlib
import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from autogpt_desktop import (
    apps,
    events,
    migrations,
    postgres,
    process,
    rabbitmq,
    runs,
    servicehost,
    supervisor,
    timing,
    valkey,
)
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ChildRegistry, ManagedProcess, wait_until
from autogpt_desktop.supervisor import Abandoned, Background, Stack, StartupError

DESKTOP = Path(__file__).resolve().parents[2]
SLEEP = "import time; time.sleep(60)"
# The RabbitMQ that `rabbitmq._server_flags` was last seen to boot with.
RABBITMQ_SEEN_BOOTING_DISTRIBUTED = "4.1.8"


@pytest.fixture
def stack(tmp_path: Path):
    data = DataDir(tmp_path / "data")
    data.prepare()
    stack = Stack(Bundle(tmp_path / "runtime"), data)
    rabbit_application(stack.bundle, "4.3.6")
    yield stack
    stack.stop()


def rabbit_application(bundle: Bundle, version: str) -> None:
    metadata = (
        bundle.rabbitmq_home / "plugins" / f"rabbit-{version}" / "ebin" / "rabbit.app"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(f'{{application, rabbit, [\n{{vsn, "{version}"}}\n]}}.\n')


def stand_in(stack: Stack, name: str, code: str = SLEEP) -> ManagedProcess:
    return ManagedProcess(
        name, [sys.executable, "-c", code], {}, stack.data.root, stack.data.logs
    )


def events_of(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


# --- the first second ---------------------------------------------------------


def test_a_machine_without_openssl_is_told_before_anything_is_created(
    stack: Stack, monkeypatch
):
    """Linux: the migrations' engine needs libssl.so.3. Found out after the
    database was created, that was a minute into the first start."""

    def refuse(bundle: Bundle) -> None:
        raise RuntimeError(migrations.OPENSSL_MISSING)

    monkeypatch.setattr(migrations, "require_engines", refuse)
    monkeypatch.setattr(
        postgres, "initialize", lambda *args: pytest.fail("a database was created")
    )

    with pytest.raises(RuntimeError, match="OpenSSL 3"):
        stack.start()

    assert stack.processes == []
    assert not stack.data.ports_file.exists()
    assert not (stack.data.postgres / "PG_VERSION").exists()


def test_the_password_file_is_still_taken_before_the_engine_check(
    stack: Stack, monkeypatch
):
    """Whatever stops a start, the new password is not left on disk."""
    waiting = stack.data.config / "reset-password"
    waiting.write_text("a-new-password-for-the-owner\n")
    monkeypatch.setattr(
        migrations,
        "require_engines",
        lambda bundle: (_ for _ in ()).throw(RuntimeError("no")),
    )

    with pytest.raises(RuntimeError):
        stack.start()

    assert not waiting.exists()


# --- the database's side runs beside the wait for the broker ------------------


@pytest.mark.parametrize("entry", ["start", "start_infrastructure"])
def test_unknown_rabbit_storage_is_rejected_before_profile_mutation(
    stack: Stack, monkeypatch, entry: str
):
    mnesia = stack.data.rabbitmq / "mnesia"
    mnesia.mkdir(parents=True)
    payload = mnesia / "persisted-queue"
    payload.write_bytes(b"must survive")
    monkeypatch.setattr(migrations, "require_engines", lambda *args: None)

    def mutated(*args):
        pytest.fail("profile changed before compatibility check")

    monkeypatch.setattr(DataDir, "prepare", mutated)
    monkeypatch.setattr(valkey, "write_config", mutated)
    monkeypatch.setattr(rabbitmq, "prepare", mutated)
    monkeypatch.setattr(stack, "launch", mutated)
    monkeypatch.setattr(stack, "start_database", mutated)
    with (
        supervisor.DataLock(stack.data.root),
        pytest.raises(rabbitmq.IncompatibleDataError, match="unknown"),
    ):
        if entry == "start":
            stack.start()
        else:
            stack.start_infrastructure({}, {}, False)
    assert payload.read_bytes() == b"must survive"
    assert not (stack.data.rabbitmq / rabbitmq.DATA_VERSION_FILE).exists()


@pytest.mark.parametrize("ready", [True, False])
def test_rabbit_writer_is_recorded_before_launch_and_ready_only_after_auth(
    stack: Stack, monkeypatch, ready: bool
):
    marker = stack.data.rabbitmq / rabbitmq.DATA_VERSION_FILE
    seen = []

    def state():
        return json.loads(marker.read_text())

    def launch(processes):
        if any(item.name == "rabbitmq" for item in processes):
            assert state() == {
                "schema_version": 1,
                "version": "4.3.6",
                "state": "starting",
            }
            seen.append("launch")
        return processes

    def await_queue(*args):
        assert state()["state"] == "starting"
        mnesia = stack.data.rabbitmq / "mnesia"
        mnesia.mkdir(exist_ok=True)
        (mnesia / "persisted-queue").write_bytes(b"broker write")
        seen.append("authentication")
        if not ready:
            raise StartupError("broker authentication failed")

    monkeypatch.setattr(valkey, "write_config", lambda *args: None)
    monkeypatch.setattr(rabbitmq, "prepare", lambda *args: None)
    monkeypatch.setattr(rabbitmq, "epmd_process", lambda *args: stand_in(stack, "epmd"))
    monkeypatch.setattr(postgres, "check_compatible", lambda *args: None)
    monkeypatch.setattr(valkey, "process", lambda *args: stand_in(stack, "valkey"))
    monkeypatch.setattr(rabbitmq, "process", lambda *args: stand_in(stack, "rabbitmq"))
    monkeypatch.setattr(stack, "launch", launch)
    monkeypatch.setattr(stack, "await_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(stack, "await_cache", lambda *args: None)
    monkeypatch.setattr(stack, "await_queue", await_queue)
    monkeypatch.setattr(valkey, "ensure_cluster", lambda *args: None)
    monkeypatch.setattr(stack, "start_database", lambda *args: None)
    secrets = {
        "RABBITMQ_DEFAULT_USER": "u",
        "RABBITMQ_DEFAULT_PASS": "p",
        "REDIS_PASSWORD": "r",
    }
    with supervisor.DataLock(stack.data.root):
        wait = stack.start_infrastructure(PORTS, secrets, False)
        if ready:
            wait()
        else:
            with pytest.raises(StartupError, match="authentication failed"):
                wait()
    assert seen == ["launch", "authentication"]
    assert state()["state"] == ("ready" if ready else "starting")
    assert (
        stack.data.rabbitmq / "mnesia" / "persisted-queue"
    ).read_bytes() == b"broker write"
    assert ("queue" in stack.phases.seconds) is ready


def test_the_migrations_run_while_the_broker_is_waited_for(stack: Stack):
    order: list[str] = []
    migrating = threading.Event()

    def migrate() -> None:
        order.append("migrations begin")
        migrating.set()
        time.sleep(0.3)
        order.append("migrations end")

    def wait_for_the_broker() -> None:
        assert migrating.wait(5), "the migrations did not start beside the wait"
        order.append("the broker answers")

    stack.database_branch = Background(migrate)
    stack.settle(wait_for_the_broker)

    assert order == ["migrations begin", "the broker answers", "migrations end"]


def test_a_broker_that_fails_does_not_cut_the_migrations_short(stack: Stack):
    """What follows a failed start is a stop of every server. The database
    must not be taken from under a migration, so the failure waits."""
    finished: list[float] = []

    def migrate() -> None:
        time.sleep(0.5)
        finished.append(time.monotonic())

    def wait_for_the_broker() -> None:
        raise StartupError(
            "rabbitmq exited while starting. See rabbitmq.log for details."
        )

    stack.database_branch = Background(migrate)
    with pytest.raises(StartupError, match="rabbitmq exited while starting"):
        stack.settle(wait_for_the_broker)

    assert finished and finished[0] <= time.monotonic()


def test_a_failed_migration_is_reported_in_its_own_words(stack: Stack):
    def migrate() -> None:
        raise RuntimeError("prisma migrate deploy failed:\nP3009")

    stack.database_branch = Background(migrate)
    with pytest.raises(RuntimeError, match="prisma migrate deploy failed"):
        stack.settle(lambda: None)


def test_when_both_sides_fail_the_database_is_what_the_user_is_told(
    stack: Stack, caplog
):
    def migrate() -> None:
        raise RuntimeError("A database update did not finish (20260101_init).")

    def wait_for_the_broker() -> None:
        raise StartupError("rabbitmq did not start. See rabbitmq.log for details.")

    stack.database_branch = Background(migrate)
    with pytest.raises(RuntimeError, match="A database update did not finish"):
        stack.settle(wait_for_the_broker)

    assert "rabbitmq did not start" in caplog.text


def test_a_database_that_failed_ends_the_wait_for_the_broker_at_once(stack: Stack):
    """Not after the four minutes a broker is given."""
    queue = stand_in(stack, "rabbitmq")
    queue.start()

    def fail() -> None:
        raise RuntimeError("initdb failed:\nno space left on device")

    stack.database_branch = Background(fail)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="initdb failed"):
            stack.settle(lambda: stack.await_ready(queue, lambda: False, timeout=240))
        assert time.monotonic() - started < 20
    finally:
        queue.stop()


def test_a_cancelled_start_says_so_and_not_what_the_other_side_made_of_it(stack: Stack):
    def migrate() -> None:
        stack.raise_if_cancelled()

    stack.stop_requested.set()
    stack.database_branch = Background(migrate)
    with pytest.raises(StartupError, match="cancelled"):
        stack.settle(stack.raise_if_cancelled)


def test_a_stop_waits_for_a_migration_that_is_running(stack: Stack):
    """However the start ended: `Stack.run` stops everything in a `finally`."""
    done: list[str] = []
    server = stand_in(stack, "postgres")
    server.graceful_stop = lambda process: (
        done.append("postgres stopped"),
        process.kill(),
    )
    stack.launch([server])

    def migrate() -> None:
        time.sleep(0.5)
        done.append("migrated")

    stack.database_branch = Background(migrate)
    stack.stop()

    assert done == ["migrated", "postgres stopped"]


# --- a side that has failed is not followed by new work on the other ----------


def broker_fails_while(stack: Stack, monkeypatch, busy_with: str) -> list[str]:
    """The database's side as `start_database` runs it, with the broker's
    side failing while it is inside `busy_with`. What it went on to do."""
    calls: list[str] = []
    inside, failed = threading.Event(), threading.Event()

    def step(name: str):
        def run(*args) -> None:
            calls.append(name)
            if name == busy_with:
                inside.set()
                assert failed.wait(10)

        return run

    monkeypatch.setattr(postgres, "initialize", step("initialize"))
    monkeypatch.setattr(postgres, "process", lambda *args: stand_in(stack, "postgres"))
    monkeypatch.setattr(postgres, "is_ready", lambda port, password: True)
    monkeypatch.setattr(postgres, "first_run_completed", lambda data: None)
    monkeypatch.setattr(
        migrations, "require_compatible_history", step("require_compatible_history")
    )
    for name in (
        "create_schemas",
        "refuse_interrupted_migration",
        "remove_owner_trigger",
    ):
        monkeypatch.setattr(supervisor.bootstrap, name, step(name))
    monkeypatch.setattr(
        supervisor.bootstrap, "apply_migrations", step("apply_migrations")
    )
    monkeypatch.setattr(
        supervisor.bootstrap, "configure_frontend_role", step("frontend role")
    )
    monkeypatch.setattr(stack, "secure_owner", step("owner"))
    monkeypatch.setattr(
        apps, "frontend_process", lambda *args: pytest.fail("a frontend was started")
    )
    servers = stack.launch([])
    ports, secrets = {"postgres": 1}, {
        "POSTGRES_PASSWORD": "p",
        "AUTOGPT_FRONTEND_DB_PASSWORD": "f",
    }

    def wait_for_the_broker() -> None:
        assert inside.wait(10)
        try:
            raise StartupError(
                "rabbitmq exited while starting. See rabbitmq.log for details."
            )
        finally:
            stack.start_failed.set()  # as `cache_and_queue` does, on its thread
            failed.set()

    stack.database_branch = Background(
        lambda: stack.start_database(servers, ports, secrets, True)
    )
    with pytest.raises(StartupError, match="rabbitmq exited while starting"):
        stack.settle(wait_for_the_broker)
    assert isinstance(stack.database_branch.failed, Abandoned)
    return calls


def test_a_database_is_not_migrated_for_a_start_that_has_already_failed(
    stack: Stack, monkeypatch
):
    """On a loaded machine that was a minute of 'Creating the database
    tables' before the broker's error, known all along, was shown."""
    calls = broker_fails_while(stack, monkeypatch, "initialize")

    assert calls == ["initialize"]
    assert [process.name for process in stack.processes] == []


def test_a_migration_that_is_running_still_runs_to_its_end(stack: Stack, monkeypatch):
    calls = broker_fails_while(stack, monkeypatch, "apply_migrations")

    # And what makes the migrated database usable follows it; no frontend.
    assert calls == [
        "initialize",
        "require_compatible_history",
        "create_schemas",
        "refuse_interrupted_migration",
        "remove_owner_trigger",
        "apply_migrations",
        "frontend role",
        "owner",
    ]
    assert stack.migrating is False


def test_a_broker_that_failed_ends_the_wait_for_the_database_at_once(stack: Stack):
    database = stand_in(stack, "postgres")
    database.start()
    stack.start_failed.set()
    started = time.monotonic()
    try:
        with pytest.raises(Abandoned):
            stack.await_ready(database, lambda: False, timeout=120)
        assert time.monotonic() - started < 20
    finally:
        database.stop()


# --- each server is asked from the moment it is started ----------------------


def test_the_cache_and_the_broker_are_timed_while_the_start_does_other_things(
    stack: Stack, monkeypatch
):
    """With a Claude sign-in the main thread waits for its detection before
    it gets to ask them; that wait was the cache's and the broker's time."""
    monkeypatch.setattr(valkey, "write_config", lambda *args: None)
    monkeypatch.setattr(rabbitmq, "prepare", lambda *args: None)
    monkeypatch.setattr(rabbitmq, "epmd_process", lambda *args: stand_in(stack, "epmd"))
    monkeypatch.setattr(rabbitmq, "epmd_is_ready", lambda port: True)
    monkeypatch.setattr(postgres, "check_compatible", lambda *args: None)
    monkeypatch.setattr(valkey, "process", lambda *args: stand_in(stack, "valkey"))
    monkeypatch.setattr(valkey, "is_ready", lambda *args: True)
    monkeypatch.setattr(valkey, "ensure_cluster", lambda *args: None)
    monkeypatch.setattr(rabbitmq, "process", lambda *args: stand_in(stack, "rabbitmq"))
    monkeypatch.setattr(rabbitmq, "is_ready", lambda *args: True)
    monkeypatch.setattr(rabbitmq, "record_ready", lambda *args: None)
    monkeypatch.setattr(stack, "start_database", lambda *args: None)
    secrets = {
        "RABBITMQ_DEFAULT_USER": "u",
        "RABBITMQ_DEFAULT_PASS": "p",
        "REDIS_PASSWORD": "r",
    }

    wait = stack.start_infrastructure(PORTS, secrets, False)

    # Nobody has made the wait yet.
    assert wait_until(
        lambda: {"cache", "queue"} <= stack.phases.seconds.keys(), 10, 0.05
    )
    assert stack.phases.seconds["queue"] < 5
    assert stack.cache_is_up.is_set()
    wait()
    stack.settle(wait)


def test_the_frontend_is_timed_from_its_launch_to_its_first_answer(
    stack: Stack, monkeypatch
):
    """Not to when the start gets round to asking, which is after the broker."""
    host_stand_ins(stack, monkeypatch)
    answers_from = time.monotonic() + 0.6
    monkeypatch.setattr(
        supervisor, "_http_ok", lambda url: time.monotonic() >= answers_from
    )

    stack.start_frontend(PORTS, SECRETS)

    assert wait_until(lambda: "frontend" in stack.phases.seconds, 10, 0.05)
    assert 0.5 <= stack.phases.seconds["frontend"] < 3


# --- the record of what is running, written from two threads -----------------


def test_a_cache_started_again_is_recorded_under_the_same_lock_as_a_launch(
    stack: Stack, monkeypatch
):
    """The database's side may be launching PostgreSQL at that moment. A
    list read before that launch and written after it would leave
    PostgreSQL out of the record a later start cleans up from."""
    held: list[bool] = []

    def free_for_another_thread() -> bool:
        taken: list[bool] = []

        def try_it() -> None:
            taken.append(stack._launching.acquire(blocking=False))
            if taken[0]:
                stack._launching.release()

        other = threading.Thread(target=try_it)
        other.start()
        other.join()
        return taken[0]

    cache = stand_in(stack, "valkey", "import sys; sys.exit(1)")
    stack.launch([cache])
    wait_until(lambda: cache.exit_code() is not None, 20, 0.05)
    cache.argv = [sys.executable, "-c", SLEEP]
    start, record = cache.start, stack.registry.record
    monkeypatch.setattr(
        cache, "start", lambda: (held.append(not free_for_another_thread()), start())
    )
    monkeypatch.setattr(
        stack.registry,
        "record",
        lambda processes: (
            held.append(not free_for_another_thread()),
            record(processes),
        ),
    )
    monkeypatch.setattr(
        valkey, "is_ready", lambda port, password: cache.exit_code() is None
    )
    monkeypatch.setattr(valkey, "set_aside", lambda data: None)
    monkeypatch.setattr(valkey, "write_config", lambda *args: None)

    stack.await_cache(cache, {"valkey": 1, "valkey_bus": 2}, "password")

    assert held == [True, True]
    assert [entry["name"] for entry in json.loads(stack.registry.path.read_text())] == [
        "valkey"
    ]


def test_the_record_is_replaced_whole(tmp_path: Path, monkeypatch):
    """Not written in place: a runtime killed in the middle would leave half
    a list, and the next start could then stop none of what was left."""
    registry = ChildRegistry(tmp_path / "children.json")
    registry.path.write_text('[{"name": "postgres", "pid": 1, "started": 1.0}]')
    written: list[Path] = []
    monkeypatch.setattr(
        process.os,
        "replace",
        lambda source, target: written.append(Path(source)) or pytest.fail("killed"),
    )

    with pytest.raises(pytest.fail.Exception):
        registry.record([])

    assert json.loads(registry.path.read_text())[0]["name"] == "postgres"
    assert written and written[0] != registry.path


def test_the_database_is_created_started_and_migrated_on_its_own_side(
    stack: Stack, monkeypatch
):
    calls: list[str] = []
    monkeypatch.setattr(
        postgres, "initialize", lambda *args: calls.append("initialize")
    )
    monkeypatch.setattr(postgres, "process", lambda *args: stand_in(stack, "postgres"))
    monkeypatch.setattr(postgres, "is_ready", lambda port, password: True)
    monkeypatch.setattr(stack, "migrate", lambda *args: calls.append("migrate"))
    monkeypatch.setattr(stack, "start_frontend", lambda *args: calls.append("frontend"))
    servers = stack.launch([])

    stack.start_database(servers, {"postgres": 1}, {"POSTGRES_PASSWORD": "p"}, True)

    assert calls == ["initialize", "migrate", "frontend"]
    assert [server.name for server in servers] == ["postgres"]
    assert "database" in stack.phases.seconds


# --- hosts load early and start their services on the go ---------------------


def host_stand_ins(stack: Stack, monkeypatch) -> dict[tuple[str, ...], dict[str, str]]:
    given: dict[tuple[str, ...], dict[str, str]] = {}

    def host_process(bundle, data, env, group, cache) -> ManagedProcess:
        given[group.services] = env
        return stand_in(stack, group.name)

    monkeypatch.setattr(apps, "host_process", host_process)
    monkeypatch.setattr(
        apps, "frontend_process", lambda *args: stand_in(stack, apps.FRONTEND)
    )
    monkeypatch.setattr(supervisor, "ProxyThread", NoProxy)
    monkeypatch.setattr(Stack, "wait_for_apps", lambda self, port: None)
    stack.env = {"AUTOGPT_PUBLIC_URL": "http://127.0.0.1:1"}
    return given


class NoProxy:
    def __init__(self, *args) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


PORTS = {name: 20000 + index for index, name in enumerate(supervisor.ports.PORT_NAMES)}
SECRETS = {"AUTOGPT_FRONTEND_DB_PASSWORD": "a", "REDIS_PASSWORD": "b"}


def test_hosts_are_started_before_the_databases_and_told_where_to_wait(
    stack: Stack, monkeypatch
):
    given = host_stand_ins(stack, monkeypatch)

    stack.load_services(PORTS, SECRETS)

    assert {process.name for process in stack.processes} == {
        "database-manager",
        "swap-credentials",
        "workers",
        "websocket",
        "api",
    }
    assert given and all(
        env[servicehost.GO_FILE_ENV] == str(stack.go_file) for env in given.values()
    )
    assert not stack.go_file.exists()
    # The services' own environment is the stack's, untouched.
    assert servicehost.GO_FILE_ENV not in stack.env
    # Every host is told which service's host closes the chats of turns that
    # were ended without their executor, and where they are listed.
    listed = stack.data.run / runs.ENDED_TURNS_FILE
    assert all(
        env[servicehost.ENDED_TURNS_ENV] == f"{apps.COPILOT_EXECUTOR}={listed}"
        for env in given.values()
    )


def test_the_go_is_given_after_the_stale_locks_are_cleared(stack: Stack, monkeypatch):
    given = host_stand_ins(stack, monkeypatch)
    cleared: list[tuple[tuple[str, ...], bool]] = []

    def clear(self, services) -> None:
        cleared.append((tuple(services), stack.go_file.exists()))

    monkeypatch.setattr(runs.Cache, "clear_stale_locks", clear)

    stack.start_apps(PORTS, SECRETS)

    assert stack.go_file.exists()
    assert {services for services, _ in cleared} == set(given)
    assert not any(
        gone for _, gone in cleared
    ), "a service could start before its locks went"
    names = [process.name for process in stack.processes]
    assert names.count(apps.FRONTEND) == 1


def test_a_go_left_by_the_last_run_does_not_let_services_start_early(
    stack: Stack, monkeypatch
):
    """The file outlives a runtime that was killed."""
    stack.go_file.write_text("go\n")
    monkeypatch.setattr(
        supervisor.settings,
        "ensure_secrets",
        lambda *args: pytest.fail("past the go file"),
    )

    with pytest.raises(pytest.fail.Exception):
        stack.start()

    assert not stack.go_file.exists()


def test_a_host_started_before_the_cache_is_up_does_not_try_to_clear_locks(monkeypatch):
    up = threading.Event()
    cache = runs.Cache(1, "password", up)
    asked: list[list[str]] = []
    monkeypatch.setattr(
        runs.Cache,
        "_delete_matching",
        lambda self, patterns: asked.append(patterns) or 0,
    )

    cache.clear_stale_locks(["executor"])
    assert asked == []

    up.set()
    cache.clear_stale_locks(["executor"])
    assert asked == [["exec_lock:*"]]


def test_the_frontend_starts_as_soon_as_the_migrations_are_in(
    stack: Stack, monkeypatch
):
    """It needs what their last step decides (whether registration is open)
    and nothing of the broker or the backend."""
    host_stand_ins(stack, monkeypatch)
    seen: dict[str, str] = {}

    def frontend_process(bundle, data, env) -> ManagedProcess:
        seen.update(env)
        return stand_in(stack, apps.FRONTEND)

    monkeypatch.setattr(apps, "frontend_process", frontend_process)
    stack.env["AUTH_ALLOW_NEW_ACCOUNTS"] = "false"  # as secure_owner leaves it

    stack.start_frontend(PORTS, SECRETS)

    assert seen["AUTH_ALLOW_NEW_ACCOUNTS"] == "false"
    assert [process.name for process in stack.processes] == [apps.FRONTEND]
    stack.start_apps(PORTS, SECRETS)
    assert [process.name for process in stack.processes].count(apps.FRONTEND) == 1


# --- the service host's side of the go ---------------------------------------

FAKE_BUNDLE = Path(__file__).parent / "fake_bundle"


class NoCache:
    def clear_stale_locks(self, services) -> None:
        pass


def start_host(
    tmp_path: Path,
    monkeypatch,
    go_file: Path | None,
    services: tuple[str, ...] = ("one",),
) -> tuple[ManagedProcess, Path, DataDir]:
    data = DataDir(tmp_path / "data")
    data.prepare()
    marks = tmp_path / "marks"
    marks.mkdir()
    known = tuple(
        apps.Service(name, f"backend.services:{name}", "port", "/") for name in services
    )
    monkeypatch.setattr(apps, "SERVICES", known)
    env = {"FAKE_MARKS": str(marks), "PYTHONDONTWRITEBYTECODE": "1"}
    if go_file:
        env[servicehost.GO_FILE_ENV] = str(go_file)
    host = apps.host_process(
        Bundle(FAKE_BUNDLE), data, env, apps.Group("host", services), NoCache()
    )
    host.start()
    return host, marks, data


def log_of(data: DataDir) -> str:
    return (data.logs / "host.log").read_text(encoding="utf-8", errors="replace")


def test_a_host_loads_and_then_waits_for_the_go(tmp_path: Path, monkeypatch):
    go_file = tmp_path / "services.go"
    host, marks, data = start_host(tmp_path, monkeypatch, go_file)
    try:
        assert wait_until(
            lambda: "waiting for the databases" in log_of(data), 30, 0.05
        ), log_of(data)
        time.sleep(0.5)
        assert not (marks / "one.started").exists(), "a service started before the go"

        go_file.write_text("go\n")

        assert wait_until((marks / "one.started").exists, 20, 0.05), log_of(data)
        # Not handed on to what the services start.
        assert servicehost.GO_FILE_ENV not in (marks / "one.environment").read_text()
    finally:
        host.stop()


@pytest.mark.parametrize("services", [("serves_itself",), ("one", "serves_itself")])
def test_an_entry_point_that_serves_by_itself_is_not_called_before_the_go(
    tmp_path: Path, monkeypatch, services: tuple[str, ...]
):
    """Upstream drift the host has a way of living with: an entry point that
    runs its service instead of handing it over. Called while the host
    loads, that service would be running before the database is migrated."""
    go_file = tmp_path / "services.go"
    host, marks, data = start_host(tmp_path, monkeypatch, go_file, services)
    try:
        assert wait_until(
            lambda: "waiting for the databases" in log_of(data), 30, 0.05
        ), log_of(data)
        time.sleep(1)
        assert list(marks.iterdir()) == [], "an entry point was called before the go"

        go_file.write_text("go\n")

        assert wait_until((marks / "itself.started").exists, 20, 0.05), log_of(data)
    finally:
        host.stop()


def test_a_host_that_is_restarted_later_does_not_wait(tmp_path: Path, monkeypatch):
    go_file = tmp_path / "services.go"
    go_file.write_text("go\n")
    host, marks, data = start_host(tmp_path, monkeypatch, go_file)
    try:
        assert wait_until((marks / "one.started").exists, 30, 0.05), log_of(data)
        assert "waiting for the databases" not in log_of(data)
    finally:
        host.stop()


def test_a_host_asked_to_stop_while_it_waits_leaves_by_itself(
    tmp_path: Path, monkeypatch
):
    host, marks, data = start_host(tmp_path, monkeypatch, tmp_path / "never.go")
    assert wait_until(
        lambda: "waiting for the databases" in log_of(data), 30, 0.05
    ), log_of(data)

    host.stop()

    assert host.exit_code() == 0, log_of(data)
    assert "stopped with code 0" in log_of(data)
    assert not (marks / "one.started").exists()


# --- asking whether a server is up --------------------------------------------


def closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_a_closed_port_is_known_to_be_closed_at_once():
    """Windows retries a refused connection for two seconds; a start that
    asked nine services in turn fell twenty seconds behind."""
    started = time.monotonic()
    assert process.listening(closed_port()) is False
    assert time.monotonic() - started < 1.5


def test_an_open_port_is_seen_even_when_nobody_accepts():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        assert process.listening(server.getsockname()[1]) is True


@pytest.mark.parametrize("module", ["postgres", "valkey", "rabbitmq"])
def test_no_client_is_pointed_at_a_server_that_is_not_listening(
    module: str, monkeypatch
):
    """psycopg2, redis and pika each sit out the system's own retries."""
    readiness = importlib.import_module(f"autogpt_desktop.{module}")
    for client in ("psycopg2", "redis", "pika"):
        monkeypatch.setitem(sys.modules, client, None)  # importing one would fail
    port = closed_port()
    arguments = (
        (port, "user", "password") if module == "rabbitmq" else (port, "password")
    )

    assert readiness.is_ready(*arguments) is False


def test_services_are_asked_all_at_once(stack: Stack, monkeypatch):
    """One that is not up yet costs its own wait, not everybody's."""
    asked_at: list[float] = []

    def slow_no(url: str) -> bool:
        asked_at.append(time.monotonic())
        time.sleep(0.4)
        return False

    monkeypatch.setattr(supervisor, "_http_ok", slow_no)
    monkeypatch.setattr(supervisor, "APP_READY_TIMEOUT_SECONDS", 1)
    started = time.monotonic()

    with pytest.raises(StartupError, match="did not become ready"):
        stack.wait_for_apps(PORTS)

    services = len(apps.health_urls(PORTS))
    rounds = len(asked_at) / services
    assert rounds == int(rounds) and rounds >= 2
    # In turn, one round alone would take 0.4 s x 9.
    assert time.monotonic() - started < 0.4 * services


# --- the broker, started as a node --------------------------------------------


def test_the_broker_is_started_as_the_node_rabbitmq_would_make_it(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home" / ".cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "home" / "local"))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "home" / "shared"))
    bundle = Bundle(tmp_path / "runtime")
    for directory in (bundle.erlang_home, bundle.rabbitmq_home):
        directory.mkdir(parents=True)
    rabbit_application(bundle, RABBITMQ_SEEN_BOOTING_DISTRIBUTED)
    data = DataDir(tmp_path / "data")
    data.prepare()
    ports = {"rabbitmq": 15000, "rabbitmq_dist": 15002, "epmd": 15001}

    env = rabbitmq.environment(bundle, data, ports)

    flags = env["RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS"]
    assert f"-sname {env['RABBITMQ_NODENAME']}" in flags
    assert (
        "-kernel inet_dist_listen_min 15002 -kernel inet_dist_listen_max 15002" in flags
    )
    assert env["RABBITMQ_DIST_PORT"] == "15002"
    assert rabbitmq.QUIET_SYSLOG in flags
    # Every VM under this environment still keeps its distribution on loopback.
    assert rabbitmq.LOOPBACK_DISTRIBUTION in env["ERL_AFLAGS"]


@pytest.mark.parametrize("version", ["4.1.8", "4.3.6", None])
def test_only_a_smoke_proven_rabbitmq_uses_the_distributed_shortcut(tmp_path, version):
    bundle = Bundle(tmp_path / "runtime")
    if version:
        rabbit_application(bundle, version)
    flags = rabbitmq.server_flags(bundle, PORTS)
    assert rabbitmq.PROVEN_DISTRIBUTED_START_VERSIONS == {
        RABBITMQ_SEEN_BOOTING_DISTRIBUTED
    }
    if version == RABBITMQ_SEEN_BOOTING_DISTRIBUTED:
        assert "-sname rabbit@localhost" in flags
    else:
        assert flags == rabbitmq.QUIET_SYSLOG


def test_a_broker_that_does_not_boot_that_way_is_started_rabbitmqs_own_way(
    stack: Stack, monkeypatch
):
    """A version of RabbitMQ that boots differently costs seconds, not the app."""
    code = (
        "import os, sys, time\n"
        f"sys.exit(3) if '-sname' in os.environ[{rabbitmq.SERVER_FLAGS!r}] else time.sleep(60)\n"
    )
    queue = stand_in(stack, "rabbitmq", code)
    queue.env = {
        rabbitmq.SERVER_FLAGS: f"{rabbitmq.QUIET_SYSLOG} -sname rabbit@localhost",
        **process.base_env(),
    }
    monkeypatch.setattr(
        rabbitmq,
        "is_ready",
        lambda *args: "-sname" not in queue.env[rabbitmq.SERVER_FLAGS],
    )
    stack.launch([queue])
    wait_until(lambda: queue.exit_code() is not None, 20, 0.05)

    stack.await_queue(queue, {"rabbitmq": 1}, "user", "password")

    assert queue.exit_code() is None
    assert queue.env[rabbitmq.SERVER_FLAGS] == rabbitmq.QUIET_SYSLOG


def test_a_broker_that_fails_its_own_way_too_fails_the_start(stack: Stack, monkeypatch):
    queue = stand_in(stack, "rabbitmq", "import sys; sys.exit(3)")
    queue.env = {rabbitmq.SERVER_FLAGS: "-sname rabbit@localhost", **process.base_env()}
    monkeypatch.setattr(rabbitmq, "is_ready", lambda *args: False)
    stack.launch([queue])

    with pytest.raises(StartupError, match="rabbitmq exited while starting"):
        stack.await_queue(queue, {"rabbitmq": 1}, "user", "password")


# --- the numbers ---------------------------------------------------------------


def test_each_phase_is_reported_as_it_ends_and_all_of_them_at_the_end(capsys, caplog):
    clock = iter([0.0, 1.0, 3.5, 40.9])
    phases = timing.Phases(clock=lambda: next(clock))

    with phases.measure("migrations"), caplog.at_level(
        "INFO", logger="autogpt_desktop"
    ):
        pass
    phases.record("database", 2.04)
    phases.record("queue", 9.81)
    with caplog.at_level("INFO", logger="autogpt_desktop"):
        phases.finish()

    reported = events_of(capsys)
    assert reported[0] == {"event": "timing", "phase": "migrations", "seconds": 2.5}
    assert reported[-1] == {
        "event": "timing",
        "phase": "total",
        "seconds": 40.9,
        "phases": {"database": 2.04, "queue": 9.81, "migrations": 2.5},
    }
    assert "started in 41 s (database 2.0, queue 9.8, migrations 2.5)" in caplog.text


def test_a_service_restarted_while_starting_keeps_its_first_time():
    phases = timing.Phases()
    phases.record("services", 21.0)
    phases.record("services", 55.0)
    assert phases.seconds == {"services": 21.0}


def test_the_shell_shows_nothing_for_a_timing():
    """Its start-up window puts the message of a `progress` on screen, and
    nothing else of what the runtime says while it starts."""
    page = (DESKTOP / "src" / "startup.js").read_text(encoding="utf-8")
    assert 'event.event === "progress" && event.message' in page, (
        "src/startup.js no longer picks what it shows by `event.event`: make sure a "
        '{"event": "timing"} line (autogpt_desktop/events.py) still shows nothing'
    )
    assert "timing" not in page


def test_a_phase_that_failed_is_not_in_the_table():
    phases = timing.Phases()
    with pytest.raises(RuntimeError), phases.measure("migrations"):
        raise RuntimeError("failed")
    assert phases.seconds == {}


def test_timings_reach_the_log_too(caplog):
    with caplog.at_level("INFO", logger="autogpt_desktop"):
        events.timing("queue", 9.84)
    assert "queue: 9.8 s" in caplog.text
