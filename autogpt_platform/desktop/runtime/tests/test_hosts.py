"""How the supervisor treats service hosts: waiting for them, restarting
them, splitting one that upstream no longer fits, and the processes it
starts and stops them through."""

import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from autogpt_desktop import apps, ports
from autogpt_desktop import process as process_module
from autogpt_desktop import servicehost, settings, supervisor
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ManagedProcess, StopEvent, wait_until
from autogpt_desktop.supervisor import Stack, StartupError

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
SLEEP = "import time; time.sleep(60)"


@pytest.fixture
def stack(tmp_path: Path):
    data = DataDir(tmp_path / "data")
    data.prepare()
    stack = Stack(Bundle(tmp_path / "runtime"), data)
    yield stack
    stack.stop()


def process(stack: Stack, name: str, code: str = SLEEP, **options) -> ManagedProcess:
    return ManagedProcess(
        name=name,
        argv=[sys.executable, "-c", code],
        env={},
        cwd=stack.data.root,
        log_dir=stack.data.logs,
        stop_timeout=2,
        **options,
    )


def host_everything(stack: Stack, **code: str) -> None:
    """A merged layout of stand-in processes; `code` replaces what one runs."""
    groups = apps.layout(merged=True)
    stack.hosts = {group.name: group for group in groups}
    names = [*stack.hosts, apps.FRONTEND]
    stack.launch([process(stack, name, code.get(name, SLEEP)) for name in names])


def events_of(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def port_table() -> dict[str, int]:
    return {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}


def test_the_app_is_ready_when_every_service_answers(stack: Stack, monkeypatch):
    host_everything(stack)
    asked: list[str] = []
    monkeypatch.setattr(supervisor, "_http_ok", lambda url: asked.append(url) or True)

    stack.wait_for_apps(port_table())

    assert sorted(asked) == sorted(apps.health_urls(port_table()).values())


@pytest.mark.parametrize("name", sorted(apps.NEEDED_TO_OPEN))
def test_a_service_the_window_needs_that_never_answers_is_named(
    stack: Stack, monkeypatch, name: str
):
    host_everything(stack)
    silent = apps.health_urls(port_table())[name]
    monkeypatch.setattr(supervisor, "_http_ok", lambda url: url != silent)
    monkeypatch.setattr(supervisor, "APP_READY_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(supervisor, "OTHER_SERVICES_SECONDS", 0)

    with pytest.raises(StartupError, match=f"no answer from {name}"):
        stack.wait_for_apps(port_table())


def test_what_the_window_needs_is_the_api_its_database_and_the_pages():
    needed = apps.NEEDED_TO_OPEN
    assert needed == {
        "database-manager",
        "swap-credentials",
        "rest",
        "websocket",
        "frontend",
    }
    assert needed <= apps.health_urls(port_table()).keys()


@pytest.mark.parametrize("name", ["scheduler", "executor", "copilot-executor"])
def test_another_service_that_never_answers_holds_the_app_up_only_so_long(
    stack: Stack, monkeypatch, caplog, name: str
):
    """Where it answers is upstream's to change (a metrics port, a path).
    That must cost a wait and a warning, not the app."""
    host_everything(stack)
    port = port_table()
    silent = apps.health_urls(port)[name]
    monkeypatch.setattr(supervisor, "_http_ok", lambda url: url != silent)
    monkeypatch.setattr(supervisor, "OTHER_SERVICES_SECONDS", 2)
    started = time.monotonic()

    stack.wait_for_apps(port)

    assert 2 <= time.monotonic() - started < 15
    where = port[apps.service(name).port]
    assert f"no answer from {name} (at {where}) after 2s" in caplog.text
    assert "SERVICES in autogpt_desktop/apps.py" in caplog.text


def test_the_wait_for_the_other_services_still_ends_when_their_host_exits(
    stack: Stack, monkeypatch
):
    host_everything(stack, workers="import sys, time; time.sleep(1); sys.exit(1)")
    needed = [apps.health_urls(port_table())[name] for name in apps.NEEDED_TO_OPEN]
    monkeypatch.setattr(supervisor, "_http_ok", lambda url: url in needed)

    with pytest.raises(StartupError, match="workers exited while starting"):
        stack.wait_for_apps(port_table())


def test_a_host_that_exits_while_starting_ends_the_wait_at_once(
    stack: Stack, monkeypatch
):
    """Not after the five minutes the app is given to become ready."""
    host_everything(stack, workers="import sys; sys.exit(3)")
    monkeypatch.setattr(supervisor, "_http_ok", lambda url: False)
    started = time.monotonic()

    with pytest.raises(
        StartupError, match=r"workers exited while starting.*workers\.log"
    ):
        stack.wait_for_apps(port_table())
    assert time.monotonic() - started < 20


def isolating(stack: Stack, monkeypatch) -> list[tuple[str, ...]]:
    """Replace apps.backend_processes with stand-ins; returns what was asked."""
    asked: list[tuple[str, ...]] = []

    def backend_processes(
        bundle, data, env, groups, cache, copilot_env=None
    ) -> list[ManagedProcess]:
        asked.append(tuple(group.name for group in groups))
        return [process(stack, group.name) for group in groups]

    monkeypatch.setattr(apps, "backend_processes", backend_processes)
    return asked


def test_a_host_upstream_no_longer_fits_is_split_into_one_per_service(
    stack: Stack, monkeypatch
):
    """The app starts with more processes instead of not starting."""
    host_everything(stack, workers=f"import sys; sys.exit({servicehost.EXIT_CONTRACT})")
    asked = isolating(stack, monkeypatch)
    monkeypatch.setattr(supervisor, "_http_ok", lambda url: len(asked) == 1)
    workers = next(
        group.services for group in apps.layout(merged=True) if group.name == "workers"
    )

    stack.wait_for_apps(port_table())

    assert [set(names) for names in asked] == [set(workers)]
    running = {
        process.name for process in stack.processes if process.exit_code() is None
    }
    assert running == {
        "database-manager",
        "swap-credentials",
        "websocket",
        "api",
        "frontend",
        *workers,
    }
    assert "workers" not in stack.hosts
    assert all(stack.hosts[name].services == (name,) for name in workers)
    recorded = json.loads(stack.registry.path.read_text())
    assert {entry["name"] for entry in recorded} == running  # for the next boot's sweep


def test_the_same_split_happens_to_a_host_that_stops_fitting_while_running(
    stack: Stack, monkeypatch
):
    host_everything(
        stack,
        workers=f"import time, sys; time.sleep(1); sys.exit({servicehost.EXIT_CONTRACT})",
    )
    asked = isolating(stack, monkeypatch)
    watching = watch_in_background(stack)

    assert wait_until(
        lambda: asked
        == [
            (
                "scheduler",
                "batch-executor",
                "notification",
                "executor",
                "copilot-executor",
            )
        ],
        15,
        0.1,
    )
    stack.stop_requested.set()
    assert watching() == 0
    assert stack.restarts == {}  # a split is not a restart against the budget


def spawns_and_exits(pid_file: Path, code: int) -> str:
    """A program that starts a sleeper, writes its pid and exits with `code`."""
    return (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {SLEEP!r}])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        f"time.sleep(1); sys.exit({code})\n"
    )


def started_then_left(stack: Stack, name: str, pid_file: Path, code: int = 0):
    """`name`, exited, and the process it left running."""
    parent = process(stack, name, spawns_and_exits(pid_file, code))
    # The interpreter itself: a virtual environment's python.exe is a
    # launcher that starts it, and would be a dead process in the middle.
    parent.argv[0] = getattr(sys, "_base_executable", sys.executable)
    parent.start()
    assert wait_until(lambda: parent.exit_code() is not None, 15, 0.1)
    orphan = psutil.Process(int(pid_file.read_text()))
    assert orphan.is_running()
    return parent, orphan


def gone(orphan: psutil.Process) -> bool:
    try:  # killed but not yet collected by whoever inherited it counts
        return not orphan.is_running() or orphan.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_a_host_that_is_split_takes_what_it_started_with_it(
    stack: Stack, monkeypatch, tmp_path: Path
):
    """A worker that connected to the database started a query engine; the
    host then leaves with EXIT_CONTRACT and is in no tier any more, so
    nothing later would stop the engine."""
    stack.hosts = {group.name: group for group in apps.layout(merged=True)}
    workers, orphan = started_then_left(
        stack, "workers", tmp_path / "engine.pid", servicehost.EXIT_CONTRACT
    )
    stack.tiers.append([workers])
    isolating(stack, monkeypatch)

    assert stack.isolate(workers)

    assert wait_until(lambda: gone(orphan), 5, 0.1)
    assert workers not in stack.processes


def test_stopping_a_process_that_already_exited_stops_what_it_started(
    stack: Stack, tmp_path: Path
):
    """A host that left by itself and has not been restarted yet when the
    app quits, or that used up its restarts."""
    parent, orphan = started_then_left(
        stack, "database-manager", tmp_path / "engine.pid", 1
    )

    parent.stop()

    assert wait_until(lambda: gone(orphan), 5, 0.1)


@posix_only
def test_a_group_whose_leaders_id_went_to_another_process_is_left_alone(
    stack: Stack, monkeypatch
):
    """The id of a leader that exited is reused only once its group is
    empty; signalling the group then would hit a stranger's."""
    leader = process(stack, "workers", "pass")
    leader.start()
    assert wait_until(lambda: leader.exit_code() is not None, 15, 0.1)
    signalled: list[int] = []
    monkeypatch.setattr(
        process_module.os, "killpg", lambda pid, sig: signalled.append(pid)
    )

    monkeypatch.setattr(process_module, "_pid_exists", lambda pid: True)
    leader.kill()
    assert signalled == []

    monkeypatch.setattr(process_module, "_pid_exists", lambda pid: False)
    leader.kill()
    assert signalled == [leader.pid]


def test_only_a_service_host_can_say_the_backend_does_not_fit(stack: Stack):
    """78 is also sysexits' EX_CONFIG, which any program may exit with:
    RabbitMQ doing so is restarted like any other exit, not blamed on the
    backend."""
    stack.hosts = {"rest": apps.Group("rest", ("rest",))}
    flag = stack.data.root / "second-start"
    code = (
        "import os, sys, time\n"
        f"flag = {str(flag)!r}\n"
        "if os.path.exists(flag): time.sleep(60)\n"
        f"open(flag, 'w').close(); sys.exit({servicehost.EXIT_CONTRACT})\n"
    )
    stack.launch([process(stack, "rabbitmq", code)])
    watching = watch_in_background(stack)

    assert wait_until(flag.exists, 15, 0.1)
    assert wait_until(lambda: bool(stack.restarts.get("rabbitmq")), 15, 0.1)
    assert wait_until(lambda: stack.processes[0].exit_code() is None, 5, 0.1)
    stack.stop_requested.set()
    assert watching() == 0


def test_a_single_service_that_cannot_run_is_fatal_at_once_and_says_where(
    stack: Stack, capsys
):
    """Restarting it would fail the same way three more times."""
    stack.hosts = {"rest": apps.Group("rest", ("rest",))}
    stack.launch(
        [process(stack, "rest", f"import sys; sys.exit({servicehost.EXIT_CONTRACT})")]
    )
    assert wait_until(lambda: stack.processes[0].exit_code() is not None, 15, 0.1)

    assert stack.watch() == 1

    error = events_of(capsys)[-1]
    assert error["fatal"] is True
    assert "rest cannot run with this version of the backend" in error["message"]
    assert "rest.log" in error["message"]


def test_a_host_that_dies_is_restarted_with_its_preparation_run_again(
    stack: Stack, tmp_path: Path
):
    prepared: list[float] = []
    flag = tmp_path / "second-start"
    code = (
        "import os, sys, time\n"
        f"flag = {str(flag)!r}\n"
        "if os.path.exists(flag): time.sleep(60)\n"
        "open(flag, 'w').close(); sys.exit(1)\n"
    )
    stack.launch(
        [process(stack, "workers", code, before_start=lambda: prepared.append(1))]
    )
    watching = watch_in_background(stack)

    assert wait_until(
        lambda: len(prepared) == 2, 15, 0.1
    )  # the stale locks, cleared again
    assert wait_until(lambda: stack.processes[0].exit_code() is None, 5, 0.1)
    stack.stop_requested.set()
    assert watching() == 0


def watch_in_background(stack: Stack):
    import threading

    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(stack.watch()))
    thread.start()

    def join() -> int:
        thread.join(15)
        return result[0]

    return join


def test_a_preparation_that_fails_does_not_keep_a_process_from_starting(
    stack: Stack, caplog
):
    def prepare() -> None:
        raise ConnectionError("the cache is not answering")

    service = process(stack, "workers", before_start=prepare)
    service.start()
    try:
        assert service.exit_code() is None
        assert (
            "preparing to start workers failed: the cache is not answering"
            in caplog.text
        )
    finally:
        service.stop()


def catalog(stack: Stack, monkeypatch, code: str = "pass", version: str = "v1") -> None:
    stack.bundle.backend_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        apps, "entry_point_argv", lambda *args: [sys.executable, "-c", code]
    )
    monkeypatch.setattr(apps, "bundle_version", lambda bundle: version)


def tend_until_settled(stack: Stack) -> None:
    def settled() -> bool:
        stack.tend_skills_catalog()
        return stack.oneshot is not None and stack.publishing is None

    assert wait_until(settled, 20, 0.1)


def test_the_skills_catalog_waits_until_the_app_has_been_up_a_minute(
    stack: Stack, monkeypatch
):
    """It is another backend interpreter; not while the app is starting."""
    catalog(stack, monkeypatch)
    stack.ready_at = time.monotonic() - supervisor.SKILLS_CATALOG_DELAY_SECONDS + 30

    stack.tend_skills_catalog()
    assert stack.oneshot is None

    stack.ready_at = time.monotonic() - supervisor.SKILLS_CATALOG_DELAY_SECONDS
    stack.tend_skills_catalog()
    assert stack.oneshot is not None and stack.oneshot.pid


def test_a_published_catalog_is_not_published_again_by_the_same_bundle(
    stack: Stack, monkeypatch
):
    catalog(stack, monkeypatch)
    stack.ready_at = time.monotonic() - supervisor.SKILLS_CATALOG_DELAY_SECONDS
    tend_until_settled(stack)
    assert stack.published_catalog() == "v1"

    again = Stack(stack.bundle, stack.data)
    again.ready_at = stack.ready_at
    again.tend_skills_catalog()
    assert (
        again.oneshot is not None and again.oneshot.pid is None
    )  # nothing was started

    updated = Stack(stack.bundle, stack.data)
    updated.ready_at = stack.ready_at
    catalog(stack, monkeypatch, version="v2")
    try:
        tend_until_settled(updated)
        assert updated.published_catalog() == "v2"
    finally:
        updated.stop()


def test_a_publish_that_failed_is_tried_again_on_the_next_start(
    stack: Stack, monkeypatch, caplog
):
    catalog(stack, monkeypatch, code="import sys; sys.exit(1)")
    stack.ready_at = time.monotonic() - supervisor.SKILLS_CATALOG_DELAY_SECONDS

    tend_until_settled(stack)

    assert stack.published_catalog() is None
    assert "the skills catalog was not published (exit code 1)" in caplog.text
    started = stack.oneshot
    stack.tend_skills_catalog()
    assert stack.oneshot is started  # once per start


class NoProxy:
    def __init__(self, *args) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def test_database_manager_stops_after_the_services_that_write_through_it(
    stack: Stack, monkeypatch
):
    order: list[str] = []

    def backend_processes(
        bundle, data, env, groups, cache, copilot_env=None
    ) -> list[ManagedProcess]:
        return [
            process(
                stack,
                group.name,
                graceful_stop=lambda p: (order.append(p.name), p.kill()),
            )
            for group in groups
        ]

    monkeypatch.setattr(apps, "backend_processes", backend_processes)
    monkeypatch.setattr(
        apps, "frontend_process", lambda *args: process(stack, apps.FRONTEND)
    )
    monkeypatch.setattr(supervisor, "ProxyThread", NoProxy)
    monkeypatch.setattr(Stack, "wait_for_apps", lambda self, port: None)
    stack.env = {"AUTOGPT_PUBLIC_URL": "http://127.0.0.1:1"}
    secret = {"AUTOGPT_FRONTEND_DB_PASSWORD": "a", "REDIS_PASSWORD": "b"}

    stack.start_apps(port_table(), secret)
    stack.stop()

    assert set(order) == {
        "database-manager",
        "swap-credentials",
        "workers",
        "websocket",
        "api",
    }
    assert order[-1] == "database-manager"


@windows_only
def test_the_stop_event_reaches_a_process_that_was_given_its_name(tmp_path: Path):
    event = StopEvent.create(f"test-{time.monotonic_ns()}")
    assert event is not None
    waiter = (
        "import ctypes, sys\n"
        "kernel32 = ctypes.WinDLL('kernel32')\n"
        "kernel32.OpenEventW.restype = ctypes.c_void_p\n"
        "kernel32.OpenEventW.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)\n"
        "kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)\n"
        "handle = kernel32.OpenEventW(0x00100000, False, sys.argv[1])\n"
        "sys.exit(40 + kernel32.WaitForSingleObject(handle, 20000))\n"
    )
    child = subprocess.Popen([sys.executable, "-c", waiter, event.name])
    time.sleep(0.5)
    assert child.poll() is None

    event.set()
    assert child.wait(10) == 40  # WAIT_OBJECT_0: signalled, not timed out

    # Manual-reset: it stays set for a host that opens it late, until the
    # next start resets it.
    late = subprocess.Popen([sys.executable, "-c", waiter, event.name])
    assert late.wait(10) == 40
    event.reset()
    after_reset = subprocess.Popen([sys.executable, "-c", waiter, event.name])
    time.sleep(0.5)
    assert after_reset.poll() is None
    after_reset.kill()


def test_there_is_no_stop_event_off_windows():
    if sys.platform != "win32":
        assert StopEvent.create("workers") is None


@windows_only
def test_killing_a_process_on_windows_kills_what_it_started_even_after_it_is_gone(
    stack: Stack, tmp_path: Path
):
    """A database-manager that exited leaves its query engine holding its
    database connections; `kill` must find it without a living parent."""
    parent, orphan = started_then_left(
        stack, "database-manager", tmp_path / "child.pid"
    )
    started = time.monotonic()

    parent.kill()

    assert wait_until(lambda: not orphan.is_running(), 5, 0.1)
    assert time.monotonic() - started < 5  # one snapshot, not one per process


@windows_only
def test_stopping_a_process_on_windows_stops_its_whole_tree(
    stack: Stack, tmp_path: Path
):
    """Seen while the process is alive: a coding agent's CLI and what that
    started, however deep."""
    pid_file = tmp_path / "grandchild.pid"
    grandchild = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    spawn = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {!r}]); time.sleep(60)"
    parent = process(stack, "workers", spawn.format(spawn.format(grandchild)))
    parent.start()
    assert wait_until(lambda: pid_file.exists() and pid_file.read_text() != "", 20, 0.1)
    deepest = psutil.Process(int(pid_file.read_text()))

    parent.stop()

    assert wait_until(lambda: not deepest.is_running(), 5, 0.1)


@windows_only
def test_psutil_still_has_the_snapshot_the_windows_tree_kill_reads():
    assert callable(getattr(psutil, process_module.PPID_SNAPSHOT, None)), (
        f"psutil.{process_module.PPID_SNAPSHOT} is gone. process._windows_descendants reads "
        "every process's parent from it in one snapshot; without it the fallback takes a "
        "snapshot per process (18 s on a busy machine) and cannot find what an exited "
        "process had started. Find psutil's replacement and update _windows_descendants."
    )


@windows_only
def test_without_that_snapshot_the_tree_kill_says_so_once(monkeypatch, caplog):
    monkeypatch.setattr(process_module, "PPID_SNAPSHOT", "_a_name_psutil_never_had")
    monkeypatch.setattr(process_module, "_warned", set())
    child = subprocess.Popen([sys.executable, "-c", SLEEP])
    try:
        process_module._windows_descendants(child.pid, time.time())
        process_module._windows_descendants(child.pid, time.time())
    finally:
        child.kill()
    assert caplog.text.count("Update process._windows_descendants") == 1


def test_every_service_is_addressed_on_the_same_host():
    """A service reads its peers' address from <ITS OWN NAME>_HOST, and in a
    host of several services "its own name" is whichever of them set it
    last. That is harmless only while every one of them is the same."""
    addresses = settings._service_addresses(port_table())
    hosts = {name: value for name, value in addresses.items() if name.endswith("_HOST")}
    assert len(hosts) > 1
    assert set(hosts.values()) == {addresses["PYRO_HOST"]}, (
        "The *_HOST values in settings._service_addresses differ. Services that share a "
        "process (apps.MERGED) would then reach each other at whichever address the last "
        "of them to start named; give such a service a host of its own first."
    )


@posix_only
def test_the_open_file_limit_is_raised_for_the_services(monkeypatch):
    import resource

    limits = {"now": (256, resource.RLIM_INFINITY)}
    monkeypatch.setattr(resource, "getrlimit", lambda what: limits["now"])
    monkeypatch.setattr(resource, "setrlimit", lambda what, new: limits.update(now=new))

    process_module.raise_file_limit()
    assert limits["now"] == (process_module.FILE_LIMIT, resource.RLIM_INFINITY)

    limits["now"] = (256, 4096)  # never past the hard limit
    process_module.raise_file_limit()
    assert limits["now"] == (4096, 4096)

    limits["now"] = (65536, 65536)  # never lowered
    process_module.raise_file_limit()
    assert limits["now"] == (65536, 65536)


def test_a_limit_that_cannot_be_raised_does_not_stop_the_app(monkeypatch, caplog):
    if sys.platform == "win32":
        process_module.raise_file_limit()  # nothing to do there
        return
    import resource

    def refuse(what, new):
        raise ValueError("not allowed to raise maximum limit")

    monkeypatch.setattr(resource, "getrlimit", lambda what: (256, 1024))
    monkeypatch.setattr(resource, "setrlimit", refuse)
    process_module.raise_file_limit()
    assert "could not raise the open-file limit from 256 to 1024" in caplog.text
