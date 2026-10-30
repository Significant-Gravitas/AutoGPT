"""The service host, run as the runtime runs it (apps.host_process and
ManagedProcess), against a stand-in backend: tests/fake_bundle.

The stand-in has only what the host depends on, so these run anywhere in
seconds. That the real backend still has that shape is
test_backend_contract.py's job.
"""

import json
import socket
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

import pytest

from autogpt_desktop import apps, servicehost
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ManagedProcess, wait_until

FAKE_BUNDLE = Path(__file__).parent / "fake_bundle"
WINDOWS = sys.platform == "win32"
# What a process that was killed rather than asked reports.
KILLED = 1 if WINDOWS else -9


class NoCache:
    def __init__(self) -> None:
        self.cleared: list[tuple[str, ...]] = []

    def clear_stale_locks(self, services) -> None:
        self.cleared.append(tuple(services))


class Hosting:
    """Starts hosts of stand-in services and looks at what they left."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.data = DataDir(tmp_path / "data")
        self.data.prepare()
        self.marks = tmp_path / "marks"
        self.marks.mkdir()
        self.cache = NoCache()
        self.ports = {"a": free_port(), "b": free_port()}
        self.monkeypatch = monkeypatch
        self.started: list[ManagedProcess] = []

    def start(
        self,
        *names: str,
        database: tuple[str, ...] = (),
        more: dict[str, str] | None = None,
    ) -> ManagedProcess:
        services = tuple(
            apps.Service(
                name,
                f"backend.{'slow:main' if name == 'slow' else f'services:{name}'}",
                "port",
                "/",
                database=name in database,
            )
            for name in names
        )
        self.monkeypatch.setattr(apps, "SERVICES", services)
        self.monkeypatch.setattr(apps, "API_SERVERS", ("web_a", "web_b"))
        env = {
            "FAKE_MARKS": str(self.marks),
            "FAKE_PORT_A": str(self.ports["a"]),
            "FAKE_PORT_B": str(self.ports["b"]),
            "PYTHONDONTWRITEBYTECODE": "1",
            **(more or {}),
        }
        process = apps.host_process(
            Bundle(FAKE_BUNDLE), self.data, env, apps.Group("host", names), self.cache
        )
        process.start()
        self.started.append(process)
        return process

    def wait_for(self, *marks: str, timeout: float = 20) -> None:
        present = wait_until(
            lambda: all(self.marked(mark) for mark in marks), timeout, 0.05
        )
        assert present, f"never saw {marks}; the host said:\n{self.log()}"

    def marked(self, mark: str) -> bool:
        return (self.marks / mark).exists()

    def log(self) -> str:
        return (self.data.logs / "host.log").read_text(
            encoding="utf-8", errors="replace"
        )

    def get(self, port: str) -> str:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.ports[port]}/", timeout=5
        ) as reply:
            return reply.read().decode()

    def answers(self, port: str) -> bool:
        try:
            return bool(self.get(port))
        except OSError:
            return False


@pytest.fixture
def hosting(tmp_path: Path, monkeypatch):
    hosting = Hosting(tmp_path, monkeypatch)
    yield hosting
    for process in hosting.started:
        process.stop()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def exit_code(process: ManagedProcess, timeout: float = 20) -> int | None:
    wait_until(lambda: process.exit_code() is not None, timeout, 0.05)
    return process.exit_code()


def test_services_share_a_process_each_in_a_thread_of_its_own(hosting: Hosting):
    hosting.start("one", "two")
    hosting.wait_for("one.started", "two.started")

    assert (hosting.marks / "one.started").read_text() == "one"
    assert (hosting.marks / "two.started").read_text() == "two"
    assert "started one (One), two (Two)" in hosting.log()


def test_a_stop_runs_every_cleanup_and_the_host_exits_by_itself(hosting: Hosting):
    """On Windows too, where the request is an event and not a signal."""
    process = hosting.start("one", "two", "forever")
    hosting.wait_for("one.started", "two.started", "forever.started")
    started = time.monotonic()

    process.stop()

    assert process.exit_code() == 0, hosting.log()  # it left; it was not killed
    assert time.monotonic() - started < apps.STOP_TIMEOUT_SECONDS
    for mark in ("one.cleaned", "two.cleaned", "forever.cleaned"):
        assert hosting.marked(mark)
    # Services whose run() returns once cleaned up got to finish in their
    # own words; the one that never returns did not hold the stop up.
    assert hosting.marked("one.terminated") and hosting.marked("two.terminated")
    assert "one (One) cleaned up in" in hosting.log()
    assert "stopped with code 0" in hosting.log()


def test_a_cleanup_that_hangs_is_given_the_budget_and_no_more(hosting: Hosting):
    process = hosting.start("stubborn", "one")
    hosting.wait_for("stubborn.started", "one.started")
    started = time.monotonic()

    process.stop()

    assert process.exit_code() == 0, hosting.log()
    assert time.monotonic() - started < apps.STOP_TIMEOUT_SECONDS
    assert hosting.marked("stubborn.cleaning") and hosting.marked("one.cleaned")
    assert "1 cleanup(s) still running after 2.5s" in hosting.log()


def test_a_cleanup_that_fails_does_not_keep_the_others_from_theirs(hosting: Hosting):
    process = hosting.start("careless", "one")
    hosting.wait_for("careless.started", "one.started")

    process.stop()

    assert process.exit_code() == 0
    assert hosting.marked("one.cleaned")
    assert "careless (Careless) cleanup failed: RuntimeError" in hosting.log()


def test_a_service_that_ends_takes_its_host_down_and_the_log_names_it(hosting: Hosting):
    """The supervisor restarts the process; a host that carried on would be
    an app with a service missing and nothing to say so."""
    process = hosting.start("quitter", "one")
    hosting.wait_for("quitter.started", "one.started")

    assert exit_code(process) == servicehost.EXIT_SERVICE_ENDED
    assert "stopping: quitter (Quitter) ended" in hosting.log()
    assert hosting.marked("one.cleaned")


def test_a_stop_during_the_imports_does_not_wait_for_them(hosting: Hosting):
    process = hosting.start("slow")
    hosting.wait_for("slow.importing")
    started = time.monotonic()

    process.stop()

    assert process.exit_code() == 0, hosting.log()
    assert time.monotonic() - started < apps.STOP_TIMEOUT_SECONDS


def test_the_hosts_arguments_are_not_left_for_the_backend_to_parse(hosting: Hosting):
    hosting.start("one")
    hosting.wait_for("one.started", "one.argv")
    assert (hosting.marks / "one.argv").read_text() == "-c"


def test_locks_are_cleared_and_the_stop_request_reset_before_every_start(
    hosting: Hosting,
):
    process = hosting.start("one")
    hosting.wait_for("one.started")
    process.stop()
    assert process.exit_code() == 0
    (hosting.marks / "one.started").unlink()

    process.start()  # as the supervisor restarts it

    hosting.wait_for("one.started")
    assert process.exit_code() is None, "the restart saw the previous stop request"
    assert hosting.cache.cleared == [("one",), ("one",)]


def test_several_services_that_stop_fitting_ask_for_a_process_each(hosting: Hosting):
    process = hosting.start("one", "hands_nothing_over")

    assert exit_code(process) == servicehost.EXIT_CONTRACT
    assert hosting.marked("nothing.called")
    assert not hosting.marked("one.started")
    assert "handed over 0 services through backend.app.run_processes" in hosting.log()
    assert "tests/backend_contract.py" in hosting.log()


def test_one_service_that_stops_fitting_runs_the_way_upstream_would(hosting: Hosting):
    """`cleanup` was renamed: the host cannot stop the service properly, but
    upstream's own run_processes can still run it, in the foreground."""
    process = hosting.start("renamed")
    hosting.wait_for("renamed.started")

    assert (hosting.marks / "renamed.started").read_text() == "MainThread"
    assert "has no cleanup()" in hosting.log()
    assert "running the service in the foreground" in hosting.log()
    process.stop()
    # Stopped as every service was before the host: without its cleanup. On
    # Windows at once; elsewhere the signal is upstream's to handle, and the
    # stand-in ignores it.
    assert process.exit_code() in (0, KILLED)
    assert not hosting.marked("renamed.cleaned")


def test_the_same_change_in_a_shared_host_asks_for_a_process_each(hosting: Hosting):
    process = hosting.start("one", "renamed")
    assert exit_code(process) == servicehost.EXIT_CONTRACT
    assert not hosting.marked("one.started")


def test_an_entry_point_that_is_gone_says_so(hosting: Hosting):
    process = hosting.start("renamed_upstream")
    assert exit_code(process) == servicehost.EXIT_CONTRACT
    assert "the entry point backend.services:renamed_upstream is gone" in hosting.log()


def test_a_worker_that_connects_to_the_database_itself_is_caught(hosting: Hosting):
    """The backend decides per process whether a query goes to the database
    or to database-manager; a service that connects would flip the others."""
    process = hosting.start("connector", "one")

    assert exit_code(process) == servicehost.EXIT_CONTRACT
    assert "connected to the database itself" in hosting.log()


def test_a_host_with_a_database_service_may_connect(hosting: Hosting):
    process = hosting.start("connector", "one", database=("connector",))
    hosting.wait_for("connector.started", "one.started")
    time.sleep(1.5)  # two turns of the watchdog
    assert process.exit_code() is None


def test_one_service_alone_may_connect(hosting: Hosting):
    process = hosting.start("connector")
    hosting.wait_for("connector.started")
    time.sleep(1.5)
    assert process.exit_code() is None


def test_two_servers_share_one_event_loop(hosting: Hosting):
    pytest.importorskip("uvicorn")
    process = hosting.start("web_a", "web_b")
    assert wait_until(
        lambda: hosting.answers("a") and hosting.answers("b"), 30, 0.1
    ), hosting.log()

    assert hosting.get("a") == hosting.get("b")  # the id of the loop each ran on

    process.stop()
    assert process.exit_code() == 0, hosting.log()
    # uvicorn was told to exit, so each app's shutdown ran (the real ones
    # disconnect from the database there).
    assert hosting.marked("a.lifespan-shutdown") and hosting.marked(
        "b.lifespan-shutdown"
    )
    assert hosting.marked("a.terminated") and hosting.marked("b.terminated")


def test_a_server_that_cannot_have_its_port_ends_the_host_at_once(hosting: Hosting):
    """As uvicorn.run would have ended the process: the supervisor's wait
    for the app must fail now, not when its five minutes run out."""
    pytest.importorskip("uvicorn")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", hosting.ports["a"]))
        squatter.listen()
        process = hosting.start("web_a", "web_b")
        assert exit_code(process, 30) == servicehost.EXIT_SERVICE_ENDED
    assert "web_a (A) ended" in hosting.log()


def host(*names: str, shared_loop: bool = False) -> servicehost.Host:
    return servicehost.Host(
        name="test",
        entries=[(name, f"module:{name}") for name in names],
        shared_loop=shared_loop,
    )


def test_a_server_that_never_goes_through_uvicorn_run_is_noticed():
    """Upstream serving some other way would put the two API servers on a
    loop each, where the second never finishes starting. That must not be a
    silent hang."""
    api = host("websocket", "rest", shared_loop=True)
    long_ago = time.monotonic() - servicehost.SERVE_DEADLINE_SECONDS - 1
    api.services = [
        servicehost.Hosted("websocket", object(), started=long_ago),
        servicehost.Hosted("rest", object(), started=long_ago),
    ]
    api.served = {"websocket"}

    assert api.not_serving() == "rest (?)"

    api.services[1].started = time.monotonic()
    assert api.not_serving() is None
    assert (
        host("rest", shared_loop=True).not_serving() is None
    )  # alone: nothing to share


class Left(Exception):
    pass


def leave(code: int):
    raise Left(code)


def test_an_entry_point_that_serves_instead_of_handing_over_is_noticed(
    monkeypatch, capsys
):
    monkeypatch.setattr(servicehost, "leave", leave)

    with pytest.raises(Left) as left:
        host("executor", "scheduler").entry_point_is_serving("executor")
    assert left.value.args == (servicehost.EXIT_CONTRACT,)

    alone = host("executor")
    alone.entry_point_is_serving("executor")  # alone, that is the old way: let it
    assert alone.foreground
    assert "runs its service itself" in capsys.readouterr().err


def test_an_entry_point_that_was_only_slow_keeps_its_graceful_stop(monkeypatch, capsys):
    """On a machine that is paging, building a service can outlast the wait
    that tells handing over from serving."""
    monkeypatch.setattr(servicehost, "ENTRY_CALL_SECONDS", 0.2)
    alone = host("executor")

    alone.call_entry_point("executor", lambda: time.sleep(1))

    assert "runs its service itself" in capsys.readouterr().err
    assert not alone.foreground  # or shut_down would skip every cleanup


class Service:
    def __init__(self, block: threading.Event) -> None:
        self.block = block
        self.started = False

    def start(self, background: bool) -> None:
        self.started = True
        self.block.wait(10)


def test_no_service_is_started_once_a_stop_was_asked_for(monkeypatch):
    """A stop during start-up cleans up what is running; a service started
    after that would get no cleanup, and would import alongside the others
    without the wait that keeps their imports apart."""
    monkeypatch.setattr(servicehost, "IMPORTS_QUIET_SECONDS", 0.2)
    block = threading.Event()
    workers = host("executor", "scheduler", "notification")
    workers.services = [
        servicehost.Hosted(name, Service(block)) for name, _ in workers.entries
    ]
    try:
        workers.stop.set()
        workers.start_services()
        assert [hosted.instance.started for hosted in workers.services] == [False] * 3

        workers.stop.clear()
        workers.start_services()
        assert wait_until(
            lambda: all(h.instance.started for h in workers.services), 5, 0.05
        )
    finally:
        block.set()


def uvicorn_whose_server(monkeypatch, serve) -> list[tuple]:
    """A stand-in uvicorn; returns the calls its own `run` received."""
    calls: list[tuple] = []

    class Config:
        def __init__(self, app, **options) -> None:
            self.app = app

    class Server:
        def __init__(self, config) -> None:
            self.started = False

    Server.serve = serve
    module = types.ModuleType("uvicorn")
    module.__dict__.update(
        Config=Config, Server=Server, run=lambda app, **o: calls.append((app, o))
    )
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return calls


async def serve_takes_something_new(server, sockets):
    """As if Server.serve had gained a required argument."""


def test_one_server_that_uvicorn_no_longer_lets_share_a_loop_is_served_upstreams_way(
    monkeypatch, capsys
):
    """The escape hatch (a host per service) must not depend on uvicorn's
    insides: there, upstream's own uvicorn.run does."""
    calls = uvicorn_whose_server(monkeypatch, serve_takes_something_new)
    rest = host("rest", shared_loop=True)
    rest.serve_on_one_loop()
    try:
        sys.modules["uvicorn"].run("the app", port=8006)
    finally:
        rest._loop.call_soon_threadsafe(rest._loop.stop)

    assert calls == [("the app", {"port": 8006})]
    assert rest.servers == []
    said = capsys.readouterr().err
    assert "uvicorn could not serve on the host's event loop (TypeError" in said
    assert "update serve_on_one_loop" in said


def test_two_servers_that_uvicorn_no_longer_lets_share_a_loop_ask_for_a_process_each(
    monkeypatch,
):
    calls = uvicorn_whose_server(monkeypatch, serve_takes_something_new)
    api = host("websocket", "rest", shared_loop=True)
    left: list[tuple] = []

    def shut_down(code: int, reason: str):
        left.append((code, reason))
        raise Left(code)

    monkeypatch.setattr(api, "shut_down", shut_down)
    api.serve_on_one_loop()
    try:
        with pytest.raises(Left):
            sys.modules["uvicorn"].run("the app", port=8006)
    finally:
        api._loop.call_soon_threadsafe(api._loop.stop)

    assert calls == []
    assert left[0][0] == servicehost.EXIT_CONTRACT
    assert "serve_on_one_loop" in left[0][1]


def test_a_server_that_crashes_after_serving_is_not_taken_for_a_misfit(monkeypatch):
    """That is a service ending: the host exits and is restarted."""

    async def serve(server):
        server.started = True
        raise RuntimeError("the server crashed")

    calls = uvicorn_whose_server(monkeypatch, serve)
    rest = host("rest", shared_loop=True)
    rest.serve_on_one_loop()
    try:
        with pytest.raises(RuntimeError, match="the server crashed"):
            sys.modules["uvicorn"].run("the app")
    finally:
        rest._loop.call_soon_threadsafe(rest._loop.stop)
    assert calls == []


def test_signal_handlers_can_only_be_installed_from_the_main_thread_still():
    """What `make_signals_thread_aware` exists for. If CPython ever allows
    it, the shim can go."""
    import signal
    import threading

    raised: list[BaseException] = []

    def install() -> None:
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except ValueError as exc:
            raised.append(exc)

    thread = threading.Thread(target=install)
    thread.start()
    thread.join()
    assert raised


# --- chats left busy by AutoPilot turns that were ended without the executor --


def ended_turns(hosting: Hosting, *sessions: str) -> tuple[Path, dict[str, str]]:
    """The list the runtime leaves (runs.ENDED_TURNS_FILE), and what tells
    the host of `one` to close what is on it."""
    listed = hosting.data.run / "ended-turns.json"
    listed.write_text(
        json.dumps(
            [
                {"session_id": session, "turn_id": f"{session}-turn"}
                for session in sessions
            ]
        ),
        encoding="utf-8",
    )
    return listed, {servicehost.ENDED_TURNS_ENV: f"one={listed}"}


def test_the_host_of_the_named_service_closes_the_chats_and_removes_the_list(
    hosting: Hosting,
):
    listed, told = ended_turns(hosting, "chat-a", "chat-b")

    hosting.start("one", "two", more=told)
    hosting.wait_for("closed-chat-a", "closed-chat-b")

    # As upstream's own Stop does when the executor does not confirm: marked
    # cancelled, and no error shown for something the user asked for.
    assert (
        hosting.marks / "closed-chat-a"
    ).read_text() == "Operation cancelled|True|chat-a-turn"
    assert wait_until(lambda: not listed.exists(), 10, 0.05)
    assert "closed the chat(s) of 2 AutoPilot turn(s)" in hosting.log()
    # Not handed on to what the services start.
    assert (
        servicehost.ENDED_TURNS_ENV
        not in (hosting.marks / "one.environment").read_text()
    )


def test_a_host_without_that_service_leaves_the_list_alone(hosting: Hosting):
    listed, told = ended_turns(hosting, "chat-a")

    hosting.start("two", more=told)
    hosting.wait_for("two.started")
    time.sleep(1)

    assert listed.exists() and not hosting.marked("closed-chat-a")


def test_closing_is_tried_again_while_it_fails(hosting: Hosting):
    listed, told = ended_turns(hosting, "chat-a")
    (hosting.marks / "closing-refused").write_text("1")

    hosting.start("one", more=told)
    hosting.wait_for(
        "closed-chat-a", timeout=servicehost.ENDED_TURNS_RETRY_SECONDS + 30
    )

    assert wait_until(lambda: not listed.exists(), 10, 0.05)
    assert (
        hosting.log().count(
            "closing the chats of stopped turns failed: ConnectionError"
        )
        == 1
    )


def test_an_upstream_that_closes_chats_some_other_way_costs_only_that(hosting: Hosting):
    """The list stays for a version that can; the services run."""
    listed, told = ended_turns(hosting, "chat-a")
    listed.write_text("not a list", encoding="utf-8")

    process = hosting.start("one", more=told)
    hosting.wait_for("one.started")

    assert wait_until(
        lambda: "cannot close the chats of turns that were stopped" in hosting.log(),
        10,
        0.05,
    )
    assert listed.exists() and process.exit_code() is None


def test_retrying_chat_closure_cannot_cancel_a_later_turn(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    listed = tmp_path / "ended-turns.json"
    listed.write_text(
        json.dumps(
            [
                {"session_id": "a", "turn_id": "old-a"},
                {"session_id": "b", "turn_id": "old-b"},
            ]
        )
    )
    calls = []
    current = {"a": "old-a", "b": "old-b"}
    refused = False

    async def close(session_id, *, turn_id, **kwargs):
        nonlocal refused
        calls.append((session_id, turn_id))
        if session_id == "b" and not refused:
            refused = True
            raise ConnectionError("temporary failure")
        if current[session_id] != turn_id:
            return False
        current[session_id] = "new-a" if session_id == "a" else "completed"
        return True

    registry = SimpleNamespace(
        mark_session_completed=close, CANCELLED_MESSAGE="Operation cancelled"
    )
    import_module = servicehost.importlib.import_module
    monkeypatch.setattr(
        servicehost.importlib,
        "import_module",
        lambda name: (
            registry
            if name == "backend.copilot.stream_registry"
            else import_module(name)
        ),
    )
    host = SimpleNamespace(
        say=lambda message: None, stop=SimpleNamespace(wait=lambda seconds: False)
    )

    servicehost.Host.close_ended_turns(host, str(listed))

    assert calls == [("a", "old-a"), ("b", "old-b"), ("a", "old-a"), ("b", "old-b")]
    assert current["a"] == "new-a"
    assert not listed.exists()


def test_legacy_chat_closure_without_turn_identity_is_not_applied_to_new_work(
    hosting: Hosting,
):
    listed, told = ended_turns(hosting, "chat-a")
    listed.write_text(json.dumps(["chat-a"]))

    hosting.start("one", more=told)
    hosting.wait_for("one.started")

    assert wait_until(lambda: "require manual recovery" in hosting.log(), 10, 0.05)
    assert listed.exists() and not hosting.marked("closed-chat-a")
