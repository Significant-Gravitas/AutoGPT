"""One process that hosts backend services, and stops them properly.

    python -c "<apps.HOST_CODE>" <package parent> --name workers \
        executor=backend.exec:main scheduler=backend.scheduler:main ...

Every backend service process is one of these, whether it hosts five
services or one. Each upstream entry point is `run_processes(SomeService())`,
so putting a collector in place of `backend.app.run_processes` before the
entry modules are imported hands over upstream's own service objects, built
with upstream's own arguments, without starting them. Each then runs in a
thread of its own, exactly as it would in a process of its own:
`service.start(background=False)`. The main thread stays the host's.

While the app starts, the supervisor starts the hosts before the databases
are up, so that the slow part (importing the backend) overlaps theirs. It
names a file in GO_FILE_ENV; a host that is given one imports its services'
modules and then waits for the file to exist. Only then does it call an
entry point: one that hands its service over costs milliseconds to call (the
five of the largest host took 4 ms together, 2026-10-03), and one that has
come to run its service itself (the case `entry_point_is_serving` is for)
starts it right there, which must not be before the databases answer and
the migrations are in.

A stop is asked for with SIGTERM, or on Windows (where a signal cannot reach
a console-less process) by setting a named event the supervisor created. The
host then calls every service's `cleanup()` at once, waits for them just
short of the time the supervisor allows, and exits. Windows services never
ran their cleanup before.

With `--shared-loop`, `uvicorn.run` serves every app on one event loop: the
backend's database client belongs to the loop that connected it, so the two
API servers can share an interpreter only if they share a loop.

The host of the copilot executor has one more job, once its services are
started: an AutoPilot turn that was ended without its executor (the user
chose to stop it as the app closed, and it did not stop: runs.py) left its
chat marked as busy, and a busy chat takes no new message. Closing it is
upstream's own function's work, the one its Stop button falls back on, and
this is the process that has it: `close_ended_turns`.

Everything this module needs from upstream:

    backend.app.run_processes            the seam the services are taken from
    AppProcess.start(background=False)   runs a service in the calling thread
    AppProcess.cleanup()                 its shutdown
    AppProcess.service_name              for the log
    backend.data.db.is_connected()       the watchdog below
    uvicorn.run                          what the two API servers call
    backend.copilot.stream_registry.mark_session_completed, CANCELLED_MESSAGE
                                         closing the chat of an ended turn

tests/backend_contract.py checks each against the real backend. When one is
broken here, a host of several services exits with EXIT_CONTRACT and the
supervisor starts each service in a host of its own; a host of one service
runs it the way upstream would have, in the foreground.

Only the standard library is imported until the services are: this module is
also imported by the supervisor and by tests that have no backend.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

EXIT_SERVICE_ENDED = 1
# sysexits.h EX_CONFIG: upstream no longer fits how this host runs services.
EXIT_CONTRACT = 78
# uvicorn's own code for a server that could not start.
EXIT_SERVER_DID_NOT_START = 3

STOP_EVENT_ENV = "AUTOGPT_DESKTOP_STOP_EVENT"
GO_FILE_ENV = "AUTOGPT_DESKTOP_GO_FILE"
GO_POLL_SECONDS = 0.05
# "<service>=<file>": the host of that service closes the chats listed in
# the file (session ids, a JSON list) and removes it.

ENDED_TURNS_ENV = "AUTOGPT_DESKTOP_ENDED_TURNS"
# Closing a chat writes through database-manager, which starts when this
# host does and may answer later than this host asks.

ENDED_TURNS_TRIES = 6
ENDED_TURNS_RETRY_SECONDS = 10.0

# An entry point builds its service and hands it over; importing is the slow
# part and happens before. One that has not returned by now is running its
# service in the foreground itself.
ENTRY_CALL_SECONDS = 60
# A service calls uvicorn.run within moments of starting.
SERVE_DEADLINE_SECONDS = 90
IMPORTS_QUIET_SECONDS = 1.0
IMPORTS_SETTLE_LIMIT_SECONDS = 30.0
# After cleanup, for services whose run() returns to log their own goodbye.
GOODBYE_SECONDS = 0.5


class ContractError(RuntimeError):
    """Upstream changed something the host depends on."""


@dataclass
class Hosted:
    name: str
    instance: Any
    thread: threading.Thread | None = None
    started: float = 0.0

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    @property
    def label(self) -> str:
        return f"{self.name} ({getattr(self.instance, 'service_name', '?')})"


@dataclass
class Host:
    name: str
    entries: list[tuple[str, str]]
    stop_budget: float = 2.5
    shared_loop: bool = False
    no_database: bool = False
    services: list[Hosted] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)
    signalled: bool = False
    servers: list[Any] = field(default_factory=list)
    served: set[str] = field(default_factory=set)
    foreground: bool = False
    # Whether the collector was put in place of backend.app.run_processes.
    seam: bool = False
    # Where the supervisor says that services may start; None: at once.
    go_file: str | None = None
    # ENDED_TURNS_ENV's value, when it was given.

    ended_turns: str | None = None
    _ending: threading.Lock = field(default_factory=threading.Lock)
    _loop: Any = None
    _loop_thread: threading.Thread | None = None
    _loop_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def merged(self) -> bool:
        return len(self.entries) > 1

    def run(self) -> NoReturn:
        self.listen_for_stop()
        self.go_file = os.environ.pop(
            GO_FILE_ENV, None
        )  # not for the services' children
        self.ended_turns = os.environ.pop(ENDED_TURNS_ENV, self.ended_turns)
        try:
            self.collect()
            if self.shared_loop:
                self.serve_on_one_loop()
        except ContractError as exc:
            self.fall_back(exc)
        make_signals_thread_aware()
        self.start_services()
        self.tend_ended_turns()
        self.supervise()

    def wait_for_the_go(self) -> None:
        """Nothing a service connects to may be up yet. The supervisor writes
        the file when it is; a host it restarts later finds it there. Made
        before anything of upstream's is called, whichever way the services
        are then run."""
        if not self.go_file or os.path.exists(self.go_file):
            return
        self.say("loaded; waiting for the databases before starting any service")
        while not (os.path.exists(self.go_file) or self.stop.is_set()):
            time.sleep(GO_POLL_SECONDS)
        if self.stop.is_set():
            park()  # the thread that saw the request ends the process

    # --- taking the services from upstream --------------------------------

    def collect(self) -> None:
        try:
            app: Any = importlib.import_module("backend.app")
            run_processes = app.run_processes
        except (ImportError, AttributeError) as exc:
            raise ContractError(f"backend.app.run_processes is gone ({exc})") from exc
        handed_over: list[Any] = []
        # Before any entry module is imported: each binds the name on import.
        app.run_processes = lambda *processes, **_: handed_over.extend(processes)
        self.seam = True
        try:
            # Imported first, all of them: the slow part, and all that is
            # done before the go.
            functions = [entry_point(entry) for _, entry in self.entries]
            self.wait_for_the_go()
            for (name, entry), function in zip(self.entries, functions, strict=True):
                taken = len(handed_over)
                self.call_entry_point(name, function)
                self.services += [
                    Hosted(name, service) for service in handed_over[taken:]
                ]
                if len(handed_over) != taken + 1:
                    raise ContractError(
                        f"{entry} handed over {len(handed_over) - taken} services through "
                        "backend.app.run_processes; one was expected"
                    )
                check_service(entry, handed_over[-1])
        finally:
            app.run_processes = run_processes

    def call_entry_point(self, name: str, function: Callable[[], object]) -> None:
        timer = threading.Timer(
            ENTRY_CALL_SECONDS, self.entry_point_is_serving, (name,)
        )
        timer.daemon = True
        timer.start()
        try:
            function()
        finally:
            timer.cancel()
            timer.join()
        if self.foreground:  # slow (a machine that is paging), not serving
            self.foreground = False
            self.say(f"{name}'s entry point handed its service over after all")

    def entry_point_is_serving(self, name: str) -> None:
        problem = (
            f"{name}'s entry point runs its service itself instead of handing it over"
        )
        if self.merged:
            self.say(f"{problem}; the services need a process each")
            leave(EXIT_CONTRACT)
        self.foreground = True
        self.say(
            f"{problem}; it runs as it did before the host, and stops without cleanup"
        )

    def fall_back(self, problem: ContractError) -> NoReturn:
        """Upstream no longer fits. Several services cannot run the old way in
        one process, so the supervisor is told to give each its own; one
        service is run the way upstream's own entry point would have."""
        self.say(f"upstream changed: {problem}")
        self.say(
            "update autogpt_desktop/servicehost.py; tests/backend_contract.py says what to"
        )
        if self.merged or (self.seam and not self.services):
            leave(EXIT_CONTRACT)
        self.wait_for_the_go()
        self.say("running the service in the foreground, as its entry point would")
        self.foreground = True
        if sys.platform != "win32":  # upstream installs its own handlers
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            if self.services:
                run_processes = importlib.import_module("backend.app").run_processes
                run_processes(*[hosted.instance for hosted in self.services])
            else:  # nothing was put in upstream's way: the entry point is as written
                entry_point(self.entries[0][1])()
        except ContractError as exc:
            self.say(f"{exc}")
            leave(EXIT_CONTRACT)
        leave(EXIT_SERVICE_ENDED)

    # --- one loop for every uvicorn server --------------------------------

    def serve_on_one_loop(self) -> None:
        """Beyond `uvicorn.run` being what is called, this leans on uvicorn's
        own shape: Config(app, **options), Server(config), Server.serve(),
        Server.started and Server.should_exit. `without_the_shared_loop` is
        what happens when one of them is no longer there."""
        try:
            uvicorn: Any = importlib.import_module("uvicorn")
            config_class, server_class = uvicorn.Config, uvicorn.Server
            upstream_run = uvicorn.run
        except (ImportError, AttributeError) as exc:
            raise ContractError(f"uvicorn.run is gone ({exc})") from exc
        import asyncio

        async def serve(server: Any) -> None:
            try:
                await server.serve()
            except SystemExit:
                # uvicorn exits when it cannot have its port. Let through,
                # that would end the loop under every other server on it,
                # with nothing left to deliver this result.
                return

        def run(app: Any, **options: Any) -> None:
            self.served.add(threading.current_thread().name)
            server: Any = None
            try:
                config = config_class(app, **options)
                server = server_class(config)
                loop = self.shared_event_loop(config)
                self.servers.append(server)
                asyncio.run_coroutine_threadsafe(serve(server), loop).result()
                started = server.started
            except Exception as exc:
                if getattr(server, "started", False):
                    raise  # it was serving: a crash, not a misfit
                self.without_the_shared_loop(exc)
                return upstream_run(app, **options)
            if not started:  # as uvicorn.run ends: the service did not start
                raise SystemExit(EXIT_SERVER_DID_NOT_START)

        uvicorn.run = run

    def without_the_shared_loop(self, problem: Exception) -> None:
        """uvicorn no longer fits `serve_on_one_loop`. Two servers cannot do
        without it; one can, and the caller serves it as upstream wrote it."""
        what = (
            f"uvicorn could not serve on the host's event loop "
            f"({type(problem).__name__}: {problem}); update serve_on_one_loop in "
            "autogpt_desktop/servicehost.py"
        )
        if self.merged:
            self.shut_down(EXIT_CONTRACT, what)
        self.say(f"{what}. Serving through uvicorn.run itself, as upstream does")
        self.servers.clear()  # nothing here can ask that server to exit

    def shared_event_loop(self, config: Any) -> Any:
        """The first server's choice of loop (uvloop where uvicorn finds it)."""
        import asyncio

        with self._loop_lock:
            if self._loop is None:
                factory = getattr(config, "get_loop_factory", lambda: None)()
                self._loop = factory() if factory else asyncio.new_event_loop()
                self._loop_thread = threading.Thread(
                    target=self._loop.run_forever, name="shared-loop", daemon=True
                )
                self._loop_thread.start()
            return self._loop

    # --- running ----------------------------------------------------------

    def start_services(self) -> None:
        for hosted in self.services:
            if self.stop.is_set():  # the ones not started have nothing to clean up
                return
            hosted.thread = threading.Thread(
                target=self.run_service, args=(hosted,), name=hosted.name, daemon=True
            )
            hosted.started = time.monotonic()
            hosted.thread.start()
            if hosted is not self.services[-1]:
                self.let_imports_settle()
        self.say(f"started {', '.join(hosted.label for hosted in self.services)}")

    def run_service(self, hosted: Hosted) -> None:
        try:
            hosted.instance.start(background=False)
        except BaseException as exc:
            if self._ending.locked():
                # Upstream cleans up again as run() returns, and a second
                # cleanup may find nothing left to shut down.

                self.say(
                    f"{hosted.label} ended with {type(exc).__name__} after its cleanup"
                )
            else:
                self.say(f"{hosted.label} raised:\n{traceback.format_exc()}")

    def let_imports_settle(self) -> None:
        """Services import more as they start (the executor loads every block
        module). Two threads importing modules that import each other can
        each be handed a half-initialised one, so the next service starts
        once this one has stopped importing."""
        deadline = time.monotonic() + IMPORTS_SETTLE_LIMIT_SECONDS
        modules, quiet_since = len(sys.modules), time.monotonic()
        while time.monotonic() < deadline and not self.stop.is_set():
            time.sleep(0.1)
            if len(sys.modules) != modules:
                modules, quiet_since = len(sys.modules), time.monotonic()
            elif time.monotonic() - quiet_since >= IMPORTS_QUIET_SECONDS:
                return

    # --- chats left busy by turns that were ended from outside ------------

    def tend_ended_turns(self) -> None:
        service, _, file = (self.ended_turns or "").partition("=")
        if (
            file
            and service in {name for name, _ in self.entries}
            and os.path.exists(file)
        ):
            threading.Thread(
                target=self.close_ended_turns,
                args=(file,),
                name="ended-turns",
                daemon=True,
            ).start()

    def close_ended_turns(self, file: str) -> None:
        """Mark each listed chat's turn as cancelled, the way upstream's own
        Stop does when the executor does not confirm, then remove the list.
        Anything that fails leaves the list for the next start; a chat left
        busy meanwhile is one its user can still free with Stop."""
        try:
            with open(file, encoding="utf-8") as stream:
                listed = json.load(stream)
            if not isinstance(listed, list):
                raise TypeError("stopped turns must be a list")
            sessions = [
                record
                for record in listed
                if isinstance(record, dict)
                and isinstance(record.get("session_id"), str)
                and isinstance(record.get("turn_id"), str)
                and record["turn_id"]
            ]
            legacy = [record for record in listed if record not in sessions]
            if legacy:
                self.say(
                    "stopped turns without turn identities require manual recovery; leaving their chats alone"
                )
            registry: Any = importlib.import_module("backend.copilot.stream_registry")
            close, cancelled = (
                registry.mark_session_completed,
                registry.CANCELLED_MESSAGE,
            )
        except Exception as exc:
            self.say(
                f"cannot close the chats of turns that were stopped: {type(exc).__name__}: {exc}"
            )
            return
        import asyncio

        async def close_all() -> None:
            for record in sessions:
                await close(
                    record["session_id"],
                    error_message=cancelled,
                    skip_error_publish=True,
                    turn_id=record["turn_id"],
                )

        for attempt in range(ENDED_TURNS_TRIES):
            try:
                asyncio.run(close_all())
                break
            except Exception as exc:
                self.say(
                    f"closing the chats of stopped turns failed: {type(exc).__name__}: {exc}"
                )
                if attempt + 1 == ENDED_TURNS_TRIES or self.stop.wait(
                    ENDED_TURNS_RETRY_SECONDS
                ):
                    return
        if legacy:
            from .runs import _write

            _write(Path(file), legacy)
        else:
            with contextlib.suppress(OSError):
                os.remove(file)
        self.say(
            f"closed the chat(s) of {len(sessions)} AutoPilot turn(s) stopped as the app closed"
        )

    def supervise(self) -> NoReturn:
        while not self.stop.wait(0.5):
            ended = [hosted.label for hosted in self.services if not hosted.running]
            if self._loop_thread and not self._loop_thread.is_alive():
                ended.append("the shared event loop")
            if ended:
                # The supervisor restarts the whole process: a service that
                # ended cannot be started twice in one interpreter.
                self.shut_down(EXIT_SERVICE_ENDED, f"{', '.join(ended)} ended")
            if self.no_database and database_is_connected(self):
                self.shut_down(
                    EXIT_CONTRACT,
                    "a service here connected to the database itself; these services "
                    "reach it through database-manager and must not share a process "
                    "with one that does",
                )
            silent = self.not_serving()
            if silent:
                self.shut_down(
                    EXIT_CONTRACT,
                    f"{silent} did not start its server through uvicorn.run",
                )
        park()  # the thread that saw the request ends the process

    def not_serving(self) -> str | None:
        if not (self.shared_loop and self.merged):
            return None
        overdue = time.monotonic() - SERVE_DEADLINE_SECONDS
        late = [
            h.label
            for h in self.services
            if h.name not in self.served and h.started < overdue
        ]
        return ", ".join(late) or None

    # --- stopping ---------------------------------------------------------

    def listen_for_stop(self) -> None:
        """Before the imports, which take seconds: a stop can come any time."""
        event = os.environ.pop(STOP_EVENT_ENV, None)  # not for the services' children
        if sys.platform == "win32":
            if event:
                wait_for_windows_event(event, self.stop, self.say)
        else:
            signal.signal(signal.SIGTERM, self.on_signal)
            signal.signal(signal.SIGINT, self.on_signal)
        threading.Thread(
            target=self.stop_when_asked, name="host-stop", daemon=True
        ).start()

    def on_signal(self, signum: int, frame: object) -> None:
        # Only this: a handler runs in the main thread, on top of whatever it
        # was doing. Setting the event here could deadlock with the main
        # thread's own wait on it.
        self.signalled = True

    def stop_when_asked(self) -> None:
        while not (self.signalled or self.stop.wait(0.1)):
            pass
        self.stop.set()
        self.shut_down(0, "asked to stop")

    def shut_down(self, code: int, reason: str) -> NoReturn:
        if not self._ending.acquire(blocking=False):
            park()
        self.say(f"stopping: {reason}")
        deadline = time.monotonic() + self.stop_budget
        refreshes = sys.modules.get("backend.integrations.credential_refresh")
        if refreshes is not None:
            drained = refreshes.drain_credential_refreshes(
                min(refreshes.REFRESH_DRAIN_SECONDS, max(0.0, self.stop_budget - 2.5))
            )
            if not drained:
                self.say("credential refresh did not drain; reconnect may be required")
        if self.foreground:  # upstream's own run owns the services
            leave(code)
        for server in self.servers:
            server.should_exit = (
                True  # uvicorn's own way out, lifespan shutdown included
            )
        running = [hosted for hosted in self.services if hosted.running]
        cleaners = [
            threading.Thread(target=self.clean_up, args=(hosted,), daemon=True)
            for hosted in running
        ]
        for cleaner in cleaners:
            cleaner.start()
        for cleaner in cleaners:
            cleaner.join(max(0.0, deadline - time.monotonic()))
        unfinished = sum(cleaner.is_alive() for cleaner in cleaners)
        served = [hosted for hosted in running if hosted.name in self.served]
        wait_for(lambda: not any(hosted.running for hosted in served), deadline)
        goodbye = min(deadline, time.monotonic() + GOODBYE_SECONDS)
        wait_for(lambda: not any(hosted.running for hosted in running), goodbye)
        if unfinished:
            self.say(
                f"{unfinished} cleanup(s) still running after {self.stop_budget:g}s"
            )
        self.say(f"stopped with code {code}")
        leave(code)

    def clean_up(self, hosted: Hosted) -> None:
        started = time.monotonic()
        try:
            hosted.instance.cleanup()
        except BaseException as exc:  # one failed cleanup must not block the others
            self.say(f"{hosted.label} cleanup failed: {type(exc).__name__}: {exc}")
        else:
            self.say(f"{hosted.label} cleaned up in {time.monotonic() - started:.2f}s")

    def say(self, message: str) -> None:
        """The services' logging is upstream's; this is the host's own voice,
        present before the backend is imported and after it has shut down."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with contextlib.suppress(OSError, ValueError):
            sys.stderr.write(f"{stamp} [host {self.name}] {message}\n")
            sys.stderr.flush()


def entry_point(entry: str) -> Callable[[], object]:
    module, _, function = entry.partition(":")
    try:
        return getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError) as exc:
        raise ContractError(f"the entry point {entry} is gone ({exc})") from exc


def check_service(entry: str, service: object) -> None:
    start = getattr(service, "start", None)
    if not callable(start) or "background" not in inspect.signature(start).parameters:
        raise ContractError(f"the service from {entry} has no start(background=...)")
    if not callable(getattr(service, "cleanup", None)):
        raise ContractError(f"the service from {entry} has no cleanup()")


def database_is_connected(host: Host) -> bool:
    try:
        return bool(importlib.import_module("backend.data.db").is_connected())
    except Exception as exc:
        host.no_database = False
        host.say(
            f"backend.data.db.is_connected() is gone ({exc}); no longer watching it"
        )
        return False


def make_signals_thread_aware() -> None:
    """`AppProcess.execute_run_command` installs signal handlers, which only
    the main thread may do. From any other thread, accept and do nothing: the
    host handles stopping. The main thread's calls pass through."""
    module: Any = signal
    install = module.signal

    def thread_aware(signalnum: int, handler: Any) -> Any:
        if threading.current_thread() is threading.main_thread():
            return install(signalnum, handler)
        return signal.SIG_DFL

    module.signal = thread_aware


def wait_for_windows_event(
    name: str, stop: threading.Event, say: Callable[[str], None]
) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenEventW.restype = ctypes.c_void_p
    kernel32.OpenEventW.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    synchronize, infinite, signalled = 0x00100000, 0xFFFFFFFF, 0
    handle = kernel32.OpenEventW(synchronize, False, name)
    if not handle:
        say(f"cannot open the stop event {name}; a stop will end this process abruptly")
        return

    def wait() -> None:
        if kernel32.WaitForSingleObject(handle, infinite) == signalled:
            stop.set()

    threading.Thread(target=wait, name="host-stop-event", daemon=True).start()


def wait_for(done: Callable[[], bool], deadline: float) -> None:
    while not done() and time.monotonic() < deadline:
        time.sleep(0.05)


def park() -> NoReturn:
    while True:
        time.sleep(3600)


def leave(code: int) -> NoReturn:
    """Without interpreter finalisation: service threads that never return
    and telemetry teardown must not be able to hold the process open."""
    with contextlib.suppress(OSError, ValueError):
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)


def main() -> NoReturn:
    parser = argparse.ArgumentParser(prog="autogpt_desktop.servicehost")
    parser.add_argument(
        "--name", required=True, help="what the supervisor calls this process"
    )
    parser.add_argument(
        "--stop-budget", type=float, default=2.5, help="seconds for cleanup"
    )
    parser.add_argument("--shared-loop", action="store_true")
    parser.add_argument("--no-database", action="store_true")
    parser.add_argument("services", nargs="+", metavar="name=module:function")
    options = parser.parse_args()
    sys.argv = sys.argv[:1]  # the backend must not see the host's arguments
    services = [service.partition("=") for service in options.services]
    Host(
        name=options.name,
        entries=[(name, entry) for name, _, entry in services],
        stop_budget=options.stop_budget,
        shared_loop=options.shared_loop,
        no_database=options.no_database,
    ).run()
