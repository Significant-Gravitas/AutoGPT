"""What the service host depends on in the backend, checked against a real one.

    <runtime>/python/python -B backend_contract.py [<runtime dir>]

Run with an interpreter that has the backend's dependencies: the bundle's.
It is a script and not a test because importing the backend configures
logging, Sentry and the multiprocessing start method for the whole
interpreter; it gets one of its own, with the environment the runtime gives
a service. test_backend_contract.py runs it when a bundle is assembled, and
build/smoke_test.py runs it on every OS before it starts anything.

Nothing is started. Each line of output is one dependency: `ok`, or `BROKEN`
with what changed upstream and what to update here. Exit code 1 if any is
broken.
"""

from __future__ import annotations

import ast
import fnmatch
import inspect
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

RUNTIME_PACKAGE_PARENT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = RUNTIME_PACKAGE_PARENT.parent / "build" / "runtime"
# In backend.app.main's list and deliberately not run by the desktop: the
# chat-bot bridge services, as in the appliance.
NOT_RUN = {"PlatformLinkingManager", "CoPilotChatBridge"}
HOST = "autogpt_desktop/servicehost.py"
APPS = "autogpt_desktop/apps.py"
# The backend setting that holds the port of each service that does not serve
# through AppService (those are asked with get_port).
PORT_SETTINGS = {
    "executor": "execution_manager_port",
    "copilot-executor": "copilot_executor_port",
    "websocket": "websocket_server_port",
    "rest": "agent_api_port",
}

# on_graph_execution's last branch: any status but queued, running or in review.
SKIPS_AN_ENDED_RUN = r'else:\s+log_metadata\.warning\(\s+f"Skipped graph execution'

# backend/copilot/executor/manager.py, where a turn that has ended gives up
# its lock, and backend/copilot/stream_registry.py, where a chat is closed
# only if it is busy.
RELEASES_A_TURNS_LOCK = (
    r"def on_run_done\(f: Future\):.*?finally:\s+if session_id in self\._task_locks:\s+"
    r"(?:logger\.\w+\([^\n]*\)\s+)?self\._task_locks\[session_id\]\.release\(\)"
)
CLOSES_ONLY_A_BUSY_CHAT = (
    r'hash_compare_and_set\(\s*redis,\s*meta_key,\s*"status",\s*'
    r'expected="running",\s*new=status'
)

# backend/api/features/chat/routes.py, where a cancel is not confirmed in time.
CLOSES_AN_UNCONFIRMED_CANCEL = (
    r"await stream_registry\.mark_session_completed\(\s+session_id,\s+"
    r"error_message=stream_registry\.CANCELLED_MESSAGE,\s+skip_error_publish=True,"
)

findings: list[tuple[bool, str]] = []
_finding_exceptions: dict[int, str] = {}
_diagnostic_output_failed = False
_EXCEPTION_TYPES = (
    (ModuleNotFoundError, "module_not_found"),
    (ImportError, "import_error"),
    (AttributeError, "attribute_error"),
    (TypeError, "type_error"),
    (ValueError, "value_error"),
    (TimeoutError, "timeout"),
    (OSError, "os_error"),
    (KeyboardInterrupt, "keyboard_interrupt"),
    (SystemExit, "system_exit"),
)
_EXCEPTION_IDS = frozenset({"none", "other", *(name for _, name in _EXCEPTION_TYPES)})


def _exception_category(exc: BaseException) -> str:
    for exception_type, category in _EXCEPTION_TYPES:
        if isinstance(exc, exception_type):
            return category
    return "other"


def _failure_count(start: int = 0) -> int:
    return sum(not ok for ok, _ in findings[start:])


def _finding_exception(start: int = 0) -> str:
    return next(
        (
            _finding_exceptions[index]
            for index in range(start, len(findings))
            if not findings[index][0] and index in _finding_exceptions
        ),
        "none",
    )


def _diagnostic(
    stage: Literal["contract_collect", "contract_check"],
    status: Literal["started", "passed", "failed", "error"],
    contract: str,
    *,
    failures: int = 0,
    exception: str = "none",
) -> None:
    global _diagnostic_output_failed
    exception = exception if exception in _EXCEPTION_IDS else "other"
    reason = "none"
    if exception == "timeout":
        reason = "timeout"
    elif exception != "none":
        reason = "exception"
    elif status == "failed":
        reason = "assertion_failed"
    message = "DESKTOP_SMOKE_DIAGNOSTIC " + json.dumps(
        {
            "version": 1,
            "stage": stage,
            "status": status,
            "reason": reason,
            "exception": exception,
            "contract": contract if contract in _CONTRACT_NAMES else "unknown",
            "child_exit": None,
            "failures": int(max(0, min(failures, 65535))),
            "start": 0,
        },
        separators=(",", ":"),
    )
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        _diagnostic_output_failed = True


def main() -> int:
    global _diagnostic_output_failed
    _diagnostic_output_failed = False
    collecting = True
    _diagnostic("contract_collect", "started", "collect")
    try:
        bundle_root = (
            Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_BUNDLE
        )
        scratch = tempfile.TemporaryDirectory(
            prefix="autogpt-contract-", ignore_cleanup_errors=True
        )
        with scratch as directory:
            services = collect(bundle_root, Path(directory))
            broken = _failure_count()
            _diagnostic(
                "contract_collect",
                "failed" if broken else "passed",
                "collect",
                failures=broken,
                exception=_finding_exception(),
            )
            collecting = False
            for check in CHECKS if services else ():
                first = len(findings)
                contract = _CONTRACT_IDS.get(check, "unknown")
                _diagnostic("contract_check", "started", contract)
                try:
                    check(services, bundle_root / "backend" / "backend")
                except Exception as exc:
                    try:
                        what = check.__name__.replace("_", " ")
                        expect(
                            False,
                            what,
                            f"The check itself failed: {type(exc).__name__}: {exc}",
                            exception=exc,
                        )
                    finally:
                        _diagnostic(
                            "contract_check",
                            "error",
                            contract,
                            failures=_failure_count(first),
                            exception=_exception_category(exc),
                        )
                except BaseException as exc:
                    _diagnostic(
                        "contract_check",
                        "error",
                        contract,
                        failures=_failure_count(first),
                        exception=_exception_category(exc),
                    )
                    raise
                else:
                    broken = _failure_count(first)
                    _diagnostic(
                        "contract_check",
                        "failed" if broken else "passed",
                        contract,
                        failures=broken,
                        exception=_finding_exception(first),
                    )
    except BaseException as exc:
        if collecting:
            _diagnostic(
                "contract_collect",
                "error",
                "collect",
                failures=_failure_count(),
                exception=_exception_category(exc),
            )
        raise
    for ok, message in findings:
        print(f"{'ok' if ok else 'BROKEN'}: {message}")
    broken = sum(not ok for ok, _ in findings)
    print(f"{len(findings) - broken} of {len(findings)} dependencies hold")
    return 1 if broken or _diagnostic_output_failed else 0


def expect(
    ok: bool, holds: str, broken: str, *, exception: BaseException | None = None
) -> None:
    findings.append((bool(ok), holds if ok else f"{holds}. {broken}"))
    if not ok and exception is not None:
        _finding_exceptions[len(findings) - 1] = _exception_category(exception)


def collect(bundle_root: Path, scratch: Path) -> dict[str, Any]:
    """The eight services, taken from upstream the way the host takes them:
    with its own collector, in one interpreter."""
    as_a_service_sees_it(bundle_root, scratch)
    from autogpt_desktop import apps, servicehost

    host = servicehost.Host(
        name="contract",
        entries=[(service.name, service.entry) for service in apps.SERVICES],
    )
    try:
        host.collect()
    except servicehost.ContractError as exc:
        expect(
            False,
            "every entry point hands one service to backend.app.run_processes",
            str(exc),
            exception=exc,
        )
        return {}
    expect(
        True,
        "every entry point hands one service, with start(background=...) and cleanup(), "
        "to backend.app.run_processes",
        "",
    )
    return {hosted.name: hosted.instance for hosted in host.services}


def as_a_service_sees_it(bundle_root: Path, scratch: Path) -> None:
    """cwd and path as apps.host_argv gives them, environment as
    settings.backend_environment does, over made-up ports and secrets."""
    sys.path.append(str(RUNTIME_PACKAGE_PARENT))
    from autogpt_desktop import ports, resources, settings
    from autogpt_desktop.layout import Bundle, DataDir

    data = DataDir(scratch)
    data.prepare()
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    bundle = Bundle(bundle_root)
    secret = settings.ensure_secrets(bundle, data)
    profile = resources.choose({}, ram_bytes=16 * 1024**3, cores=6)
    os.environ.update(
        settings.backend_environment(
            bundle, data, port, secret, {}, profile.backend_env
        )
    )
    os.chdir(bundle.backend_dir)
    sys.path.insert(0, "")
    sys.argv = sys.argv[:1]


def service_names_and_classes(services: dict[str, Any], backend: Path) -> None:
    for name, service in services.items():
        label = getattr(service, "service_name", None)
        expect(
            isinstance(label, str) and bool(label),
            f"{name}: service_name is a string ({label!r})",
            f"The host's log names services with it; see Hosted.label in {HOST}.",
        )
    tree = ast.parse((backend / "app.py").read_text(encoding="utf-8"))
    upstream = {
        node.func.id
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", "") == "run_processes"
        for node in ast.walk(call)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } - {"run_processes"}
    ours = {type(service).__name__ for service in services.values()}
    expect(
        upstream - NOT_RUN == ours,
        f"backend.app.main runs the services the desktop runs, and {sorted(NOT_RUN)}",
        f"Upstream now runs {sorted(upstream)}; the desktop runs {sorted(ours)}. A new "
        f"service needs an entry in SERVICES and a group in {APPS} (and a port in ports.py); "
        "a removed one has to go from there.",
    )


def nobody_is_connected(services: dict[str, Any], backend: Path) -> None:
    """Building the services must not connect to the database, and only the
    ones marked `database` may ever do so themselves."""
    from autogpt_desktop import apps

    try:
        from backend.data import db

        connected = db.is_connected()
    except Exception as exc:
        expect(
            False,
            "backend.data.db.is_connected() exists",
            f"{exc}. The watchdog in {HOST} uses it.",
            exception=exc,
        )
        return
    expect(
        connected is False,
        "backend.data.db.is_connected() is False before any service starts",
        f"The watchdog in {HOST} would end the workers host at once.",
    )
    connects = re.compile(r"\b(db|database|prisma)\.connect\(\)")
    for name, service in services.items():
        source = Path(inspect.getsourcefile(type(service)) or "").read_text(
            encoding="utf-8"
        )
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        marked = apps.service(name).database
        expect(
            bool(connects.search(code)) == marked,
            f"{name} {'connects' if marked else 'does not connect'} to the database itself",
            f"Its module now says otherwise. Set `database` on its entry in SERVICES in {APPS} "
            "to match: only services that never connect may share the workers host.",
        )


def api_servers_call_uvicorn_run(services: dict[str, Any], backend: Path) -> None:
    from autogpt_desktop import apps

    for name in apps.API_SERVERS:
        source = inspect.getsource(type(services[name]).run)
        expect(
            "uvicorn.run(" in source,
            f"{name} serves by calling uvicorn.run",
            f"That is the call {HOST} replaces to put the API servers on one event loop "
            "(serve_on_one_loop). Intercept what it calls now, or take it out of API_SERVERS "
            f"and the api group in {APPS}.",
        )


def services_answer_where_the_health_table_says(
    services: dict[str, Any], backend: Path
) -> None:
    from autogpt_desktop import apps, ports

    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    for service in apps.SERVICES:
        instance = services[service.name]
        get_port: Callable[[], int] | None = getattr(type(instance), "get_port", None)
        if service.health == "/health_check":
            expect(
                get_port is not None and get_port() == port[service.port],
                f"{service.name} listens on the {service.port} port and has /health_check",
                f"Update its port or path in SERVICES in {APPS} (and the *_PORT names in "
                "settings._service_addresses).",
            )
    source = (backend / "util" / "service.py").read_text(encoding="utf-8")
    expect(
        '"/health_check"' in source,
        "AppService serves /health_check",
        f"Update SERVICES in {APPS}.",
    )
    for name in ("executor", "copilot-executor"):
        run = inspect.getsource(type(services[name]).run)
        expect(
            "start_http_server(" in run,
            f"{name} serves its metrics over HTTP",
            f"/metrics is how the supervisor knows it is up; update SERVICES in {APPS}.",
        )


def the_other_services_listen_on_the_ports_they_are_given(
    services: dict[str, Any], backend: Path
) -> None:
    """The executors' metrics servers and the two API servers: the runtime
    probes each on the port it handed out through settings._service_addresses."""
    from backend.util.settings import Config

    from autogpt_desktop import apps, ports

    config = Config()
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    for name, setting in PORT_SETTINGS.items():
        given = port[apps.service(name).port]
        run = inspect.getsource(type(services[name]).run)
        expect(
            getattr(config, setting, None) == given and setting in run,
            f"{name} listens on the port given as {setting.upper()}",
            f"The backend reads {getattr(config, setting, None)!r} for {setting} and its run() "
            f"{'uses' if setting in run else 'no longer uses'} it. The supervisor waits for "
            f"{name} there: update the *_PORT names in settings._service_addresses and its "
            f"entry in SERVICES in {APPS}.",
        )


def uvicorn_can_serve_on_the_hosts_loop(
    services: dict[str, Any], backend: Path
) -> None:
    import uvicorn

    update = f"Update serve_on_one_loop in {HOST}: it builds the server and awaits it itself."

    async def app(scope: Any, receive: Any, send: Any) -> None:
        pass

    server: Any = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None)
    )
    serve = getattr(server, "serve", None)
    needed = [
        parameter.name
        for parameter in inspect.signature(serve).parameters.values()
        if parameter.default is parameter.empty
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    expect(
        inspect.iscoroutinefunction(serve) and not needed,
        "uvicorn.Server(uvicorn.Config(app, **options)).serve() is a coroutine needing no arguments",
        f"It now needs {needed}. {update}",
    )
    expect(
        getattr(server, "started", None) is False
        and getattr(server, "should_exit", None) is False,
        "a uvicorn server has `started` and `should_exit`, both False before it serves",
        f"The host reads the first to tell a failed start and sets the second to stop. {update}",
    )


def the_scheduler_holds_the_connections_the_limit_counts(
    services: dict[str, Any], backend: Path
) -> None:
    from autogpt_desktop import resources

    source = (backend / "executor" / "scheduler.py").read_text(encoding="utf-8")
    pools = source.count("create_engine(")
    bounded = (
        source.count("pool_size=self.db_pool_size()")
        == pools
        == source.count("max_overflow=0")
    )
    expect(
        pools == resources.SCHEDULER_POOLS
        and bounded
        and "config.scheduler_db_pool_size" in source,
        f"the scheduler keeps {resources.SCHEDULER_POOLS} connection pools of "
        "scheduler_db_pool_size, without overflow",
        f"It now creates {pools} engines, or sizes them differently. Count its connections "
        "again and update SCHEDULER_POOLS and POSTGRES_LIMITS in autogpt_desktop/resources.py.",
    )


def pool_sizes_come_from_the_environment(
    services: dict[str, Any], backend: Path
) -> None:
    from backend.util.settings import Config

    from autogpt_desktop import resources

    config = Config()
    wanted = resources.choose({}, ram_bytes=16 * 1024**3, cores=6).backend_env
    for name, value in wanted.items():
        expect(
            str(getattr(config, name.lower(), None)) == value,
            f"{name} sets the backend's {name.lower()}",
            "Update the names in autogpt_desktop/resources.py.",
        )


def lock_keys_are_the_ones_cleared(services: dict[str, Any], backend: Path) -> None:
    from backend.copilot.executor.utils import get_session_lock_key

    from autogpt_desktop import runs

    update = "Update LOCK_PATTERNS in autogpt_desktop/runs.py."
    manager = (backend / "executor" / "manager.py").read_text(encoding="utf-8")
    key = re.search(r'key=f"(exec_lock:)\{', manager)
    expect(
        bool(key)
        and fnmatch.fnmatch("exec_lock:some-run", runs.LOCK_PATTERNS["executor"]),
        "the executor locks a run under exec_lock:<id>",
        update,
    )
    expect(
        fnmatch.fnmatch(
            get_session_lock_key("some-session"), runs.LOCK_PATTERNS["copilot-executor"]
        ),
        f"the copilot executor locks a session under {get_session_lock_key('<id>')}",
        update,
    )


def runs_are_cancelled_and_settled_under_the_runtimes_names(
    services: dict[str, Any], backend: Path
) -> None:
    """What the runtime does to runs from outside the executors when the
    user quits with agents running (autogpt_desktop/runs.py)."""
    import json

    from backend.copilot.executor import utils as copilot
    from backend.executor import utils as graph

    from autogpt_desktop import runs

    update = "Update the names at the top of autogpt_desktop/runs.py."
    expect(
        graph.GRAPH_EXECUTION_QUEUE_NAME == runs.GRAPH_RUN_QUEUE,
        f"agent runs are queued on {runs.GRAPH_RUN_QUEUE}",
        f"They are queued on {graph.GRAPH_EXECUTION_QUEUE_NAME} now. {update}",
    )
    expect(
        copilot.COPILOT_EXECUTION_QUEUE_NAME == runs.COPILOT_TURN_QUEUE,
        f"AutoPilot turns are queued on {runs.COPILOT_TURN_QUEUE}",
        f"They are queued on {copilot.COPILOT_EXECUTION_QUEUE_NAME} now. {update}",
    )
    exchanges = (
        (
            "an agent run",
            graph.GRAPH_EXECUTION_CANCEL_EXCHANGE,
            runs.GRAPH_CANCEL_EXCHANGE,
        ),
        (
            "an AutoPilot turn",
            copilot.COPILOT_CANCEL_EXCHANGE,
            runs.COPILOT_CANCEL_EXCHANGE,
        ),
    )
    for what, exchange, ours in exchanges:
        kind = str(getattr(exchange.type, "value", exchange.type))
        expect(
            exchange.name == ours and kind == "fanout",
            f"a request to cancel {what} goes to the fanout exchange {ours}",
            f"It goes to the {kind} exchange {exchange.name} now. {update} A request is published "
            "with an empty routing key, to every executor (Queue.cancel).",
        )
    requests = (
        (graph.CancelExecutionEvent, runs.GRAPH_CANCEL_FIELD),
        (copilot.CancelCoPilotEvent, runs.COPILOT_SESSION_FIELD),
    )
    for model, name in requests:
        error = None
        try:
            read = getattr(
                model.model_validate_json(json.dumps({name: "an-id"})), name, None
            )
        except Exception as exc:
            error = exc
            read = f"{type(exc).__name__}: {exc}"
        expect(
            read == "an-id",
            f"{model.__name__} is read from {{{name!r}: ...}} alone",
            f"The executor now makes of it: {read}. {update} Queue.cancel writes it.",
            exception=error,
        )
    expect(
        {runs.COPILOT_SESSION_FIELD, "turn_id"}
        <= copilot.CoPilotExecutionEntry.model_fields.keys(),
        f"a turn's message names its session in {runs.COPILOT_SESSION_FIELD!r} and its turn_id",
        f"{update} Queue.drop_turns finds the turns of a stopped session by it.",
    )
    manager = (backend / "executor" / "manager.py").read_text(encoding="utf-8")
    expect(
        bool(re.search(SKIPS_AN_ENDED_RUN, manager)),
        "the executor skips the message of a run whose status says it has ended",
        "on_graph_execution in backend/executor/manager.py no longer does. Ending a run in the "
        "database (runs.reconcile, runs.end_stopped, runs.sweep_orphans) relies on it: a run "
        "the user stopped would start again.",
    )


def a_chat_can_be_closed_from_outside_its_turn(
    services: dict[str, Any], backend: Path
) -> None:
    """What the copilot executor's host calls for a turn that was ended
    without the executor (servicehost.close_ended_turns)."""
    update = f"Update close_ended_turns in {HOST}."
    try:
        from backend.copilot import stream_registry

        close = stream_registry.mark_session_completed
        cancelled = stream_registry.CANCELLED_MESSAGE
    except Exception as exc:
        expect(
            False,
            "backend.copilot.stream_registry closes a chat's turn",
            f"{exc}. {update}",
            exception=exc,
        )
        return
    parameters = inspect.signature(close).parameters
    needed = [
        name
        for name, parameter in parameters.items()
        if parameter.default is parameter.empty
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    fits = (
        inspect.iscoroutinefunction(close)
        and needed == ["session_id"]
        and {"error_message", "skip_error_publish", "turn_id"} <= parameters.keys()
    )
    expect(
        fits and isinstance(cancelled, str),
        "stream_registry.mark_session_completed(session_id, error_message=CANCELLED_MESSAGE, "
        "skip_error_publish=True, turn_id=original_turn) is a coroutine that needs nothing else",
        f"It now takes {list(parameters)} and needs {needed}. {update}",
    )
    routes = (backend / "api" / "features" / "chat" / "routes.py").read_text(
        encoding="utf-8"
    )
    expect(
        bool(re.search(CLOSES_AN_UNCONFIRMED_CANCEL, routes)),
        "upstream's own Stop closes a chat that way when the executor does not confirm",
        "The chat routes no longer do. The host copies that call: find what Stop does now for "
        f"a turn nobody confirms as cancelled. {update}",
    )


def turns_are_locked_while_they_run_and_only_busy_chats_are_closed(
    services: dict[str, Any], backend: Path
) -> None:
    """What counting AutoPilot turns, and ending them from outside, takes
    for granted (autogpt_desktop/runs.py: Cache.active, end_stopped,
    reconcile)."""
    copilot = backend / "copilot"
    processor = (copilot / "executor" / "processor.py").read_text(encoding="utf-8")
    expect(
        "cluster_lock.refresh()" in processor,
        "the copilot executor renews a turn's lock while the turn goes on",
        "backend/copilot/executor/processor.py no longer does. A turn of more than five "
        "minutes would stop counting as in progress and be left out of a stop: see "
        "Cache.active in autogpt_desktop/runs.py.",
    )
    manager = (copilot / "executor" / "manager.py").read_text(encoding="utf-8")
    expect(
        bool(re.search(RELEASES_A_TURNS_LOCK, manager, re.DOTALL)),
        "the copilot executor gives up a turn's lock when the turn has ended",
        "on_run_done in backend/copilot/executor/manager.py no longer releases "
        "self._task_locks[session_id]. runs.Cache.active counts locks as active work.",
    )
    registry = (copilot / "stream_registry.py").read_text(encoding="utf-8")
    closing = re.search(
        r"^async def mark_session_completed\(.*?(?=^(?:async def|def|class|@)\b|\Z)",
        registry,
        re.DOTALL | re.MULTILINE,
    )
    expect(
        bool(closing and re.search(CLOSES_ONLY_A_BUSY_CHAT, closing.group(0))),
        'mark_session_completed changes only a chat whose status is "running"',
        'It no longer swaps the status from "running" alone. runs.reconcile lists every '
        f"recorded turn's chat for close_ended_turns in {HOST} to close, finished or not: a "
        "finished chat would now be marked failed. List only chats whose message was found.",
    )
    expect(
        bool(
            closing
            and 'guard = ("turn_id", turn_id) if turn_id else None' in closing.group(0)
            and re.search(
                r'expected="running",\s*new=status,\s*guard=guard', closing.group(0)
            )
        ),
        "mark_session_completed atomically guards the original turn identity",
        f"Without that guard, retries in {HOST} can cancel a later turn in the same chat.",
    )


CHECKS = (
    service_names_and_classes,
    nobody_is_connected,
    api_servers_call_uvicorn_run,
    services_answer_where_the_health_table_says,
    the_other_services_listen_on_the_ports_they_are_given,
    uvicorn_can_serve_on_the_hosts_loop,
    pool_sizes_come_from_the_environment,
    the_scheduler_holds_the_connections_the_limit_counts,
    lock_keys_are_the_ones_cleared,
    runs_are_cancelled_and_settled_under_the_runtimes_names,
    a_chat_can_be_closed_from_outside_its_turn,
    turns_are_locked_while_they_run_and_only_busy_chats_are_closed,
)
_CONTRACT_IDS = {check: check.__name__ for check in CHECKS}
_CONTRACT_NAMES = frozenset({"collect", "unknown", *_CONTRACT_IDS.values()})


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # the backend's telemetry threads must not hold this open
