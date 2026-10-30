"""AutoGPT's own processes: the backend services and the Next server.

The same services the appliance's supervisord runs
(supervisor/supervisord.conf) minus the opt-in chat-bot ones. Each is
upstream's own entry point, called through the bundled interpreter rather
than the console scripts pip generates: those carry absolute shebangs that
break as soon as the app bundle is moved.

Every service process is a service host (servicehost.py), whether it hosts
one service or several. By default they are grouped, because each
interpreter that imports the backend costs about 600 MB:

    database-manager   alone: it is the others' way to the database
    swap-credentials   alone: owns a separate database client and event loop
    workers            the five services that never connect to the database
    websocket          reads through database-manager, without a Prisma client
    api                REST owns its database client

`isolated()` is one host per service: what the `isolated` profile runs, and
what a group is replaced with when upstream stops fitting it.
`copilot_alone()` takes the copilot executor out of its group, when it is
the one service that runs in the user's own home directory (claude_code.py).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from autogpt_desktop import servicehost
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ManagedProcess, StopEvent, base_env
from autogpt_desktop.runs import Cache


@dataclass(frozen=True)
class Service:
    name: str
    entry: str
    # Where it answers once it is up: its name in ports.PORT_NAMES, and a
    # path. Services built on the backend's AppService have /health_check;
    # the two executors serve only their metrics.
    port: str
    health: str
    # Connects to the database itself (Prisma) instead of through
    # database-manager. The backend decides per process which of the two a
    # query takes, so such a service cannot share one with the others.
    database: bool = False


SERVICES = (
    Service(
        "database-manager",
        "autogpt_desktop.backend_entries:database_main",
        "database_api",
        "/health_check",
        database=True,
    ),
    Service(
        "swap-credentials",
        "autogpt_desktop.backend_entries:swap_main",
        "swap_credential",
        "/health_check",
        database=True,
    ),
    Service(
        "scheduler", "backend.scheduler:main", "execution_scheduler", "/health_check"
    ),
    Service(
        "batch-executor",
        "backend.batch_executor:main",
        "batch_executor",
        "/health_check",
    ),
    Service(
        "notification", "backend.notification:main", "notification", "/health_check"
    ),
    Service("executor", "backend.exec:main", "execution_manager", "/metrics"),
    Service(
        "copilot-executor",
        "backend.copilot.executor.__main__:main",
        "copilot_executor",
        "/metrics",
    ),
    Service("websocket", "backend.ws:main", "websocket", "/health"),
    Service("rest", "backend.rest:main", "agent_api", "/health", database=True),
)
DATABASE_MANAGER = "database-manager"
COPILOT_EXECUTOR = "copilot-executor"
FRONTEND = "frontend"
# The two that call `uvicorn.run` themselves (the others serve through
# AppService, each on a loop of its own).
API_SERVERS = ("websocket", "rest")

# What the window talks to, and their way to the database. The app is ready
# when these answer; the rest are waited for only so long (supervisor.py).
NEEDED_TO_OPEN = frozenset(
    {DATABASE_MANAGER, "swap-credentials", *API_SERVERS, FRONTEND}
)

SKILLS_CATALOG_ENTRY = "backend.cli.publish_skills_catalog:main"

# How long a service gets between being asked to stop and being killed. The
# appliance measured this (single-container/supervisor/supervisord.conf): a
# service finishes its own cleanup well inside a second, then sits in
# third-party telemetry teardown for several more, so waiting longer buys
# nothing. The host exits by itself once cleanup is done, just inside it.
STOP_TIMEOUT_SECONDS = 3
HOST_STOP_BUDGET_SECONDS = STOP_TIMEOUT_SECONDS - 0.5
REFRESH_STOP_TIMEOUT_SECONDS = 30
REFRESH_SERVICES = frozenset(
    {
        "executor",
        "copilot-executor",
        "rest",
        "websocket",
        "scheduler",
        "swap-credentials",
    }
)

# The package is not on a service's path (its cwd is the backend), and it is
# appended, never put first: its parent directory in the bundle also holds
# directories named prisma, site and backend.
HOST_CODE = (
    "import sys; sys.path.append(sys.argv.pop(1)); "
    "from autogpt_desktop.servicehost import main; main()"
)
PACKAGE_PARENT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Group:
    """Services that share a process, named as the supervisor and the logs
    name it."""

    name: str
    services: tuple[str, ...]

    @property
    def merged(self) -> bool:
        return len(self.services) > 1


# The executor first: it loads every block module as it starts, and the
# host starts the next service once that has finished.
MERGED = (
    Group("database-manager", ("database-manager",)),
    Group("swap-credentials", ("swap-credentials",)),
    Group(
        "workers",
        ("executor", "copilot-executor", "scheduler", "batch-executor", "notification"),
    ),
    Group("websocket", ("websocket",)),
    Group("api", ("rest",)),
)


def layout(merged: bool) -> tuple[Group, ...]:
    return MERGED if merged else isolated(MERGED)


def isolated(groups: tuple[Group, ...]) -> tuple[Group, ...]:
    """A host for each service of `groups`, in the appliance's start order."""
    members = {name for group in groups for name in group.services}
    return tuple(Group(s.name, (s.name,)) for s in SERVICES if s.name in members)


def copilot_alone(groups: tuple[Group, ...]) -> tuple[Group, ...]:
    """`groups` with the copilot executor in a process of its own, for when
    that process is given the user's real home directory (`backend_processes`).
    A service that shared it would be in that home too, and the graph
    executor is one that starts the Claude Code CLI itself (the orchestrator
    block), with none of AutoPilot's flags against loading the user's own
    settings, hooks and MCP servers. It costs one more interpreter."""
    apart: list[Group] = []
    for group in groups:
        others = tuple(name for name in group.services if name != COPILOT_EXECUTOR)
        if others == group.services or not others:
            apart.append(group)
        else:
            apart += [
                Group(group.name, others),
                Group(COPILOT_EXECUTOR, (COPILOT_EXECUTOR,)),
            ]
    return tuple(apart)


def service(name: str) -> Service:
    return next(service for service in SERVICES if service.name == name)


def health_urls(port: dict[str, int]) -> dict[str, str]:
    """What answers once the app is up, by service."""
    urls = {s.name: f"http://127.0.0.1:{port[s.port]}{s.health}" for s in SERVICES}
    urls[FRONTEND] = f"http://127.0.0.1:{port[FRONTEND]}/"
    return urls


def health_port(name: str) -> str:
    """The name in ports.PORT_NAMES of the port `name` answers on."""
    return FRONTEND if name == FRONTEND else service(name).port


def backend_processes(
    bundle: Bundle,
    data: DataDir,
    env: dict[str, str],
    groups: tuple[Group, ...],
    cache: Cache,
    copilot_env: dict[str, str] | None = None,
) -> list[ManagedProcess]:
    """`copilot_env`, when given, is the environment of the copilot
    executor's process: the user's real home, for AutoPilot on their own
    Claude Code sign-in (settings.claude_code_host_environment). Every other
    process gets `env`."""
    return [
        host_process(
            bundle, data, environment_of(group, env, copilot_env), group, cache
        )
        for group in groups
    ]


def environment_of(
    group: Group, env: dict[str, str], copilot_env: dict[str, str] | None
) -> dict[str, str]:
    """`copilot_env` only for a process that hosts the copilot executor and
    nothing else (`copilot_alone`): no other service is to run in the user's
    home, whatever layout this is called with."""
    return copilot_env if copilot_env and group.services == (COPILOT_EXECUTOR,) else env


def host_process(
    bundle: Bundle, data: DataDir, env: dict[str, str], group: Group, cache: Cache
) -> ManagedProcess:
    stop_event = StopEvent.create(group.name)
    env = {**base_env(), **env}
    if stop_event:
        env[servicehost.STOP_EVENT_ENV] = stop_event.name
    return ManagedProcess(
        name=group.name,
        argv=host_argv(bundle, group),
        env=env,
        cwd=bundle.backend_dir,
        log_dir=data.logs,
        # Elsewhere the default, SIGTERM to the host alone, is the request.
        graceful_stop=(lambda _: stop_event.set()) if stop_event else None,
        stop_timeout=host_stop_timeout(group),
        before_start=_before_start(group, cache, stop_event),
    )


def host_argv(bundle: Bundle, group: Group) -> list[str]:
    members = [service(name) for name in group.services]
    flags = [
        "--name",
        group.name,
        "--stop-budget",
        f"{host_stop_timeout(group) - 0.5:g}",
    ]
    if any(member.name in API_SERVERS for member in members):
        flags.append("--shared-loop")
    if group.merged and not any(member.database for member in members):
        flags.append("--no-database")
    entries = [f"{member.name}={member.entry}" for member in members]
    return [str(bundle.python), "-c", HOST_CODE, str(PACKAGE_PARENT), *flags, *entries]


def host_stop_timeout(group: Group) -> float:
    return (
        REFRESH_STOP_TIMEOUT_SECONDS
        if REFRESH_SERVICES.intersection(group.services)
        else STOP_TIMEOUT_SECONDS
    )


def _before_start(
    group: Group, cache: Cache, stop_event: StopEvent | None
) -> Callable[[], None]:
    def prepare() -> None:
        if stop_event:
            stop_event.reset()  # a restart must not find the last stop's request
        cache.clear_stale_locks(group.services)

    return prepare


def frontend_process(
    bundle: Bundle, data: DataDir, env: dict[str, str]
) -> ManagedProcess:
    return ManagedProcess(
        name=FRONTEND,
        argv=[*bundle.node_command(), str(bundle.frontend_server)],
        env={**base_env(), **env, "ELECTRON_RUN_AS_NODE": "1"},
        cwd=bundle.frontend_server.parent,
        log_dir=data.logs,
        stop_timeout=STOP_TIMEOUT_SECONDS,
    )


def entry_point_argv(bundle: Bundle, entry: str, *args: str) -> list[str]:
    """A one-shot tool of the backend's (not a service: no host)."""
    module, _, function = entry.partition(":")
    code = f"import sys; from {module} import {function} as m; sys.exit(m())"
    return [str(bundle.python), "-c", code, *args]


def bundle_version(bundle: Bundle) -> str:
    """Changes whenever the bundled backend does: the bundle carries no
    version number of its own. From names and sizes, not contents: reading
    two thousand files takes ten seconds under a virus scanner."""
    import hashlib

    digest = hashlib.sha256(sys.version.encode())
    backend = bundle.backend_dir
    for path in sorted(backend.rglob("*")):
        if path.suffix in (".py", ".prisma", ".sql"):
            digest.update(
                f"{path.relative_to(backend).as_posix()}:{path.stat().st_size};".encode()
            )
    return digest.hexdigest()
