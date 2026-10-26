"""The desktop runtime against the appliance it is a port of.

autogpt_platform/single-container is upstream's; the desktop runtime does by
hand what its scripts do. Each test here reads one appliance file as text and
compares it with what the desktop does, so that an upstream change which
needs a desktop change fails the upstream sync with what changed and where to
port it, and everything else flows through.

The files are read as their statements, not as their layout: comments, blank
lines, indentation and how a command is wrapped over lines make no difference.

Every intentional difference is an entry in a table below, with its reason.
A table entry that no longer matches anything fails too: waivers do not
outlive what they waive.

Where the desktop mirrors a piece by hand and nothing finer can be compared,
the piece's digest is pinned (HAND_PORTED): any change to its statements
fails with "review the diff and re-pin".

Nothing here imports the backend, the proxy or the build: those are read as
text, so the tests run wherever the unit tests do.
"""

from __future__ import annotations

import ast
import configparser
import functools
import hashlib
import importlib.util
import json
import re
import shlex
import sys
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from autogpt_desktop import apps, ports, settings, valkey
from autogpt_desktop.layout import Bundle, DataDir

DESKTOP = Path(__file__).resolve().parents[2]
# Upstream's tree. Only ever read through `upstream()`.
PLATFORM = DESKTOP.parent
RUNTIME = DESKTOP / "runtime" / "autogpt_desktop"
BUILD = DESKTOP / "build"

APPLIANCE = "single-container"
ENTRYPOINT = f"{APPLIANCE}/entrypoint.sh"
COMMON = f"{APPLIANCE}/common.sh"
BOOTSTRAP = f"{APPLIANCE}/bootstrap.sh"
DOCKERFILE = f"{APPLIANCE}/Dockerfile"
SUPERVISORD = f"{APPLIANCE}/supervisor/supervisord.conf"
NGINX = f"{APPLIANCE}/nginx/nginx.conf"
RABBITMQ_CONF = f"{APPLIANCE}/rabbitmq/rabbitmq.conf"
HEALTHCHECK = f"{APPLIANCE}/healthcheck.sh"
RUN_FRONTEND = f"{APPLIANCE}/run-frontend.sh"
RUN_APP = f"{APPLIANCE}/run-app.sh"
RUN_SERVICE = f"{APPLIANCE}/run-service.sh"
PROMOTE_ADMIN = f"{APPLIANCE}/promote-admin.sh"
BACKEND_DOCKERFILE = "backend/Dockerfile"
PYPROJECT = "backend/pyproject.toml"
PACKAGE_JSON = "frontend/package.json"
THIS = "desktop/runtime/tests/test_appliance_parity.py"

NAME = r"[A-Z][A-Z0-9_]*"


# --- reading -----------------------------------------------------------------


def upstream(relative: str) -> str:
    """An upstream file as text. A Windows checkout has CRLF in every one of
    them that .gitattributes does not force to LF."""
    path = PLATFORM / relative
    assert path.is_file(), (
        f"autogpt_platform/{relative} is gone or was renamed upstream. Find where it went and "
        f"update {THIS} and whatever in desktop/ ports it."
    )
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def desktop(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def drift(relative: str, change: str, action: str) -> str:
    return f"autogpt_platform/{relative}: {change}. {action}"


def statements(text: str) -> list[str]:
    """The statements of a shell script, a Dockerfile or an nginx or
    RabbitMQ configuration: comment lines and blank lines dropped (a
    Dockerfile allows a comment inside a continued instruction), continued
    lines joined, trailing comments cut and whitespace collapsed."""
    kept = [line for line in text.split("\n") if not line.lstrip().startswith("#")]
    joined = "\n".join(kept).replace("\\\n", " ")
    lines = (" ".join(_without_comment(line).split()) for line in joined.split("\n"))
    return [line for line in lines if line]


def _without_comment(line: str) -> str:
    for match in re.finditer(r"\s#", line):
        before = line[: match.start()]
        if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
            return before
    return line


def python_statements(text: str) -> list[str]:
    lines = (line.rstrip() for line in text.split("\n"))
    return [line for line in lines if line and not line.lstrip().startswith("#")]


def shell_function(relative: str, name: str) -> list[str]:
    """The statements of a function of a script."""
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", upstream(relative), re.S | re.M)
    assert match, drift(
        relative,
        f"the shell function {name}() is gone or no longer starts at column 0",
        f"Find where its work went and update {THIS}.",
    )
    return statements(match.group(1))


def words(relative: str, text: str) -> list[str]:
    """`text` split as a shell splits it."""
    try:
        return shlex.split(text)
    except ValueError as exc:
        raise AssertionError(
            drift(
                relative, f"cannot read `{text}` ({exc})", f"Teach {THIS} the new form."
            )
        ) from exc


def desktop_has(symbol: str) -> bool:
    """Whether `module.name` or `module.Class.method` is defined in the
    runtime package, without importing it."""
    module, *names = symbol.split(".")
    body = ast.parse(desktop(RUNTIME / f"{module}.py")).body
    for name in names:
        found = next(
            (
                node
                for node in body
                if isinstance(
                    node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                )
                and node.name == name
            ),
            None,
        )
        if found is None:
            return False
        body = found.body
    return True


def missing_symbols(table: dict[str, str]) -> dict[str, str]:
    return {name: symbol for name, symbol in table.items() if not desktop_has(symbol)}


def build_artifacts() -> ModuleType:
    """build/artifacts.py: a module of a script directory, not of a package."""
    spec = importlib.util.spec_from_file_location(
        "desktop_parity_artifacts", BUILD / "artifacts.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


# --- the desktop's environment, with inputs that can be told apart -------------

PORT = {name: 21001 + index for index, name in enumerate(ports.PORT_NAMES)}
PUBLIC_URL = f"http://127.0.0.1:{PORT['public']}"


class Secrets(dict[str, str]):
    """Every secret exists, and the ones asked for are remembered."""

    def __missing__(self, key: str) -> str:
        self[key] = f"secret-of-{key.lower().replace('_', '-')}"
        return self[key]


@functools.cache
def complete_bundle() -> Bundle:
    """A bundle with the files backend_environment looks for before it names
    them (ffmpeg): the environment of a complete install is what is compared."""
    directory = tempfile.TemporaryDirectory(prefix="parity-bundle-")
    complete_bundle.keep = directory  # removed when the interpreter exits
    bundle = Bundle(Path(directory.name))
    bundle.ffmpeg.parent.mkdir(parents=True)
    bundle.ffmpeg.touch()
    return bundle


def environments(
    user: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str], Secrets]:
    data, secret = DataDir(Path("data")), Secrets()
    backend = settings.backend_environment(
        complete_bundle(), data, PORT, secret, user or {}
    )
    frontend = settings.frontend_environment(backend, PORT, secret, data)
    return backend, frontend, secret


# The appliance owns its network namespace and fixes these; the desktop
# allocates each (ports.py).
APPLIANCE_PORTS = {
    "5432": "postgres",
    "17000": "valkey",
    "5672": "rabbitmq",
    "3001": "frontend",
}

# common.sh AUTOGPT_<key>_PORT -> the name in ports.PORT_NAMES.
PORT_KEYS = {
    "WEBSOCKET": "websocket",
    "EXECUTION_MANAGER": "execution_manager",
    "EXECUTION_SCHEDULER": "execution_scheduler",
    "DATABASE_API": "database_api",
    "SWAP_CREDENTIAL_SERVICE": "swap_credential",
    "AGENT_API": "agent_api",
    "NOTIFICATION_SERVICE": "notification",
    "COPILOT_EXECUTOR": "copilot_executor",
    "PLATFORM_LINKING_SERVICE": "platform_linking",
    "COPILOT_CHAT_BRIDGE": "copilot_chat_bridge",
    "BATCH_EXECUTOR": "batch_executor",
}


def localise(value: str) -> str:
    """An appliance value with its fixed ports replaced by the desktop's."""
    return re.sub(
        r"(?<![0-9])(" + "|".join(APPLIANCE_PORTS) + r")(?![0-9])",
        lambda match: str(PORT[APPLIANCE_PORTS[match.group(1)]]),
        value,
    )


# --- 0. the appliance's files ----------------------------------------------------

# Every file of single-container/ (its licences and its own tests apart), and
# what holds the desktop to it. A new file is new behaviour until shown
# otherwise.
APPLIANCE_FILES = {
    ".env.example": "documentation of the operator's settings; the scripts are what is compared",
    ".trivyignore.yaml": "the image's vulnerability scan",
    "Dockerfile": "sections 8 and 9",
    "Dockerfile.dockerignore": "the image build",
    "README.md": "documentation",
    "docker-bake.hcl": "the image build",
    "bootstrap.sh": "section 5; the build takes the frontend role's SQL out of it",
    "common.sh": "sections 2 and 7",
    "disabled-service.sh": "stands in for a chat-bot service that is switched off: not ported",
    "entrypoint.sh": "sections 2, 5 and 6",
    "fatal_listener.py": "HAND_PORTED",
    "healthcheck.sh": "section 7",
    "nginx/nginx.conf": "section 4",
    "probe.py": "HAND_PORTED",
    "promote-admin.sh": "test_the_owner_is_made_admin_where_promote_admin_does_it",
    "python/sitecustomize.py": "copied into the bundle as it is (COPIED_VERBATIM)",
    "rabbitmq/rabbitmq.conf": "section 6",
    "run-app.sh": "HAND_PORTED, and its exports in section 2",
    "run-frontend.sh": "section 3",
    "run-optional-app.sh": "starts the chat-bot services: not ported; its exports in section 2",
    "run-service.sh": "HAND_PORTED",
    "runtime_config.py": "copied into the bundle as it is (COPIED_VERBATIM)",
    "supervisor/supervisord.conf": "section 1",
    "watchdog.sh": "HAND_PORTED",
}
NOT_COMPARED = ("licenses", "tests", "__pycache__")


def test_every_file_of_the_appliance_is_accounted_for():
    root = PLATFORM / APPLIANCE
    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not set(path.relative_to(root).parts) & set(NOT_COMPARED)
    }
    new = present - set(APPLIANCE_FILES)
    assert not new, drift(
        APPLIANCE,
        f"new file(s) {sorted(new)}",
        "See what each does in the appliance. Port it to the desktop runtime and cover it with "
        f"a test here, or say why not; either way name it in APPLIANCE_FILES in {THIS}.",
    )
    gone = set(APPLIANCE_FILES) - present
    assert not gone, drift(
        APPLIANCE,
        f"{sorted(gone)} are gone or were renamed",
        f"Find where their work went, update what in desktop/ ports it, and APPLIANCE_FILES in {THIS}.",
    )


# --- 1. services ---------------------------------------------------------------

# Programs of supervisord.conf that are not backend services, and what does
# their work on the desktop.
REPLACED = {
    "bootstrap": "supervisor.Stack.migrate",
    "next": "apps.frontend_process",
    "nginx": "proxy.build_app",
    "watchdog": "supervisor.Stack.watch",
    "postgres": "postgres.process",
    "valkey-0": "valkey.process",
    "rabbitmq": "rabbitmq.process",
    "fatal-exit": "supervisor.Stack.watch",  # the event listener
}
NOT_RUN = {
    "valkey-1": "one node owns every slot; three only mirror production (valkey.py)",
    "valkey-2": "one node owns every slot; three only mirror production (valkey.py)",
    "falkordb": "a Linux Redis module under the SSPL: not shipped, Graphiti memory is off",
    "copilot-bot": "opt-in chat-bot bridge (AUTOGPT_ENABLE_BOT_SERVICES), not ported",
    "platform-linking-manager": "opt-in chat-bot bridge (AUTOGPT_ENABLE_BOT_SERVICES), not ported",
}
# What a program is started with besides its command: `environment=` and the
# NAME=value words of an `env` in front of the command. A program that is not
# here is started with nothing, which is what every backend service is; they
# get their environment from entrypoint.sh (section 2). RabbitMQ's is compared
# name by name in section 6.
PROGRAM_ENVIRONMENT = {
    # The PATH of a cleared environment. On the desktop every bundled server
    # is started by its full path, with process.base_env.
    "fatal-exit": {"PATH"},
    # LANG: process.base_env keeps the user's. PGDATA: postgres.process passes -D.
    "postgres": {"PATH", "LANG", "PGDATA"},
    "valkey-0": {"PATH", "LANG"},
    "valkey-1": {"PATH", "LANG"},
    "valkey-2": {"PATH", "LANG"},
    "falkordb": {"PATH", "LANG", "AUTOGPT_RUNTIME_DIR"},
    # Where nginx may write in the container: the proxy writes nothing.
    "nginx": {"PATH", "AUTOGPT_HOME", "AUTOGPT_CACHE_DIR", "AUTOGPT_RUNTIME_DIR"},
}
# The stop tier of the data stores (`[group:state]`): supervisor.Stack.stop
# stops the same ones last.
STATE_TIER = {"postgres", "valkey-0", "valkey-1", "valkey-2", "rabbitmq", "falkordb"}
# `stopwaitsecs` of every service: what apps.STOP_TIMEOUT_SECONDS was chosen
# against (a service is done within a second, then sits in telemetry teardown).
SERVICE_STOP_WAIT = "1"


@dataclass(frozen=True)
class Program:
    environment: dict[str, str]
    executable: str  # without its directory
    arguments: tuple[str, ...]
    options: dict[str, str]


def supervisor_conf() -> configparser.ConfigParser:
    conf = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=(";",))
    try:
        conf.read_string(upstream(SUPERVISORD))
    except configparser.Error as exc:
        raise AssertionError(
            drift(
                SUPERVISORD, f"cannot be read as an ini file ({exc})", f"Update {THIS}."
            )
        ) from exc
    return conf


def supervisor_programs() -> dict[str, Program]:
    conf = supervisor_conf()
    programs = {}
    for section in conf.sections():
        if not section.startswith(("program:", "eventlistener:")):
            continue
        options = dict(conf[section])
        tokens = words(SUPERVISORD, options.get("command", ""))
        environment = dict(
            re.findall(rf'({NAME})="?([^",]*)"?', options.get("environment", ""))
        )
        if tokens and tokens[0].rsplit("/", 1)[-1] == "env":
            tokens = tokens[1:]
            while tokens and (
                tokens[0].startswith("-") or re.match(rf"{NAME}=", tokens[0])
            ):
                name, separator, value = tokens.pop(0).partition("=")
                if separator:
                    environment[name] = value
        assert tokens, drift(
            SUPERVISORD, f"[{section}] has no command", f"Update {THIS}."
        )
        programs[section.split(":", 1)[1]] = Program(
            environment, tokens[0].rsplit("/", 1)[-1], tuple(tokens[1:]), options
        )
    return programs


def appliance_services() -> tuple[dict[str, str], set[str], set[str]]:
    """(services as name -> entry point, optional programs, the other programs)."""
    scripts = tomllib.loads(upstream(PYPROJECT))["tool"]["poetry"]["scripts"]
    services: dict[str, str] = {}
    optional: set[str] = set()
    others: set[str] = set()
    for name, program in supervisor_programs().items():
        arguments = program.arguments
        if program.executable == "run-optional-app.sh":
            optional.add(name)
        elif (
            program.executable == "run-app.sh"
            and len(arguments) == 2
            and arguments[1] in scripts
        ):
            services[arguments[0]] = scripts[arguments[1]]
        else:
            others.add(name)
    return services, optional, others


def test_the_services_are_the_ones_supervisord_runs():
    services, optional, others = appliance_services()
    assert services["database-manager"] == "backend.db:main"
    services["database-manager"] = "autogpt_desktop.backend_entries:database_main"
    services["swap-credentials"] = "autogpt_desktop.backend_entries:swap_main"
    ours = {service.name: service.entry for service in apps.SERVICES}
    added = {name: entry for name, entry in services.items() if name not in ours}
    removed = sorted(set(ours) - set(services) - others)
    moved = {
        name: f"appliance {services[name]}, desktop {ours[name]}"
        for name in services.keys() & ours.keys()
        if services[name] != ours[name]
    }
    assert not (added or removed or moved), drift(
        SUPERVISORD,
        "the services it runs through `run-app.sh <name> <script>` (the script resolved through "
        f"[tool.poetry.scripts] of {PYPROJECT}) differ from the desktop's: new {added}, gone "
        f"{removed}, another entry point {moved}",
        "Update SERVICES in desktop/runtime/autogpt_desktop/apps.py (and a group of MERGED; a "
        "new service also needs its port in ports.PORT_NAMES and settings._service_addresses).",
    )
    unknown = (optional | others) - set(REPLACED) - set(NOT_RUN)
    assert not unknown, drift(
        SUPERVISORD,
        f"program(s) {sorted(unknown)} are new, or are no longer started as `run-app.sh <name> "
        "<script>` with nothing else",
        "Port each to the desktop runtime (a service that gained arguments: pass the same in "
        "apps.host_argv) and name what replaces it in REPLACED, or add it to NOT_RUN with the "
        f"reason, in {THIS}.",
    )
    stale = (set(REPLACED) | set(NOT_RUN)) - optional - others
    assert not stale, drift(
        SUPERVISORD,
        f"{sorted(stale)} are no longer programs",
        "Remove them from REPLACED / NOT_RUN and remove what replaced them if it has no other use.",
    )
    assert optional <= set(NOT_RUN), (
        f"{sorted(optional - set(NOT_RUN))} are optional in the appliance; the desktop has no "
        "optional services yet"
    )
    assert not missing_symbols(
        REPLACED
    ), f"REPLACED names desktop code that is gone: {missing_symbols(REPLACED)}"


def test_every_service_is_in_exactly_one_host():
    hosted = [name for group in apps.MERGED for name in group.services]
    assert sorted(hosted) == sorted(
        service.name for service in apps.SERVICES
    ), "every service of apps.SERVICES belongs to exactly one group of apps.MERGED"


def test_programs_are_started_with_the_environment_the_desktop_knows_of():
    programs = supervisor_programs()
    different = {
        name: sorted(set(program.environment) ^ PROGRAM_ENVIRONMENT.get(name, set()))
        for name, program in programs.items()
        if name != "rabbitmq"
        and set(program.environment) != PROGRAM_ENVIRONMENT.get(name, set())
    }
    assert not different, drift(
        SUPERVISORD,
        f"these programs are started with other variables than before: {different}",
        "A backend service's: set the same in backend_environment in "
        "desktop/runtime/autogpt_desktop/settings.py (or for that service alone in "
        "apps.host_process). A bundled server's: in the module that replaces it (REPLACED). "
        f"Then record it in PROGRAM_ENVIRONMENT in {THIS}.",
    )
    assert set(PROGRAM_ENVIRONMENT) <= set(programs), drift(
        SUPERVISORD,
        f"{sorted(set(PROGRAM_ENVIRONMENT) - set(programs))} are no longer programs",
        "Remove them from PROGRAM_ENVIRONMENT.",
    )


def test_the_stop_tiers_are_the_ones_the_desktop_stops_in():
    conf = supervisor_conf()
    state = set(conf.get("group:state", "programs", fallback="").split(","))
    assert state == STATE_TIER, drift(
        SUPERVISORD,
        f"the data stores' stop tier ([group:state]) is now {sorted(state)}",
        "Stop the same ones last in Stack.stop in desktop/runtime/autogpt_desktop/supervisor.py, "
        f"then update STATE_TIER in {THIS}.",
    )
    services, _, _ = appliance_services()
    waits = {
        name: supervisor_programs()[name].options.get("stopwaitsecs")
        for name in services
        if name in supervisor_programs()
    }
    assert set(waits.values()) <= {SERVICE_STOP_WAIT}, drift(
        SUPERVISORD,
        f"the services are now given {waits} seconds to stop (stopwaitsecs)",
        "Read the comment above [group:runtime] for what was measured, reconsider "
        "STOP_TIMEOUT_SECONDS in desktop/runtime/autogpt_desktop/apps.py, then update "
        f"SERVICE_STOP_WAIT in {THIS}.",
    )


# --- 2. the backend's environment ------------------------------------------------

# Exported by the appliance and not set by settings.backend_environment.
ENV_NOT_SET = {
    "AUTOGPT_ENABLE_BOT_SERVICES": "the chat-bot services are not ported",
    "AUTH_ALLOW_NEW_ACCOUNTS": "set once the accounts are counted (supervisor.Stack.secure_owner)",
    "PGDATA": "postgres.py passes the data directory to each tool",
    "POSTGRES_USER": "read by the image's own scripts only",
    "POSTGRES_DB": "read by the image's own scripts only",
    "RABBITMQ_CONFIG_FILE": "the server's alone: rabbitmq.environment",
    "RABBITMQ_MNESIA_BASE": "the server's alone: rabbitmq.environment",
    "RABBITMQ_NODENAME": "the server's alone: rabbitmq.environment",
    "GRAPHITI_FALKORDB_HOST": "no FalkorDB",
    "GRAPHITI_FALKORDB_PORT": "no FalkorDB",
    "CODEX_TEMP_ROOT": "a tmpfs path of the container (/dev/shm); nothing in backend/ reads "
    "the variable (searched 2026-10-03)",
}
# Set by both, to values that differ on purpose.
ENV_DIFFERS = {
    "AUTOGPT_PUBLIC_URL": "the loopback address of the allocated public port",
    "PRISMA_SCHEMA": "a path inside the bundle",
    "PYTHONPATH": "a path inside the bundle",
    "WORKSPACE_STORAGE_DIR": "a path inside the data directory",
}
# What the operator of an appliance may set (`export X="${X:-default}"`,
# normalize_integer, normalize_toggle). Here settings.env is the operator.
# These the desktop fixes instead.
ENV_NOT_OVERRIDABLE = {
    "BEHAVE_AS": "fixed to `local`: the appliance leaves it open to test entitlement gating "
    "of a hosted deployment, which a desktop install never is",
}
# Keys of runtime.env the desktop never reads.
SECRETS_NOT_READ = {
    "AUTOGPT_RUNTIME_CONFIG_VERSION": "the file's own format version",
    "GRAPHITI_FALKORDB_PASSWORD": "no FalkorDB",
}

TEMPLATE = re.compile(rf"\$\{{({NAME})(?::-([^}}]*))?\}}")
EXPORTING = re.compile(r"(?:export|(?:declare|typeset) -\w*x\w*) (.+)")
ASSIGNING = re.compile(rf"(?:(?:readonly|local|declare)(?: -\w+)* )?({NAME})=")
NORMALIZING = re.compile(rf"normalize_(toggle|integer) ({NAME}) (\S+)(?: (\d+) (\d+))?")


def exports(relative: str) -> dict[str, str | None]:
    """NAME -> the value it is exported with (None when the script does not
    say): `export`, `declare -x`, a name assigned earlier and exported by
    name, and what normalize_integer and normalize_toggle export, with their
    defaults."""
    found: dict[str, str | None] = {}
    assigned: dict[str, str | None] = {}
    for line in statements(upstream(relative)):
        if match := NORMALIZING.match(line):
            found[match.group(2)] = match.group(3)
        elif match := EXPORTING.match(line):
            for token in words(relative, match.group(1)):
                name, separator, value = token.partition("=")
                if re.fullmatch(NAME, name):
                    found[name] = (
                        value if separator else assigned.get(name, found.get(name))
                    )
        elif match := ASSIGNING.match(line):
            try:
                token = next(
                    t for t in shlex.split(line) if t.startswith(f"{match.group(1)}=")
                )
                assigned[match.group(1)] = token.partition("=")[2]
            except (
                ValueError,
                StopIteration,
            ):  # a command substitution over several lines
                assigned[match.group(1)] = None
    return found


def appliance_exports() -> dict[str, str | None]:
    """Everything entrypoint.sh and common.sh export to the services. The
    whole files, not one function, so a variable exported from a new function
    is seen too."""
    return {**exports(COMMON), **exports(ENTRYPOINT)}


def normalized() -> dict[str, tuple[str, ...]]:
    """NAME -> (default,) of a toggle or (default, lowest, highest) of an integer."""
    found = {}
    for relative in (COMMON, ENTRYPOINT):
        for line in statements(upstream(relative)):
            if match := NORMALIZING.match(line):
                found[match.group(2)] = tuple(
                    group for group in match.groups()[2:] if group
                )
    return found


def runtime_keys() -> set[str]:
    """The keys of runtime.env: the one `case` arm load_runtime_config accepts."""
    match = re.search(
        r'case "\$\{name\}" in ([A-Z0-9_ |]+)\)',
        " ".join(shell_function(COMMON, "load_runtime_config")),
    )
    assert match, drift(
        COMMON,
        "load_runtime_config no longer lists the keys of runtime.env in one `case` arm",
        f"Update runtime_keys() in {THIS}.",
    )
    return {key.strip() for key in match.group(1).split("|")}


def expand(name: str, value: str, known: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        variable, default = match.group(1), match.group(2)
        if variable == name and default is not None:
            return default
        assert variable in known, drift(
            ENTRYPOINT,
            f"{name} is now built from ${{{variable}}}, which this test cannot translate",
            "Set it the same way in settings.backend_environment and add the variable to "
            "`known` in test_the_backend_environment_has_the_appliances_values.",
        )
        return known[variable]

    return localise(TEMPLATE.sub(replace, value))


def test_every_variable_the_appliance_exports_is_set_or_waived():
    backend, _, _ = environments()
    exported = set(appliance_exports())
    assert len(exported) > 60, drift(
        ENTRYPOINT,
        f"only {len(exported)} exports were found",
        f"Fix exports() in {THIS}.",
    )
    missing = exported - set(backend) - set(ENV_NOT_SET)
    assert not missing, drift(
        ENTRYPOINT,
        f"it (or common.sh) now exports {sorted(missing)}",
        "Set them in backend_environment in desktop/runtime/autogpt_desktop/settings.py, or "
        "add them to ENV_NOT_SET with the reason.",
    )
    stale = set(ENV_NOT_SET) - exported
    assert not stale, drift(
        ENTRYPOINT,
        f"{sorted(stale)} are no longer exported",
        "Remove them from ENV_NOT_SET.",
    )
    now_set = set(ENV_NOT_SET) & set(backend)
    assert (
        not now_set
    ), f"{sorted(now_set)} are set by the desktop now; remove them from ENV_NOT_SET"


def test_the_backend_environment_has_the_appliances_values():
    backend, _, secret = environments()
    exported = appliance_exports()
    known = {
        **{key: secret[key] for key in runtime_keys()},
        **{name: bounds[0] for name, bounds in normalized().items()},
        **{f"AUTOGPT_{key}_PORT": str(PORT[name]) for key, name in PORT_KEYS.items()},
        "AUTOGPT_PUBLIC_URL": PUBLIC_URL,
    }
    waived = set(ENV_NOT_SET) | set(ENV_DIFFERS)
    different = {}
    for name, value in exported.items():
        if value is None or name in waived or name not in backend:
            continue  # a name the desktop lacks is the test above's to report
        expected = expand(name, value, known)
        if backend[name] != expected:
            different[name] = f"appliance {expected!r}, desktop {backend[name]!r}"
    assert not different, drift(
        ENTRYPOINT,
        f"values differ from the desktop's: {different}",
        "Change backend_environment in desktop/runtime/autogpt_desktop/settings.py to match, "
        "or add the name to ENV_DIFFERS with the reason.",
    )
    stale = set(ENV_DIFFERS) - set(exported)
    assert not stale, drift(
        ENTRYPOINT,
        f"{sorted(stale)} are no longer exported",
        "Remove them from ENV_DIFFERS.",
    )


def test_what_the_operator_may_override_the_user_may_override():
    bounds = normalized()
    overridable = {
        name
        for name, value in appliance_exports().items()
        if value
        and any(
            m.group(1) == name and m.group(2) is not None
            for m in TEMPLATE.finditer(value)
        )
    } | set(bounds)
    assert overridable >= set(ENV_NOT_OVERRIDABLE), drift(
        ENTRYPOINT,
        f"{sorted(set(ENV_NOT_OVERRIDABLE) - overridable)} can no longer be overridden there either",
        "Remove them from ENV_NOT_OVERRIDABLE.",
    )
    fixed = set()
    for name in overridable - set(ENV_NOT_OVERRIDABLE) - set(ENV_NOT_SET):
        chosen = "set-by-the-user"
        if len(bounds.get(name, ())) == 3:  # an integer: another one inside its bounds
            default, lowest, highest = bounds[name]
            chosen = lowest if lowest != default else highest
        elif name in bounds:
            chosen = "false" if bounds[name][0] == "true" else "true"
        backend, _, _ = environments({name: chosen})
        if backend.get(name) != chosen:
            fixed.add(name)
    assert not fixed, drift(
        ENTRYPOINT,
        f"an operator can set {sorted(fixed)}, and settings.env cannot",
        "Read them from `user` in backend_environment in "
        "desktop/runtime/autogpt_desktop/settings.py, or add them to ENV_NOT_OVERRIDABLE with "
        "the reason.",
    )


def test_the_database_pool_settings_have_the_appliances_bounds():
    integers = {
        name: tuple(int(bound) for bound in bounds)
        for name, bounds in normalized().items()
        if len(bounds) == 3
    }
    assert integers == settings.DB_SETTINGS, drift(
        ENTRYPOINT,
        f"normalize_integer now bounds {integers}; the desktop has {settings.DB_SETTINGS}",
        "Update DB_SETTINGS in desktop/runtime/autogpt_desktop/settings.py (a setting that is "
        "not the database's needs its own handling in backend_environment).",
    )
    for name, (default, lowest, highest) in settings.DB_SETTINGS.items():
        for refused in (str(lowest - 1), str(highest + 1), "many", "-1", ""):
            backend, _, _ = environments({name: refused} if refused else {})
            assert backend[name] == str(default), (name, refused)
    backend, _, _ = environments({"DB_CONNECT_TIMEOUT": "7", "DB_POOL_TIMEOUT": "9"})
    assert "connect_timeout=7&pool_timeout=9" in backend["DATABASE_URL"]
    assert backend["DIRECT_URL"].endswith("connect_timeout=7")


def test_every_generated_secret_reaches_the_services():
    keys = runtime_keys()
    _, _, secret = environments()
    unread = keys - set(secret) - set(SECRETS_NOT_READ)
    assert not unread, drift(
        COMMON,
        f"runtime.env has the key(s) {sorted(unread)}, which the desktop generates (the same "
        "runtime_config.py) and hands to nothing",
        "Pass them on in backend_environment in desktop/runtime/autogpt_desktop/settings.py, or "
        "add them to SECRETS_NOT_READ with the reason.",
    )
    gone = set(secret) - keys - set(settings.DESKTOP_SECRETS)
    assert not gone, drift(
        COMMON,
        f"runtime.env no longer has {sorted(gone)}, which the desktop's environment is built from",
        "See what replaced them upstream and update backend_environment in "
        "desktop/runtime/autogpt_desktop/settings.py.",
    )
    stale = set(SECRETS_NOT_READ) - keys
    assert not stale, drift(
        COMMON,
        f"{sorted(stale)} left runtime.env",
        "Remove them from SECRETS_NOT_READ.",
    )


# The environment the images give every process: the ENV instructions of the
# appliance's last stage and of the backend stage it is built on
# (server-base). Where upstream puts new defaults for self-hosting. Not set
# on the desktop at all:
IMAGE_ENV_NOT_SET = {
    "DEBIAN_FRONTEND": "for the image's own apt-get",
    "AGENT_BROWSER_EXECUTABLE_PATH": "KNOWN GAP: no browser is bundled for AutoPilot's "
    "browsing tool (README, Known limitations)",
    "ERLANG_INSTALL_PATH_PREFIX": "the rabbitmq image's layout; rabbitmq.environment sets ERLANG_HOME",
    "OPENSSL_INSTALL_PATH_PREFIX": "the rabbitmq image's layout",
    "RABBITMQ_HOME": "the server's alone: rabbitmq.environment",
    "RABBITMQ_DATA_DIR": "the rabbitmq image's layout; rabbitmq.environment sets RABBITMQ_BASE",
    "RUNNING_UNDER_SYSTEMD": "read by the rabbitmq image's launcher only",
    "PGDATA": "postgres.py passes the data directory to each tool",
    "LANG": "the user's own locale is kept; Python is told PYTHONUTF8=1 instead",
    "LANGUAGE": "the user's own locale is kept",
    "LC_ALL": "the user's own locale is kept",
    "PRISMA_BINARY_CACHE_DIR": "the engines are named one by one (PRISMA_*_ENGINE_BINARY)",
}
IMAGE_ENV_DIFFERS = {
    "PATH": "the user's, with the bundle's tools/bin in front (settings._bundled_tools); the "
    "image's adds its servers and the backend's scripts, which the desktop starts by full path",
    "PRISMA_QUERY_ENGINE_BINARY": "a path inside the bundle",
    "AUTOGPT_PUBLIC_URL": "the loopback address of the allocated public port",
}
APPLIANCE_STAGE = "single-container"
BACKEND_STAGE = "server-base"


def dockerfile(relative: str) -> list[tuple[str, str, str]]:
    """(stage, instruction, arguments) of every instruction."""
    found = []
    stage = ""
    for line in statements(upstream(relative)):
        instruction, _, arguments = line.partition(" ")
        if instruction.upper() == "FROM":
            named = re.search(r" AS (\S+)$", arguments, re.I)
            stage = named.group(1) if named else f"stage {len(found)}"
        found.append((stage, instruction.upper(), arguments))
    return found


def stage_of(relative: str, stage: str) -> list[tuple[str, str]]:
    found = [(i, arguments) for s, i, arguments in dockerfile(relative) if s == stage]
    assert found, drift(
        relative,
        f"there is no stage named {stage} any more",
        f"See which stage is the image the services run in and update {THIS}.",
    )
    return found


def stage_env(relative: str, stage: str) -> dict[str, str]:
    environment: dict[str, str] = {}
    for instruction, arguments in stage_of(relative, stage):
        if instruction != "ENV":
            continue
        tokens = words(relative, arguments)
        if all("=" in token for token in tokens):
            environment.update(token.split("=", 1) for token in tokens)
        else:  # the older `ENV NAME value`
            environment[tokens[0]] = " ".join(tokens[1:])
    return environment


def test_the_images_environment_is_set_or_waived():
    last = dockerfile(DOCKERFILE)[-1][0]
    assert last == APPLIANCE_STAGE, drift(
        DOCKERFILE,
        f"its last stage is {last}, not {APPLIANCE_STAGE}",
        f"Update {THIS}.",
    )
    image = {
        **stage_env(BACKEND_DOCKERFILE, BACKEND_STAGE),
        **stage_env(DOCKERFILE, APPLIANCE_STAGE),
    }
    backend, frontend, _ = environments()
    ours = {**frontend, **backend}
    waived = set(IMAGE_ENV_NOT_SET) | set(IMAGE_ENV_DIFFERS)
    wrong = {
        name: f"image {localise(value)!r}, desktop {ours.get(name)!r}"
        for name, value in image.items()
        if name not in waived
        and localise(value) not in (backend.get(name), frontend.get(name))
    }
    assert not wrong, drift(
        f"{DOCKERFILE} (or {BACKEND_DOCKERFILE}, stage {BACKEND_STAGE})",
        f"the image's ENV differs from the desktop's environment: {wrong}",
        "Set them in backend_environment or frontend_environment in "
        "desktop/runtime/autogpt_desktop/settings.py, or add them to IMAGE_ENV_NOT_SET / "
        "IMAGE_ENV_DIFFERS with the reason.",
    )
    stale = waived - set(image)
    assert not stale, drift(
        DOCKERFILE, f"{sorted(stale)} left the image's ENV", "Remove the waivers."
    )
    unset = set(IMAGE_ENV_DIFFERS) - set(ours)
    assert (
        not unset
    ), f"IMAGE_ENV_DIFFERS says the desktop sets {sorted(unset)}; it does not"
    now_set = set(IMAGE_ENV_NOT_SET) & set(ours)
    assert (
        not now_set
    ), f"{sorted(now_set)} are set by the desktop now; move them out of IMAGE_ENV_NOT_SET"


# What the appliance's other scripts export to what they start.
SCRIPT_EXPORTS = {RUN_APP: {"HOME", "XDG_CACHE_HOME"}}


def test_services_get_what_the_launch_scripts_export():
    scripts = sorted(path.name for path in (PLATFORM / APPLIANCE).glob("*.sh"))
    backend, _, _ = environments()
    for script in scripts:
        relative = f"{APPLIANCE}/{script}"
        if relative in (COMMON, ENTRYPOINT):
            continue  # the tests above
        names = set(exports(relative))
        expected = SCRIPT_EXPORTS.get(relative, set())
        assert names == expected, drift(
            relative,
            f"it exports {sorted(names)} to what it starts, not {sorted(expected)}",
            "Set the same in backend_environment in desktop/runtime/autogpt_desktop/settings.py "
            f"(or where the desktop does that script's work) and update SCRIPT_EXPORTS in {THIS}.",
        )
        assert names <= set(backend)


# What an operator can set that the appliance's scripts act on, beyond the
# exports above: every `${NAME:-default}`, `${NAME-default}`, `${NAME:?}` and
# `${NAME:=default}` they read.
OPERATOR_SETTINGS = {
    "AUTH_REQUIRE_EMAIL_VERIFICATION": "refused unless false there; fixed to false here",
    "AUTOGPT_ASSET_DIR": "the image's layout (layout.Bundle)",
    "AUTOGPT_BACKEND_DIR": "the image's layout (layout.Bundle)",
    "AUTOGPT_FRONTEND_DIR": "the image's layout (layout.Bundle)",
    "AUTOGPT_PYTHON": "the image's layout (layout.Bundle)",
    "AUTOGPT_DB_INIT_SQL": "the image's layout (layout.Bundle)",
    "AUTOGPT_CACHE_DIR": "the data directory's layout (layout.DataDir)",
    "AUTOGPT_HOME": "the data directory's layout (layout.DataDir)",
    "AUTOGPT_RUNTIME_DIR": "the data directory's layout (layout.DataDir)",
    "AUTOGPT_RUNTIME_ENV": "the data directory's layout (layout.DataDir)",
    "AUTOGPT_READY_FILE": "readiness is an event to the shell (events.ready), not a file",
    "AUTOGPT_STARTUP_TIMEOUT": "supervisor.APP_READY_TIMEOUT_SECONDS",
    "AUTOGPT_ENABLE_BOT_SERVICES": "the chat-bot services are not ported",
    "AUTOGPT_ENABLE_LEGACY_AUTH": "no install predates Better Auth: both legacy secrets are blank",
    "AUTOGPT_PUBLIC_URL": "the loopback address of the allocated public port",
    "AUTOGPT_PUBLISH_SKILLS": "published in the background once per bundle version; no switch "
    "(supervisor.Stack.tend_skills_catalog)",
    "BEHAVE_AS": ENV_NOT_OVERRIDABLE["BEHAVE_AS"],
    "CHAT_DAILY_COST_LIMIT_MICRODOLLARS": "honoured (settings.env)",
    "CHAT_WEEKLY_COST_LIMIT_MICRODOLLARS": "honoured (settings.env)",
    "VAPID_CLAIM_EMAIL": "honoured (settings.env)",
    "JWT_VERIFY_KEY": "legacy auth only: blank",
    "SUPABASE_JWT_SECRET": "legacy auth only: blank",
    "PGDATA": "postgres.py passes the data directory to each tool",
    "POSTGRES_BINDIR": "the image's layout (layout.Bundle.postgres_bin)",
}


def test_every_setting_the_appliance_reads_is_accounted_for():
    scripts = sorted((PLATFORM / APPLIANCE).glob("*.sh"))
    assert scripts, f"autogpt_platform/{APPLIANCE} has no shell scripts any more"
    read = set()
    for script in scripts:
        read |= set(
            re.findall(
                rf"\$\{{({NAME}):?[-?=+]", upstream(f"{APPLIANCE}/{script.name}")
            )
        )
    unknown = read - set(OPERATOR_SETTINGS)
    assert not unknown, drift(
        f"{APPLIANCE}/*.sh",
        f"the scripts now read the setting(s) {sorted(unknown)}",
        "Honour them in the desktop runtime (settings.env is the operator's file here), or say "
        f"why not, in OPERATOR_SETTINGS in {THIS}.",
    )
    stale = set(OPERATOR_SETTINGS) - read
    assert not stale, drift(
        f"{APPLIANCE}/*.sh",
        f"{sorted(stale)} are no longer read",
        "Remove them from OPERATOR_SETTINGS.",
    )


# --- 3. the frontend's environment -------------------------------------------------

FRONTEND_NOT_PASSED = {
    **{
        f"AUTH_{provider}_CLIENT_{kind}": "no social sign-in: it sends the whole window to the "
        "provider, and the shell keeps the window on the app (src/navigation.js)"
        for provider in ("DISCORD", "GITHUB", "GOOGLE")
        for kind in ("ID", "SECRET")
    },
    "AUTH_CALLBACK_URL": "social sign-in only",
    "SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS": "legacy auth only",
    "SUPABASE_JWT_SECRET": "legacy auth only",
}
# The fixed part of run-frontend.sh's environment the desktop does not set.
FRONTEND_FIXED_NOT_SET = {
    "PATH": "process.base_env keeps the user's",
    "USER": "process.base_env keeps the user's",
    "LOGNAME": "process.base_env keeps the user's",
    "LANG": "process.base_env keeps the user's",
    "LC_ALL": "the user's own locale is kept",
}
FRONTEND_FIXED_DIFFERS = {
    "HOME": "a path inside the data directory",
    "XDG_CACHE_HOME": "a path inside the data directory",
    "DATABASE_URL": "loopback TCP with a password: there is no Unix socket on Windows",
}


def shell_arrays(lines: list[str]) -> list[tuple[str, bool, str]]:
    """(name, whether it is appended to, the words) of every `name=( ... )`."""
    found = re.findall(r"(\w+)(\+?)=\(([^()]*)\)", " ".join(lines))
    return [(name, bool(plus), body) for name, plus, body in found]


def frontend_lists() -> set[str]:
    lists = {
        name: body.split()
        for name, _, body in shell_arrays(statements(upstream(RUN_FRONTEND)))
        if name.endswith("_FRONTEND_ENV")
    }
    assert set(lists) == {"REQUIRED_FRONTEND_ENV", "OPTIONAL_FRONTEND_ENV"}, drift(
        RUN_FRONTEND,
        f"the lists of what Next is passed are now {sorted(lists)}, not REQUIRED_FRONTEND_ENV "
        "and OPTIONAL_FRONTEND_ENV",
        f"Update frontend_lists() in {THIS}.",
    )
    return {name for names in lists.values() for name in names}


def test_the_frontend_is_passed_what_run_frontend_passes():
    passed = frontend_lists()
    ours = set(settings.FRONTEND_PASSTHROUGH)
    assert passed - set(FRONTEND_NOT_PASSED) == ours, drift(
        RUN_FRONTEND,
        f"it passes {sorted(passed - set(FRONTEND_NOT_PASSED) - ours)} that the desktop does not, "
        f"and the desktop passes {sorted(ours - passed)} that it does not",
        "Update FRONTEND_PASSTHROUGH in desktop/runtime/autogpt_desktop/settings.py, or add the "
        "name to FRONTEND_NOT_PASSED with the reason.",
    )
    stale = set(FRONTEND_NOT_PASSED) - passed
    assert not stale, drift(
        RUN_FRONTEND,
        f"{sorted(stale)} are no longer passed",
        "Remove them from FRONTEND_NOT_PASSED.",
    )


def test_the_frontend_gets_the_fixed_environment_run_frontend_builds():
    text = upstream(RUN_FRONTEND)
    blocks = [
        body
        for name, appended, body in shell_arrays(
            shell_function(RUN_FRONTEND, "build_frontend_environment")
        )
        if name == "frontend_env" and not appended
    ]
    assert len(blocks) == 1, drift(
        RUN_FRONTEND,
        "build_frontend_environment no longer fills frontend_env=( ... ) once",
        f"Update {THIS}.",
    )
    fixed = dict(token.partition("=")[::2] for token in words(RUN_FRONTEND, blocks[0]))
    _, frontend, _ = environments()
    waived = set(FRONTEND_FIXED_NOT_SET) | set(FRONTEND_FIXED_DIFFERS)
    wrong = {
        name: f"appliance {localise(value)!r}, desktop {frontend.get(name)!r}"
        for name, value in fixed.items()
        if name not in waived and frontend.get(name) != localise(value)
    }
    assert not wrong, drift(
        RUN_FRONTEND,
        f"the Next server's fixed environment differs: {wrong}",
        "Update frontend_environment in desktop/runtime/autogpt_desktop/settings.py, or add the "
        "name to FRONTEND_FIXED_NOT_SET / FRONTEND_FIXED_DIFFERS with the reason.",
    )
    assert not waived - set(fixed), drift(
        RUN_FRONTEND,
        f"{sorted(waived - set(fixed))} left frontend_env",
        "Remove the waivers.",
    )
    assert set(FRONTEND_FIXED_DIFFERS) <= set(frontend)
    assert f"user={settings.FRONTEND_DB_ROLE}'" in text, drift(
        RUN_FRONTEND,
        f"the Next server no longer connects as {settings.FRONTEND_DB_ROLE}",
        "Update FRONTEND_DB_ROLE in desktop/runtime/autogpt_desktop/settings.py.",
    )
    assert any(
        line.endswith("set -- node /app/frontend/server.js")
        for line in shell_function(RUN_FRONTEND, "main")
    ), drift(
        RUN_FRONTEND,
        "the Next server is no longer `node /app/frontend/server.js` with nothing else",
        "Update apps.frontend_process.",
    )


# --- 4. the proxy ----------------------------------------------------------------

API_REDIRECTS = [
    r"~^(/(?:api|external-api)(?:/.*)?$) $autogpt_public_url/_agpt$1",
    r"~^https?://(?:localhost|127\.0\.0\.1):3001(/.*)$ $autogpt_public_url$1",
    r"~^https?://[^/]+(/(?:api|external-api)(?:/.*)?$) $autogpt_public_url/_agpt$1",
]
SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}
BYTES = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def proxy_source() -> ast.Module:
    return ast.parse(desktop(RUNTIME / "proxy.py"))


def proxy_constant(name: str) -> ast.expr:
    for node in proxy_source().body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == name
        ):
            return node.value
    raise AssertionError(f"proxy.py no longer defines {name}; update {THIS}")


def proxy_pattern(name: str) -> str:
    call = proxy_constant(name)
    assert isinstance(call, ast.Call), f"proxy.{name} is no longer re.compile(...)"
    return ast.literal_eval(call.args[0])


def product(node: ast.expr) -> int:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return product(node.left) * product(node.right)
    return ast.literal_eval(node)


def nginx_http() -> list[str]:
    """The statements of the `http { }` block: what decides how a request is
    answered. What is before it is the process's own (workers, pid, log)."""
    lines = statements(upstream(NGINX))
    assert "http {" in lines and "server {" in lines, drift(
        NGINX, "there is no `server {` in an `http {` block any more", f"Update {THIS}."
    )
    return lines[lines.index("http {") :]


def nginx_locations() -> dict[str, list[str]]:
    """location (modifier and path) -> its statements."""
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in nginx_http():
        if match := re.fullmatch(r"location ((?:[=~*^]+ )?\S+) \{", line):
            current = blocks.setdefault(match.group(1), [])
        elif current is not None:
            current.append(line)
    return blocks


# Compared by value in test_the_proxy_routes_what_nginx_routes, however spelt.
NGINX_COMPARED = (r"proxy_read_timeout .*;", r"client_max_body_size .*;", r"\}")


def nginx_ported() -> list[str]:
    """The http block for its digest: the locations in a fixed order (nginx
    matches exact and prefix locations the same in any order), and without
    what is compared by value."""
    lines = nginx_http()
    first = next(i for i, line in enumerate(lines) if line.startswith("location "))
    ordered = [
        line
        for location, block in sorted(nginx_locations().items())
        for line in (f"location {location}", *block)
    ]
    return quiet([*lines[:first], *ordered], *NGINX_COMPARED)


def directive(lines: list[str], name: str, units: dict[str, int]) -> int | None:
    """An nginx time or size, in seconds or bytes."""
    for line in lines:
        if match := re.fullmatch(rf"{name} (\d+)([a-zA-Z]*);", line):
            unit = match.group(2).lower()
            assert unit in units, drift(
                NGINX, f"`{line}` has a unit this test does not know", f"Update {THIS}."
            )
            return int(match.group(1)) * units[unit]
    return None


def test_the_proxy_routes_what_nginx_routes():
    route = next(
        node
        for node in proxy_source().body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_route"
    )
    literals = {
        node.value
        for node in ast.walk(route)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    timeouts = {
        keywords["api"]: keywords["read_timeout"]
        for node in ast.walk(route)
        if isinstance(node, ast.Call)
        and {"api", "read_timeout"}
        <= (
            keywords := {k.arg: getattr(k.value, "value", None) for k in node.keywords}
        ).keys()
    }
    assert set(timeouts) == {
        True,
        False,
    }, "proxy._route no longer passes read_timeout= and api= to the API and to Next"
    locations = nginx_locations()
    # Exact and prefix locations match the same in any order.
    expected = {
        "= /healthz",
        "= /_agpt/ws",
        "= /_agpt/ws/",
        f"~ {proxy_pattern('PRIVATE_API')}",
        "/_agpt/",
        "/",
    }
    update = "Route the same in _route in desktop/runtime/autogpt_desktop/proxy.py, then update this test."
    assert set(locations) == expected, drift(
        NGINX,
        f"its locations gained {sorted(set(locations) - expected)} and lost "
        f"{sorted(expected - set(locations))} (a changed pattern shows as both)",
        update,
    )
    assert {
        "/healthz",
        "/_agpt/ws",
        "/_agpt/ws/",
        "/_agpt/",
    } <= literals, "proxy._route lost a route"

    def read_timeout(location: str) -> int | None:
        return directive(locations[location], "proxy_read_timeout", SECONDS)

    assert {True: read_timeout("/_agpt/"), False: read_timeout("/")} == timeouts, drift(
        NGINX,
        f"proxy_read_timeout for the API and for Next is {read_timeout('/_agpt/')}s and "
        f"{read_timeout('/')}s; the proxy waits {timeouts[True]}s and {timeouts[False]}s",
        update,
    )
    # The proxy puts no limit on a websocket (heartbeat=None); a day is none.
    assert read_timeout("= /_agpt/ws") == read_timeout("= /_agpt/ws/") == 86400, drift(
        NGINX, "the websocket's read timeout is no longer a day", update
    )
    limit = directive(nginx_http(), "client_max_body_size", BYTES)
    assert limit == product(proxy_constant("MAX_BODY")), drift(
        NGINX,
        f"client_max_body_size is now {limit} bytes",
        "Update MAX_BODY in desktop/runtime/autogpt_desktop/proxy.py.",
    )
    redirects = sorted(
        match.group(1)
        for line in nginx_http()
        if (match := re.fullmatch(r"proxy_redirect (.+);", line))
    )
    assert redirects == API_REDIRECTS, drift(
        NGINX,
        f"its proxy_redirect rules are {redirects}",
        "Port them to _rewrite_location and API_REDIRECT in "
        "desktop/runtime/autogpt_desktop/proxy.py, then update API_REDIRECTS in this test.",
    )


# --- 5. start-up: entrypoint.sh and bootstrap.sh --------------------------------------

ENTRYPOINT_STEPS = {
    "prepare_directories": "layout.DataDir.prepare",
    "load_runtime_config": "settings.ensure_secrets",
    "configure_environment": "settings.backend_environment",
    "write_valkey_configs": "valkey.write_config",
    "initialize_postgres": "postgres.initialize",
    "write_rabbitmq_config": "rabbitmq.prepare",
}
ENTRYPOINT_STEPS_NOT_PORTED = {
    "initialize_backend_config": "nothing writes or reads backend/config.json: the backend's "
    "Settings.save has no caller (test_bundle_integrity.py holds the list of backend files "
    "that resolve paths beside the code)",
    "write_falkordb_config": "no FalkorDB",
}
BOOTSTRAP_STEPS = {
    "load_runtime_config": "settings.ensure_secrets",
    "wait_for_infrastructure": "supervisor.Stack.start_infrastructure",
    "ensure_valkey_cluster": "valkey.ensure_cluster",
    # Signs in as the generated user. That `guest` is absent follows from the
    # same rabbitmq.conf lines (default_user), and is not asked again.
    "verify_rabbitmq_user": "rabbitmq.is_ready",
    "migrate_database": "bootstrap.apply_migrations",
    # In the background, a minute after the app is ready: the appliance blocks
    # the boot on it for up to ten.
    "publish_skills_catalog": "supervisor.Stack.publish_skills_catalog",
    # Its SQL is taken out of bootstrap.sh by the build (frontend_role_sql.py),
    # so a changed grant needs nothing here.
    "configure_frontend_database_role": "bootstrap.configure_frontend_role",
    "publish_readiness": "events.ready",
}
# Statements of bootstrap.sh's main() that are not steps.
NOT_STEPS = (
    r'rm -f "\$\{AUTOGPT_READY_FILE\}"',
    r"log .*",
)


SHELL_WORDS = {"fi", "else", "then", "do", "done", "esac"}


def steps(relative: str) -> list[str]:
    lines = shell_function(relative, "main")
    return [
        line
        for line in lines
        if re.fullmatch(r"[a-z_]+", line) and line not in SHELL_WORDS
    ]


def check_steps(
    relative: str, ported: dict[str, str], not_ported: dict[str, str], where: str
) -> None:
    found = steps(relative)
    expected = {**ported, **not_ported}
    assert set(found) == set(expected), drift(
        relative,
        f"main() gained the step(s) {sorted(set(found) - set(expected))} and lost "
        f"{sorted(set(expected) - set(found))}",
        f"Port a new step to {where} and name what does it in this test, or say why it is not "
        "ported; remove a lost one.",
    )
    assert not missing_symbols(
        ported
    ), f"this test names desktop code that is gone: {missing_symbols(ported)}"


def test_every_entrypoint_step_is_ported():
    """What main() does besides its steps is pinned in HAND_PORTED."""
    check_steps(
        ENTRYPOINT,
        ENTRYPOINT_STEPS,
        ENTRYPOINT_STEPS_NOT_PORTED,
        "supervisor.Stack.start",
    )
    ensure = [
        line
        for line in shell_function(ENTRYPOINT, "main")
        if "runtime_config.py" in line
    ]
    assert len(ensure) == 1 and 'runtime_config.py" ensure --path' in ensure[0], drift(
        ENTRYPOINT,
        f"main() no longer creates the secrets with one `runtime_config.py ensure`: {ensure}",
        "Update ensure_secrets in desktop/runtime/autogpt_desktop/settings.py.",
    )


def test_every_bootstrap_step_is_ported():
    check_steps(BOOTSTRAP, BOOTSTRAP_STEPS, {}, "supervisor.Stack.start")
    unknown = [
        line
        for line in shell_function(BOOTSTRAP, "main")
        if line not in BOOTSTRAP_STEPS
        and not any(re.fullmatch(pattern, line) for pattern in NOT_STEPS)
    ]
    assert not unknown, drift(
        BOOTSTRAP,
        f"main() does something new: {unknown}",
        "Port it, then allow the statement in NOT_STEPS.",
    )


def test_the_database_is_migrated_the_way_bootstrap_does_it():
    """The whole function is pinned in HAND_PORTED; these are the parts the
    desktop repeats word for word."""
    migrate = shell_function(BOOTSTRAP, "migrate_database")
    ours = desktop(RUNTIME / "bootstrap.py")
    update = "Update desktop/runtime/autogpt_desktop/bootstrap.py."
    calls = [line for line in migrate if re.fullmatch(r"[a-z_]+", line)]
    assert calls == ["report_interrupted_migration"], drift(
        BOOTSTRAP, f"migrate_database now calls {calls} besides psql and prisma", update
    )
    assert '--file="${INIT_SQL}"' in " ".join(migrate) and "00-init.sql" in upstream(
        BOOTSTRAP
    ), drift(BOOTSTRAP, "the schemas are no longer created from 00-init.sql", update)
    assert "prisma migrate deploy" in migrate, drift(
        BOOTSTRAP, "migrations are no longer a plain `prisma migrate deploy`", update
    )
    assert re.search(
        r'"migrate",\s*"deploy"', ours
    ), "bootstrap.apply_migrations no longer runs `migrate deploy`"
    interrupted = "WHERE finished_at IS NULL AND rolled_back_at IS NULL"
    assert interrupted in " ".join(
        shell_function(BOOTSTRAP, "report_interrupted_migration")
    ), drift(BOOTSTRAP, "an interrupted migration is recognised differently", update)
    assert interrupted in ours, "bootstrap.refuse_interrupted_migration lost its query"


def test_the_skills_catalog_is_published_with_the_appliances_command():
    scripts = tomllib.loads(upstream(PYPROJECT))["tool"]["poetry"]["scripts"]
    commands = [
        match
        for line in shell_function(BOOTSTRAP, "publish_skills_catalog")
        if (match := re.fullmatch(r"timeout \d+ (\S+)((?: --\S+)*)", line))
    ]
    assert len(commands) == 1, drift(
        BOOTSTRAP,
        "publish_skills_catalog no longer runs one `timeout N <script> --flags`",
        f"Update {THIS}.",
    )
    script, flags = commands[0].group(1), commands[0].group(2).split()
    assert scripts.get(script) == apps.SKILLS_CATALOG_ENTRY, drift(
        PYPROJECT,
        f"the appliance publishes the catalog with `{script}` = {scripts.get(script)!r}",
        "Update SKILLS_CATALOG_ENTRY in desktop/runtime/autogpt_desktop/apps.py.",
    )
    supervisor = desktop(RUNTIME / "supervisor.py")
    missing = [flag for flag in flags if f'"{flag}"' not in supervisor]
    assert flags and not missing, drift(
        BOOTSTRAP,
        f"the catalog is published with {flags}",
        "Pass the same in publish_skills_catalog in desktop/runtime/autogpt_desktop/supervisor.py.",
    )


def test_the_owner_is_made_admin_where_promote_admin_does_it():
    """The desktop makes the first account an admin itself (bootstrap.py);
    promote-admin.sh is upstream's statement of what an admin is."""
    promote = " ".join(statements(upstream(PROMOTE_ADMIN)))
    ours = desktop(RUNTIME / "bootstrap.py")
    update = "Update owner_sql and IDENTITY_TABLE in desktop/runtime/autogpt_desktop/bootstrap.py."
    assert "UPDATE platform.\"UserAuthIdentity\" SET role = 'admin'" in promote, drift(
        PROMOTE_ADMIN,
        "an account is no longer made admin by its role in UserAuthIdentity",
        update,
    )
    assert (
        'IDENTITY_TABLE = "UserAuthIdentity"' in ours and "SET role = 'admin'" in ours
    ), "bootstrap.py no longer sets role = 'admin' in UserAuthIdentity; update this test"


# /data paths the appliance manages that the desktop's data directory lacks.
DATA_PATHS_NOT_KEPT = {
    "valkey/17000": "one node, which keeps its files in valkey/",
    "valkey/17001": "one node",
    "valkey/17002": "one node",
    "falkordb": "no FalkorDB",
}
# What the image redirects into /data with a symlink, so it survives an
# upgrade and is not written beside the code.
REDIRECTS_NOT_MADE = {
    "/data/config/backend.json": "nothing writes or reads backend/config.json, see "
    "initialize_backend_config in ENTRYPOINT_STEPS_NOT_PORTED",
    "/data/cache/next": "the Next server is built to keep no cache on disk (build/next_config.py)",
}


def test_the_data_directory_has_the_appliances_directories():
    managed = set(
        re.findall(
            r"/data/([a-z0-9/-]+)",
            " ".join(shell_function(ENTRYPOINT, "prepare_directories")),
        )
    )
    data = DataDir(Path("data"))
    kept = set()
    for name in dir(DataDir):
        if isinstance(getattr(DataDir, name), property):
            path = getattr(data, name).relative_to("data")
            kept |= {path.as_posix(), *(parent.as_posix() for parent in path.parents)}
    missing = managed - kept - set(DATA_PATHS_NOT_KEPT)
    assert not missing, drift(
        ENTRYPOINT,
        f"prepare_directories now manages /data/{{{', '.join(sorted(missing))}}}",
        "Add the directory to DataDir in desktop/runtime/autogpt_desktop/layout.py and point "
        "whatever uses it there, or add it to DATA_PATHS_NOT_KEPT with the reason.",
    )
    assert not set(DATA_PATHS_NOT_KEPT) - managed, drift(
        ENTRYPOINT,
        f"{sorted(set(DATA_PATHS_NOT_KEPT) - managed)} are no longer managed",
        "Remove the waivers.",
    )


def test_what_the_image_redirects_into_the_data_volume_is_known():
    redirected = set(
        re.findall(
            r"\bln -s[a-z]* (/data/\S+) ", " ".join(statements(upstream(DOCKERFILE)))
        )
    )
    assert redirected == set(REDIRECTS_NOT_MADE), drift(
        DOCKERFILE,
        f"the image now symlinks {sorted(redirected)} into the data volume",
        "Something else the backend or Next writes beside its code: redirect it into the data "
        "directory on the desktop too, or record it in REDIRECTS_NOT_MADE.",
    )
    assert (
        BUILD / "next_config.py"
    ).is_file(), "build/next_config.py is gone; REDIRECTS_NOT_MADE says it is why Next needs no cache"


# --- 6. the bundled services' configuration ------------------------------------------

# Where the desktop differs: the appliance's value the difference was
# accepted for (it fails when that moves), and why.
VALKEY_DIFFERS = {
    "dir": (
        "/data/valkey/%s",
        "relative to the server's cwd: the Windows build cannot read drive letters",
    )
}
RABBITMQ_ENV_NOT_SET = {
    "LANG": "process.base_env keeps the user's",
    "ERLANG_INSTALL_PATH_PREFIX": IMAGE_ENV_NOT_SET["ERLANG_INSTALL_PATH_PREFIX"],
    "OPENSSL_INSTALL_PATH_PREFIX": IMAGE_ENV_NOT_SET["OPENSSL_INSTALL_PATH_PREFIX"],
    "RABBITMQ_DATA_DIR": IMAGE_ENV_NOT_SET["RABBITMQ_DATA_DIR"],
    "RUNNING_UNDER_SYSTEMD": IMAGE_ENV_NOT_SET["RUNNING_UNDER_SYSTEMD"],
}
INITDB_DIFFERS = {
    "--auth-local": ("peer", "no Unix socket: --auth covers every connection"),
    "--auth-host": ("scram-sha-256", "--auth=scram-sha-256 covers every connection"),
    "--locale": (
        "C.UTF-8",
        "C: Windows and macOS have no C.UTF-8 locale; the encoding is still UTF8",
    ),
}
POSTGRES_CONF_DIFFERS = {
    "port": (
        "5432",
        "passed with -p on every start, so a moved port never leaves the file stale",
    ),
    "unix_socket_directories": (
        "'/run/postgresql'",
        "off (-c unix_socket_directories=): loopback TCP on every OS",
    ),
}


def moved_waivers(
    waivers: dict[str, tuple[str, str]], appliance: dict[str, str]
) -> dict:
    """Waivers whose appliance value is not the one they were written for."""
    return {
        name: f"was {was!r}, is {appliance.get(name)!r}"
        for name, (was, _) in waivers.items()
        if appliance.get(name) != was
    }


def test_valkey_is_configured_like_an_appliance_node(tmp_path: Path):
    lines = dict(
        re.findall(
            r"""printf ['"]([a-z-]+) ([^'"]*)\\n['"]""",
            "\n".join(shell_function(ENTRYPOINT, "write_valkey_configs")),
        )
    )
    assert len(lines) >= 10, drift(
        ENTRYPOINT,
        "write_valkey_configs no longer prints one directive per printf",
        f"Update {THIS}.",
    )
    data = DataDir(tmp_path)
    valkey.write_config(data, 17000, 27000, "password")
    ours = dict(
        line.split(" ", 1)
        for line in desktop(data.valkey / valkey.CONFIG_NAME).splitlines()
    )
    wrong = {
        directive: f"appliance {value!r}, desktop {ours.get(directive)!r}"
        for directive, value in lines.items()
        if directive not in VALKEY_DIFFERS
        and (directive not in ours or ("%s" not in value and ours[directive] != value))
    }
    update = (
        "Update write_config in desktop/runtime/autogpt_desktop/valkey.py, or add the directive "
        "to VALKEY_DIFFERS with the appliance's value and the reason."
    )
    assert not wrong, drift(
        ENTRYPOINT,
        f"write_valkey_configs differs from the desktop's valkey.conf: {wrong}",
        update,
    )
    moved = moved_waivers(VALKEY_DIFFERS, lines)
    assert not moved, drift(
        ENTRYPOINT,
        f"directives the desktop sets differently on purpose changed: {moved}",
        update,
    )


def test_rabbitmq_is_configured_like_the_appliances():
    ours = desktop(RUNTIME / "rabbitmq.py")
    conf = statements(upstream(RABBITMQ_CONF))
    written = re.findall(
        r"printf ['\"]([a-z_]+ = [^'\"]*)\\n['\"]",
        "\n".join(shell_function(ENTRYPOINT, "write_rabbitmq_config")),
    )
    lines = [line.replace(":5672", ":{port}") for line in conf]
    lines += [line.split(" = ")[0] + " = " for line in written]
    missing = [line for line in lines if line not in ours]
    assert len(written) >= 3 and not missing, drift(
        f"{RABBITMQ_CONF} (and write_rabbitmq_config in entrypoint.sh)",
        f"the appliance's rabbitmq.conf has {lines}; the desktop's lacks {missing}",
        "Update prepare in desktop/runtime/autogpt_desktop/rabbitmq.py.",
    )
    given = supervisor_programs()["rabbitmq"].environment
    unset = [
        name
        for name in given
        if name not in RABBITMQ_ENV_NOT_SET and f'"{name}"' not in ours
    ]
    assert not unset, drift(
        SUPERVISORD,
        f"RabbitMQ is started with {unset}",
        "Set them in environment in desktop/runtime/autogpt_desktop/rabbitmq.py, or add them to "
        "RABBITMQ_ENV_NOT_SET with the reason.",
    )
    stale = set(RABBITMQ_ENV_NOT_SET) - set(given)
    assert not stale, drift(
        SUPERVISORD,
        f"RabbitMQ is no longer started with {sorted(stale)}",
        "Remove them from RABBITMQ_ENV_NOT_SET.",
    )
    fixed = {
        name: given.get(name) for name in ("RABBITMQ_NODENAME", "ERL_EPMD_ADDRESS")
    }
    assert all(value and f'"{value}"' in ours for value in fixed.values()), drift(
        SUPERVISORD,
        f"RabbitMQ's node name or epmd address changed: {fixed}",
        "Update desktop/runtime/autogpt_desktop/rabbitmq.py.",
    )


def test_postgres_is_initialised_like_the_appliances():
    function = shell_function(ENTRYPOINT, "initialize_postgres")
    ours = desktop(RUNTIME / "postgres.py")
    update = (
        "Update initialize in desktop/runtime/autogpt_desktop/postgres.py, or add it to "
        "INITDB_DIFFERS / POSTGRES_CONF_DIFFERS with the appliance's value and the reason."
    )
    commands = [line for line in function if re.search(r'/initdb" --', line)]
    assert len(commands) == 1, drift(
        ENTRYPOINT,
        "initialize_postgres no longer runs initdb once, with flags",
        f"Update {THIS}.",
    )
    flags = {
        flag: value
        for flag, value in re.findall(
            r" (--[a-z-]+)(?:=(\S+))?", commands[0].split('/initdb"')[1]
        )
    }
    wrong = [
        f"{flag}={value}" if value else flag
        for flag, value in flags.items()
        if flag not in INITDB_DIFFERS
        and (
            f'"{flag}"'
            if not value
            else f"{flag}=" if "$" in value else f"{flag}={value}"
        )
        not in ours
    ]
    assert not wrong, drift(ENTRYPOINT, f"initdb is run with {wrong}", update)
    moved = moved_waivers(INITDB_DIFFERS, flags)
    assert not moved, drift(
        ENTRYPOINT,
        f"initdb flags the desktop sets differently on purpose changed: {moved}",
        update,
    )
    conf = dict(
        line.split(" = ", 1)
        for line in re.findall(
            r'printf "(?:\\n)?([a-z_]+ = [^"]*?)\\n"', "\n".join(function)
        )
    )
    assert len(conf) >= 3, drift(
        ENTRYPOINT,
        "initialize_postgres no longer appends to postgresql.conf with printf",
        f"Update {THIS}.",
    )
    wrong = [
        f"{name} = {value}"
        for name, value in conf.items()
        if name not in POSTGRES_CONF_DIFFERS and f"{name} = {value}" not in ours
    ]
    assert not wrong, drift(ENTRYPOINT, f"postgresql.conf is given {wrong}", update)
    moved = moved_waivers(POSTGRES_CONF_DIFFERS, conf)
    assert not moved, drift(
        ENTRYPOINT,
        f"postgresql.conf settings the desktop sets differently on purpose changed: {moved}",
        update,
    )


# --- pieces the desktop mirrors by hand, with no finer check than a digest -------------


@dataclass(frozen=True)
class Ported:
    relative: str
    lines: Callable[[], list[str]]
    sha256: str
    mirror: str  # what in the desktop does the same


def quiet(lines: list[str], *dropped: str) -> list[str]:
    """Without what is said to the log, and without `dropped` (patterns)."""
    patterns = (r"log .*", *dropped)
    return [line for line in lines if not any(re.fullmatch(p, line) for p in patterns)]


def script(relative: str, *dropped: str) -> Callable[[], list[str]]:
    return lambda: quiet(statements(upstream(relative)), *dropped)


def function(relative: str, name: str, *dropped: str) -> Callable[[], list[str]]:
    return lambda: quiet(shell_function(relative, name), *dropped)


def python(relative: str) -> Callable[[], list[str]]:
    return lambda: python_statements(upstream(relative))


# Digests of the statements (see `statements`): a comment, a blank line, a
# log message or a reflowed command changes nothing.
HAND_PORTED = {
    "run-service.sh": Ported(
        RUN_SERVICE,
        # The FalkorDB arm starts a server the desktop does not ship.
        script(RUN_SERVICE, r"falkordb\)", r"exec /opt/falkordb/.*"),
        "aa7f27a7b44f263588f233f64d6bd6eb97cd5b6427e82e7d9fe97bb8bef2db4f",
        "how each bundled server is started: postgres.process, valkey.process, rabbitmq.process",
    ),
    "run-app.sh": Ported(
        RUN_APP,
        script(RUN_APP),
        "1088e5d000cabeb103411736b175a0946c7bd8559efa27e460fde9b67368d97c",
        "how each service is started: apps.host_process (there is no ready file to wait for: "
        "supervisor.Stack.start_apps runs once the migrations are done)",
    ),
    "rabbitmq.conf": Ported(
        RABBITMQ_CONF,
        script(RABBITMQ_CONF),
        "1f19da9f442afece9eb39120e0f861a046ca4b651955fc544b2ef5242297d013",
        "rabbitmq.prepare writes the same lines",
    ),
    "nginx.conf, the http block": Ported(
        NGINX,
        lambda: nginx_ported(),
        "c903e02a2cdac656b44de081cef29de437fd41e9cd128291dd51069eb01478c5",
        "proxy.py: the forwarded headers, buffering, upstreams' paths and redirects are ported "
        "by hand; a header added for every response belongs in proxy._http",
    ),
    "entrypoint.sh main()": Ported(
        ENTRYPOINT,
        function(ENTRYPOINT, "main"),
        "d901f1e66b03e1321a42d960f0d072d7ec9bdde6493a49e9d2787b116035a608",
        "supervisor.Stack.start (the steps themselves are named in ENTRYPOINT_STEPS)",
    ),
    "entrypoint.sh configure_environment() apart from its exports": Ported(
        ENTRYPOINT,
        function(ENTRYPOINT, "configure_environment", r"export .*"),
        "6369ea557245c4e95058cf90ddbe71a68d03e56f7d3e52363b0c41973713f171",
        "settings.backend_environment: what it checks and normalises, and the helpers it calls",
    ),
    "bootstrap.sh wait_for_infrastructure()": Ported(
        BOOTSTRAP,
        function(BOOTSTRAP, "wait_for_infrastructure"),
        "0f3624ecec5e00749618eacd0fc575f5f264b7fc1d073806e02b90a3b4c9519a",
        "supervisor.Stack.start_infrastructure: which servers are waited for, how, and how long",
    ),
    "bootstrap.sh ensure_valkey_cluster()": Ported(
        BOOTSTRAP,
        function(BOOTSTRAP, "ensure_valkey_cluster"),
        "9af95a0126e09dd30315a2996339bf43207d2e956ce40af87721d3024453b715",
        "valkey.ensure_cluster",
    ),
    "bootstrap.sh verify_rabbitmq_user()": Ported(
        BOOTSTRAP,
        function(BOOTSTRAP, "verify_rabbitmq_user"),
        "afeb480f4970471ffc8c8630fde628e079d2f32f5725113b594df393253baa61",
        "rabbitmq.is_ready signs in as the generated user",
    ),
    "bootstrap.sh migrate_database()": Ported(
        BOOTSTRAP,
        function(BOOTSTRAP, "migrate_database"),
        "727850352057fee4c47e4214d90d0d8213ebdd4cea0fdfe3a3e3b231e9fb6af1",
        "supervisor.Stack.migrate: bootstrap.create_schemas, refuse_interrupted_migration, "
        "apply_migrations",
    ),
    "healthcheck.sh main()": Ported(
        HEALTHCHECK,
        function(HEALTHCHECK, "main"),
        "795b11775bfd4d0c440c420ff8060542efe8bf4a30f53a985b2103381e5509a1",
        "what counts as healthy: supervisor.Stack.wait_for_apps and Stack.watch (the checks "
        "themselves are compared in section 7)",
    ),
    "watchdog.sh": Ported(
        f"{APPLIANCE}/watchdog.sh",
        script(f"{APPLIANCE}/watchdog.sh"),
        "7ff4c2df3fc0e94fbc9936f2ec23092b338786b8680b2bb07264049f76286c13",
        "when a running appliance is restarted: supervisor.Stack.watch and may_restart",
    ),
    "fatal_listener.py": Ported(
        f"{APPLIANCE}/fatal_listener.py",
        python(f"{APPLIANCE}/fatal_listener.py"),
        "321367ebe25f3723d226d064f976004fc0f0eb09ca72ad6252e46b0ee38ea6f6",
        "which program's death ends the appliance: supervisor.Stack.watch",
    ),
    "probe.py": Ported(
        f"{APPLIANCE}/probe.py",
        python(f"{APPLIANCE}/probe.py"),
        "68263319a92f046ac49a74237f083489dab1053c64b55ab808d1a0e7bcfa46a1",
        "what `ready` means for each server: postgres.is_ready, valkey.is_ready and "
        "ensure_cluster, rabbitmq.is_ready, supervisor._http_ok",
    ),
}


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


@pytest.mark.parametrize("piece", HAND_PORTED)
def test_a_hand_ported_piece_is_the_one_that_was_ported(piece: str):
    ported = HAND_PORTED[piece]
    found = digest(ported.lines())
    assert found == ported.sha256, drift(
        ported.relative,
        f"{piece} changed, and the desktop mirrors it by hand ({ported.mirror})",
        f"Review the diff (git log -p -- autogpt_platform/{ported.relative}), port what matters "
        f'and re-pin: "{found}" for "{piece}" in HAND_PORTED in {THIS}.',
    )


# --- 7. health ---------------------------------------------------------------------

# Where the appliance asks a service and the desktop asks elsewhere.
HEALTH_DIFFERS = {
    "execution_manager": "the executor's only HTTP server is its Prometheus one, which "
    "answers /metrics (and, as it happens, any path)",
    "copilot_executor": "as the executor",
}
# Asked only when the chat-bot services are switched on.
HEALTH_NOT_ASKED = {
    "copilot_chat_bridge": NOT_RUN["copilot-bot"],
    "platform_linking": NOT_RUN["platform-linking-manager"],
}
INFRASTRUCTURE_PROBES = {
    "pg_isready": "postgres.is_ready",
    "redis --port 17000 --cluster": "valkey.ensure_cluster",
    "amqp --port 5672": "rabbitmq.is_ready",
}
INFRASTRUCTURE_NOT_PROBED = {"redis --port 6380": "no FalkorDB"}
PROBED = re.compile(
    r"http://127\.0\.0\.1:(?:\$\{AUTOGPT_([A-Z_]+)_PORT\}|(\d+))"
    r"(\$\{AUTOGPT_INTERNAL_HEALTH_PATH\}|/[\w/]*)"
)


def common_ports() -> dict[str, str]:
    return dict(
        match.groups()
        for line in statements(upstream(COMMON))
        if (
            match := re.fullmatch(
                r"(?:readonly|declare -r) AUTOGPT_([A-Z_]+)_PORT=(\d+)", line
            )
        )
    )


def test_every_fixed_port_of_the_appliance_is_allocated():
    fixed = common_ports()
    assert fixed.keys() == PORT_KEYS.keys(), drift(
        COMMON,
        f"its service ports are now {sorted(fixed)}",
        "Add a new one to PORT_NAMES in desktop/runtime/autogpt_desktop/ports.py, to "
        "_service_addresses in settings.py and to PORT_KEYS in this test; remove a lost one.",
    )
    assert set(PORT_KEYS.values()) <= set(
        ports.PORT_NAMES
    ), "PORT_KEYS names a port ports.PORT_NAMES lacks"
    assert not set(common_ports().values()) & set(
        APPLIANCE_PORTS
    ), "a service port collides with APPLIANCE_PORTS"


def probed() -> tuple[dict[str, str], set[str]]:
    """(port name -> the path asked, the ports asked only when the optional
    services run): the URL lists of check_application_services."""
    internal = [
        match.group(1)
        for line in statements(upstream(COMMON))
        if (match := re.fullmatch(r"readonly AUTOGPT_INTERNAL_HEALTH_PATH=(\S+)", line))
    ]
    lists = [
        (appended, words(HEALTHCHECK, body))
        for _, appended, body in shell_arrays(
            shell_function(HEALTHCHECK, "check_application_services")
        )
    ]
    lists = [
        (appended, urls)
        for appended, urls in lists
        if urls and urls[0].startswith("http")
    ]
    assert len(internal) == 1 and len(lists) == 2, drift(
        HEALTHCHECK,
        "check_application_services no longer builds one list of URLs and appends the optional "
        "services' to it",
        f"Update probed() in {THIS}.",
    )
    asked: dict[str, str] = {}
    always: set[str] = set()
    for appended, urls in lists:
        for url in urls:
            match = PROBED.fullmatch(url)
            port = match and (
                PORT_KEYS.get(match.group(1))
                if match.group(1)
                else {"3001": apps.FRONTEND, "3000": "public"}.get(match.group(2))
            )
            assert match and port, drift(
                HEALTHCHECK,
                f"it probes {url}, which is no port this test knows",
                f"See test_every_fixed_port_of_the_appliance_is_allocated, then update {THIS}.",
            )
            path = internal[0] if match.group(3).startswith("$") else match.group(3)
            if not appended:
                always.add(port)
            asked[port] = asked[port] if appended and port in always else path
    return asked, set(asked) - always


def test_the_services_are_asked_where_the_healthcheck_asks():
    asked, optional = probed()
    internal = asked.get("database_api")
    ours = {service.port: service.health for service in apps.SERVICES}
    ours[apps.FRONTEND] = "/"
    ours["public"] = "/healthz"  # answered by the proxy itself (proxy._route)
    update = (
        "Update the port and health path of the service in SERVICES in "
        "desktop/runtime/autogpt_desktop/apps.py, or add it to HEALTH_DIFFERS / HEALTH_NOT_ASKED "
        "with the reason."
    )
    stale = {
        port: reason
        for port, reason in HEALTH_DIFFERS.items()
        if asked.get(port) != internal or ours.get(port) == asked.get(port)
    }
    assert not stale, drift(
        HEALTHCHECK,
        f"these are no longer asked at {internal} while the desktop asks elsewhere: {stale}",
        "Ask the desktop's service where the appliance now asks it, and remove it from "
        "HEALTH_DIFFERS.",
    )
    ours.update({port: asked[port] for port in HEALTH_DIFFERS})
    expected = {
        port: path for port, path in asked.items() if port not in HEALTH_NOT_ASKED
    }
    expected["swap_credential"] = "/health_check"
    assert expected == ours, drift(
        HEALTHCHECK, f"it probes {expected}; the desktop waits for {ours}", update
    )
    assert set(HEALTH_NOT_ASKED) == optional, drift(
        HEALTHCHECK,
        f"the services probed only when switched on are now {sorted(optional)}, not "
        f"{sorted(HEALTH_NOT_ASKED)}",
        "One that is now always probed is now always run: port it (section 1). "
        + update,
    )


def test_the_bundled_servers_are_probed_like_the_healthcheck_probes_them():
    probes = re.findall(
        r'"\$\{POSTGRES_BINDIR\}/(pg_isready)|"\$\{PROBE\[@\]\}" (\w+ --port \d+(?: --cluster)?)',
        "\n".join(shell_function(HEALTHCHECK, "check_infrastructure")),
    )
    found = {a or b for a, b in probes}
    expected = set(INFRASTRUCTURE_PROBES) | set(INFRASTRUCTURE_NOT_PROBED)
    assert found == expected, drift(
        HEALTHCHECK,
        f"check_infrastructure probes {sorted(found)}",
        "A new bundled server: port it and its readiness probe to the desktop runtime, then "
        "update INFRASTRUCTURE_PROBES in this test.",
    )
    assert not missing_symbols(INFRASTRUCTURE_PROBES)


# --- 8. versions and what the images install -----------------------------------------

BASE_IMAGES = {
    "node": "build/artifacts.py NODE_VERSION",
    "rabbitmq": "build/artifacts.py RABBITMQ_VERSION and ERLANG_VERSION",
    "falkordb/falkordb-server": "not shipped (SSPL)",
    "autogpt-backend": "the backend itself: build_runtime.step_backend",
    "autogpt-backend-base": "the backend's dependencies: build_runtime.step_deps",
}
# The appliance runs PostgreSQL 15 from PGDG. The desktop takes a prebuilt
# PostgreSQL 18 on Windows and macOS (POSTGRES_BUNDLE) and builds 16 on Linux
# (POSTGRES_SOURCE_VERSION): the newest that fit each platform when it was
# chosen, and a data directory can never move to another major, so the
# desktop's majors do not follow the appliance's. Upstream's migrations are
# written for this one; never ship an older server than it.
APPLIANCE_POSTGRES_MAJOR = "15"
APPLIANCE_PACKAGES = {
    "ca-certificates": "the OS's, and certifi in the bundle",
    "curl": "build-time only",
    "gnupg": "build-time only",
    "postgresql-common": "Debian's packaging",
    "bash": "the appliance's scripts; the desktop's are Python",
    "libgomp1": "an OpenMP runtime for native code in the image; which code is not recorded "
    "upstream, and the desktop bundle has not needed one",
    "netcat-openbsd": "the appliance's scripts",
    "nginx": "proxy.py",
    "openssl": "the appliance's scripts",
    "postgresql-15": "artifacts: postgres",
    "postgresql-client-15": "artifacts: postgres",
    "postgresql-15-pgvector": "artifacts: postgres (in the bundle), pgvector (built on Linux)",
    "procps": "the appliance's scripts",
    "supervisor": "supervisor.py",
    "tini": "the shell is the parent process",
    "util-linux": "the appliance's scripts (setpriv, runuser)",
    "valkey-server": "artifacts: valkey",
    "valkey-tools": "the cluster is formed over the redis client (valkey.ensure_cluster)",
}
# What the backend's own image installs for the services to call.
BACKEND_PACKAGES = {
    "python3.13": "artifacts: python",
    "python3-pip": "build-time only",
    "ffmpeg": "bundled as tools/bin/ffmpeg (build/bundled_tools.py, layout.Bundle.ffmpeg)",
    "imagemagick": "KNOWN GAP: not bundled (README, Known limitations)",
    "jq": "KNOWN GAP: a CLI tool for AutoPilot's shell, which is itself unavailable",
    "ripgrep": "KNOWN GAP: a CLI tool for AutoPilot's shell, which is itself unavailable",
    "tree": "KNOWN GAP: a CLI tool for AutoPilot's shell, which is itself unavailable",
    "bubblewrap": "KNOWN GAP: Linux only; AutoPilot's sandboxed shell is unavailable (README, Known limitations)",
    "libatomic1": "for the Node that Debian's Prisma CLI runs on; the desktop runs the "
    "Prisma CLI on its own Node",
    "chromium": "KNOWN GAP: no browser for AutoPilot's browsing tool (README, Known limitations)",
    "fonts-liberation": "chromium's",
}
# Debian packages the images pin to a version (`name=version`): the version,
# and what the desktop's own pin was checked against it.
PINNED_PACKAGES: dict[str, tuple[str, str]] = {}
# `npm install -g` in the backend's image: the version, and why it is not in
# the bundle. There is no pin of the desktop's to compare, so a new version
# is a re-pin here, after reading what changed for the day it is bundled.
BACKEND_NPM = {
    "agent-browser": ("0.37.1", "KNOWN GAP: not bundled, see chromium"),
}
# `pip install` in the backend's image, outside the locked dependencies.
BACKEND_PIP: dict[str, str] = {}
# What the desktop bundles of the above, by where the runtime finds it.
BUNDLED_TOOLS = {"ffmpeg": "layout.Bundle.ffmpeg"}

# (what it is recorded under, the program, what parts a package from its version)
INSTALLERS = (
    ("apt", "apt-get", "="),
    ("npm", "npm", "@"),
    ("pip", "pip", "[=<>~!]"),
    ("pip", "pip3", "[=<>~!]"),
)


def installed(relative: str, stage: str) -> dict[str, dict[str, str | None]]:
    """installer -> package -> its version (None when unpinned), for every
    RUN of the stage."""
    found: dict[str, dict[str, str | None]] = {"apt": {}, "npm": {}, "pip": {}}
    for instruction, arguments in stage_of(relative, stage):
        if instruction != "RUN":
            continue
        for command in re.split(r"&&|\|\||[;|]", arguments):
            tokens = [token.strip("\"'") for token in command.split()]
            for installer, program, separator in INSTALLERS:
                if (
                    program not in tokens
                    or "install" not in tokens[tokens.index(program) :]
                ):
                    continue
                for package in tokens[
                    tokens.index("install", tokens.index(program)) + 1 :
                ]:
                    if package.startswith("-"):
                        continue
                    # From the second character: an npm scope starts with @.
                    cut = re.search(separator, package[1:])
                    name = package[: cut.start() + 1] if cut else package
                    version = package[cut.start() + 1 :].lstrip("=@") if cut else None
                    found[installer][name] = version
    return found


def stage_images(relative: str) -> list[tuple[str, str | None]]:
    """(image, version) of every FROM that is not an earlier stage."""
    stages: set[str] = set()
    images = []
    for stage, instruction, arguments in dockerfile(relative):
        if instruction != "FROM":
            continue
        source = next(
            token for token in arguments.split() if not token.startswith("--")
        )
        match = re.fullmatch(
            r"([^:@\s]+)(?::v?(\d+(?:\.\d+)*)[^@\s]*)?(?:@\S+)?", source
        )
        assert match, drift(
            relative, f"cannot read `FROM {arguments}`", f"Update {THIS}."
        )
        if match.group(1) not in stages:
            images.append((match.group(1), match.group(2)))
        stages.add(stage)
    return images


def major(version: str) -> str:
    found = re.search(r"\d+", version)
    return found.group(0) if found else version


def test_the_pinned_versions_follow_the_appliance():
    artifacts = build_artifacts()
    docker = upstream(DOCKERFILE)
    update = "in desktop/build/artifacts.py (with the artifacts' digests)"
    images = stage_images(DOCKERFILE)
    unknown = {image for image, _ in images} - set(BASE_IMAGES)
    assert not unknown, drift(
        DOCKERFILE,
        f"the image is now also built from {sorted(unknown)}",
        "A new bundled component: add it to the desktop bundle (build/artifacts.py, "
        "build/build_runtime.py) or record why not in BASE_IMAGES.",
    )
    stale = set(BASE_IMAGES) - {image for image, _ in images}
    assert not stale, drift(
        DOCKERFILE,
        f"the image is no longer built from {sorted(stale)}",
        "See what replaced each (what the desktop bundles in its place may need to follow), "
        "then remove it from BASE_IMAGES.",
    )
    versions: dict[str, set[str | None]] = {}
    for image, version in images:
        versions.setdefault(image, set()).add(version)
    assert versions["rabbitmq"] == {artifacts.RABBITMQ_VERSION}, drift(
        DOCKERFILE,
        f"RabbitMQ is {sorted(map(str, versions['rabbitmq']))}; the desktop bundles {artifacts.RABBITMQ_VERSION}",
        f"Update RABBITMQ_VERSION {update}, and ERLANG_VERSION if that RabbitMQ needs a newer Erlang.",
    )
    # The patch is the desktop's own (newest of the major); the major is the frontend's.
    ours = major(artifacts.NODE_VERSION)
    node_majors = {major(str(version)) for version in versions["node"]}
    node_update = (
        f"Update NODE_VERSION {update}, and Electron in desktop/package.json to a release "
        "with that Node major."
    )
    assert node_majors == {ours}, drift(
        DOCKERFILE,
        f"Node is {sorted(map(str, versions['node']))}; the desktop bundles {artifacts.NODE_VERSION}",
        node_update,
    )
    engines = json.loads(upstream(PACKAGE_JSON))["engines"]["node"]
    assert major(engines) == ours, drift(
        PACKAGE_JSON,
        f"the frontend wants Node {engines}; the desktop bundles {artifacts.NODE_VERSION}",
        node_update,
    )
    majors = set(re.findall(r"\bpostgresql-(?:client-)?(\d+)\b", docker))
    assert majors == {APPLIANCE_POSTGRES_MAJOR}, drift(
        DOCKERFILE,
        f"PostgreSQL moved from {APPLIANCE_POSTGRES_MAJOR} to {sorted(majors)}",
        "Upstream's migrations may now use newer features: check that the desktop's servers "
        "(POSTGRES_BUNDLE, POSTGRES_SOURCE_VERSION in desktop/build/artifacts.py) are at least "
        "that major, then update APPLIANCE_POSTGRES_MAJOR in this test. Never change the major "
        "an installed data directory is on (postgres.check_compatible).",
    )
    assert int(artifacts.POSTGRES_SOURCE_VERSION.split(".")[0]) >= int(
        APPLIANCE_POSTGRES_MAJOR
    )
    pythons = set(re.findall(r"\bpython(3\.\d+)\b", upstream(BACKEND_DOCKERFILE)))
    assert pythons == {artifacts.PYTHON_VERSION.rsplit(".", 1)[0]}, drift(
        BACKEND_DOCKERFILE,
        f"the backend runs on Python {sorted(pythons)}; the desktop bundles {artifacts.PYTHON_VERSION}",
        f"Update PYTHON_VERSION and PYTHON_BUILD {update}, the `--python` of the build commands "
        "in desktop/README.md, and site_packages in build/build_runtime.py.",
    )


def check_packages(
    relative: str,
    stage: str,
    found: dict[str, str | None],
    known: dict[str, str],
    table: str,
) -> None:
    assert set(found) == set(known), drift(
        f"{relative} (stage {stage})",
        f"it now installs {sorted(set(found) - set(known))} and no longer "
        f"{sorted(set(known) - set(found))}",
        "Something a service may call: bundle it (desktop/build/artifacts.py) or record why "
        f"not in {table} in {THIS}.",
    )
    pinned = {name: version for name, version in found.items() if version}
    recorded = {
        name: version for name, (version, _) in PINNED_PACKAGES.items() if name in known
    }
    assert pinned == recorded, drift(
        f"{relative} (stage {stage})",
        f"it pins {pinned}; PINNED_PACKAGES records {recorded}",
        "Check the version the desktop bundles of it (desktop/build/artifacts.py) against the "
        "pin, then record the pin in PINNED_PACKAGES.",
    )


def test_what_the_images_install_is_bundled_or_waived():
    appliance = installed(DOCKERFILE, APPLIANCE_STAGE)
    check_packages(
        DOCKERFILE,
        APPLIANCE_STAGE,
        appliance["apt"],
        APPLIANCE_PACKAGES,
        "APPLIANCE_PACKAGES",
    )
    backend = installed(BACKEND_DOCKERFILE, BACKEND_STAGE)
    check_packages(
        BACKEND_DOCKERFILE,
        BACKEND_STAGE,
        backend["apt"],
        BACKEND_PACKAGES,
        "BACKEND_PACKAGES",
    )
    where = f"{BACKEND_DOCKERFILE} (stage {BACKEND_STAGE})"
    npm = {**appliance["npm"], **backend["npm"]}
    assert npm == {name: version for name, (version, _) in BACKEND_NPM.items()}, drift(
        where,
        f"it (or the appliance's last stage) installs with npm {npm}",
        "A tool the services call: bundle it, or a new version of one: read what changed. "
        f"Then record it in BACKEND_NPM in {THIS}.",
    )
    pip = {**appliance["pip"], **backend["pip"]}
    assert set(pip) == set(BACKEND_PIP), drift(
        where,
        f"it (or the appliance's last stage) installs with pip {pip}",
        "A tool outside the locked dependencies: add it to the bundle "
        f"(desktop/build/build_runtime.py) or record why not in BACKEND_PIP in {THIS}.",
    )
    assert not missing_symbols(
        BUNDLED_TOOLS
    ), f"BUNDLED_TOOLS names desktop code that is gone: {missing_symbols(BUNDLED_TOOLS)}"


# --- 9. the build --------------------------------------------------------------------

FRONTEND_STAGE = "frontend"
FRONTEND_BUILD_DIFFERS = {
    "DATABASE_URL": "a placeholder either way; there is no Unix socket to name on Windows",
    "NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY": "false: desktop does not bundle FalkorDB",
    "NEXT_SKIP_BUILD_CHECKS": "desktop only",
    "CI": "desktop only",
    "NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID": "desktop does not send feature-flag identities",
    "NEXT_PUBLIC_POSTHOG_KEY": "desktop reporting uses the shell consent gate only",
    "NEXT_PUBLIC_POSTHOG_HOST": "desktop reporting uses the shell consent gate only",
    "NEXT_PUBLIC_COOKIEBOT_CBID": "desktop preferences replace the web consent manager",
    "NEXT_PUBLIC_COOKIEBOT_GEOREGIONS": "desktop preferences replace the web consent manager",
    "NEXT_PUBLIC_GOOGLE_ADS_ID": "desktop does not load advertising trackers",
    "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS": "desktop does not load advertising trackers",
    "NEXT_PUBLIC_GA_MEASUREMENT_ID": "desktop reporting uses the shell consent gate only",
    "DATAFAST_API_KEY": "desktop reporting uses the shell consent gate only",
    "SENTRY_AUTH_TOKEN": "desktop does not upload source maps",
    "SENTRY_DSN": "desktop reporting uses the shell consent gate only",
    "DISABLE_SENTRY": "desktop reporting uses the shell consent gate only",
    "OTEL_SDK_DISABLED": "desktop does not export content traces",
}
# What build_runtime.step_assets copies out of upstream's tree as it is.
COPIED_VERBATIM = {
    "db/init/00-init.sql": 'PLATFORM / "db" / "init" / "00-init.sql"',
    f"{APPLIANCE}/runtime_config.py": 'appliance / "runtime_config.py"',
    f"{APPLIANCE}/python/sitecustomize.py": 'appliance / "python" / "sitecustomize.py"',
    BOOTSTRAP: 'appliance / "bootstrap.sh"',
    "LICENSE.md": 'PLATFORM / "LICENSE.md"',
}
# What the appliance's last stage copies out of its other stages, and what
# puts the same in the desktop bundle.
COPIED_FROM_STAGES = {
    "frontend:/usr/local/bin/node": "build_runtime.step_node",
    "frontend:/usr/local/LICENSE": "build_runtime.step_node",
    "node-runtime:/usr/local/lib/node_modules/npm": "npm, for the image's own Prisma CLI: not shipped",
    "rabbitmq:/opt/erlang": "build_runtime.step_erlang",
    "rabbitmq:/opt/openssl": "the Erlang builds the desktop takes carry their own",
    "rabbitmq:/opt/rabbitmq": "build_runtime.step_rabbitmq",
    "falkordb:/usr/local/bin/redis-server": "not shipped (SSPL)",
    "falkordb:/var/lib/falkordb/bin/falkordb.so": "not shipped (SSPL)",
    "frontend:/app/.next/standalone": "build_runtime.step_frontend",
    "frontend:/app/.next/static": "build_runtime.step_frontend",
    "frontend:/app/public": "build_runtime.step_frontend",
    "backend-code:/app/autogpt_platform/autogpt_libs": "build_runtime.step_deps",
    "backend-code:/app/autogpt_platform/backend": "build_runtime.step_backend",
    "backend-code:/app/docs": "build_runtime.step_backend (the markdown under docs/)",
}


def build_constant(name: str) -> object:
    for node in ast.parse(desktop(BUILD / "build_runtime.py")).body:
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"build/build_runtime.py no longer defines {name}")


def test_the_frontend_is_built_with_the_appliances_settings():
    theirs = stage_env(DOCKERFILE, FRONTEND_STAGE)
    ours = build_constant("FRONTEND_BUILD_ENV")
    assert isinstance(ours, dict)
    assert ours["NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY"] == "false"
    different = {
        name: f"appliance {theirs.get(name)!r}, desktop {ours.get(name)!r}"
        for name in sorted(set(theirs) | set(ours))
        if name not in FRONTEND_BUILD_DIFFERS and theirs.get(name) != ours.get(name)
    }
    assert not different, drift(
        DOCKERFILE,
        f"the frontend is built with different settings: {different}",
        "Update FRONTEND_BUILD_ENV in desktop/build/build_runtime.py, or add the name to "
        "FRONTEND_BUILD_DIFFERS with the reason.",
    )
    builds = [
        arguments
        for instruction, arguments in stage_of(DOCKERFILE, FRONTEND_STAGE)
        if instruction == "RUN" and "pnpm build" in arguments
    ]
    assert len(builds) == 1 and re.match(
        r"pnpm run generate:api && NODE_OPTIONS=\S+ pnpm build( |$)", builds[0]
    ), drift(
        DOCKERFILE,
        f"the frontend is no longer built with `pnpm run generate:api && pnpm build`: {builds}",
        "Do the same in step_frontend in desktop/build/build_runtime.py, then update this test.",
    )


def test_the_files_the_build_copies_still_exist():
    build = desktop(BUILD / "build_runtime.py")
    for relative, expression in COPIED_VERBATIM.items():
        assert (
            expression in build
        ), f"build/build_runtime.py no longer copies {relative}; remove it from COPIED_VERBATIM"
        assert (PLATFORM / relative).is_file(), drift(
            relative,
            "the file is gone or was renamed, and the desktop bundle ships it",
            "Find where it went and update step_assets in desktop/build/build_runtime.py (and the "
            "path filters of .github/workflows/platform-desktop-build.yml).",
        )


def test_what_the_image_ships_is_known():
    """Every COPY into the appliance's last stage: a file of the appliance
    (APPLIANCE_FILES), a file the desktop build copies too, or something out
    of another stage that the desktop gets its own way."""
    ours = f"autogpt_platform/{APPLIANCE}/"
    files: set[str] = set()
    stages: set[str] = set()
    for instruction, arguments in stage_of(DOCKERFILE, APPLIANCE_STAGE):
        if instruction != "COPY":
            continue
        tokens = arguments.split()
        origin = next(
            (t.split("=", 1)[1] for t in tokens if t.startswith("--from=")), None
        )
        sources = [token for token in tokens if not token.startswith("--")][:-1]
        if origin:
            stages |= {f"{origin}:{source}" for source in sources}
        else:
            files |= set(sources)
    unknown = {
        source
        for source in files
        if not (
            source.removeprefix("autogpt_platform/") in COPIED_VERBATIM
            or source.startswith(ours)
            and (
                source.removeprefix(ours) in APPLIANCE_FILES
                or source.removeprefix(ours).split("/")[0] in NOT_COMPARED
                or any(
                    name.startswith(f"{source.removeprefix(ours)}/")
                    for name in APPLIANCE_FILES
                )
            )
        )
    }
    assert not unknown, drift(
        DOCKERFILE,
        f"the image now also ships {sorted(unknown)}",
        "Something the appliance's scripts or services read at run time: copy it into the "
        "bundle in step_assets in desktop/build/build_runtime.py and use it where the "
        "appliance does, then add it to COPIED_VERBATIM.",
    )
    assert stages == set(COPIED_FROM_STAGES), drift(
        DOCKERFILE,
        f"its last stage now also takes {sorted(stages - set(COPIED_FROM_STAGES))} from other "
        f"stages and no longer {sorted(set(COPIED_FROM_STAGES) - stages)}",
        "Put the same in the desktop bundle (desktop/build/build_runtime.py) or say why not, in "
        f"COPIED_FROM_STAGES in {THIS}.",
    )
