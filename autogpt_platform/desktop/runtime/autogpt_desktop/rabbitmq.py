"""RabbitMQ on a bundled Erlang runtime.

The appliance copies /opt/erlang and /opt/rabbitmq out of the official image.
Here the same two trees sit under the bundle root, and everything RabbitMQ
would otherwise derive from the user's profile (data dirs, the Erlang cookie,
the epmd and distribution ports) is pinned so it cannot collide with another
Erlang installation on the machine.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from autogpt_desktop import install, winlinks
from autogpt_desktop.layout import EXE, SCRIPT, Bundle, DataDir, write_private
from autogpt_desktop.process import ManagedProcess, base_env, listening, run_tool

NODE_NAME = "rabbit@localhost"
INETRC_NAME = "erl_inetrc"
LOOPBACK_DISTRIBUTION = "-kernel inet_dist_use_interface {127,0,0,1}"
# RabbitMQ starts its syslog client while it boots, before it knows logging
# goes to the console. In its default UDP mode the client opens a socket on
# every interface at once (it takes no bind address), which is enough for
# Windows Firewall to prompt. In TCP mode it connects only when it has a
# message to send, and with console logging it never does.
QUIET_SYSLOG = "-syslog protocol {rfc5424,tcp}"
# Read by rabbitmq-server alone, of RabbitMQ's scripts.
SERVER_FLAGS = "RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS"
WINDOWS = sys.platform == "win32"
DATA_VERSION_FILE = "desktop-version.json"
PROVEN_DISTRIBUTED_START_VERSIONS = frozenset({"4.1.8"})
METADATA_BYTES = 64 * 1024
METADATA_ENTRIES = 256
VERSION_PATTERN = r"(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})\.(?:0|[1-9][0-9]{0,3})"
APPLICATION_TOKEN = re.compile(
    r"""\s+|%[^\r\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|"""
    r"[a-z][A-Za-z0-9_@]*|-?[0-9]+(?:\.[0-9]+(?:[eE][+-]?[0-9]+)?)?|<<|>>|[{}\[\],.]",
    re.DOTALL,
)


class IncompatibleDataError(RuntimeError):
    pass


def _metadata_text(path: Path, limit: int = METADATA_BYTES) -> str:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or _is_link(info) or info.st_size > limit:
        raise ValueError("invalid metadata file")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("metadata size limit exceeded")
    return content.decode("utf-8")


def _is_link(info: os.stat_result) -> bool:
    return (
        stat.S_ISLNK(info.st_mode)
        or bool(
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )
        or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)
    )


def _entries(path: Path) -> list[Path]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return []
    if not stat.S_ISDIR(info.st_mode) or _is_link(info):
        raise ValueError("invalid metadata directory")
    result = []
    with os.scandir(path) as entries:
        for entry in entries:
            if len(result) == METADATA_ENTRIES:
                raise ValueError("metadata directory entry limit exceeded")
            result.append(Path(entry.path))
    return result


def _version(value: object) -> tuple[int, ...]:
    if not isinstance(value, str) or not re.fullmatch(VERSION_PATTERN, value):
        raise ValueError("invalid RabbitMQ version")
    return tuple(int(part) for part in value.split("."))


def _application_version(content: str) -> str:
    """Parse the literal terms used by generated rabbit.app files, without evaluation.

    Unsupported Erlang syntax is rejected. Only a complete application term's
    own vsn property can establish the version; nested settings cannot.
    """
    tokens = []
    offset = 0
    while offset < len(content):
        match = APPLICATION_TOKEN.match(content, offset)
        if match is None:
            raise ValueError("unsupported application metadata syntax")
        token = match.group()
        offset = match.end()
        if not token.isspace() and not token.startswith("%"):
            tokens.append(token)
    position = 0

    def take() -> str:
        nonlocal position
        if position == len(tokens):
            raise ValueError("incomplete application metadata")
        token = tokens[position]
        position += 1
        return token

    def term(depth: int = 0):
        if depth > 32:
            raise ValueError("application metadata nesting limit exceeded")
        token = take()
        closing = {"{": "}", "[": "]", "<<": ">>"}.get(token)
        if closing is None:
            if token in ("}", "]", ">>", ",", "."):
                raise ValueError("unexpected application metadata delimiter")
            return token
        members = []
        if position < len(tokens) and tokens[position] == closing:
            take()
        else:
            while True:
                members.append(term(depth + 1))
                separator = take()
                if separator == closing:
                    break
                if separator != ",":
                    raise ValueError("invalid application metadata sequence")
        if token == "<<":
            if any(
                not isinstance(member, str)
                or not (member.startswith('"') or re.fullmatch(r"[0-9]+", member))
                for member in members
            ):
                raise ValueError("unsupported application binary literal")
            return "<<>>"
        return tuple(members) if token == "{" else members

    application = term()
    if take() != "." or position != len(tokens):
        raise ValueError("unexpected data after application term")
    if (
        not isinstance(application, tuple)
        or len(application) != 3
        or application[0] not in ("application", "'application'")
        or application[1] not in ("rabbit", "'rabbit'")
        or not isinstance(application[2], list)
    ):
        raise ValueError("unexpected RabbitMQ application")
    properties = {}
    for item in application[2]:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not re.fullmatch(r"[a-z][A-Za-z0-9_@]*|'[a-z][A-Za-z0-9_@]*'", item[0])
        ):
            raise ValueError("invalid application property")
        key = item[0].strip("'")
        if key in properties:
            raise ValueError("duplicate application property")
        properties[key] = item[1]
    version = properties.get("vsn")
    if not isinstance(version, str) or not version.startswith('"'):
        raise ValueError("missing application version string")
    version = version[1:-1]
    _version(version)
    return version


def bundled_version(bundle: Bundle) -> str:
    """Read the application's version, never a version inferred from its directory."""
    try:
        candidates = [
            entry
            for entry in _entries(bundle.rabbitmq_home / "plugins")
            if entry.name == "rabbit" or entry.name.startswith("rabbit-")
        ]
        if len(candidates) != 1:
            raise ValueError("ambiguous RabbitMQ application")
        for directory in (candidates[0], candidates[0] / "ebin"):
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or _is_link(info):
                raise ValueError("invalid RabbitMQ application directory")
        content = _metadata_text(candidates[0] / "ebin" / "rabbit.app")
        return _application_version(content)
    except (OSError, ValueError) as exc:
        raise IncompatibleDataError(
            "Cannot verify the bundled RabbitMQ version from rabbit.app metadata. "
            "Open Doctor and repair or reinstall this AutoGPT build before starting it."
        ) from exc


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    record = dict(pairs)
    if len(record) != len(pairs):
        raise ValueError("duplicate metadata fields")
    return record


def _stored_version(data: DataDir) -> tuple[str, str] | None:
    path = data.rabbitmq / DATA_VERSION_FILE
    try:
        content = _metadata_text(path, limit=1024)
    except FileNotFoundError:
        return None
    record = json.loads(content, object_pairs_hook=_unique_fields)
    if not isinstance(record, dict):
        raise TypeError("invalid stored version metadata")
    if (
        set(record) != {"schema_version", "version", "state"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["state"] not in ("starting", "ready")
    ):
        raise ValueError("unsupported stored version metadata")
    _version(record["version"])
    return record["version"], record["state"]


def _has_storage(data: DataDir) -> bool:
    files = {
        DATA_VERSION_FILE,
        ".erlang.cookie",
        "rabbitmq.conf",
        "enabled_plugins",
        INETRC_NAME,
        "erl_crash.dump",
    }
    for entry in _entries(data.rabbitmq):
        info = entry.lstat()
        if _is_link(info):
            raise ValueError("linked RabbitMQ scaffold entry")
        if entry.name in ("mnesia", "log"):
            valid = stat.S_ISDIR(info.st_mode)
        elif entry.name in files or re.fullmatch(
            rf"\.{re.escape(DATA_VERSION_FILE)}\.[0-9a-f]{{16}}\.tmp", entry.name
        ):
            valid = stat.S_ISREG(info.st_mode)
        else:
            valid = False
        if not valid:
            raise ValueError("unrecognized RabbitMQ storage layout")
    storage = _entries(data.rabbitmq / "mnesia")
    for entry in storage:
        info = entry.lstat()
        if _is_link(info) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise ValueError("invalid RabbitMQ storage entry")
    return bool(storage)


def _incompatible(data: DataDir, reason: str) -> IncompatibleDataError:
    return IncompatibleDataError(
        f"{reason} RabbitMQ startup is blocked to preserve {data.rabbitmq}. "
        "Open Doctor for diagnostics and use a validated staged RabbitMQ upgrade "
        "with stable feature flags enabled at each step. "
        "This check has not migrated or reset your data."
    )


def check_compatible(bundle: Bundle, data: DataDir) -> str:
    """Read-only preflight; call under the profile lock before prepare or any launch.

    Cross-series changes need a separate, validated staged upgrade. A starting
    record is an attempted writer, not evidence of readiness or enabled flags.
    """
    target = bundled_version(bundle)
    try:
        populated = _has_storage(data)
        stored = _stored_version(data)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise _incompatible(
            data, "RabbitMQ version metadata is unreadable or invalid."
        ) from exc
    if stored is None:
        if populated:
            raise _incompatible(
                data, "The version that last wrote RabbitMQ data is unknown."
            )
        return target
    previous, state = stored
    old_version, new_version = _version(previous), _version(target)
    if new_version < old_version:
        raise _incompatible(
            data,
            f"RabbitMQ {previous} {'attempted to open' if state == 'starting' else 'last opened'} "
            f"this data; bundled {target} would be an unsupported downgrade.",
        )
    if new_version[:2] != old_version[:2]:
        raise _incompatible(
            data,
            f"RabbitMQ {previous} to {target} requires a staged upgrade. "
            "RabbitMQ 4.1 must pass through 4.2 before 4.3; a version record alone "
            "does not prove the required stable feature flags are enabled.",
        )
    return target


def record_starting(bundle: Bundle, data: DataDir) -> str:
    """After preflight, record the attempted writer before launching the broker.

    Keep this record on failure: a newer broker may have changed storage before
    it became ready. The supervisor must hold the profile lock throughout.
    """
    version = check_compatible(bundle, data)
    _write_version(data, version, "starting")
    return version


def record_ready(data: DataDir, version: str) -> None:
    """Call only after the launched broker passes authenticated AMQP readiness.

    `version` is the value returned by record_starting for that launch. This
    records a successful start, not completion of a cross-series migration.
    """
    try:
        _version(version)
        if _stored_version(data) != (version, "starting") or not _has_storage(data):
            raise ValueError("no matching RabbitMQ start")
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise _incompatible(
            data, "Cannot confirm the RabbitMQ startup version record."
        ) from exc
    _write_version(data, version, "ready")


def _write_version(data: DataDir, version: str, state: str) -> None:
    write_private(
        data.rabbitmq / DATA_VERSION_FILE,
        json.dumps({"schema_version": 1, "version": version, "state": state}) + "\n",
    )


def server_flags(bundle: Bundle, ports: dict[str, int]) -> str:
    """Use the distributed-start shortcut only for versions proven to boot with it."""
    try:
        version = bundled_version(bundle)
    except IncompatibleDataError:
        return QUIET_SYSLOG
    return (
        _server_flags(ports)
        if version in PROVEN_DISTRIBUTED_START_VERSIONS
        else QUIET_SYSLOG
    )


def prepare(data: DataDir, port: int, user: str, password: str) -> None:
    forget_dangling_aliases()
    base = data.rabbitmq
    (base / "mnesia").mkdir(parents=True, exist_ok=True)
    write_private(
        base / "rabbitmq.conf",
        "\n".join(
            [
                f"listeners.tcp.default = 127.0.0.1:{port}",
                "distribution.listener.interface = 127.0.0.1",
                "loopback_users.guest = true",
                "log.console = true",
                "log.console.level = info",
                "log.file = false",
                f"default_user = {user}",
                f"default_pass = {password}",
                "default_vhost = /",
            ]
        )
        + "\n",
    )
    (base / "enabled_plugins").write_text("[].\n", encoding="utf-8")
    # The node is named rabbit@localhost. Answer that name from a static
    # table and leave everything else to the OS resolver; Erlang's own DNS
    # client would otherwise open a wildcard UDP socket and wait out a 2s
    # timeout on every boot.
    (base / INETRC_NAME).write_text(
        '{lookup, [file, native]}.\n{host, {127,0,0,1}, ["localhost"]}.\n',
        encoding="utf-8",
    )
    cookie = base / ".erlang.cookie"
    if not cookie.exists():
        descriptor = os.open(cookie, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(secrets.token_hex(20).upper())


def environment(bundle: Bundle, data: DataDir, ports: dict[str, int]) -> dict[str, str]:
    # Created here if need be, unlike the bundle's directories: nothing is
    # ever created inside the bundle.
    data.rabbitmq.mkdir(parents=True, exist_ok=True)
    base = _short(data.rabbitmq)
    erlang = _short(bundle.erlang_home)
    env = {
        # Not the user's whole environment: Erlang and RabbitMQ read dozens of
        # variables (ERL_FLAGS, RABBITMQ_NODE_PORT, ...) that an Erlang
        # developer may have set, and any of them could undo the wiring below.
        **base_env(),
        "PATH": os.pathsep.join(
            [str(Path(erlang) / "bin"), os.environ.get("PATH", "")]
        ),
        "ERLANG_HOME": erlang,
        "RABBITMQ_HOME": _short(bundle.rabbitmq_home),
        "RABBITMQ_BASE": base,
        "RABBITMQ_MNESIA_BASE": str(Path(base) / "mnesia"),
        "RABBITMQ_CONFIG_FILE": str(Path(base) / "rabbitmq.conf"),
        "RABBITMQ_ENABLED_PLUGINS_FILE": str(Path(base) / "enabled_plugins"),
        "RABBITMQ_LOGS": "-",
        # Logging goes to the console. Should a file log ever be switched
        # on, its default place is under RABBITMQ_HOME: inside the bundle.
        "RABBITMQ_LOG_BASE": str(Path(base) / "log"),
        "RABBITMQ_NODENAME": NODE_NAME,
        "RABBITMQ_DIST_PORT": str(ports["rabbitmq_dist"]),
        "ERL_EPMD_PORT": str(ports["epmd"]),
        "ERL_EPMD_ADDRESS": "127.0.0.1",
        "ERL_INETRC": str(Path(base) / INETRC_NAME),
        # Erlang distribution listens on every interface unless told
        # otherwise, and it starts listening before rabbitmq.conf is read.
        # ERL_AFLAGS reaches every VM started under this environment: the
        # server, rabbitmqctl, and the short-lived `epmd-starter` node that
        # rabbitmq-server.bat launches. A non-loopback socket from any of
        # them is needless exposure, and on Windows raises a firewall prompt.
        "ERL_AFLAGS": _erlang_flags(bundle),
        SERVER_FLAGS: server_flags(bundle, ports),
        "ERL_CRASH_DUMP": str(Path(base) / "erl_crash.dump"),
        # Erlang finds its cookie in the home directory; point every flavour
        # of "home" at the data dir so the server and rabbitmqctl agree.
        "HOME": base,
        "USERPROFILE": base,
    }
    if WINDOWS:
        drive, rest = os.path.splitdrive(base)
        env["HOMEDRIVE"] = drive
        env["HOMEPATH"] = rest or "\\"
    return env


def _server_flags(ports: dict[str, int]) -> str:
    """The server's VM is started as a distributed node, under the name and
    on the port RabbitMQ would give it.

    Left to itself RabbitMQ starts the VM undistributed and, while it boots,
    makes sure there is a port mapper by starting a second Erlang VM that
    exits at once (`epmd-starter`), checks the distribution port and looks
    for a node of the same name, and only then turns distribution on. That
    is two to three seconds of every start on Windows, and here there is
    nothing to find out: the port mapper is this runtime's own and already
    answers (supervisor.py), and the name and both ports are this install's.
    A VM that is already the node RabbitMQ wants skips all of it
    (rabbit_prelaunch_dist: "Erlang distribution already running").

    tests/test_start.py names the RabbitMQ version this was seen with;
    `start_the_plain_way` is what a version that boots differently gets."""
    port = ports["rabbitmq_dist"]
    return (
        f"{QUIET_SYSLOG} -sname {NODE_NAME} "
        f"-kernel inet_dist_listen_min {port} -kernel inet_dist_listen_max {port}"
    )


def start_the_plain_way(process: ManagedProcess) -> bool:
    """Take `_server_flags` off a broker's process, so that its next start
    is RabbitMQ's own. False when they were off already."""
    if process.env.get(SERVER_FLAGS) == QUIET_SYSLOG:
        return False
    process.env[SERVER_FLAGS] = QUIET_SYSLOG
    return True


def _erlang_flags(bundle: Bundle) -> str:
    flags = LOOPBACK_DISTRIBUTION
    if any(bundle.erlang_patches.glob("*.beam")):
        # Shadows OTP's inet_udp and inet_tcp so that sockets opened without
        # an address bind loopback (build/erlang_patches.py). Forward
        # slashes: the value is split on spaces and backslashes are escapes.
        patches = _short(bundle.erlang_patches).replace(os.sep, "/")
        if WINDOWS:
            patches = f'"{patches}"'
        flags += f" -pa {patches}"
    return flags


def process(bundle: Bundle, data: DataDir, ports: dict[str, int]) -> ManagedProcess:
    env = environment(bundle, data, ports)
    return ManagedProcess(
        name="rabbitmq",
        argv=[_script(bundle, "rabbitmq-server")],
        env=env,
        cwd=data.rabbitmq,
        log_dir=data.logs,
        graceful_stop=lambda _: _stop(bundle, env),
        stop_timeout=30,
    )


def epmd_process(
    bundle: Bundle, data: DataDir, ports: dict[str, int]
) -> ManagedProcess:
    """Erlang's port mapper, which RabbitMQ and rabbitmqctl find each other
    through. Left alone, the first Erlang VM starts one as a daemon that
    outlives everything and that nothing here would know to stop. Started
    first and in the foreground it is a child like any other, and the VMs
    use the one they find running."""
    epmd = next(bundle.erlang_home.glob(f"erts-*/bin/epmd{EXE}"))
    return ManagedProcess(
        name="epmd",
        argv=[str(epmd)],
        env=environment(bundle, data, ports),
        cwd=data.rabbitmq,
        log_dir=data.logs,
    )


def epmd_is_ready(port: int) -> bool:
    return listening(port)


def is_ready(port: int, user: str, password: str) -> bool:
    if not listening(port):
        return False
    import pika
    import pika.exceptions

    # pika logs every refused attempt at ERROR; while the broker boots those
    # are expected and would bury the runtime's own log.
    logging.getLogger("pika").setLevel(logging.CRITICAL)
    parameters = pika.ConnectionParameters(
        host="127.0.0.1",
        port=port,
        credentials=pika.PlainCredentials(user, password),
        connection_attempts=1,
        socket_timeout=3,
        blocked_connection_timeout=3,
    )
    try:
        pika.BlockingConnection(parameters).close()
        return True
    except pika.exceptions.AMQPError:
        return False


def _stop(bundle: Bundle, env: dict[str, str]) -> None:
    run_tool(
        [_script(bundle, "rabbitmqctl"), "-n", NODE_NAME, "stop"],
        env=env,
        capture_output=True,
        timeout=25,
    )


QUEUE_COUNTS = ("messages_ready", "messages_unacknowledged", "consumers")
# What rabbitmqctl is told to give up after, in seconds. Left to itself it
# waits a minute for a broker that is slow, and on Windows the limit set on
# running it ends only the script, not the Erlang VM whose output is waited
# for: the tool has to end by itself.
QUEUE_COUNTS_SECONDS = 10


def queue_counts(
    bundle: Bundle, env: dict[str, str], queue: str
) -> dict[str, int] | None:
    """How many messages of `queue` wait, how many a consumer has been handed
    and has not settled, and how many consumers it has (QUEUE_COUNTS). None
    when the broker did not say. Through rabbitmqctl, which costs a second
    Erlang VM for a few seconds: a client of the broker is told the first
    and the last, and not the one in the middle."""
    command = [
        _script(bundle, "rabbitmqctl"),
        *(
            "-n",
            NODE_NAME,
            "--timeout",
            str(QUEUE_COUNTS_SECONDS),
            "--quiet",
            "--formatter",
            "json",
        ),
        *("list_queues", "name", *QUEUE_COUNTS),
    ]
    try:
        listed = run_tool(command, env=env, capture_output=True, timeout=20)
        queues = json.loads(listed.stdout)
        found = next(row for row in queues if row["name"] == queue)
        return {name: int(found[name]) for name in QUEUE_COUNTS}
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        TypeError,
        KeyError,
        StopIteration,
    ):
        return None


def _script(bundle: Bundle, name: str) -> str:
    # Through the space-free alias: the scripts locate each other from $0.
    return str(Path(_short(bundle.rabbitmq_home)) / "sbin" / f"{name}{SCRIPT}")


def _short(path: Path) -> str:
    """A spelling of `path` with no spaces in it.

    RabbitMQ's launch scripts (batch on Windows, sh elsewhere) source and
    execute unquoted paths, and both the bundle and the data directory
    normally contain a space or the user's name: `%LOCALAPPDATA%` on Windows,
    `~/Library/Application Support` on macOS.
    """
    if sys.platform == "win32":
        return _windows_alias(path)
    if not any(character.isspace() for character in str(path)):
        return str(path)
    link = _alias_root() / hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if link.is_symlink() and Path(os.readlink(link)) != path:
        link.unlink()
    if not link.is_symlink():
        link.symlink_to(path, target_is_directory=True)
    return str(link)


def _alias_root() -> Path:
    """Where the space-free symlinks live: the user's cache directory. Not
    the temp directory, which macOS and systemd clear of anything untouched
    for a few days; the broker would lose its data path while running."""
    assert sys.platform != "win32"
    root = _cache_alias_root()
    if any(character.isspace() for character in str(root)):
        root = _temp_alias_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _cache_alias_root() -> Path:
    if sys.platform == "darwin":
        cache = Path.home() / "Library" / "Caches"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    # A folder of its own for each install: a variant of the app shares
    # none with the normal app.
    return cache / install.name() / "links"


def forget_dangling_aliases() -> None:
    """Links to folders that are gone: a data directory that was moved or
    deleted, an app that was run from a disk image since unmounted, a test
    run's temporary folder. Nothing else takes them away, and each start is a
    chance to. Windows does the same for its junctions (winlinks.junction)."""
    if sys.platform == "win32":
        return
    for root in (_cache_alias_root(), _temp_alias_root()):
        forget_dangling(root)


def forget_dangling(root: Path) -> None:
    """Only what `_short` could have made: a link, never a file or a folder
    somebody put there."""
    with contextlib.suppress(OSError):
        for entry in list(os.scandir(root)):
            # exists() follows the link; lexists() is true of a dangling one.
            if entry.is_symlink() and not os.path.exists(entry.path):
                with contextlib.suppress(OSError):
                    os.unlink(entry.path)


def _temp_alias_root() -> Path:
    """Stands in for the cache folder when that one's own path has a space."""
    assert sys.platform != "win32"
    return (
        Path(tempfile.gettempdir()) / f"{install.name().lower()}-desktop-{os.getuid()}"
    )


def alias_roots() -> list[Path]:
    """Every folder this install's aliases may be in, whether it exists or
    not. Nothing is created: this is for clearing up (build/smoke_test.py)."""
    if sys.platform != "win32":
        return [_cache_alias_root(), _temp_alias_root()]
    try:
        user = winlinks.current_user_sid()
    except OSError:
        return []
    found = winlinks.roots(install.name(), os.environ, user, winlinks.short_name)
    return [root.path for root in found]


def _windows_alias(path: Path) -> str:
    """The directory's 8.3 name, which has neither spaces nor non-ASCII
    characters; where the volume keeps no short names, a junction to it
    from a place whose path is plain (winlinks.py)."""
    short = winlinks.short_name(str(path))
    if winlinks.plain(short):
        return short
    alias = winlinks.alias(path, install.name())
    if alias:
        return alias
    if not any(character.isspace() for character in short):
        return short  # not plain, but nothing the scripts are known to break on
    raise RuntimeError(
        f"AutoGPT cannot run from a folder with a space in its name ({path}) "
        "on a drive that has short file names turned off, and it could not make "
        "a link to that folder from a place without one (see the log). Install "
        "it, and keep its data, in folders without spaces."
    )
