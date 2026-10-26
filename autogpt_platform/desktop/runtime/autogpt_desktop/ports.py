"""Port allocation.

The appliance owns its network namespace and hardcodes every port. A desktop
shares the machine with whatever else the user runs, so each port is picked
free on first boot and remembered, then re-used for as long as it stays free.
Keeping the public port stable matters most: it is the origin that browser
sessions and OAuth redirect URIs are tied to.
"""

from __future__ import annotations

import json
import logging
import os
import random
import socket
import sys
import time
from pathlib import Path

logger = logging.getLogger("autogpt_desktop")

# Below the ephemeral range of every supported OS (Linux starts at 32768,
# Windows and macOS at 49152), so a remembered port is never handed out to
# some other program's outbound connection between boots.
PORT_RANGE = range(15000, 32000)
PUBLIC_PORT_PATIENCE_SECONDS = 5

# Tried first on an install that has no port yet, so that most installs share
# one address and OAuth redirect URIs can be documented once. Unassigned in
# the IANA registry (18464-18515, checked 2026-10-03) and nobody's default.
# An install that already has a port keeps it: moving it signs the user out
# and breaks the redirect URIs they registered.
PREFERRED = {"public": 18473}
# The shell names the public port of the install it belongs to. A variant of
# the app (src/identity.js) prefers a port of its own, so that it and the
# normal app can both keep their address while both are installed.
PUBLIC_PORT_VARIABLE = "AUTOGPT_DESKTOP_PUBLIC_PORT"
# Where the shell puts the public ports of variants (src/identity.js
# VARIANT_PORT_FIRST and VARIANT_PORT_COUNT). No install picks a port for
# anything from here by chance: a port is remembered for good, and the
# database of one install would sit on the address of another.
VARIANT_PUBLIC_PORTS = range(20000, 30000)

PORT_NAMES = (
    "public",
    "frontend",
    "postgres",
    "valkey",
    "valkey_bus",
    "rabbitmq",
    "rabbitmq_dist",
    "epmd",
    "websocket",
    "execution_manager",
    "execution_scheduler",
    "database_api",
    "swap_credential",
    "agent_api",
    "notification",
    "copilot_executor",
    "platform_linking",
    "copilot_chat_bridge",
    "batch_executor",
)


def allocate(path: Path, names: tuple[str, ...] = PORT_NAMES) -> dict[str, int]:
    stored = _read(path)
    # A replacement must not land on a port another service already has on
    # record, or that service would be moved too.
    remembered = {port for port in stored.values() if isinstance(port, int)}
    ports: dict[str, int] = {}
    for name in names:
        port = stored.get(name)
        if not (isinstance(port, int) and _is_free(port, wait=name == "public")):
            port = None
        if port is None and name not in stored:
            port = _preferred(name)
        if port is None or port in ports.values():
            port = _free_port(exclude=set(ports.values()) | remembered)
            if name in stored:
                logger.warning(f"the {name} port {stored[name]} is taken; using {port}")
        ports[name] = port
    if ports != stored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ports, indent=2) + "\n", encoding="utf-8")
    return ports


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _preferred(name: str) -> int | None:
    port = _preferences().get(name)
    return port if port is not None and _is_free(port) else None


def _preferences() -> dict[str, int]:
    """PREFERRED, with the public port the shell asked for. A value that is
    not a port of PORT_RANGE means no preference at all: falling back to the
    default would put a variant on the normal app's port."""
    asked = os.environ.get(PUBLIC_PORT_VARIABLE)
    if asked is None:
        return PREFERRED
    if asked.isdecimal() and int(asked) in PORT_RANGE:
        return {**PREFERRED, "public": int(asked)}
    logger.warning(
        f"ignoring {PUBLIC_PORT_VARIABLE}={asked!r}: not a port in {PORT_RANGE}"
    )
    return {name: port for name, port in PREFERRED.items() if name != "public"}


def _is_free(port: int, *, wait: bool = False) -> bool:
    """`wait` gives a busy port a few seconds to come free before giving up
    on it. Worth it for the public port only: moving that one signs the user
    out and breaks the redirect URIs of their OAuth apps, and the usual
    holder is a previous run of this app that is still shutting down."""
    if port not in PORT_RANGE:
        return False
    deadline = time.monotonic() + (PUBLIC_PORT_PATIENCE_SECONDS if wait else 0)
    while not _can_listen(port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)
    return True


def _can_listen(port: int) -> bool:
    """Whether a server started now could have the port.

    Outside Windows a plain bind is refused for up to a minute after the
    port's last listener closed a connection (TIME_WAIT), which the proxy
    does on every stop; a restart seconds later would read that as "taken"
    and move the app. Servers get past it with SO_REUSEADDR, so the probe
    asks the same way. On macOS that option also lets a bind succeed beside
    another program listening on every address, hence the connection attempt.
    On Windows a plain bind is already exact, and SO_REUSEADDR would bind
    over a live listener."""
    if sys.platform == "win32":
        return _can_bind(port, reuse_address=False)
    return _can_bind(port, reuse_address=True) and not _answers(port)


def _can_bind(port: int, *, reuse_address: bool) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if reuse_address:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _answers(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.5)
        return client.connect_ex(("127.0.0.1", port)) == 0


def _free_port(exclude: set[int]) -> int:
    # The normal app's address is kept clear by every install, and so is
    # this install's own and that of every variant there could be.
    reserved = exclude | set(PREFERRED.values()) | set(_preferences().values())
    for _ in range(1000):
        port = random.choice(PORT_RANGE)
        if port in reserved or port in VARIANT_PUBLIC_PORTS:
            continue
        if _is_free(port):
            return port
    raise RuntimeError("no free local port found")
