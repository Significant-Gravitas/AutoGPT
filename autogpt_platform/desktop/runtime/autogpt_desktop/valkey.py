"""Valkey as a single-node cluster.

The backend only speaks RedisCluster (backend/data/redis_client.py), but a
cluster does not need three nodes: one node that owns all 16384 slots is a
complete cluster. The appliance runs three to mirror production; a desktop
gains nothing from the extra memory and processes.
"""

from __future__ import annotations

import contextlib
import logging
import time

from autogpt_desktop.layout import Bundle, DataDir, write_private
from autogpt_desktop.process import ManagedProcess, base_env, listening, wait_until

logger = logging.getLogger("autogpt_desktop")

CONFIG_NAME = "valkey.conf"
LAST_SLOT = 16383


def write_config(data: DataDir, port: int, bus_port: int, password: str) -> None:
    # Valkey runs with its data directory as cwd and every path relative, so
    # the same config works for the MSYS2-built Windows server, which does not
    # understand drive-letter paths.
    lines = [
        "bind 127.0.0.1",
        "protected-mode yes",
        f"port {port}",
        "dir ./",
        "appendonly yes",
        "appendfsync everysec",
        "cluster-enabled yes",
        "cluster-config-file nodes.conf",
        "cluster-node-timeout 5000",
        "cluster-require-full-coverage no",
        "cluster-announce-ip 127.0.0.1",
        f"cluster-announce-port {port}",
        # Without this the bus listens on port + 10000, which may be taken.
        f"cluster-port {bus_port}",
        f"cluster-announce-bus-port {bus_port}",
        f"requirepass {password}",
        f"masterauth {password}",
        "daemonize no",
        'logfile ""',
    ]
    write_private(data.valkey / CONFIG_NAME, "\n".join(lines) + "\n")


def set_aside(data: DataDir) -> None:
    """Move data the server cannot load out of its way. What Valkey holds is
    caches, locks and replay buffers, so starting empty costs the user
    nothing, while failing to start costs them the app."""
    target = data.valkey.with_name(f"valkey.unreadable-{int(time.time())}")
    data.valkey.rename(target)
    data.valkey.mkdir()
    logger.warning(f"Valkey could not load its data; it was moved to {target}")


def process(bundle: Bundle, data: DataDir, port: int, password: str) -> ManagedProcess:
    return ManagedProcess(
        name="valkey",
        argv=[str(bundle.valkey_server), CONFIG_NAME],
        env=base_env(),
        cwd=data.valkey,
        log_dir=data.logs,
        graceful_stop=lambda _: _shutdown(port, password),
        stop_timeout=15,
        windows_quote_executable=True,
    )


def is_ready(port: int, password: str) -> bool:
    if not listening(port):
        return False
    import redis

    try:
        return bool(_client(port, password).ping())
    except redis.RedisError:
        return False


def ensure_cluster(port: int, password: str, timeout: float = 60) -> None:
    """Assign every slot to the only node on first boot; afterwards the node
    restores its slots from nodes.conf and this is a no-op."""
    client = _client(port, password)
    if _cluster_state(client) == "ok":
        return
    owned = client.execute_command("CLUSTER", "SLOTS")
    if not owned:
        logger.info("forming the single-node Valkey cluster")
        client.execute_command("CLUSTER", "ADDSLOTSRANGE", 0, LAST_SLOT)
    if not wait_until(lambda: _cluster_state(client) == "ok", timeout):
        raise RuntimeError("the Valkey cluster did not become healthy")


def _cluster_state(client) -> str:
    info = client.execute_command("CLUSTER", "INFO")
    if isinstance(info, bytes):
        info = info.decode()
    for line in str(info).splitlines():
        key, _, value = line.partition(":")
        if key == "cluster_state":
            return value.strip()
    return "unknown"


def _client(port: int, password: str):
    import redis

    return redis.Redis(
        host="127.0.0.1",
        port=port,
        password=password,
        socket_timeout=5,
        socket_connect_timeout=3,
        decode_responses=True,
    )


def _shutdown(port: int, password: str) -> None:
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry

    # No retries: the server closes the connection as its reply, and a
    # retrying client would spend its whole budget reconnecting to a server
    # that is gone (each refused loopback connect takes ~2s on Windows).
    client = redis.Redis(
        host="127.0.0.1",
        port=port,
        password=password,
        socket_timeout=5,
        socket_connect_timeout=3,
        retry=Retry(NoBackoff(), 0),
    )
    with contextlib.suppress(redis.RedisError):
        client.shutdown()
