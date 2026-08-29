"""Sharded pubsub reconnect across a real `docker restart` of a shard,
against a private 3-shard cluster on isolated host ports. Gated on
`E2E_REDIS_CLUSTER_RESTART=1` + `docker` on PATH, marked `pytest.mark.slow`."""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import socket
import subprocess
import time
from uuid import uuid4

import pytest

# Disjoint from the dev-compose ports (17000-17002) so both stacks coexist.
ISOLATED_PROJECT = "redis-restart-test"
ISOLATED_PORTS = (27110, 27111, 27112)
ISOLATED_BUS_PORTS = (37110, 37111, 37112)


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _isolated_enabled() -> bool:
    return os.getenv("E2E_REDIS_CLUSTER_RESTART", "").lower() in ("1", "true", "yes")


cluster_restart_only = pytest.mark.skipif(
    not (_docker_available() and _isolated_enabled()),
    reason=(
        "isolated docker cluster restart e2e: requires docker + E2E_REDIS_CLUSTER_RESTART=1"
    ),
)


def _run(cmd: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _wait_port(port: int, *, deadline_s: float = 60.0) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"port {port} never opened within {deadline_s:.0f}s")


def _start_isolated_cluster() -> None:
    """Spin up a private 3-shard cluster via raw `docker run` + one-shot
    `redis-cli --cluster create`."""
    network = f"{ISOLATED_PROJECT}-net"
    _run(["docker", "network", "create", network])  # may exist; ignore exit
    for i, (port, bus) in enumerate(zip(ISOLATED_PORTS, ISOLATED_BUS_PORTS)):
        name = f"{ISOLATED_PROJECT}-redis-{i}"
        _run(["docker", "rm", "-f", name])
        rc = _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--network",
                network,
                "--network-alias",
                f"redis-{i}",
                "-p",
                f"{port}:{port}",
                "redis:7",
                "redis-server",
                "--port",
                str(port),
                "--cluster-enabled",
                "yes",
                "--cluster-config-file",
                "nodes.conf",
                "--cluster-node-timeout",
                "5000",
                "--cluster-require-full-coverage",
                "no",
                "--cluster-announce-hostname",
                f"redis-{i}",
                "--cluster-announce-port",
                str(port),
                "--cluster-announce-bus-port",
                str(bus),
                "--cluster-preferred-endpoint-type",
                "hostname",
            ]
        )
        if rc.returncode != 0:
            raise RuntimeError(f"docker run redis-{i} failed: {rc.stderr}")
    for port in ISOLATED_PORTS:
        _wait_port(port)
    rc = _run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "redis:7",
            "redis-cli",
            "--cluster",
            "create",
            f"redis-0:{ISOLATED_PORTS[0]}",
            f"redis-1:{ISOLATED_PORTS[1]}",
            f"redis-2:{ISOLATED_PORTS[2]}",
            "--cluster-replicas",
            "0",
            "--cluster-yes",
        ]
    )
    if rc.returncode != 0:
        raise RuntimeError(f"cluster create failed: {rc.stderr}")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        info = _run(
            [
                "docker",
                "exec",
                f"{ISOLATED_PROJECT}-redis-0",
                "redis-cli",
                "-p",
                str(ISOLATED_PORTS[0]),
                "cluster",
                "info",
            ]
        )
        if "cluster_state:ok" in info.stdout:
            return
        time.sleep(0.5)
    raise TimeoutError("isolated cluster never reached cluster_state:ok")


def _wait_cluster_ok(timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = _run(
            [
                "docker",
                "exec",
                f"{ISOLATED_PROJECT}-redis-0",
                "redis-cli",
                "-p",
                str(ISOLATED_PORTS[0]),
                "cluster",
                "info",
            ]
        )
        if "cluster_state:ok" in info.stdout:
            return True
        time.sleep(0.5)
    return False


def _teardown_isolated_cluster() -> None:
    for i in range(3):
        _run(["docker", "rm", "-f", f"{ISOLATED_PROJECT}-redis-{i}"])
    _run(["docker", "network", "rm", f"{ISOLATED_PROJECT}-net"])


@pytest.fixture(scope="module")
def isolated_cluster():
    """Module-scoped: tests share one cluster lifecycle."""
    _start_isolated_cluster()
    try:
        yield
    finally:
        _teardown_isolated_cluster()


@pytest.mark.asyncio
@pytest.mark.slow
@cluster_restart_only
async def test_subscriber_survives_shard_restart(isolated_cluster, monkeypatch) -> None:
    """Subscriber must receive a post-`docker restart` SPUBLISH after
    reopening the sharded-pubsub client (the broker drops the socket on
    restart; production's `with_pubsub` loop reconnects the same way)."""
    # Must override REDIS_CLUSTER_HOST/PORT too — those take precedence
    # over REDIS_HOST/PORT and a stray .env would point us at the dev cluster.
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", str(ISOLATED_PORTS[0]))
    monkeypatch.setenv("REDIS_CLUSTER_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_CLUSTER_PORT", str(ISOLATED_PORTS[0]))
    monkeypatch.setenv("REDIS_USE_ANNOUNCED_ADDRESS", "false")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    import backend.data.redis_client as rc

    importlib.reload(rc)

    # Restart whichever container owns the keyslot, not a guess.
    cluster = rc.get_redis()
    target_tag = f"restart-{uuid4().hex[:8]}"
    channel = "{" + target_tag + "}/restart-test"
    owner = cluster.get_node_from_key(channel)
    port_to_idx = {p: i for i, p in enumerate(ISOLATED_PORTS)}
    target_idx = port_to_idx.get(owner.port)
    assert (
        target_idx is not None
    ), f"owner port {owner.port} not in known set {ISOLATED_PORTS}"
    target_container = f"{ISOLATED_PROJECT}-redis-{target_idx}"

    client = await rc.connect_sharded_pubsub_async(channel)
    pubsub = client.pubsub()
    await pubsub.execute_command("SSUBSCRIBE", channel)
    pubsub.channels[channel] = None  # type: ignore[index]

    received: list[str] = []

    async def _drain_one() -> str | None:
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "smessage":
                    return msg["data"]
        except Exception:
            return None
        return None

    try:
        async_cluster = await rc.get_redis_async()
        await async_cluster.execute_command("SPUBLISH", channel, "before-restart")

        first = await asyncio.wait_for(_drain_one(), timeout=6.0)
        received.append(first or "")
        assert received == [
            "before-restart"
        ], f"pre-restart publish did not arrive: {received}"

        # Restart the shard that owns the slot.
        rc_restart = _run(["docker", "restart", "--time", "1", target_container])
        assert rc_restart.returncode == 0, rc_restart.stderr

        assert _wait_cluster_ok(
            timeout_s=30
        ), "isolated cluster never re-converged to state=ok after restart"
        # Hold a small grace window for shard's gossip to settle.
        await asyncio.sleep(1.0)

        # Old socket is dead — open a fresh sharded-pubsub connection.
        try:
            await pubsub.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass
        rc._async_clients.clear()

        client2 = await rc.connect_sharded_pubsub_async(channel)
        pubsub2 = client2.pubsub()
        try:
            await pubsub2.execute_command("SSUBSCRIBE", channel)
            pubsub2.channels[channel] = None  # type: ignore[index]

            # Drain the SSUBSCRIBE confirm.
            async for _msg in pubsub2.listen():
                break

            async def _drain_after() -> str | None:
                async for msg in pubsub2.listen():
                    if msg.get("type") == "smessage":
                        return msg["data"]
                return None

            async_cluster_2 = await rc.get_redis_async()
            await async_cluster_2.execute_command("SPUBLISH", channel, "after-restart")

            data = await asyncio.wait_for(_drain_after(), timeout=15.0)
            assert (
                data == "after-restart"
            ), f"subscriber did not receive post-restart event (got {data!r})"
        finally:
            try:
                await pubsub2.execute_command("SUNSUBSCRIBE", channel)
            except Exception:
                pass
            try:
                await pubsub2.aclose()
            except Exception:
                pass
            await client2.aclose()
    finally:
        try:
            await pubsub.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass
        await rc.disconnect_async()
        # Undo monkeypatched env BEFORE reloading so subsequent tests see the
        # original REDIS_HOST/PORT — otherwise the module captures the
        # isolated cluster's port (27110) which is torn down right after this
        # test, and any later test that touches redis hangs on conn_retry.
        monkeypatch.undo()
        importlib.reload(rc)
