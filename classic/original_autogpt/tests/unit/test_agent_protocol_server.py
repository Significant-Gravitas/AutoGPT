import asyncio
import contextlib
import socket
from pathlib import Path

import pytest
from autogpt.app.agent_protocol_server import AgentProtocolServer
from autogpt.app.config import AppConfig

from forge.agent_protocol.database import AgentDB
from forge.file_storage import FileStorage
from forge.llm.providers import MultiProvider


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _non_loopback_ip() -> str | None:
    # Connecting a UDP socket sends nothing; it only picks the outgoing interface.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
        except OSError:
            return None
        ip = s.getsockname()[0]
    return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


async def _accepts(host: str, port: int) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 1)
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    await writer.wait_closed()
    return True


@contextlib.asynccontextmanager
async def _running(server: AgentProtocolServer, port: int, probe: str = "127.0.0.1"):
    task = asyncio.create_task(server.start(port=port))
    try:
        for _ in range(100):
            if await _accepts(probe, port):
                break
            if task.done():
                task.result()  # surface the startup error
                pytest.fail("server exited before listening")
            await asyncio.sleep(0.05)
        else:
            pytest.fail(f"server did not start listening on {probe}")
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.fixture
def server(
    config: AppConfig,
    storage: FileStorage,
    llm_provider: MultiProvider,
    tmp_path: Path,
) -> AgentProtocolServer:
    return AgentProtocolServer(
        app_config=config,
        database=AgentDB(f"sqlite:///{tmp_path / 'ap_server.db'}"),
        file_storage=storage,
        llm_provider=llm_provider,
    )


@pytest.fixture
def lan_ip() -> str:
    ip = _non_loopback_ip()
    if ip is None:
        pytest.skip("no non-loopback IPv4 address on this machine")
    return ip


@pytest.mark.parametrize("host", [None, ""], ids=["unset", "empty"])
async def test_server_listens_on_localhost_only_by_default(
    server: AgentProtocolServer,
    lan_ip: str,
    host: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    if host is None:
        monkeypatch.delenv("AP_SERVER_HOST", raising=False)
    else:
        monkeypatch.setenv("AP_SERVER_HOST", host)
    port = _free_port()
    async with _running(server, port):
        assert not await _accepts(lan_ip, port)


async def test_server_listens_on_all_interfaces_when_host_is_0_0_0_0(
    server: AgentProtocolServer, lan_ip: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AP_SERVER_HOST", "0.0.0.0")
    port = _free_port()
    async with _running(server, port):
        assert await _accepts(lan_ip, port)


async def test_server_starts_on_configured_host_when_loopback_port_is_taken(
    server: AgentProtocolServer, lan_ip: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("AP_SERVER_HOST", lan_ip)
    port = _free_port()
    with socket.socket() as loopback:
        loopback.bind(("127.0.0.1", port))
        async with _running(server, port, probe=lan_ip):
            assert await _accepts(lan_ip, port)
