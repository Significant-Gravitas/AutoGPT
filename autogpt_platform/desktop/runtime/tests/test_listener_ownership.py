import os
import socket
import sys
from unittest.mock import Mock

import psutil
import pytest
from autogpt_desktop import ports, postgres, rabbitmq, valkey
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ManagedProcess, base_env, wait_until
from autogpt_desktop.supervisor import Stack, StartupError


@pytest.fixture
def stack(tmp_path):
    data = DataDir(tmp_path / "data")
    data.prepare()
    instance = Stack(Bundle(tmp_path / "runtime"), data)
    yield instance
    instance.stop()


@pytest.fixture
def listener():
    with socket.socket() as server:
        for port in ports.PORT_RANGE:
            try:
                server.bind(("127.0.0.1", port))
                break
            except OSError:
                continue
        else:
            pytest.fail("no free fixture port")
        server.listen()
        server.setblocking(False)
        yield server


def child(stack, name, code):
    proc = ManagedProcess(
        name,
        [sys.executable, "-c", code],
        base_env(),
        stack.data.root,
        stack.data.logs,
    )
    stack.launch([proc])
    return proc


@pytest.mark.parametrize("name", ["postgres", "valkey", "rabbitmq"])
def test_readiness_never_contacts_an_unowned_listener(stack, listener, name):
    port = listener.getsockname()[1]
    proc = child(stack, name, "import time; time.sleep(60)")
    proc.ready_port = port
    probes = {
        "postgres": lambda: postgres.is_ready(port, "fixture-password"),
        "valkey": lambda: valkey.is_ready(port, "fixture-password"),
        "rabbitmq": lambda: rabbitmq.is_ready(port, "fixture-user", "fixture-password"),
    }
    with pytest.raises(StartupError):
        stack.await_ready(proc, probes[name], timeout=0)
    with pytest.raises(BlockingIOError):
        listener.accept()


def test_cache_shutdown_does_not_authenticate_to_another_listener(
    stack, listener, monkeypatch
):
    cache = valkey.process(
        stack.bundle, stack.data, listener.getsockname()[1], "fixture-password"
    )
    cache.argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    stack.launch([cache])
    shutdown = Mock()
    monkeypatch.setattr(valkey, "_shutdown", shutdown)
    cache.stop_timeout = 0.1
    cache.stop()
    shutdown.assert_not_called()
    assert cache.exit_code() is not None


def test_remembered_foreign_port_is_replaced(stack, listener):
    port = listener.getsockname()[1]
    assert port in ports.PORT_RANGE
    stack.data.ports_file.write_text('{"postgres": %d}' % port)
    assert (
        ports.allocate(stack.data.ports_file, names=("postgres",))["postgres"] != port
    )


@pytest.mark.parametrize("wrapper", [False, True])
def test_owned_listener_and_script_child_are_accepted(stack, wrapper):
    address = stack.data.root / "address"
    program = (
        "import socket,time,pathlib; "
        "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
        f"pathlib.Path({str(address)!r}).write_text(str(s.getsockname()[1])); "
        "time.sleep(60)"
    )
    if wrapper:
        program = (
            f"import subprocess,sys; subprocess.run([sys.executable,'-c',{program!r}])"
        )
    proc = child(stack, "service", program)
    assert wait_until(lambda: address.exists() and address.stat().st_size > 0, 15)
    proc.ready_port = int(address.read_text())
    assert proc.owns_listener()
    probe = Mock(return_value=True)
    stack.await_ready(proc, probe, timeout=1)
    assert probe.called
    proc.stop()
    assert not proc.owns_listener()


def test_adopted_postgres_uses_server_identity_without_popen(stack, listener):
    proc = postgres.WindowsServer(
        "postgres",
        ["postgres", "-D", str(stack.data.postgres)],
        {},
        stack.data.postgres,
        stack.data.logs,
    )
    proc.server = psutil.Process(os.getpid())
    proc.ready_port = listener.getsockname()[1]
    assert proc.popen is None
    assert proc.owns_listener()


def test_connection_inspection_failure_is_not_readiness(stack, listener, monkeypatch):
    proc = child(stack, "service", "import time; time.sleep(60)")
    proc.ready_port = listener.getsockname()[1]
    monkeypatch.setattr(
        psutil.Process, "net_connections", Mock(side_effect=psutil.AccessDenied())
    )
    assert not proc.owns_listener()


def test_final_readiness_check_also_requires_ownership(stack, monkeypatch):
    proc = child(stack, "service", "import time; time.sleep(60)")
    proc.ready_port = 12345
    monkeypatch.setattr(proc, "owns_listener", Mock(side_effect=[True, False]))
    probe = Mock(return_value=True)
    with pytest.raises(StartupError):
        stack.await_ready(proc, probe, timeout=1)
    probe.assert_called_once()


def test_service_factories_configure_the_readiness_port(stack, monkeypatch):
    monkeypatch.setattr(rabbitmq, "environment", lambda *args: {})
    monkeypatch.setattr(rabbitmq, "_script", lambda *args: "rabbitmq-server")
    assert postgres.process(stack.bundle, stack.data, 19001).ready_port == 19001
    assert (
        valkey.process(stack.bundle, stack.data, 19002, "fixture-password").ready_port
        == 19002
    )
    assert (
        rabbitmq.process(stack.bundle, stack.data, {"rabbitmq": 19003}).ready_port
        == 19003
    )


def test_owned_cache_keeps_graceful_shutdown(stack, monkeypatch):
    cache = valkey.process(stack.bundle, stack.data, 19002, "fixture-password")
    cache.argv = [sys.executable, "-c", "import time; time.sleep(60)"]
    stack.launch([cache])
    monkeypatch.setattr(cache, "owns_listener", lambda: True)
    shutdown = Mock(side_effect=lambda *args: cache.popen.terminate())
    monkeypatch.setattr(valkey, "_shutdown", shutdown)
    cache.stop()
    shutdown.assert_called_once_with(19002, "fixture-password")
    assert cache.exit_code() is not None
