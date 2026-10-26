"""Exercise the bundled MSYS2 binary without ever launching its default config."""

import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from autogpt_desktop import ports, valkey
from autogpt_desktop import process as process_module
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ManagedProcess, wait_until

DESKTOP = Path(__file__).resolve().parents[2]
PROFILES = ["Nick", "John Smith", "O'Brien", "Jürgen", "O''Brien", "O'Brien's"]


@pytest.fixture(params=PROFILES)
def installation(tmp_path, request):
    if sys.platform != "win32":
        pytest.skip("MSYS2 Windows binary")
    configured = os.environ.get("AUTOGPT_DESKTOP_RUNTIME")
    source = Bundle(Path(configured) if configured else DESKTOP / "build/runtime")
    if not source.valkey_server.is_file():
        pytest.skip("no assembled Windows Valkey bundle")
    bundle = Bundle(tmp_path / request.param / "runtime")
    bundle.valkey_server.parent.mkdir(parents=True)
    shutil.copy2(source.valkey_server, bundle.valkey_server)
    for library in source.valkey_server.parent.glob("*.dll"):
        shutil.copy2(library, bundle.valkey_server.parent / library.name)
    checker = bundle.valkey_server.with_name("valkey-check-rdb.exe")
    shutil.copy2(bundle.valkey_server, checker)
    data = DataDir(tmp_path / request.param / "data")
    data.prepare()
    return bundle, checker, data


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("quote", [False, True])
def test_executable_quoting_is_opt_in_and_windows_only(
    tmp_path, monkeypatch, windows, quote
):
    captured = []
    monkeypatch.setattr(process_module, "WINDOWS", windows)
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda command, **kwargs: captured.append(command),
    )
    argv = [r"C:\O'Brien\valkey-server.exe", "valkey.conf"]
    child = ManagedProcess(
        name="probe",
        argv=argv,
        env={},
        cwd=tmp_path,
        log_dir=tmp_path,
        windows_quote_executable=quote,
    )
    child.start()
    expected = f'"{argv[0]}" valkey.conf' if windows and quote else argv
    assert captured == [expected]


def assert_arguments_survive(checker, data, monkeypatch):
    # The same binary dispatches this name to a checker that always exits,
    # even if MSYS2 swallows every argument. A server --version probe would
    # instead start an unconfigured network listener when that flag is lost.
    with monkeypatch.context() as patch:
        patch.setattr(valkey, "CONFIG_NAME", "--version")
        child = valkey.process(SimpleNamespace(valkey_server=checker), data, 1, "")
    child.start()
    try:
        assert child.popen.wait(timeout=10) == 0
        assert "valkey-check-rdb v=" in (data.logs / "valkey.log").read_text()
    finally:
        child.stop()


def test_windows_valkey_preserves_arguments(installation, monkeypatch):
    _, checker, data = installation
    assert_arguments_survive(checker, data, monkeypatch)


def test_windows_valkey_loads_its_loopback_configuration(installation, monkeypatch):
    psutil = pytest.importorskip("psutil")
    redis = pytest.importorskip("redis")
    bundle, checker, data = installation
    assert_arguments_survive(checker, data, monkeypatch)
    port = ports._free_port(exclude=set())
    bus = ports._free_port(exclude={port})
    password = "valkey-path-regression-only"
    valkey.write_config(data, port, bus, password)
    child = valkey.process(bundle, data, port, password)
    child.start()
    try:
        assert wait_until(lambda: valkey.is_ready(port, password), 20, 0.1)
        client = redis.Redis(
            host="127.0.0.1",
            port=port,
            password=password,
            socket_timeout=3,
            decode_responses=True,
        )
        assert client.config_get("port")["port"] == str(port)
        assert client.config_get("bind")["bind"] == "127.0.0.1"
        assert client.config_get("cluster-enabled")["cluster-enabled"] == "yes"
        listeners = {
            (connection.laddr.ip, connection.laddr.port)
            for connection in psutil.Process(child.pid).net_connections(kind="inet")
            if connection.status == psutil.CONN_LISTEN
        }
        assert listeners == {("127.0.0.1", port), ("127.0.0.1", bus)}
    finally:
        child.stop()
    assert child.exit_code() == 0
    assert not psutil.pid_exists(child.pid)
