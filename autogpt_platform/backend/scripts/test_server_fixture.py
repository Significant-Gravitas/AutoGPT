import os
import subprocess
import sys
import time
from collections.abc import Iterator
from multiprocessing import Process
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend import conftest as fixtures
from backend.api import rest_api
from backend.util.exceptions import DatabaseError
from backend.util.process import AppProcess
from backend.util.test import SpinTestServer

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "names,ci,collect_only,methods",
    [
        ([], False, False, ["forkserver"]),
        (["server"], True, False, ["forkserver"]),
        (["server"], False, True, ["forkserver"]),
        (["server"], False, False, ["spawn"]),
    ],
)
def test_preload_only_for_local_service_tests(
    monkeypatch, names, ci, collect_only, methods
):
    monkeypatch.delenv("CI", raising=False)
    if ci:
        monkeypatch.setenv("CI", "true")
    item = Mock(spec=pytest.Function)
    item.config = Mock()
    item.fixturenames = names
    item.config.getoption.return_value = collect_only
    with (
        patch.object(fixtures, "_preload_started", False),
        patch.object(fixtures, "get_all_start_methods", return_value=methods),
        patch.object(fixtures, "Thread") as thread,
    ):
        fixtures.pytest_itemcollected(item)
    thread.assert_not_called()


def test_service_collection_starts_only_one_preload(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    item = Mock(spec=pytest.Function)
    item.config = Mock()
    item.fixturenames = ["server"]
    item.config.getoption.return_value = False
    with (
        patch.object(fixtures, "_preload_started", False),
        patch.object(fixtures, "get_all_start_methods", return_value=["forkserver"]),
        patch.object(fixtures, "set_forkserver_preload") as configure,
        patch.object(fixtures, "Thread") as thread,
    ):
        fixtures.pytest_itemcollected(item)
        fixtures.pytest_itemcollected(item)
    configure.assert_called_once_with(["scripts.server_preload"])
    thread.return_value.start.assert_called_once_with()


@pytest.mark.parametrize("failure", ["configure", "start"])
def test_preload_setup_failure_does_not_mark_it_started(monkeypatch, failure):
    monkeypatch.delenv("CI", raising=False)
    item = Mock(spec=pytest.Function)
    item.config = Mock()
    item.fixturenames = ["server"]
    item.config.getoption.return_value = False
    with (
        patch.object(fixtures, "_preload_started", False),
        patch.object(fixtures, "get_all_start_methods", return_value=["forkserver"]),
        patch.object(fixtures, "set_forkserver_preload") as configure,
        patch.object(fixtures, "Thread") as thread,
    ):
        operation = configure if failure == "configure" else thread.return_value.start
        operation.side_effect = RuntimeError("preload setup failed")
        with pytest.raises(RuntimeError, match="preload setup failed"):
            fixtures.pytest_itemcollected(item)
        assert not fixtures._preload_started
        operation.side_effect = None
        fixtures.pytest_itemcollected(item)
        assert fixtures._preload_started


def test_server_starts_on_demand_and_cleans_up(pytester: pytest.Pytester):
    events = pytester.path / "events.txt"
    backend_dir = Path(__file__).resolve().parents[1]
    pytester.makeconftest(
        f"""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, {str(backend_dir)!r})
from backend.conftest import *
import backend.conftest as fixtures

events = Path({str(events)!r})

def record(message):
    with events.open("a") as stream:
        stream.write(message + "\\n")

class FakeServer:
    def __init__(self):
        self.agent_server = self

    async def __aenter__(self):
        record("start")
        return self

    async def __aexit__(self, *args):
        record("stop")

    async def test_create_graph(self, graph, user_id):
        return SimpleNamespace(id="graph", sub_graphs=[])

    async def test_create_store_listing(self):
        return SimpleNamespace(listing_id="listing")

async def cleanup(server, graphs, listings):
    assert graphs == [("graph", "user")]
    assert listings == ["listing"]
    record("cleanup")

module = ModuleType("backend.util.test")
module.SpinTestServer = FakeServer
sys.modules["backend.util.test"] = module
fixtures._delete_test_graphs = cleanup
fixtures._warm_test_forkserver = lambda: None
"""
    )
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    pytester.makepyfile(
        f"""
from pathlib import Path

def test_01_pure_unit_test():
    assert not Path({str(events)!r}).exists()

async def test_02_server_test(server):
    await server.agent_server.test_create_graph({{}}, user_id="user")
    await server.agent_server.test_create_store_listing()
"""
    )

    result = pytester.runpytest_subprocess("-q", "-p", "no:syrupy")

    result.assert_outcomes(passed=2)
    assert events.read_text().splitlines() == ["start", "cleanup", "stop"]


def test_preloaded_services_have_no_background_threads_or_connections():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import gc
import os
import sys
import threading

os.environ.pop("PYTEST_CURRENT_TEST", None)
import scripts.server_preload
from backend.data import db

assert "pytest" in sys.modules
assert threading.active_count() == 1, threading.enumerate()
assert not db.is_connected()
assert gc.get_freeze_count() > 0
""",
        ],
        check=True,
    )


async def test_a_fixture_user_is_retried_once_through_the_uncached_lookup():
    create = AsyncMock(side_effect=[DatabaseError("Event loop is closed"), None])
    with patch("backend.data.user.get_or_create_user_with_status", create):
        await fixtures._create_user_with_loop_retry({"sub": "user"})
    assert create.await_count == 2


@pytest.fixture
def stand_in_services(mocker) -> Iterator[tuple[dict[str, int], dict[str, Process]]]:
    """Have ``SpinTestServer`` start a stand-in process for each service, and
    skip its database and Redis work. A service named in the yielded exit
    codes gets a stand-in that exits at once with that code; the others sleep
    until stopped. Also yields the stand-ins, by service name."""
    exit_codes: dict[str, int] = {}
    stand_ins: dict[str, Process] = {}

    def start(service: AppProcess, **kwargs) -> int:
        exit_code = exit_codes.get(service.service_name)
        if exit_code is None:
            process = Process(target=time.sleep, args=(600,), daemon=True)
            process.start()
        else:
            process = Process(target=os._exit, args=(exit_code,), daemon=True)
            process.start()
            process.join()
        service.process = stand_ins[service.service_name] = process
        return process.pid or 0

    mocker.patch.object(AppProcess, "start", autospec=True, side_effect=start)
    mocker.patch.dict(rest_api.app.dependency_overrides)
    mocker.patch("backend.util.test.initialize_blocks", AsyncMock())
    mocker.patch("backend.util.test.create_default_user", AsyncMock())
    mocker.patch("backend.data.db.connect", AsyncMock())
    mocker.patch("backend.data.db.disconnect", AsyncMock())
    mocker.patch("backend.data.redis_client.disconnect_async", AsyncMock())
    yield exit_codes, stand_ins
    for process in stand_ins.values():
        process.kill()
        process.join()


async def test_the_server_starts_every_service_but_the_rest_api(stand_in_services):
    """Tests call the REST API in process, so its own server process would
    only run start-up work (data migrations included) against the test
    database."""
    _, stand_ins = stand_in_services

    async with SpinTestServer():
        assert sorted(stand_ins) == [
            "DatabaseManager",
            "ExecutionManager",
            "NotificationManager",
            "Scheduler",
        ]

    assert not any(process.is_alive() for process in stand_ins.values())


async def test_the_server_refuses_a_service_that_exited_during_start_up(
    stand_in_services,
):
    exit_codes, stand_ins = stand_in_services
    exit_codes["Scheduler"] = 3

    with pytest.raises(RuntimeError) as refused:
        async with SpinTestServer():
            pass

    assert "Scheduler (exit code 3)" in str(refused.value)
    assert "DatabaseManager" not in str(refused.value)
    # The others are stopped too: one left running would hang the session
    # at exit, which joins every child process.
    assert len(stand_ins) > 1
    assert not any(process.is_alive() for process in stand_ins.values())
