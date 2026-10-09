import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from backend import conftest as fixtures

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
