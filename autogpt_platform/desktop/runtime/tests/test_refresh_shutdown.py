"""Refresh draining precedes service teardown without starting any processes."""

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from autogpt_desktop import apps, servicehost


class HostExit(BaseException):
    pass


@pytest.mark.parametrize("foreground", [False, True])
def test_host_drains_refreshes_before_uvicorn_and_cleanup(monkeypatch, foreground):
    events = []

    def drain(timeout):
        events.append("drain")
        assert timeout == 25
        return True

    class Server:
        @property
        def should_exit(self):
            return False

        @should_exit.setter
        def should_exit(self, _value):
            events.append("server_stop")

    class Cleaner:
        def __init__(self, target, args, **_kwargs):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

        def join(self, _timeout):
            pass

        def is_alive(self):
            return False

    def leave(_code):
        events.append("exit")
        raise HostExit()

    monkeypatch.setitem(
        sys.modules,
        "backend.integrations.credential_refresh",
        SimpleNamespace(drain_credential_refreshes=drain, REFRESH_DRAIN_SECONDS=25),
    )
    monkeypatch.setattr(servicehost.threading, "Thread", Cleaner)
    monkeypatch.setattr(servicehost, "wait_for", lambda *_args: True)
    monkeypatch.setattr(servicehost, "leave", leave)
    hosted = servicehost.Hosted(
        "worker", SimpleNamespace(cleanup=lambda: events.append("cleanup"))
    )
    hosted.thread = SimpleNamespace(is_alive=lambda: True)
    host = servicehost.Host(
        "test",
        [],
        stop_budget=29.5,
        foreground=foreground,
        services=[hosted],
        servers=[Server()],
    )
    with pytest.raises(HostExit):
        host.shut_down(0, "test")
    assert events == (
        ["drain", "exit"] if foreground else ["drain", "server_stop", "cleanup", "exit"]
    )


def test_only_refresh_capable_groups_get_extended_host_grace():
    assert apps.host_stop_timeout(apps.Group("database", ("database-manager",))) == 3
    assert apps.host_stop_timeout(apps.Group("notification", ("notification",))) == 3
    for name in (
        "executor",
        "copilot-executor",
        "scheduler",
        "rest",
        "websocket",
        "swap-credentials",
    ):
        assert apps.host_stop_timeout(apps.Group(name, (name,))) >= 30


def test_worker_drains_before_stopping_its_event_loop():
    root = Path(__file__).resolve().parents[3] / "backend/backend"
    path = root / "copilot/executor/processor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    processor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CoPilotProcessor"
    )
    method = next(
        node
        for node in processor.body
        if isinstance(node, ast.FunctionDef) and node.name == "cleanup"
    )
    events = []

    async def storage():
        pass

    def run_coroutine(coro, _loop):
        coro.close()
        events.append("storage")
        return SimpleNamespace(result=lambda **_: None)

    namespace = {
        "drain_credential_refreshes": lambda: events.append("drain") or True,
        "shutdown_workspace_storage": storage,
        "asyncio": SimpleNamespace(run_coroutine_threadsafe=run_coroutine),
        "logger": logging.getLogger("worker-drain-test"),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    loop = SimpleNamespace(
        stop=lambda: None, call_soon_threadsafe=lambda _: events.append("loop_stop")
    )
    processor = SimpleNamespace(
        tid=1,
        execution_loop=loop,
        execution_thread=SimpleNamespace(join=lambda **_: events.append("join")),
    )
    namespace["cleanup"](processor)
    assert events == ["drain", "storage", "loop_stop", "join"]


@pytest.mark.asyncio
async def test_pending_codex_connection_can_be_removed_without_refresh_or_logout():
    root = Path(__file__).resolve().parents[3] / "backend/backend"
    path = root / "api/features/integrations/codex.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "revoke_codex_credentials"
    )

    class Pending(RuntimeError):
        pass

    namespace = {
        "CredentialRefreshRequired": Pending,
        "IntegrationCredentialsManager": object,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    manager = SimpleNamespace(
        acquire_lease=AsyncMock(side_effect=Pending()),
        delete_pending_refresh=AsyncMock(),
    )
    assert (
        await namespace["revoke_codex_credentials"](manager, "user-a", "credential-a")
        is False
    )
    manager.delete_pending_refresh.assert_awaited_once_with(
        "user-a", "credential-a", "codex"
    )
