import asyncio
import socket
import threading

import pytest
from aiohttp import web
from autogpt_desktop import proxy


@pytest.mark.parametrize(
    "phase, cleanup_fails",
    [
        ("build", False),
        ("setup", False),
        ("setup", True),
        ("bind", False),
        ("bind", True),
    ],
)
def test_failed_start_cleans_up_in_its_thread(
    monkeypatch, caplog, phase, cleanup_fails
):
    original_setup = web.AppRunner.setup
    original_cleanup = web.AppRunner.cleanup
    startup_error = RuntimeError("injected startup failure")
    sessions = []
    cleanup_calls = []

    async def setup(runner):
        await original_setup(runner)
        sessions.append(runner.app[proxy.CLIENT])
        if phase == "setup":
            raise startup_error

    async def cleanup(runner):
        cleanup_calls.append((threading.current_thread(), asyncio.get_running_loop()))
        await original_cleanup(runner)
        if cleanup_fails:
            raise RuntimeError("injected cleanup failure")

    def fail_build(_upstreams):
        raise startup_error

    def reject_scheduling(coroutine, _loop):
        coroutine.close()
        pytest.fail("shutdown scheduled cleanup on a stopped event loop")

    monkeypatch.setattr(web.AppRunner, "setup", setup)
    monkeypatch.setattr(web.AppRunner, "cleanup", cleanup)
    if phase == "build":
        monkeypatch.setattr(proxy, "build_app", fail_build)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        service = proxy.ProxyThread(proxy.Upstreams("", "", "", ""), port)
        try:
            with pytest.raises(RuntimeError, match="the local web server failed"):
                service.start()
            monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", reject_scheduling)
            service.stop()
            service.stop()
            assert not service.thread.is_alive()
            assert service.loop.is_closed()
            assert all(session.closed for session in sessions)
            if phase == "bind":
                assert isinstance(service.failure, OSError)
            else:
                assert service.failure is startup_error
            assert cleanup_calls == (
                [] if phase == "build" else [(service.thread, service.loop)]
            )
            if cleanup_fails:
                assert "injected cleanup failure" in caplog.text
        finally:
            service.thread.join(2)
            assert not service.thread.is_alive()
            if not service.loop.is_closed():
                if service.runner is not None:
                    service.loop.run_until_complete(original_cleanup(service.runner))
                service.loop.close()
