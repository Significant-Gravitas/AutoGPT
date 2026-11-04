"""SpinTestServer's service lifecycle: which service processes it starts,
and how start-up and the session's end fail when one of them dies."""

import os
import time
from collections.abc import Iterator
from multiprocessing import Process
from unittest.mock import AsyncMock

import pytest

from backend.api import rest_api
from backend.util.process import AppProcess
from backend.util.test import SpinTestServer


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


async def test_a_failed_start_up_stops_the_services_and_keeps_its_error(
    stand_in_services, mocker
):
    """Cleanup after a failed start-up can fail too. The services are still
    stopped, and the start-up error is the one raised."""
    _, stand_ins = stand_in_services
    mocker.patch(
        "backend.util.test.create_default_user",
        AsyncMock(side_effect=RuntimeError("start-up failed")),
    )
    mocker.patch(
        "backend.data.db.disconnect",
        AsyncMock(side_effect=RuntimeError("disconnect failed")),
    )

    with pytest.raises(RuntimeError, match="start-up failed"):
        async with SpinTestServer():
            pass

    assert len(stand_ins) == 4
    assert not any(process.is_alive() for process in stand_ins.values())


async def test_a_service_that_fails_to_stop_does_not_keep_the_others_running(
    stand_in_services, mocker
):
    _, stand_ins = stand_in_services
    real_stop = AppProcess.stop

    def stop(service: AppProcess) -> None:
        if service.service_name == "Scheduler":
            raise RuntimeError("Scheduler failed to stop")
        real_stop(service)

    mocker.patch.object(AppProcess, "stop", autospec=True, side_effect=stop)

    with pytest.raises(RuntimeError, match="Scheduler failed to stop"):
        async with SpinTestServer():
            pass

    others = [name for name in stand_ins if name != "Scheduler"]
    assert len(others) == 3
    assert not any(stand_ins[name].is_alive() for name in others)


async def test_a_dead_service_is_still_named_when_cleanup_fails(
    stand_in_services, mocker
):
    """A failing disconnect at the end must not hide the service that died:
    the dead service is the error raised, with the cleanup error as its
    cause."""
    _, stand_ins = stand_in_services
    mocker.patch(
        "backend.data.db.disconnect",
        AsyncMock(side_effect=RuntimeError("disconnect failed")),
    )

    with pytest.raises(RuntimeError, match="during the session") as reported:
        async with SpinTestServer():
            stand_ins["Scheduler"].kill()
            stand_ins["Scheduler"].join()

    assert "Scheduler (exit code -9)" in str(reported.value)
    assert str(reported.value.__cause__) == "disconnect failed"
    assert not any(process.is_alive() for process in stand_ins.values())


async def test_the_server_names_a_service_that_died_during_the_session(
    stand_in_services,
):
    """Without a preload (CI) the services take seconds to start, so one can
    still be starting when start-up finishes and die after it. The session's
    end names it, instead of the run passing with a dead service."""
    _, stand_ins = stand_in_services

    with pytest.raises(RuntimeError) as reported:
        async with SpinTestServer():
            stand_ins["Scheduler"].kill()
            stand_ins["Scheduler"].join()

    assert "during the session" in str(reported.value)
    assert "Scheduler (exit code -9)" in str(reported.value)
    assert "DatabaseManager" not in str(reported.value)
    assert not any(process.is_alive() for process in stand_ins.values())


async def test_a_service_that_dies_while_the_session_disconnects_is_named(
    stand_in_services, mocker
):
    """The services are checked after the disconnects, just before they are
    stopped, so one that dies while the session disconnects is named too."""
    _, stand_ins = stand_in_services

    def die_while_disconnecting() -> None:
        stand_ins["Scheduler"].kill()
        stand_ins["Scheduler"].join()

    mocker.patch(
        "backend.data.redis_client.disconnect_async",
        AsyncMock(side_effect=die_while_disconnecting),
    )

    with pytest.raises(RuntimeError, match=r"Scheduler \(exit code -9\)"):
        async with SpinTestServer():
            pass

    assert not any(process.is_alive() for process in stand_ins.values())


async def test_an_error_in_the_session_is_not_replaced_by_a_dead_service(
    stand_in_services,
):
    _, stand_ins = stand_in_services

    with pytest.raises(ValueError, match="session failed") as failure:
        async with SpinTestServer():
            stand_ins["Scheduler"].kill()
            stand_ins["Scheduler"].join()
            raise ValueError("session failed")

    assert "during the session" not in str(failure.value)
    assert not any(process.is_alive() for process in stand_ins.values())
