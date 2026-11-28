"""The REST server's probes: /health answers for the process alone, so a
database outage restarts no pod and empties no load balancer; /health/ready
answers for the database."""

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock

import fastapi.testclient
import pytest

from backend.api import rest_api
from backend.data import db

_HOST_PORT = "db.internal:5432"
_PROBE_ERROR = f"Can't reach database server at `{_HOST_PORT}`"
_UNAVAILABLE = {"status": "unhealthy", "detail": "database unavailable"}


@pytest.fixture
def client():
    with contextlib.closing(
        fastapi.testclient.TestClient(rest_api.app, raise_server_exceptions=False)
    ) as client:
        yield client


def _patch_probe(mocker, *, connected: bool = True, query=None):
    mocker.patch.object(db, "is_connected", return_value=connected)
    return mocker.patch.object(
        db,
        "query_raw_with_schema",
        query or AsyncMock(return_value=[{"health_check": 1}]),
    )


def test_health_answers_for_the_process_while_the_database_is_down(client, mocker):
    probe = _patch_probe(mocker, query=AsyncMock(side_effect=RuntimeError("down")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    probe.assert_not_called()


def test_health_is_a_503_when_the_client_is_not_connected(client, mocker):
    _patch_probe(mocker, connected=False)

    response = client.get("/health")

    assert response.status_code == 503


def test_ready_is_200_when_the_database_answers(client, mocker):
    probe = _patch_probe(mocker)

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    probe.assert_awaited_once()


@pytest.mark.parametrize(
    "connected, query",
    [
        (True, AsyncMock(side_effect=RuntimeError(_PROBE_ERROR))),
        (True, AsyncMock(return_value=[])),
        (True, AsyncMock(return_value=[{"health_check": 0}])),
        (False, AsyncMock(return_value=[{"health_check": 1}])),
    ],
    ids=["unreachable", "no-row", "wrong-row", "not-connected"],
)
def test_ready_is_a_503_that_names_no_host_when_the_database_fails(
    client, mocker, connected, query
):
    _patch_probe(mocker, connected=connected, query=query)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == _UNAVAILABLE
    assert _HOST_PORT not in response.text


def test_ready_logs_why_the_database_failed(client, mocker, caplog):
    _patch_probe(mocker, query=AsyncMock(side_effect=RuntimeError(_PROBE_ERROR)))

    with caplog.at_level(logging.WARNING, logger=rest_api.logger.name):
        client.get("/health/ready")

    assert any(_HOST_PORT in record.getMessage() for record in caplog.records)


def test_ready_is_a_503_when_the_database_outlasts_the_probe(client, mocker):
    mocker.patch.object(rest_api, "HEALTH_READY_DB_TIMEOUT_SECONDS", 0.05)

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    _patch_probe(mocker, query=hang)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == _UNAVAILABLE
