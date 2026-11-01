import asyncio
from unittest.mock import AsyncMock

import fastapi.testclient

from backend.api import rest_api
from backend.data import db

client = fastapi.testclient.TestClient(rest_api.app, raise_server_exceptions=False)

_DOWN_HOST_PORT = "127.0.0.1:5432"
_PROBE_ERROR = f"Can't reach database server at `{_DOWN_HOST_PORT}`"


def _patch_probe(mocker, *, connected=True, query=None):
    mocker.patch.object(db, "is_connected", return_value=connected)
    mocker.patch.object(
        db,
        "query_raw_with_schema",
        query if query is not None else AsyncMock(return_value=[{"health_check": 1}]),
    )


def test_health_is_200_when_the_probe_succeeds(mocker):
    _patch_probe(mocker)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_is_503_when_the_probe_raises_and_hides_the_host(mocker):
    _patch_probe(mocker, query=AsyncMock(side_effect=RuntimeError(_PROBE_ERROR)))

    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body == {"status": "unhealthy", "detail": "database unavailable"}
    assert _DOWN_HOST_PORT not in response.text
    assert "127.0.0.1" not in response.text


def test_health_is_503_when_the_probe_hangs_past_the_timeout(mocker):
    mocker.patch.object(rest_api, "HEALTH_DB_TIMEOUT_SECONDS", 0.05)

    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    _patch_probe(mocker, query=hang)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "detail": "database unavailable",
    }


def test_health_is_503_when_the_client_is_not_connected(mocker):
    _patch_probe(mocker, connected=False)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "detail": "database unavailable",
    }
    db.query_raw_with_schema.assert_not_called()


def test_health_live_is_200_even_when_the_probe_raises(mocker):
    _patch_probe(mocker, query=AsyncMock(side_effect=RuntimeError(_PROBE_ERROR)))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    db.query_raw_with_schema.assert_not_called()
