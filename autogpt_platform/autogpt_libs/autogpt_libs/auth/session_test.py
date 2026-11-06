from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Security
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from autogpt_libs.auth import session
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.session import requires_live_session


@pytest.fixture
def session_lookup(mocker: MockerFixture) -> AsyncMock:
    mocker.patch(
        "autogpt_libs.auth.dependencies._is_banned", AsyncMock(return_value=False)
    )
    return mocker.patch.object(
        session, "_is_session_live", AsyncMock(return_value=True)
    )


@pytest.mark.asyncio
async def test_a_live_session_passes_the_payload_through(session_lookup):
    payload = {"sub": "user-1", "sid": "session-1"}

    assert await requires_live_session(payload) is payload
    session_lookup.assert_awaited_once_with("session-1", "user-1")


@pytest.mark.asyncio
async def test_an_ended_session_is_refused(session_lookup):
    session_lookup.return_value = False

    with pytest.raises(HTTPException) as error:
        await requires_live_session({"sub": "user-1", "sid": "session-1"})

    assert error.value.status_code == 401
    assert error.value.detail["code"] == "session_revoked"


@pytest.mark.asyncio
async def test_a_token_without_a_subject_is_refused(session_lookup):
    with pytest.raises(HTTPException) as error:
        await requires_live_session({"sid": "session-1"})

    assert error.value.status_code == 401
    session_lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"sub": "user-1"}, {"sub": "user-1", "sid": ""}])
async def test_a_token_minted_before_sid_existed_passes(session_lookup, payload):
    assert await requires_live_session(payload) is payload
    session_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_lookup_answers_503_not_a_pass(session_lookup):
    session_lookup.side_effect = RuntimeError("database down")

    with pytest.raises(HTTPException) as error:
        await requires_live_session({"sub": "user-1", "sid": "session-1"})

    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_under_impersonation_the_session_checked_is_the_admins(
    session_lookup,
):
    """Header impersonation keeps the admin as ``sub``; the session is theirs."""
    await requires_live_session({"sub": "admin-1", "sid": "admin-session"})

    session_lookup.assert_awaited_once_with("admin-session", "admin-1")


@pytest.mark.asyncio
async def test_without_the_backend_the_check_answers_503_not_401(
    mocker: MockerFixture,
):
    mocker.patch(
        "autogpt_libs.auth.dependencies._is_banned", AsyncMock(return_value=False)
    )
    mocker.patch.dict("sys.modules", {"backend.data": None})

    with pytest.raises(HTTPException) as error:
        await requires_live_session({"sub": "user-1", "sid": "session-1"})

    assert error.value.status_code == 503


def test_a_route_behind_it_answers_401_once_the_session_ends(session_lookup):
    app = FastAPI()

    @app.post("/mint", dependencies=[Security(requires_live_session)])
    def mint():
        return {"ok": True}

    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": "user-1",
        "sid": "session-1",
    }
    client = TestClient(app)

    assert client.post("/mint").status_code == 200
    session_lookup.return_value = False
    assert client.post("/mint").status_code == 401
